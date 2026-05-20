import copy
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from core.frontend.graph import FrontendGraph
from core.state import parse_end_node_to_output, parse_input_to_state
from utils.logger import logger


RUNNING = "running"
PAUSED = "paused"
COMPLETED = "completed"
STOPPED = "stopped"
FAILED = "failed"


TERMINAL_STATUSES = {COMPLETED, STOPPED, FAILED}
LANGGRAPH_TASK_NAME_PATTERN = re.compile(r"During task with name ['\"]([^'\"]+)['\"]")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FlowRun:
    run_id: str
    thread_id: str
    flow_id: str
    inputs: Dict[str, Any]
    graph: FrontendGraph
    state_graph: Any
    config: Dict[str, Any]
    status: str = RUNNING
    pause_requested: bool = False
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    pending_patch: Dict[str, Any] = field(default_factory=dict)
    current_state: Optional[Dict[str, Any]] = None
    current_checkpoint_config: Optional[Dict[str, Any]] = None
    next_nodes: list = field(default_factory=list)
    active_nodes: list = field(default_factory=list)
    failed_node: Optional[str] = None
    error_type: Optional[str] = None
    started_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    finished_at: Optional[str] = None
    events: list = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    worker: Optional[threading.Thread] = None


class FlowRunManager:
    def __init__(self):
        self._runs: Dict[str, FlowRun] = {}
        self._lock = threading.RLock()
        self._checkpointer = InMemorySaver()

    def create_run(self, flow_id: str, graph_data: Dict[str, Any], inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        run_id = str(uuid4())
        graph = FrontendGraph.from_payload(graph_data)
        state_graph = graph.compile_graph(
            checkpointer_type="memory",
            checkpointer=self._checkpointer,
            interrupt_before=graph.get_interrupt_node_names(),
        )
        config = copy.deepcopy(graph.config)
        config["configurable"]["thread_id"] = run_id

        state = copy.deepcopy(graph.state)
        request_inputs = inputs or {}
        state = parse_input_to_state(request_inputs, state, start_node=graph.get_start_node())

        run = FlowRun(
            run_id=run_id,
            thread_id=run_id,
            flow_id=str(flow_id),
            inputs=request_inputs,
            graph=graph,
            state_graph=state_graph,
            config=config,
        )
        with self._lock:
            self._runs[run_id] = run

        with run.lock:
            self._add_event(run, "created", "Run created", {"inputs": request_inputs})
        self._start_worker(run, state)
        return self.serialize_run(run)

    def get_run(self, run_id: str) -> Dict[str, Any]:
        return self.serialize_run(self._get_run(run_id))

    def pause_run(self, run_id: str) -> Dict[str, Any]:
        run = self._get_run(run_id)
        with run.lock:
            if run.status in TERMINAL_STATUSES:
                return self.serialize_run(run)
            run.pause_requested = True
            self._add_event(run, "pause_requested", "Pause requested; run will stop before the next node")
            self._refresh_state(run)
            if run.next_nodes and not (run.worker and run.worker.is_alive()):
                run.status = PAUSED
                self._add_event(run, "paused", "Run paused before next node", {"next_nodes": run.next_nodes})
            else:
                run.status = RUNNING
        return self.serialize_run(run)

    def patch_run(self, run_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        run = self._get_run(run_id)
        with run.lock:
            if run.status == STOPPED:
                raise ValueError("Stopped run cannot be patched")
            if not isinstance(patch, dict):
                raise ValueError("Patch must be a JSON object")
            run.pending_patch = self._deep_merge(run.pending_patch, patch)
            self._add_event(run, "patch_queued", "Patch queued", {"patch": patch})
        return self.serialize_run(run)

    def resume_run(self, run_id: str, patch: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        run = self._get_run(run_id)
        with run.lock:
            if run.status == STOPPED:
                raise ValueError("Stopped run cannot be resumed")
            if run.status == COMPLETED:
                return self.serialize_run(run)
            if run.status == FAILED:
                raise ValueError("Failed run cannot be resumed")
            if run.status != PAUSED:
                raise ValueError("Only paused runs can be resumed")
            if patch:
                run.pending_patch = self._deep_merge(run.pending_patch, patch)
            self._apply_pending_patch(run)
            run.pause_requested = False
            run.status = RUNNING
            self._add_event(run, "resumed", "Run resumed", {"next_nodes": run.next_nodes})

        self._start_worker(run, None)
        return self.serialize_run(run)

    def stop_run(self, run_id: str) -> Dict[str, Any]:
        run = self._get_run(run_id)
        with run.lock:
            if run.status != COMPLETED:
                run.status = STOPPED
                run.finished_at = run.finished_at or _now_iso()
                self._add_event(run, "stopped", "Run stopped by user")
        return self.serialize_run(run)

    def run_process_compat(self, flow_id: str, graph_data: Dict[str, Any], inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        run_data = self.create_run(flow_id, graph_data, inputs)
        run = self._get_run(run_data["run_id"])
        if run.worker:
            run.worker.join()
        if run.status == PAUSED:
            self.resume_run(run.run_id)
            if run.worker:
                run.worker.join()
        if run.status == FAILED:
            raise ValueError(run.error)
        return run.result or {}

    def _start_worker(self, run: FlowRun, graph_input: Any):
        with run.lock:
            if run.worker and run.worker.is_alive():
                return
            self._add_event(run, "worker_started", "Run worker started")
            worker = threading.Thread(
                target=self._run_graph,
                args=(run.run_id, graph_input),
                name=f"flow-run-{run.run_id}",
                daemon=True,
            )
            run.worker = worker
            worker.start()

    def _run_graph(self, run_id: str, graph_input: Any):
        run = self._get_run(run_id)
        try:
            current_input = graph_input
            while True:
                with run.lock:
                    self._refresh_state(run)
                    run.active_nodes = list(run.next_nodes)
                result = run.state_graph.invoke(input=current_input, config=run.config)
                with run.lock:
                    if run.status == STOPPED:
                        self._refresh_state(run)
                        run.active_nodes = []
                        return

                    self._refresh_state(run)
                    if run.next_nodes:
                        run.active_nodes = []
                        self._add_event(run, "checkpoint", "Reached node boundary", {"next_nodes": run.next_nodes})
                        if run.pause_requested:
                            run.pause_requested = False
                            run.status = PAUSED
                            self._add_event(run, "paused", "Run paused before next node", {"next_nodes": run.next_nodes})
                            return
                        current_input = None
                        continue

                    run.current_state = result
                    run.result = parse_end_node_to_output(result)
                    run.status = COMPLETED
                    run.active_nodes = []
                    run.finished_at = run.finished_at or _now_iso()
                    self._add_event(run, "completed", "Run completed", {"result": run.result})
                    return
        except Exception as exc:
            logger.exception(f"Flow run {run_id} failed")
            with run.lock:
                if run.status != STOPPED:
                    self._refresh_state(run)
                    run.status = FAILED
                    run.failed_node = self._infer_failed_node(exc, run)
                    run.error_type = type(exc).__name__
                    run.error = str(exc)
                    run.active_nodes = []
                    run.finished_at = run.finished_at or _now_iso()
                    self._add_event(run, "failed", "Run failed", {
                        "node": run.failed_node,
                        "error_type": run.error_type,
                        "error": run.error,
                        "next_nodes": run.next_nodes,
                    })

    def _apply_pending_patch(self, run: FlowRun):
        if not run.pending_patch:
            return

        patch = run.pending_patch
        if "inputs" in patch:
            if not isinstance(patch["inputs"], dict):
                raise ValueError("Patch inputs must be a JSON object")
            run.inputs = self._deep_merge(run.inputs, patch["inputs"])
            state_patch = self._deep_merge(patch.get("state", {}), {
                "fields": self._input_patch_to_field_patch(run, patch["inputs"])
            })
            patch = self._deep_merge(patch, {"state": state_patch})

        if "config" in patch:
            config_patch = patch["config"]
            if not isinstance(config_patch, dict):
                raise ValueError("Patch config must be a JSON object")
            run.config = self._deep_merge(run.config, config_patch)

        if "configurable" in patch:
            configurable_patch = patch["configurable"]
            if not isinstance(configurable_patch, dict):
                raise ValueError("Patch configurable must be a JSON object")
            run.config["configurable"] = self._deep_merge(run.config.get("configurable", {}), configurable_patch)

        state_patch = patch.get("state", {})
        if "fields" in patch:
            state_patch = self._deep_merge(state_patch, {"fields": patch["fields"]})

        if state_patch:
            if not isinstance(state_patch, dict):
                raise ValueError("Patch state must be a JSON object")
            current_state = self._get_state_values(run)
            next_state = copy.deepcopy(current_state)
            if "fields" in state_patch:
                self._apply_field_patch(next_state, state_patch["fields"])
                state_patch = {k: v for k, v in state_patch.items() if k != "fields"}
            next_state = self._deep_merge(next_state, state_patch)
            update_config = run.current_checkpoint_config or run.config
            run.config = run.state_graph.update_state(update_config, next_state)
            run.current_state = next_state
            self._add_event(run, "state_updated", "Run state updated from patch", {"state": next_state})

        run.pending_patch = {}

    def _apply_field_patch(self, state: Dict[str, Any], fields_patch: Dict[str, Any]):
        if not isinstance(fields_patch, dict):
            raise ValueError("Patch fields must be a JSON object")
        state.setdefault("fields", {})
        for field_name, patch_value in fields_patch.items():
            if isinstance(patch_value, dict) and "field_value" in patch_value:
                value = patch_value["field_value"]
            else:
                value = patch_value

            field = state["fields"].get(field_name)
            if hasattr(field, "field_value"):
                field.field_value = value
            elif isinstance(field, dict):
                field["field_value"] = value
            else:
                state["fields"][field_name] = value

    def _input_patch_to_field_patch(self, run: FlowRun, inputs_patch: Dict[str, Any]) -> Dict[str, Any]:
        field_patch = {}
        start_node = run.graph.get_start_node()
        for key, value in inputs_patch.items():
            field_patch[key] = value
            if start_node:
                field_patch[f"{start_node.name}/{key}"] = value
        return field_patch

    def _refresh_state(self, run: FlowRun):
        try:
            state_snapshot = run.state_graph.get_state(run.config)
            run.current_state = getattr(state_snapshot, "values", None) or run.current_state or {}
            run.current_checkpoint_config = getattr(state_snapshot, "config", None) or run.current_checkpoint_config or run.config
            run.next_nodes = list(getattr(state_snapshot, "next", None) or [])
        except Exception:
            run.current_state = run.current_state or {}
            run.current_checkpoint_config = run.current_checkpoint_config or run.config
            run.next_nodes = []

    def _get_state_values(self, run: FlowRun) -> Dict[str, Any]:
        self._refresh_state(run)
        return copy.deepcopy(run.current_state or {})

    def _get_run(self, run_id: str) -> FlowRun:
        with self._lock:
            run = self._runs.get(run_id)
        if not run:
            raise KeyError(f"Run not found: {run_id}")
        return run

    def serialize_run(self, run: FlowRun) -> Dict[str, Any]:
        with run.lock:
            self._refresh_state(run)
            return {
                "run_id": run.run_id,
                "thread_id": run.thread_id,
                "flow_id": run.flow_id,
                "status": run.status,
                "started_at": run.started_at,
                "updated_at": run.updated_at,
                "finished_at": run.finished_at,
                "inputs": run.inputs,
                "state": self._jsonable_state(run.current_state),
                "configurable": self._jsonable_state(run.config.get("configurable", {})),
                "next_nodes": run.next_nodes,
                "active_nodes": run.active_nodes,
                "failed_node": run.failed_node,
                "result": self._jsonable_state(run.result),
                "error": run.error,
                "error_type": run.error_type,
                "pending_patch": self._jsonable_state(run.pending_patch),
                "events": self._jsonable_state(run.events),
            }

    def _infer_failed_node(self, exc: Exception, run: FlowRun) -> Optional[str]:
        for text in self._iter_exception_text(exc):
            match = LANGGRAPH_TASK_NAME_PATTERN.search(text)
            if match:
                return match.group(1)
        if run.active_nodes:
            return run.active_nodes[0]
        if run.next_nodes:
            return run.next_nodes[0]
        return None

    def _iter_exception_text(self, exc: Exception):
        seen = set()
        stack = [exc]
        while stack:
            current = stack.pop()
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            yield str(current)
            yield repr(current)
            for note in getattr(current, "__notes__", []) or []:
                yield str(note)
            stack.append(getattr(current, "__cause__", None))
            stack.append(getattr(current, "__context__", None))

    def _jsonable_state(self, value: Any):
        if isinstance(value, dict):
            return {k: self._jsonable_state(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._jsonable_state(item) for item in value]
        if hasattr(value, "dict"):
            return value.dict()
        if hasattr(value, "content"):
            return value.content
        return value

    def _deep_merge(self, base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(base)
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def _add_event(self, run: FlowRun, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        run.updated_at = _now_iso()
        run.events.append({
            "time": run.updated_at,
            "type": event_type,
            "message": message,
            "data": self._jsonable_state(data or {}),
        })
        if len(run.events) > 200:
            run.events = run.events[-200:]


flow_run_manager = FlowRunManager()
