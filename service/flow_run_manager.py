import copy
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from core.frontend.graph import FrontendGraph
from core.state import StateField, parse_end_node_to_output, parse_input_to_state
from utils.logger import logger


RUNNING = "running"
PAUSED = "paused"
COMPLETED = "completed"
STOPPED = "stopped"
FAILED = "failed"


TERMINAL_STATUSES = {COMPLETED, STOPPED, FAILED}
LANGGRAPH_TASK_NAME_PATTERN = re.compile(r"During task with name ['\"]([^'\"]+)['\"]")
MASKED_SECRET = "********"
SECRET_KEYWORDS = (
    "api_key",
    "apikey",
    "access_key",
    "secret",
    "token",
    "password",
    "authorization",
)
CHECKPOINT_CONFIG_KEYS = ("checkpoint_id", "checkpoint_ns")


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
    rollback_state: Optional[Dict[str, Any]] = None
    rollback_checkpoint_config: Optional[Dict[str, Any]] = None
    rollback_next_nodes: list = field(default_factory=list)
    discard_worker_result: bool = False
    resume_from_checkpoint: bool = False
    failed_node: Optional[str] = None
    error_type: Optional[str] = None
    started_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    finished_at: Optional[str] = None
    events: list = field(default_factory=list)
    boundary_visits: Dict[str, int] = field(default_factory=dict)
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
            worker_alive = bool(run.worker and run.worker.is_alive())
            if worker_alive:
                run.discard_worker_result = True
                self._restore_pause_checkpoint(run)
                run.status = PAUSED
                self._add_event(run, "paused", "Run paused before the active node; in-flight result will be discarded", {
                    "active_nodes": run.active_nodes,
                    "next_nodes": run.next_nodes,
                })
            else:
                self._refresh_state(run)
                run.status = PAUSED
                run.pause_requested = False
                self._add_event(run, "paused", "Run paused before next node", {"next_nodes": run.next_nodes})
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
            if run.worker and run.worker.is_alive():
                raise ValueError("当前节点的后台调用还在返回中，结果会被丢弃；请稍后再继续运行")
            if patch:
                run.pending_patch = self._deep_merge(run.pending_patch, patch)
            self._validate_pending_patch(run)
            self._apply_pending_patch(run)
            run.pause_requested = False
            run.discard_worker_result = False
            run.boundary_visits = {}
            if self._is_end_boundary(run):
                self._complete_run(run, run.current_state, "Run completed from end checkpoint")
                return self.serialize_run(run)
            run.status = RUNNING
            run.resume_from_checkpoint = bool(run.current_checkpoint_config and run.next_nodes)
            self._add_event(run, "resumed", "Run resumed", {"next_nodes": run.next_nodes})

        self._start_worker(run, None)
        return self.serialize_run(run)

    def stop_run(self, run_id: str) -> Dict[str, Any]:
        run = self._get_run(run_id)
        with run.lock:
            if run.status != COMPLETED:
                run.status = STOPPED
                run.active_nodes = []
                run.discard_worker_result = True
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
                should_complete = False
                invoke_config = None
                with run.lock:
                    self._ensure_runtime_config(run)
                    if current_input is not None or not (run.current_checkpoint_config and run.next_nodes):
                        self._refresh_state(run)
                    if self._is_end_boundary(run):
                        self._complete_run(run, run.current_state, "Run completed at end checkpoint")
                        should_complete = True
                    run.active_nodes = list(run.next_nodes)
                    self._capture_rollback_checkpoint(run)
                    invoke_config = self._build_invoke_config(run, use_checkpoint=current_input is None)
                if should_complete:
                    return
                result = run.state_graph.invoke(input=current_input, config=invoke_config)
                with run.lock:
                    if run.status == STOPPED:
                        self._refresh_state(run)
                        run.active_nodes = []
                        return
                    if run.discard_worker_result:
                        self._discard_to_pause_checkpoint(run)
                        run.pause_requested = False
                        run.discard_worker_result = False
                        run.resume_from_checkpoint = False
                        self._add_event(run, "worker_discarded", "In-flight node result discarded after pause", {
                            "next_nodes": run.next_nodes,
                        })
                        return

                    self._refresh_state(run)
                    run.resume_from_checkpoint = False
                    if run.next_nodes:
                        if self._is_end_boundary(run):
                            self._complete_run(run, result or run.current_state, "Run completed at end boundary")
                            return
                        if not self._record_boundary_visit(run):
                            run.status = FAILED
                            run.error_type = "RuntimeError"
                            run.error = f"Run did not advance past checkpoint before {run.next_nodes}"
                            run.failed_node = run.next_nodes[0] if run.next_nodes else None
                            run.active_nodes = []
                            run.finished_at = run.finished_at or _now_iso()
                            self._add_event(run, "failed", "Run stopped because checkpoint did not advance", {
                                "node": run.failed_node,
                                "next_nodes": run.next_nodes,
                                "error": run.error,
                            })
                            return
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
                    self._complete_run(run, result, "Run completed")
                    return
        except Exception as exc:
            logger.exception(f"Flow run {run_id} failed")
            with run.lock:
                if run.discard_worker_result and run.status == PAUSED:
                    self._discard_to_pause_checkpoint(run)
                    run.pause_requested = False
                    run.discard_worker_result = False
                    run.resume_from_checkpoint = False
                    self._add_event(run, "worker_discarded", "In-flight node error discarded after pause", {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "next_nodes": run.next_nodes,
                    })
                    return
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
        self._ensure_runtime_config(run)
        input_field_patch = None
        if "inputs" in patch:
            if not isinstance(patch["inputs"], dict):
                raise ValueError("Patch inputs must be a JSON object")
            run.inputs = self._deep_merge(run.inputs, patch["inputs"])
            input_field_patch = self._input_patch_to_field_patch(run, patch["inputs"])

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
        if input_field_patch:
            state_patch = self._deep_merge(state_patch, {"fields": input_field_patch})

        if state_patch:
            if not isinstance(state_patch, dict):
                raise ValueError("Patch state must be a JSON object")
            current_state = self._get_state_values(run)
            next_state = copy.deepcopy(current_state)
            protected_fields = set((state_patch.get("fields") or {}).keys())
            if "fields" in state_patch:
                self._apply_field_patch(next_state, state_patch["fields"])
                state_patch = {k: v for k, v in state_patch.items() if k != "fields"}
            next_state = self._deep_merge(next_state, state_patch)
            self._propagate_field_relations(next_state, protected_fields)
            update_config = run.current_checkpoint_config or run.config
            checkpoint_config = run.state_graph.update_state(
                update_config,
                next_state,
                as_node=self._get_checkpoint_as_node(run),
            )
            run.current_checkpoint_config = checkpoint_config
            run.config = self._strip_checkpoint_config(run.config)
            self._ensure_runtime_config(run)
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
            if self._is_secret_key(field_name) and self._is_masked_secret_value(value):
                continue

            field = state["fields"].get(field_name)
            if hasattr(field, "field_value"):
                field.field_value = value
            elif isinstance(field, dict):
                field["field_value"] = value
            else:
                state["fields"][field_name] = StateField(field_name=field_name, field_value=value)

    def _propagate_field_relations(self, state: Dict[str, Any], protected_fields=None):
        protected_fields = protected_fields or set()
        fields = state.get("fields", {})
        if not isinstance(fields, dict):
            return

        for _ in range(len(fields)):
            changed = False
            for field_name, field in fields.items():
                if field_name in protected_fields:
                    continue
                relation = self._get_field_attr(field, "field_relation")
                if not relation or relation not in fields:
                    continue
                relation_value = self._get_field_attr(fields[relation], "field_value")
                if self._get_field_attr(field, "field_value") != relation_value:
                    self._set_field_attr(field, "field_value", relation_value)
                    changed = True
            if not changed:
                break

    def _get_field_attr(self, field: Any, key: str):
        if hasattr(field, key):
            return getattr(field, key)
        if isinstance(field, dict):
            return field.get(key)
        return field if key == "field_value" else None

    def _is_secret_key(self, key: Any) -> bool:
        if key is None:
            return False
        normalized = str(key).lower().replace("-", "_")
        return any(keyword in normalized for keyword in SECRET_KEYWORDS)

    def _is_masked_secret_value(self, value: Any) -> bool:
        return isinstance(value, str) and value.strip() == MASKED_SECRET

    def _mask_secret_value(self, value: Any) -> Any:
        if value in (None, ""):
            return value
        return MASKED_SECRET

    def _set_field_attr(self, field: Any, key: str, value: Any):
        if hasattr(field, key):
            setattr(field, key, value)
        elif isinstance(field, dict):
            field[key] = value

    def _validate_pending_patch(self, run: FlowRun):
        if not run.pending_patch:
            return

        editable = self._build_editable_patch(run)["editable"]
        patch = run.pending_patch
        errors = []

        allowed_inputs = set((editable.get("inputs") or {}).keys())
        if isinstance(patch.get("inputs"), dict):
            for key in patch["inputs"].keys():
                if key not in allowed_inputs:
                    errors.append(f"inputs.{key}")

        field_patch = {}
        if isinstance(patch.get("state"), dict) and isinstance(patch["state"].get("fields"), dict):
            field_patch.update(patch["state"]["fields"])
        if isinstance(patch.get("fields"), dict):
            field_patch.update(patch["fields"])
        allowed_fields = set((editable.get("fields") or {}).keys())
        for key in field_patch.keys():
            if key not in allowed_fields:
                errors.append(f"fields.{key}")

        configurable_patch = {}
        if isinstance(patch.get("config"), dict) and isinstance(patch["config"].get("configurable"), dict):
            configurable_patch.update(patch["config"]["configurable"])
        if isinstance(patch.get("configurable"), dict):
            configurable_patch.update(patch["configurable"])
        allowed_configurable = set((editable.get("configurable") or {}).keys())
        for key in configurable_patch.keys():
            if key not in allowed_configurable:
                errors.append(f"configurable.{key}")

        if errors:
            raise ValueError(
                "这些字段当前暂停点不会生效，已禁止修改："
                + ", ".join(errors)
                + "。如果要修改已执行节点的输入或 prompt，请重新 Test Run。"
            )

    def _build_editable_patch(self, run: FlowRun) -> Dict[str, Any]:
        editable_nodes = self._get_editable_nodes(run)
        relation_sources = self._get_relation_sources_for_nodes(run, editable_nodes)
        state_fields = (run.current_state or {}).get("fields", {})

        editable_fields = {}
        locked_fields = {}
        for key, field in state_fields.items():
            if "/" not in key:
                continue
            node_name = key.split("/", 1)[0]
            value = self._get_field_attr(field, "field_value")
            if node_name in editable_nodes or key in relation_sources:
                editable_fields[key] = value
            else:
                locked_fields[key] = value

        editable_configurable = {}
        locked_configurable = {}
        for key, value in (run.config.get("configurable") or {}).items():
            node_name = key.split("/", 1)[0] if "/" in key else None
            if node_name in editable_nodes:
                editable_configurable[key] = value
            elif key not in {"thread_id"} and not key.startswith("checkpoint_"):
                locked_configurable[key] = value

        editable_inputs = {}
        locked_inputs = dict(run.inputs or {})
        start_node = run.graph.get_start_node()
        if start_node:
            start_names = {start_node.name, start_node.display_name}
            if start_names & editable_nodes:
                editable_inputs = dict(run.inputs or {})
                locked_inputs = {}
            else:
                for field in start_node.input or []:
                    source_names = {f"{start_name}/{field.name}" for start_name in start_names}
                    if source_names & relation_sources:
                        value = run.inputs.get(field.name)
                        if value is None:
                            for source_name in source_names:
                                if source_name in state_fields:
                                    value = self._get_field_attr(state_fields[source_name], "field_value")
                                    break
                        editable_inputs[field.name] = value
                        locked_inputs.pop(field.name, None)

        template_fields = self._build_template_fields(
            run=run,
            editable_fields=editable_fields,
            relation_sources=relation_sources,
            editable_input_keys=set(editable_inputs.keys()),
        )

        return {
            "editable": {
                "inputs": editable_inputs,
                "fields": editable_fields,
                "configurable": editable_configurable,
            },
            "template": {
                "inputs": editable_inputs,
                "fields": template_fields,
                "configurable": editable_configurable,
            },
            "locked": {
                "inputs": locked_inputs,
                "fields": locked_fields,
                "configurable": locked_configurable,
            },
            "editable_nodes": sorted(editable_nodes),
            "locked_nodes": sorted(self._get_all_node_names(run) - editable_nodes),
            "note": self._build_editable_note(run, editable_nodes),
        }

    def _build_template_fields(
            self,
            run: FlowRun,
            editable_fields: Dict[str, Any],
            relation_sources: set,
            editable_input_keys: set) -> Dict[str, Any]:
        state_fields = (run.current_state or {}).get("fields", {})
        start_node = run.graph.get_start_node()
        editable_input_sources = set()
        if start_node:
            for key in editable_input_keys:
                editable_input_sources.add(f"{start_node.name}/{key}")
                editable_input_sources.add(f"{start_node.display_name}/{key}")

        start_next_nodes = set(run.next_nodes or run.active_nodes or [])
        template_fields = {}
        for key, value in editable_fields.items():
            field = state_fields.get(key)
            relation = self._get_field_attr(field, "field_relation")
            node_name = key.split("/", 1)[0] if "/" in key else None

            if key in relation_sources and key not in editable_input_sources:
                template_fields[key] = value
            elif node_name in start_next_nodes and not relation:
                template_fields[key] = value
        return template_fields

    def _get_all_node_names(self, run: FlowRun) -> set:
        return {node.display_name for node in run.graph.get_nodes() if node.display_name}

    def _get_editable_nodes(self, run: FlowRun) -> set:
        start_nodes = set(run.next_nodes or run.active_nodes or [])
        if not start_nodes:
            return set()

        adjacency = {}
        for source, target in (run.graph.edges or {}).items():
            adjacency.setdefault(source, set()).add(target)
        for source, conditions in (run.graph._condition_edges or {}).items():
            for condition in conditions:
                target = condition.get("target")
                if target:
                    adjacency.setdefault(source, set()).add(target)

        editable = set()
        stack = list(start_nodes)
        while stack:
            node = stack.pop()
            if node in editable:
                continue
            editable.add(node)
            stack.extend(adjacency.get(node, []))
        return editable

    def _get_relation_sources_for_nodes(self, run: FlowRun, editable_nodes: set) -> set:
        relation_sources = set()
        state_fields = (run.current_state or {}).get("fields", {})
        for key, field in state_fields.items():
            if "/" not in key:
                continue
            node_name = key.split("/", 1)[0]
            if node_name not in editable_nodes:
                continue
            relation = self._get_field_attr(field, "field_relation")
            if relation:
                relation_sources.add(relation)
        return relation_sources

    def _build_editable_note(self, run: FlowRun, editable_nodes: set) -> str:
        if run.status != PAUSED:
            return "只有暂停状态可以应用修正。"
        if not editable_nodes:
            return "当前没有可继续执行的节点；如需修改输入或 prompt，请重新 Test Run。"
        return "只能修改当前暂停点之后会执行的节点参数，以及会影响这些节点的 state 字段；已执行节点的 prompt/API 参数需要重新 Test Run 才会生效。"

    def _input_patch_to_field_patch(self, run: FlowRun, inputs_patch: Dict[str, Any]) -> Dict[str, Any]:
        field_patch = {}
        start_node = run.graph.get_start_node()
        for key, value in inputs_patch.items():
            field_patch[key] = value
            if start_node:
                field_patch[f"{start_node.name}/{key}"] = value
        return field_patch

    def _get_checkpoint_as_node(self, run: FlowRun) -> Optional[str]:
        next_node = run.next_nodes[0] if run.next_nodes else None
        if not next_node:
            return None

        start_node = run.graph.get_start_node()
        start_names = {"start"}
        if start_node:
            start_names.update({start_node.name, start_node.display_name})
        if next_node in start_names:
            return "__start__"

        incoming_sources = [edge.source for edge in run.graph.get_edges() if edge.target == next_node]
        if not incoming_sources:
            return None

        source = incoming_sources[0]
        if "if_condition" in source:
            condition_sources = [edge.source for edge in run.graph.get_edges() if edge.target == source]
            return condition_sources[0] if condition_sources else None
        return source

    def _record_boundary_visit(self, run: FlowRun) -> bool:
        checkpoint_id = (run.current_checkpoint_config or {}).get("configurable", {}).get("checkpoint_id")
        boundary_key = f"{checkpoint_id or '-'}:{','.join(run.next_nodes or [])}"
        run.boundary_visits[boundary_key] = run.boundary_visits.get(boundary_key, 0) + 1
        return run.boundary_visits[boundary_key] <= 2

    def _refresh_state(self, run: FlowRun):
        try:
            self._ensure_runtime_config(run)
            state_snapshot = run.state_graph.get_state(run.config)
            run.current_state = getattr(state_snapshot, "values", None) or run.current_state or {}
            run.current_checkpoint_config = getattr(state_snapshot, "config", None) or run.current_checkpoint_config or run.config
            run.next_nodes = list(getattr(state_snapshot, "next", None) or [])
        except Exception:
            run.current_state = run.current_state or {}
            run.current_checkpoint_config = run.current_checkpoint_config or run.config
            run.next_nodes = []

    def _capture_rollback_checkpoint(self, run: FlowRun):
        run.rollback_state = copy.deepcopy(run.current_state or {})
        run.rollback_checkpoint_config = copy.deepcopy(run.current_checkpoint_config or run.config)
        run.rollback_next_nodes = list(run.next_nodes or [])

    def _restore_pause_checkpoint(self, run: FlowRun):
        if run.rollback_state is not None:
            run.current_state = copy.deepcopy(run.rollback_state)
        if run.rollback_checkpoint_config:
            run.current_checkpoint_config = copy.deepcopy(run.rollback_checkpoint_config)
            self._ensure_runtime_config(run)
        if run.rollback_next_nodes:
            run.next_nodes = list(run.rollback_next_nodes)
        run.active_nodes = []

    def _discard_to_pause_checkpoint(self, run: FlowRun):
        if self._checkpoint_has_id(run.rollback_checkpoint_config):
            self._restore_pause_checkpoint(run)
        else:
            self._refresh_state(run)
            run.active_nodes = []

    def _ensure_runtime_config(self, run: FlowRun):
        base_config = copy.deepcopy(run.graph.config)
        base_config.setdefault("configurable", {})
        base_config["configurable"]["thread_id"] = run.thread_id
        run.config = self._strip_checkpoint_config(self._deep_merge(base_config, run.config or {}))

    def _strip_checkpoint_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cleaned = copy.deepcopy(config or {})
        configurable = cleaned.setdefault("configurable", {})
        for key in CHECKPOINT_CONFIG_KEYS:
            configurable.pop(key, None)
        return cleaned

    def _build_invoke_config(self, run: FlowRun, use_checkpoint: bool = True) -> Dict[str, Any]:
        config = self._strip_checkpoint_config(run.config)
        config.setdefault("configurable", {})
        config["configurable"]["thread_id"] = run.thread_id
        if use_checkpoint and run.current_checkpoint_config:
            checkpoint_configurable = (run.current_checkpoint_config or {}).get("configurable", {})
            for key in CHECKPOINT_CONFIG_KEYS:
                if key in checkpoint_configurable:
                    config["configurable"][key] = checkpoint_configurable[key]
        return config

    def _checkpoint_has_id(self, config: Optional[Dict[str, Any]]) -> bool:
        return bool((config or {}).get("configurable", {}).get("checkpoint_id"))

    def _is_end_boundary(self, run: FlowRun) -> bool:
        if not run.next_nodes:
            return False
        end_node = run.graph.get_end_node()
        end_names = {"end"}
        if end_node:
            end_names.update({end_node.name, end_node.display_name})
        return all(node in end_names for node in run.next_nodes)

    def _complete_run(self, run: FlowRun, state: Any, message: str):
        run.current_state = state or run.current_state or {}
        run.result = parse_end_node_to_output(run.current_state)
        run.status = COMPLETED
        run.active_nodes = []
        run.next_nodes = []
        run.finished_at = run.finished_at or _now_iso()
        self._add_event(run, "completed", message, {"result": run.result})

    def _get_state_values(self, run: FlowRun) -> Dict[str, Any]:
        if run.status == RUNNING or run.current_state is None:
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
            if run.status == RUNNING and not run.resume_from_checkpoint:
                self._refresh_state(run)
            return {
                "run_id": run.run_id,
                "thread_id": run.thread_id,
                "flow_id": run.flow_id,
                "status": run.status,
                "started_at": run.started_at,
                "updated_at": run.updated_at,
                "finished_at": run.finished_at,
                "inputs": self._jsonable_state(run.inputs),
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
                "worker_alive": bool(run.worker and run.worker.is_alive()),
                "editable_patch": self._jsonable_state(self._build_editable_patch(run)),
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

    def _jsonable_state(self, value: Any, key_path: str = ""):
        if self._is_secret_key(key_path):
            return self._mask_secret_value(value)
        if isinstance(value, dict):
            return {k: self._jsonable_state(v, f"{key_path}.{k}" if key_path else str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [self._jsonable_state(item, key_path) for item in value]
        if hasattr(value, "dict"):
            return self._jsonable_state(value.dict(), key_path)
        if hasattr(value, "content"):
            return value.content
        return value

    def _deep_merge(self, base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(base)
        for key, value in patch.items():
            if self._is_secret_key(key) and self._is_masked_secret_value(value):
                continue
            if (
                    self._is_secret_key(key)
                    and isinstance(value, dict)
                    and self._is_masked_secret_value(value.get("field_value"))):
                continue
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
