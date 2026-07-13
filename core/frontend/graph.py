import os
import sqlite3
import threading
from typing import Callable, Dict, List, Optional

from langgraph.checkpoint.memory import InMemorySaver

from core.frontend.edge import FrontendEdge
from core.frontend.node import FrontendNode
from core.initial import NODE_FUNCTIONS, create_dynamic_state_graph
from core.state import AppState, StateField
from utils.logger import logger


def _warn_unsupported_and_fallback(name: str) -> InMemorySaver:
    """Log a warning and fall back to InMemorySaver.

    ``postgres`` is reserved for P0'-b (RFC §7.1). Until the real
    implementation lands, callers that request it still receive an in-memory
    checkpointer plus a warning log so tests/dev can keep running without
    silently masking the upgrade.
    """
    logger.warning(
        "checkpointer_type=%r is not implemented yet; "
        "falling back to InMemorySaver. Scheduled for P0'-b.",
        name,
    )
    return InMemorySaver()


_sqlite_saver = None
_sqlite_saver_lock = threading.Lock()


def get_sqlite_saver():
    """Process-wide durable SqliteSaver (survives restarts, unlike memory).

    DB path comes from ``HELIXFLOW_CHECKPOINT_DB`` (default
    ``data/checkpoints.db``). A single shared connection with
    ``check_same_thread=False`` — SqliteSaver serializes access internally.
    """
    global _sqlite_saver
    with _sqlite_saver_lock:
        if _sqlite_saver is None:
            from langgraph.checkpoint.sqlite import SqliteSaver
            db_path = os.getenv('HELIXFLOW_CHECKPOINT_DB', os.path.join('data', 'checkpoints.db'))
            directory = os.path.dirname(db_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            connection = sqlite3.connect(db_path, check_same_thread=False)
            _sqlite_saver = SqliteSaver(connection)
        return _sqlite_saver


# Factory map: checkpointer_type -> zero-arg callable returning a saver.
# ``postgres`` is a stub entry that degrades to InMemorySaver.
_CHECKPOINTER_FACTORY: Dict[str, Callable[[], object]] = {
    'memory': lambda: InMemorySaver(),
    'sqlite': get_sqlite_saver,
    'postgres': lambda: _warn_unsupported_and_fallback('postgres'),
}


class FrontendGraph:

    def __init__(self, nodes: List[FrontendNode],
        edges: List[FrontendEdge],) -> None:
        self._nodes = nodes
        self._edges = edges
        self._condition_edges = {}
        self._build_graph()


    def get_nodes(self):
        return self._nodes

    def get_edges(self):
        return self._edges

    def _build_graph(self) -> None:
        """Build node/edge/config structures.

        NOTE: ``self.state`` is a template only. Request paths must not mutate
        it directly — take a fresh copy via :meth:`fresh_initial_state` or
        ``copy.deepcopy`` (B5 fix — no shared mutable state across concurrent
        requests; ``service.flow_run_manager`` deep-copies it per run).
        """
        self.nodes = self._build_nodes()
        self.edges = self._build_edges()
        self.state = self._build_states()
        self.config = self._build_node_params()

    def compile_graph(
            self,
            checkpointer_type: str = 'memory',
            checkpointer=None,
            interrupt_before: Optional[List[str]] = None,
            interrupt_after: Optional[List[str]] = None):
        """Compile the StateGraph.

        Parameters
        ----------
        checkpointer_type:
            One of ``memory`` / ``sqlite`` / ``postgres``. ``memory`` and
            ``sqlite`` (durable, see :func:`get_sqlite_saver`) are fully
            implemented; ``postgres`` degrades to an in-memory saver with a
            warning. Ignored when ``checkpointer`` is passed explicitly.
        checkpointer:
            An already-constructed checkpointer instance (used by
            ``service.flow_run_manager`` to share a saver across pause/resume).
        interrupt_before / interrupt_after:
            Forwarded to ``StateGraph.compile`` for node-level breakpoints.
        """
        state_graph = create_dynamic_state_graph(self.nodes, self.edges, self._condition_edges)
        if checkpointer is None:
            if checkpointer_type not in _CHECKPOINTER_FACTORY:
                raise ValueError(
                    f"checkpointer_type={checkpointer_type!r} not supported in P0'-a"
                )
            checkpointer = _CHECKPOINTER_FACTORY[checkpointer_type]()
        compile_kwargs = {"checkpointer": checkpointer}
        if interrupt_before is not None:
            compile_kwargs["interrupt_before"] = interrupt_before
        if interrupt_after is not None:
            compile_kwargs["interrupt_after"] = interrupt_after
        return state_graph.compile(**compile_kwargs)

    def get_interrupt_node_names(self) -> List[str]:
        end_node = self.get_end_node()
        end_names = {"end"}
        if end_node:
            end_names.update({end_node.name, end_node.display_name})
        return [node_name for node_name in self.nodes.keys() if node_name not in end_names]

    def fresh_initial_state(self) -> AppState:
        """Build a fresh per-request initial state (B5 fix).

        Must be called for every incoming request; previously this was cached
        on ``self.state`` and mutated across concurrent requests, causing
        field bleed-through.
        """
        return self._build_states()

    @classmethod
    def from_payload(cls, payload: Dict) -> 'FrontendGraph':
        # Parse the json payload and create a new FrontendGraph object
        if 'data' in payload:
            payload = payload['data']
        try:
            nodes = []
            edges = []
            for node in payload['nodes']:
                fnode = node["data"]
                nodes.append(FrontendNode(**fnode))
            for edge in payload['edges']:
                edges.append(FrontendEdge(id=edge['id'], source=edge['source'], target=edge['target'], sourceHandle=edge['sourceHandle'], targetHandle=edge['targetHandle']))
            return cls(nodes, edges)
        except KeyError as exc:
            raise ValueError(
                f"Invalid payload. Expected keys 'nodes' and 'edges'. Found {list(payload.keys())}"
            ) from exc
        except Exception as exc:
            raise ValueError(f"Invalid payload. {exc}") from exc

    def _build_nodes(self):
        nodes: dict = {}
        for node in self._nodes:
            if node.name in NODE_FUNCTIONS:
                if 'if_condition' in node.name:
                    # 如果是条件判断节点，将条件添加到condition_edges中
                    sub_edges = []
                    for param in node.params:
                        item = {}
                        item['param'] = param
                        item['target'] = None
                        sub_edges.append(item)
                    self._condition_edges[node.display_name] = sub_edges
                else:
                    nodes[node.display_name] = NODE_FUNCTIONS.get(node.name).function
        return nodes

    def _build_edges(self):
        # source -> [targets]。多目标支持 fan-out 并行分支；配合 if_condition
        # 的回边可以表达循环（reflection / retry / 多轮检索）。
        edges: dict = {}
        for edge in self._edges:
            if 'if_condition' in edge.source:
                # 如果是条件判断节点，将条件添加到condition_edges中
                for condition in self._condition_edges[edge.source]:
                    if condition['param'].name == edge.sourceHandle:
                        condition['target'] = edge.target
            else:
                edges.setdefault(edge.source, []).append(edge.target)
        return edges

    def _build_states(self) -> AppState:
        # Build a fresh initial state from node input/output declarations.
        # 从节点的输入输出中提取参数，构建初始状态。每次请求都重新构建，
        # 不在 ``__init__`` 时持久化到 ``self``（B5 修复）。
        state: AppState = {"messages": [], "fields": {}}
        for node in self._nodes:
            if node.input is not None:
                for input in node.input:
                    name = node.display_name + "/" + input.name
                    if input.reference:
                        # 如果是引用类型，将value设置为None，将关联的字段设置为value
                        inputField = StateField(field_name=name,field_value=None,field_relation=input.value,field_type=input.field_type)
                    else:
                        inputField = StateField(field_name=name, field_value=input.value, field_relation=None,
                                                field_type=input.field_type)
                    state['fields'][name] = inputField
            if node.output is not None:
                for output in node.output:
                    name = node.display_name + "/" + output.name
                    if output.reference:
                        outputField = StateField(field_name=name, field_value=None, field_relation=output.value,
                                                field_type=output.field_type)
                    else:
                        outputField = StateField(field_name=name, field_value=output.value, field_relation=None,
                                                field_type=output.field_type)
                    state['fields'][name] = outputField
        return state

    def _build_node_params(self) -> None:
        # Build the configuration for the nodes
        # 从节点中提取参数，构建配置，用于后续的graph调用
        config = {"configurable": {}}
        for node in self._nodes:
            if node.params is not None:
                if 'if_condition' in node.name:
                    config["configurable"][node.display_name] = self._condition_edges[node.display_name]
                else:
                    for param in node.params:
                        config["configurable"][node.display_name+"/"+param.name] = param.value


        config["configurable"]["_edges"] = self.edges
        return config

    def get_start_node(self):
        for node in self._nodes:
            if node.name == 'start':
                return node

    def get_end_node(self):
        for node in self._nodes:
            if node.name == 'end':
                return node


def compile_graph(data: Dict, checkpointer_type: str = 'memory'):
    """Module-level helper: build a graph from a payload and compile it.

    The ``checkpointer_type`` default is ``'memory'`` so callers (and
    router.flow_manage.update_flow's online-validation path) can invoke this
    without supplying the argument — B1 fix.
    """
    graph = FrontendGraph.from_payload(data)
    return graph.compile_graph(checkpointer_type)
