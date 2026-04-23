from typing import Callable, Dict, List

from langgraph.checkpoint.memory import InMemorySaver

from core.frontend.edge import FrontendEdge
from core.frontend.node import FrontendNode
from core.initial import NODE_FUNCTIONS, create_dynamic_state_graph
from core.state import AppState, StateField
from utils.logger import logger


def _warn_unsupported_and_fallback(name: str) -> InMemorySaver:
    """Log a warning and fall back to InMemorySaver.

    ``sqlite`` and ``postgres`` are reserved for P0'-b (RFC §7.1). Until their
    real implementations land, callers that request them still receive an
    in-memory checkpointer plus a warning log so tests/dev can keep running
    without silently masking the upgrade.
    """
    logger.warning(
        "checkpointer_type=%r is not implemented in P0'-a; "
        "falling back to InMemorySaver. Scheduled for P0'-b.",
        name,
    )
    return InMemorySaver()


# Factory map: checkpointer_type -> zero-arg callable returning a saver
# ``sqlite`` / ``postgres`` are stub entries that degrade to InMemorySaver.
_CHECKPOINTER_FACTORY: Dict[str, Callable[[], object]] = {
    'memory': lambda: InMemorySaver(),
    'sqlite': lambda: _warn_unsupported_and_fallback('sqlite'),
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

        NOTE: We intentionally do NOT persist initial state here. Each request
        must call :meth:`fresh_initial_state` to obtain an isolated copy
        (B5 fix — no shared mutable state across concurrent requests).
        """
        self.nodes = self._build_nodes()
        self.edges = self._build_edges()
        self.config = self._build_node_params()

    def compile_graph(self, checkpointer_type: str = 'memory'):
        """Compile the StateGraph using the requested checkpointer.

        Parameters
        ----------
        checkpointer_type:
            One of ``memory`` / ``sqlite`` / ``postgres``. ``memory`` is the
            only fully-implemented backend in P0'-a; the others degrade to an
            in-memory saver with a warning.
        """
        state_graph = create_dynamic_state_graph(self.nodes, self.edges, self._condition_edges)
        if checkpointer_type not in _CHECKPOINTER_FACTORY:
            raise ValueError(
                f"checkpointer_type={checkpointer_type!r} not supported in P0'-a"
            )
        checkpointer = _CHECKPOINTER_FACTORY[checkpointer_type]()
        return state_graph.compile(checkpointer=checkpointer)

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
        edges: dict = {}
        for edge in self._edges:
            if 'if_condition' in edge.source:
                # 如果是条件判断节点，将条件添加到condition_edges中
                for condition in self._condition_edges[edge.source]:
                    if condition['param'].name == edge.sourceHandle:
                        condition['target'] = edge.target
            else:
                edges[edge.source] = edge.target
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
