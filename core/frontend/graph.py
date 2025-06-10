from typing import List, Dict, TypedDict
from core.frontend.node import FrontendNode
from core.frontend.edge import FrontendEdge
from langgraph.graph.graph import CompiledGraph
from core.initial import NODE_FUNCTIONS, create_dynamic_state_graph
from core.state import StateField, AppState
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
        """Builds the graph from the nodes and edges."""
        self.nodes = self._build_nodes()
        self.edges = self._build_edges()
        self.state = self._build_states()
        self.config = self._build_node_params()

    def compile_graph(self):
        state_graph = create_dynamic_state_graph(self.nodes, self.edges, self._condition_edges)
        return state_graph.compile()




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
        # Build the initial state of the graph
        # 从节点的输入输出中提取参数，构建初始状态
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

def compile_graph(data: Dict) -> CompiledGraph:
    graph = FrontendGraph.from_payload(data)
    return graph.compile_graph()