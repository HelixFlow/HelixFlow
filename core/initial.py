import importlib
from langgraph.graph import StateGraph
from typing import TypedDict, List
from core.frontend.node import FrontendNode
import os
import importlib.util
import inspect
import functools
from utils.logger import logger
from core.state import AppState
from core.frontend.node import StartNode, EndNode, IfConditionNode
from core.builtin.base import start_node, end_node
from core.builtin.if_condition import if_condition
class AgentState(TypedDict):
    messages: list
    node_fields: dict
    def __init__(self, messages: list, node_fields: dict):
        self.messages = messages
        self.node_fields = node_fields





def load_nodes_from_directory() -> List[FrontendNode]:
    # Load all custom nodes from the core/builtin directory
    nodes = []
    directory = os.path.join(os.getcwd(), "core", "builtin")

    print(f"Loading nodes from directory: {directory}")

    for filename in os.listdir(directory):
        if filename.endswith('.py'):
            module_name = filename[:-3]  # 去掉 .py 扩展名
            module_path = os.path.join(directory, filename)

            logger.debug(f"Processing file: {filename} as module: {module_name}")

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            logger.debug(f"Module {module_name} loaded successfully.")

            for name, obj in inspect.getmembers(module):
                if callable(obj):
                    logger.debug(f"Found callable: {name}")
                    # Check if the callable has a _node_config annotation
                    if hasattr(obj, '_node_config'):
                        config = getattr(obj, '_node_config')
                        logger.debug(f"Callable {name} has _node_config: {config}")
                        # Parse annotation to create a new FrontendNode object
                        node = FrontendNode(
                            name=config['name'],
                            display_name=config.get('display_name', ''),
                            description=config.get('description', ''),
                            documentation=config.get('documentation', ''),
                            input=config.get('input', []),
                            output=config.get('output', []),
                            params=config.get('parameters', []),
                            function=obj
                        )
                        nodes.append(node)
                        logger.debug(f"Node {config['name']} created and added to nodes list.")
                    else:
                        logger.debug(f"Callable {name} does not have node_config attribute.")
                else:
                    logger.debug(f"Member {name} is not callable.")
        else:
            logger.debug(f"Skipping non-Python file: {filename}")

    logger.debug(f"Total nodes loaded: {len(nodes)}")
    # add base node to the nodes
    startNode = StartNode()
    startNode.function = start_node
    endNode = EndNode()
    endNode.function = end_node
    ifConditionNode = IfConditionNode()
    nodes.append(ifConditionNode)
    nodes.append(startNode)
    nodes.append(endNode)
    return nodes
ALL_NODES = load_nodes_from_directory()
NODE_FUNCTIONS = {node.name: node for node in load_nodes_from_directory()}




def create_dynamic_state_graph(node_functions: dict, edges: dict, condition_edge) -> StateGraph:
    state_graph = StateGraph(AppState)
    condition_source = {}
    # Add nodes to the StateGraph
    for node_name, node_function in node_functions.items():
        logger.debug(f"Adding node {node_name} to state graph")
        state_graph.add_node(node_name, node_function)

    # Add edges to define transitions between nodes
    for from_node, to_node in edges.items():
        if 'if_condition' in from_node:
            continue
        if 'if_condition' in to_node:
            condition_source[to_node] = from_node
            continue
        logger.debug(f"Adding edge from {from_node} to {to_node}")
        state_graph.add_edge(from_node, to_node)
    for node, conditions in condition_edge.items():
        path_map = {}
        for condition in conditions:
            path_map[condition['target']] = condition['target']
        logger.debug(f"Adding conditional edges from {condition_source[node]} for node {node} to {path_map}")
        state_graph.add_conditional_edges(condition_source[node], if_condition, path_map)

    state_graph.set_entry_point('start')
    state_graph.set_finish_point('end')

    return state_graph

