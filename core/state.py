import operator
from pydantic import BaseModel
from typing import Annotated, Any, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage
from core.frontend.node import StartNode

class StateField(BaseModel):
    field_name: str
    field_value: Optional[Any] = None
    field_relation: Optional[str] = None
    field_type: Optional[str] = 'str'

class AppState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    fields: dict = {}



def parse_input_to_state(input: dict, state, start_node: StartNode) -> AppState:
    for field in start_node.input:
        name = start_node.name + "/" + field.name
        if field.name in input.keys():
            field.value = input[field.name]
        state['fields'][name] = StateField(field_name=name, field_value=field.value)
    for key, value in input.items():
        state['fields'][key] = StateField(field_name=key, field_value=value)
    return state

def get_field_from_state(state: AppState, node_name: str) -> dict:
    # Get the fields from the state by the node name
    # 获取state中当前node的所有字段
    fields = {}
    for key, value in state['fields'].items():
        if key.startswith(node_name):
            k = key.split('/')[1]
            if hasattr(value, "field_value"):
                fields[k] = value.field_value
            elif isinstance(value, dict):
                fields[k] = value.get("field_value")
            else:
                fields[k] = value
    return fields


def update_state_by_relation(state: AppState) -> AppState:
    # Update the state by the relation of the fields
    for key, value in list(state['fields'].items()):
        if not hasattr(value, "field_relation"):
            value = StateField(field_name=key, field_value=value)
            state['fields'][key] = value
        if value.field_relation and value.field_relation in state['fields']:
            relation_field = state['fields'][value.field_relation]
            if not hasattr(relation_field, "field_value"):
                relation_field = StateField(field_name=value.field_relation, field_value=relation_field)
                state['fields'][value.field_relation] = relation_field
            state['fields'][key].field_value = relation_field.field_value
    return state

def parse_end_node_to_output(state: AppState) -> dict:
    # Parse the end node to output
    # 将state中的字段解析为输出
    update_state_by_relation(state)
    output = {}
    for field in state['fields']:
        if field.startswith('end/'):
            state_field = state['fields'][field]
            value = state_field.field_value if hasattr(state_field, "field_value") else state_field
            field = field.split('/')[1]
            output[field] = value
    return output
