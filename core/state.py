import operator
from pydantic import BaseModel
from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage
from core.frontend.node import StartNode

class StateField(BaseModel):
    field_name: str
    field_value: Optional[str] = None
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
            fields[k] = value.field_value
    return fields


def update_state_by_relation(state: AppState) -> AppState:
    # Update the state by the relation of the fields
    for key, value in state['fields'].items():
        if value.field_relation:
            state['fields'][key].field_value = state['fields'][value.field_relation].field_value
    return state

def parse_end_node_to_output(state: AppState) -> dict:
    # Parse the end node to output
    # 将state中的字段解析为输出
    output = {}
    for field in state['fields']:
        if field.startswith('end/'):
            value = state['fields'][field].field_value
            field = field.split('/')[1]
            output[field] = value
    return output