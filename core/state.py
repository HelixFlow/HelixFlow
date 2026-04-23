import operator
from typing import Annotated, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from core.frontend.node import StartNode


class StateField(BaseModel):
    field_name: str
    field_value: Optional[str] = None
    field_relation: Optional[str] = None
    field_type: Optional[str] = 'str'


def merge_fields(left: dict, right: dict) -> dict:
    """Reducer for ``AppState.fields`` (B5 fix).

    LangGraph applies reducers when a node returns a partial update: the new
    dict is merged into the previous one rather than replacing it wholesale.
    Semantics:

    * ``None`` inputs are treated as empty dicts.
    * Keys from ``right`` win over ``left`` (last-write-wins).
    * Neither input is mutated.
    """
    if left is None and right is None:
        return {}
    if left is None:
        return dict(right)
    if right is None:
        return dict(left)
    return {**left, **right}


class AppState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # B5: explicit reducer prevents partial updates from nuking unrelated keys
    # and eliminates the cross-request state-pointer aliasing that caused
    # field bleed-through under concurrent invocations.
    fields: Annotated[dict, merge_fields]


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
