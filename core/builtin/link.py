from core.frontend.field import FrontendField, InputField, OutputField
from core.frontend.annotation import node_config
from core.state import AppState, get_field_from_state, update_state_by_relation

links = FrontendField(name='links', display_name='links', type='orther', default=None, rqeuired=True, show=True)

@node_config(name='call_link',
                description='工具调用',
                inputs=[],
                outputs=[],
                parameters=[links])
def call_link(state: AppState, config):
    current_node = config['metadata']['langgraph_node']
    fields = get_field_from_state(state, current_node)
    # question = fields['question'].field_value
    links = config["configurable"][current_node+"/links"]
    #todo get links info from state
    state


    update_state_by_relation(state)

    return state