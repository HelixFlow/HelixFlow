from core.frontend.field import FrontendField, InputField, OutputField
from core.frontend.annotation import node_config
from langchain_community.chat_models import ChatOpenAI
from core.state import AppState, get_field_from_state, update_state_by_relation

model = FrontendField(name='model_name', display_name='model', type='string', default="gpt-3.5-turbo", rqeuired=True, show=True)
api_key = FrontendField(name='openai_api_key', display_name='api_key', type='string', default="sk-*****", rqeuired=True, show=True)
api_base = FrontendField(name='openai_api_base', display_name='api_base', type='string', default="https://api.chatanywhere.tech/v1", rqeuired=True, show=True)
prompts = FrontendField(name='prompts', display_name='prompts', type='string', default="请回答我的问题{{question}}", rqeuired=True, show=True)
question = InputField(name='question', display_name='question', type='string',rqeuired=True, show=True)
answer = OutputField(name='answer', display_name='answer', type='string',rqeuired=True, show=True,eidtable=False )

@node_config(name='call_model',
                description='LLM',
                inputs=[question],
                outputs=[answer],
                parameters=[prompts, model,api_key,api_base])
def call_model(state: AppState, config):
    current_node = config['metadata']['langgraph_node']
    fields = get_field_from_state(state, current_node)
    # question = fields['question'].field_value
    prompts = config["configurable"][current_node+"/prompts"]
    model = config["configurable"][current_node+"/model_name"]
    api_key = config["configurable"][current_node+"/openai_api_key"]
    api_base = config["configurable"][current_node+"/openai_api_base"]
    # todo 变量如果是 list 遍历到问题中 （retrieval）

    chat = ChatOpenAI(model=model, openai_api_key=api_key,
                     openai_api_base=api_base)

    response = chat.invoke(prompts.format(**fields))
    state['fields'][current_node+"/answer"].field_value = getattr(response, "content", response)
    update_state_by_relation(state)

    return state
