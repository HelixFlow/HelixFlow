import re
from urllib.parse import urlparse

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

URL_PATTERN = re.compile(r"https?://[^\s,，]+")


def normalize_openai_base_url(value: str) -> str:
    raw_value = str(value or "").strip()
    match = URL_PATTERN.search(raw_value)
    if match:
        raw_value = match.group(0)
    if raw_value and not raw_value.startswith(("http://", "https://")):
        raw_value = f"https://{raw_value}"

    parsed = urlparse(raw_value)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("openai_api_base 配置不合法，请填写类似 https://api.example.com/v1 的完整地址")
    return raw_value.rstrip("/")


@node_config(name='call_model',
                description='LLM',
                inputs=[question],
                outputs=[answer],
                parameters=[prompts, model,api_key,api_base])
def call_model(state: AppState, config):
    current_node = config['metadata']['langgraph_node']
    fields = get_field_from_state(state, current_node)
    # question = fields['question'].field_value
    prompts = str(config["configurable"][current_node+"/prompts"] or "")
    model = str(config["configurable"][current_node+"/model_name"] or "").strip()
    api_key = str(config["configurable"][current_node+"/openai_api_key"] or "").strip()
    api_base = normalize_openai_base_url(config["configurable"][current_node+"/openai_api_base"])
    # todo 变量如果是 list 遍历到问题中 （retrieval）

    chat = ChatOpenAI(model=model, openai_api_key=api_key,
                     openai_api_base=api_base)

    response = chat.invoke(prompts.format(**fields))
    state['fields'][current_node+"/answer"].field_value = getattr(response, "content", response)
    update_state_by_relation(state)

    return state
