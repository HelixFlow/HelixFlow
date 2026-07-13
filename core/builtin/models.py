import re
from urllib.parse import urlparse

from core.frontend.field import FrontendField, InputField, OutputField
from core.frontend.annotation import node_config
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from core.state import AppState, get_field_from_state, update_state_by_relation

# NOTE: 这些 kwargs 必须用 FrontendField 的真实字段名（field_type/value/required/
# editable）。历史版本误拼 rqeuired/eidtable、误用 type=/default=，全部被
# Pydantic 静默丢弃，导致 /operators/all 广告的 schema 缺 required/默认值。
model = FrontendField(name='model_name', display_name='model', field_type='str', value="gpt-3.5-turbo", required=True, show=True)
api_key = FrontendField(name='openai_api_key', display_name='api_key', field_type='str', value="sk-*****", required=True, show=True)
api_base = FrontendField(name='openai_api_base', display_name='api_base', field_type='str', value="https://api.chatanywhere.tech/v1", required=True, show=True)
prompts = FrontendField(name='prompts', display_name='prompts', field_type='str', value="请回答我的问题{{question}}", required=True, show=True)
memory = FrontendField(name='memory', display_name='memory', field_type='bool', value=False, required=False, show=True,
                       description='开启后将会话历史 (state.messages) 一并送入模型，实现多轮对话')
question = InputField(name='question', display_name='question', field_type='str', required=True, show=True)
answer = OutputField(name='answer', display_name='answer', field_type='str', required=True, show=True, editable=False)

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
                parameters=[prompts, model,api_key,api_base,memory])
def call_model(state: AppState, config):
    current_node = config['metadata']['langgraph_node']
    fields = get_field_from_state(state, current_node)
    # question = fields['question'].field_value
    prompts = str(config["configurable"][current_node+"/prompts"] or "")
    model = str(config["configurable"][current_node+"/model_name"] or "").strip()
    api_key = str(config["configurable"][current_node+"/openai_api_key"] or "").strip()
    api_base = normalize_openai_base_url(config["configurable"][current_node+"/openai_api_base"])
    use_memory = str(config["configurable"].get(current_node+"/memory") or "").lower() in ("true", "1", "yes", "on")
    # todo 变量如果是 list 遍历到问题中 （retrieval）

    chat = ChatOpenAI(model=model, openai_api_key=api_key,
                     openai_api_base=api_base)

    rendered = prompts.format(**fields)
    request_message = HumanMessage(content=rendered)
    history = list(state.get('messages') or []) if use_memory else []
    response = chat.invoke(history + [request_message])
    state['fields'][current_node+"/answer"].field_value = getattr(response, "content", response)
    update_state_by_relation(state)

    # messages 通道用 add_messages reducer（按 id 去重），只需带上新增消息；
    # 历史消息由 checkpointer 沿 thread_id 保留，实现多轮记忆。
    return {'fields': state['fields'], 'messages': [request_message, response]}
