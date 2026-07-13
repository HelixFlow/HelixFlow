"""ReAct agent node: LLM with tool calling in a bounded loop.

Unlike ``call_model`` (single-shot LLM), this node binds registry tools to the
model and iterates model → tool_calls → tool results → model until the model
answers without tool calls or ``max_iterations`` is reached. The full exchange
is appended to ``AppState.messages`` so downstream nodes (and multi-turn
conversations via a reused thread_id) see the history.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from core.frontend.annotation import node_config
from core.frontend.field import FrontendField, InputField, OutputField
from core.state import AppState, get_field_from_state, update_state_by_relation
from core.tools import get_tools
from utils.logger import logger

system_prompt = FrontendField(name='prompts', display_name='system_prompt', field_type='str',
                              value='你是一个乐于助人的助手，可以调用工具解决问题。', required=True, show=True)
model = FrontendField(name='model_name', display_name='model', field_type='str', value='gpt-4o', required=True, show=True)
api_key = FrontendField(name='openai_api_key', display_name='api_key', field_type='str', value='sk-*****', required=True, show=True)
api_base = FrontendField(name='openai_api_base', display_name='api_base', field_type='str',
                         value='https://api.chatanywhere.tech/v1', required=True, show=True)
tools_field = FrontendField(name='tools', display_name='tools', field_type='str', value='', required=False, show=True,
                            description='逗号分隔的工具名，可用工具见 GET /helixflow/tools/')
max_iterations = FrontendField(name='max_iterations', display_name='max_iterations', field_type='int', value=5,
                               required=False, show=True, description='ReAct 循环上限，防止死循环')
memory = FrontendField(name='memory', display_name='memory', field_type='bool', value=False, required=False, show=True,
                       description='开启后将会话历史 (state.messages) 一并送入模型，实现多轮记忆')
question = InputField(name='question', display_name='question', field_type='str', required=True, show=True)
answer = OutputField(name='answer', display_name='answer', field_type='str', required=True, show=True, editable=False)


def _parse_tool_names(raw) -> list:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(name).strip() for name in raw if str(name).strip()]
    return [name.strip() for name in str(raw).replace('，', ',').split(',') if name.strip()]


@node_config(name='agent',
             description='Agent (LLM + tools, ReAct loop)',
             inputs=[question],
             outputs=[answer],
             parameters=[system_prompt, model, api_key, api_base, tools_field, max_iterations, memory])
def agent(state: AppState, config):
    from core.builtin.models import normalize_openai_base_url

    current_node = config['metadata']['langgraph_node']
    configurable = config['configurable']
    fields = get_field_from_state(state, current_node)

    prompt_template = str(configurable.get(f'{current_node}/prompts') or '')
    model_name = str(configurable.get(f'{current_node}/model_name') or '').strip()
    key = str(configurable.get(f'{current_node}/openai_api_key') or '').strip()
    base_url = normalize_openai_base_url(configurable.get(f'{current_node}/openai_api_base'))
    tool_names = _parse_tool_names(configurable.get(f'{current_node}/tools'))
    try:
        loop_limit = max(1, int(configurable.get(f'{current_node}/max_iterations') or 5))
    except (TypeError, ValueError):
        loop_limit = 5
    use_memory = str(configurable.get(f'{current_node}/memory') or '').lower() in ('true', '1', 'yes', 'on')

    tools = get_tools(tool_names)
    tool_map = {tool.name: tool for tool in tools}

    llm = ChatOpenAI(model=model_name, api_key=key, base_url=base_url)
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    try:
        prompt_text = prompt_template.format(**fields) if prompt_template else ''
    except (KeyError, IndexError):
        prompt_text = prompt_template
    prelude = [SystemMessage(content=prompt_text)] if prompt_text else []
    history = list(state.get('messages') or []) if use_memory else []

    new_messages = [HumanMessage(content=str(fields.get('question') or ''))]
    response = AIMessage(content='')
    for iteration in range(loop_limit):
        response = llm_with_tools.invoke(prelude + history + new_messages)
        new_messages.append(response)
        tool_calls = getattr(response, 'tool_calls', None) or []
        if not tool_calls:
            break
        for tool_call in tool_calls:
            name = tool_call.get('name')
            tool = tool_map.get(name)
            if tool is None:
                result = f'未知工具: {name}'
            else:
                try:
                    result = tool.invoke(tool_call.get('args') or {})
                except Exception as exc:
                    result = f'工具 {name} 执行失败: {exc}'
            logger.info(f'[agent:{current_node}] iter={iteration} tool={name} -> {str(result)[:200]}')
            new_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call.get('id') or name))
    else:
        logger.warning(f'[agent:{current_node}] reached max_iterations={loop_limit} with pending tool calls')

    state['fields'][f'{current_node}/answer'].field_value = getattr(response, 'content', str(response))
    update_state_by_relation(state)
    return {'fields': state['fields'], 'messages': new_messages}
