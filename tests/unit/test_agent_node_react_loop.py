"""Unit tests for the agent node's ReAct tool-calling loop (no real LLM)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

import core.builtin.agent as agent_module
from core.state import StateField


class _StubChat:
    """ChatOpenAI stand-in: emits scripted responses, records bound tools."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.bound_tools = None
        self.invocations = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.invocations.append(list(messages))
        return self._responses.pop(0)


def _make_state_and_config(node_name="agent_1", **param_overrides):
    state = {
        "messages": [],
        "fields": {
            f"{node_name}/question": StateField(field_name=f"{node_name}/question", field_value="3+4等于几?"),
            f"{node_name}/answer": StateField(field_name=f"{node_name}/answer"),
        },
    }
    params = {
        f"{node_name}/prompts": "你是计算助手",
        f"{node_name}/model_name": "gpt-4o",
        f"{node_name}/openai_api_key": "sk-test",
        f"{node_name}/openai_api_base": "https://api.example.com/v1",
        f"{node_name}/tools": "calculator",
        f"{node_name}/max_iterations": 5,
        f"{node_name}/memory": False,
    }
    params.update(param_overrides)
    config = {"metadata": {"langgraph_node": node_name}, "configurable": params}
    return state, config


def test_agent_executes_tool_then_answers(monkeypatch):
    tool_call_round = AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expression": "3+4"}, "id": "tc-1"}],
    )
    final_round = AIMessage(content="答案是 7")
    stub = _StubChat([tool_call_round, final_round])
    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **kwargs: stub)

    state, config = _make_state_and_config()
    result = agent_module.agent(state, config)

    assert result["fields"]["agent_1/answer"].field_value == "答案是 7"
    assert [tool.name for tool in stub.bound_tools] == ["calculator"]
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "7"
    assert tool_messages[0].tool_call_id == "tc-1"
    # 第二次调用模型时必须带上工具结果
    assert any(isinstance(m, ToolMessage) for m in stub.invocations[1])


def test_agent_without_tools_single_shot(monkeypatch):
    stub = _StubChat([AIMessage(content="直接回答")])
    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **kwargs: stub)

    state, config = _make_state_and_config(**{"agent_1/tools": ""})
    result = agent_module.agent(state, config)

    assert result["fields"]["agent_1/answer"].field_value == "直接回答"
    assert stub.bound_tools is None  # 没选工具就不 bind
    assert len(stub.invocations) == 1


def test_agent_respects_max_iterations(monkeypatch):
    looping = AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "tc-loop"}],
    )
    stub = _StubChat([looping, looping, looping])
    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **kwargs: stub)

    state, config = _make_state_and_config(**{"agent_1/max_iterations": 3})
    agent_module.agent(state, config)

    assert len(stub.invocations) == 3  # 到达上限即停，不无限循环


def test_agent_memory_includes_history(monkeypatch):
    stub = _StubChat([AIMessage(content="记得")])
    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **kwargs: stub)

    state, config = _make_state_and_config(**{"agent_1/memory": True, "agent_1/tools": ""})
    state["messages"] = [AIMessage(content="上一轮的回答")]
    agent_module.agent(state, config)

    sent = stub.invocations[0]
    assert any(getattr(m, "content", "") == "上一轮的回答" for m in sent)
