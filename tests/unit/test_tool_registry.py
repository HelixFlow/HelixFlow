"""Unit tests for the agent tool registry and built-in tools."""

from __future__ import annotations

import pytest
from langchain_core.tools import tool

from core.tools import TOOL_REGISTRY, get_tools, list_tools, register_tool


def test_builtin_tools_registered():
    assert {"calculator", "current_time", "http_get", "json_extract"} <= set(TOOL_REGISTRY)


def test_list_tools_shape():
    entries = list_tools()
    assert all({"name", "description", "args_schema"} <= set(entry) for entry in entries)


def test_get_tools_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown tools"):
        get_tools(["calculator", "no_such_tool"])


def test_register_tool_rejects_duplicate():
    @tool
    def calculator(expression: str) -> str:
        """dup"""
        return expression

    with pytest.raises(ValueError, match="already registered"):
        register_tool(calculator)


def test_calculator_evaluates_arithmetic():
    calc = TOOL_REGISTRY["calculator"]
    assert calc.invoke({"expression": "2 + 3 * 4"}) == "14"
    assert calc.invoke({"expression": "(1 + 1) ** 3"}) == "8"


def test_calculator_rejects_code_execution():
    calc = TOOL_REGISTRY["calculator"]
    result = calc.invoke({"expression": "__import__('os').system('id')"})
    assert result.startswith("计算失败")


def test_json_extract_path():
    extract = TOOL_REGISTRY["json_extract"]
    payload = '{"data": {"items": [{"name": "helix"}]}}'
    assert extract.invoke({"json_text": payload, "path": "data.items.0.name"}) == "helix"
