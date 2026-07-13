"""Tool registry for agent nodes.

Tools are plain LangChain ``BaseTool`` objects registered by name. The agent
node binds a user-selected subset via ``ChatOpenAI.bind_tools`` and executes
calls in a ReAct loop. Register a custom tool with :func:`register_tool`, or
drop a ``@tool``-decorated function in ``core/tools/builtin_tools.py``.
"""

from typing import Dict, List

from langchain_core.tools import BaseTool

TOOL_REGISTRY: Dict[str, BaseTool] = {}


def register_tool(tool: BaseTool) -> BaseTool:
    if tool.name in TOOL_REGISTRY:
        raise ValueError(f"Tool {tool.name!r} is already registered")
    TOOL_REGISTRY[tool.name] = tool
    return tool


def get_tools(names: List[str]) -> List[BaseTool]:
    missing = [name for name in names if name not in TOOL_REGISTRY]
    if missing:
        raise ValueError(f"Unknown tools: {missing}. Available: {sorted(TOOL_REGISTRY)}")
    return [TOOL_REGISTRY[name] for name in names]


def list_tools() -> List[dict]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "args_schema": tool.args,
        }
        for tool in TOOL_REGISTRY.values()
    ]


from core.tools import builtin_tools  # noqa: E402,F401  (registers builtins)
