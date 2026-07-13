"""Built-in tools available to agent nodes."""

import ast
import datetime
import json
import operator as _op

import requests
from langchain_core.tools import tool

from core.tools import register_tool

_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}
_UNARY_OPS = {ast.UAdd: _op.pos, ast.USub: _op.neg}


def _safe_eval(node):
    # AST 白名单求值：只允许数字字面量和算术运算，杜绝任意代码执行
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """计算一个算术表达式，支持 + - * / // % ** 和括号，例如 "(3 + 4) * 2"。"""
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval")))
    except Exception as exc:
        return f"计算失败: {exc}"


@tool
def current_time(timezone_offset_hours: int = 8) -> str:
    """获取当前日期时间（ISO 格式）。timezone_offset_hours 为相对 UTC 的小时偏移，默认 +8（北京时间）。"""
    tz = datetime.timezone(datetime.timedelta(hours=timezone_offset_hours))
    return datetime.datetime.now(tz).isoformat()


@tool
def http_get(url: str, timeout_seconds: int = 10) -> str:
    """对指定 URL 发起 HTTP GET 请求，返回响应文本（截断到 4000 字符）。仅支持 http/https。"""
    if not url.startswith(("http://", "https://")):
        return "请求失败: 仅支持 http/https URL"
    try:
        response = requests.get(url, timeout=min(timeout_seconds, 30))
        return response.text[:4000]
    except Exception as exc:
        return f"请求失败: {exc}"


@tool
def json_extract(json_text: str, path: str) -> str:
    """从 JSON 文本中按点号路径提取值，例如 path="data.items.0.name"。"""
    try:
        value = json.loads(json_text)
        for part in path.split('.'):
            if isinstance(value, list):
                value = value[int(part)]
            else:
                value = value[part]
        return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    except Exception as exc:
        return f"提取失败: {exc}"


register_tool(calculator)
register_tool(current_time)
register_tool(http_get)
register_tool(json_extract)
