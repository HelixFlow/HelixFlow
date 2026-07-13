import re
from typing import Any, Dict, List, Optional


CONSTRAINT_STARTERS = (
    "primary",
    "constraint",
    "unique",
    "key",
    "index",
    "foreign",
    "check",
)

TYPE_STOP_WORDS = {
    "not",
    "null",
    "default",
    "comment",
    "primary",
    "constraint",
    "unique",
    "references",
    "collate",
    "character",
    "generated",
    "identity",
    "auto_increment",
    "enable",
    "disable",
}

TIME_FIELD_HINTS = ("event", "time", "date", "dt", "timestamp", "gmt", "create", "update")


def _strip_identifier(value: str) -> str:
    value = (value or "").strip().rstrip(";")
    if value.startswith("[") and value.endswith("]"):
        return value[1:-1]
    return value.strip("`\"")


def _split_top_level(value: str, delimiter: str = ",") -> List[str]:
    parts: List[str] = []
    start = 0
    depth = 0
    quote: Optional[str] = None
    i = 0
    while i < len(value):
        char = value[i]
        if quote:
            if char == quote:
                if i + 1 < len(value) and value[i + 1] == quote:
                    i += 1
                else:
                    quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif char == delimiter and depth == 0:
            parts.append(value[start:i].strip())
            start = i + 1
        i += 1
    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _remove_sql_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--[^\n\r]*", " ", sql)
    return sql


def _find_matching_paren(sql: str, open_index: int) -> int:
    depth = 0
    quote: Optional[str] = None
    for index in range(open_index, len(sql)):
        char = sql[index]
        if quote:
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("DDL 括号不完整，未找到 CREATE TABLE 字段定义结束位置")


def _extract_create_table(sql: str) -> tuple[str, str]:
    match = re.search(
        r"create\s+(?:global\s+temporary\s+)?table\s+(?:if\s+not\s+exists\s+)?([`\"\[\]\w.$]+)\s*\(",
        sql,
        flags=re.I,
    )
    if not match:
        raise ValueError("未找到 CREATE TABLE 语句")
    table_name = _strip_identifier(match.group(1).split("$")[-1])
    open_index = sql.find("(", match.end() - 1)
    close_index = _find_matching_paren(sql, open_index)
    return table_name, sql[open_index + 1:close_index]


def _extract_quoted_after_keyword(line: str, keyword: str) -> Optional[str]:
    match = re.search(rf"\b{keyword}\s+'((?:''|[^'])*)'", line, flags=re.I)
    if not match:
        match = re.search(rf'\b{keyword}\s+"([^"]*)"', line, flags=re.I)
    if not match:
        return None
    return match.group(1).replace("''", "'")


def _parse_comment_statements(raw_sql: str) -> tuple[Optional[str], Dict[str, str]]:
    table_comment = None
    column_comments: Dict[str, str] = {}

    for match in re.finditer(
        r"comment\s+on\s+table\s+([`\"\[\]\w.]+)\s+is\s+'((?:''|[^'])*)'",
        raw_sql,
        flags=re.I,
    ):
        table_comment = match.group(2).replace("''", "'")

    for match in re.finditer(
        r"comment\s+on\s+column\s+([`\"\[\]\w.]+)\s+is\s+'((?:''|[^'])*)'",
        raw_sql,
        flags=re.I,
    ):
        column_ref = match.group(1)
        column_comments[_strip_identifier(column_ref.split(".")[-1])] = match.group(2).replace("''", "'")

    return table_comment, column_comments


def _parse_primary_keys(definitions: List[str]) -> List[str]:
    keys: List[str] = []
    for definition in definitions:
        lower = definition.strip().lower()
        if lower.startswith("primary key") or " primary key " in lower:
            match = re.search(r"primary\s+key\s*\((.*?)\)", definition, flags=re.I | re.S)
            if match:
                keys.extend(_strip_identifier(part.split()[0]) for part in _split_top_level(match.group(1)))
                continue
            if not lower.startswith("primary key"):
                keys.append(_strip_identifier(definition.split()[0]))
    return list(dict.fromkeys(key for key in keys if key))


def _parse_indexes(definitions: List[str], raw_sql: str) -> List[Dict[str, Any]]:
    indexes: List[Dict[str, Any]] = []
    for definition in definitions:
        lower = definition.strip().lower()
        if lower.startswith(("key ", "index ", "unique key", "unique index")):
            match = re.search(r"(?:unique\s+)?(?:key|index)\s+([`\"\[\]\w]+)?\s*\((.*?)\)", definition, flags=re.I | re.S)
            if match:
                indexes.append({
                    "name": _strip_identifier(match.group(1) or ""),
                    "columns": [_strip_identifier(part.split()[0]) for part in _split_top_level(match.group(2))],
                    "unique": lower.startswith("unique"),
                })

    for match in re.finditer(
        r"create\s+(unique\s+)?index\s+([`\"\[\]\w]+)\s+on\s+[`\"\[\]\w.]+\s*\((.*?)\)",
        raw_sql,
        flags=re.I | re.S,
    ):
        indexes.append({
            "name": _strip_identifier(match.group(2)),
            "columns": [_strip_identifier(part.split()[0]) for part in _split_top_level(match.group(3))],
            "unique": bool(match.group(1)),
        })
    return indexes


def _parse_column_definition(definition: str, comment_map: Dict[str, str], primary_keys: List[str]) -> Optional[Dict[str, Any]]:
    line = definition.strip().rstrip(",")
    if not line:
        return None
    first_word = line.split(None, 1)[0].strip().lower()
    if first_word in CONSTRAINT_STARTERS:
        return None

    name_match = re.match(r"(`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|\w+)\s+(.*)$", line, flags=re.S)
    if not name_match:
        return None

    column_name = _strip_identifier(name_match.group(1))
    rest = name_match.group(2).strip()
    type_parts: List[str] = []
    for token in re.findall(r"[^\s]+", rest):
        normalized = token.strip().strip(",").lower()
        if normalized in TYPE_STOP_WORDS:
            break
        type_parts.append(token)
    raw_type = " ".join(type_parts).strip().rstrip(",")
    comment = _extract_quoted_after_keyword(line, "comment") or comment_map.get(column_name)
    default_value = None
    default_match = re.search(r"\bdefault\s+((?:'[^']*')|(?:\"[^\"]*\")|[^\s,]+)", line, flags=re.I)
    if default_match:
        default_value = default_match.group(1).strip("'\"")

    normalized_type, flink_type, type_risk = map_business_type(raw_type)
    return {
        "name": column_name,
        "raw_type": raw_type,
        "normalized_type": normalized_type,
        "flink_type": flink_type,
        "nullable": not bool(re.search(r"\bnot\s+null\b", line, flags=re.I)),
        "default": default_value,
        "comment": comment or "",
        "primary_key": column_name in primary_keys or bool(re.search(r"\bprimary\s+key\b", line, flags=re.I)),
        "business_meaning": comment or "",
        "risk": type_risk,
    }


def map_business_type(raw_type: str) -> tuple[str, str, Optional[str]]:
    raw = (raw_type or "").strip()
    normalized = re.sub(r"\s+", " ", raw).lower()
    precision_match = re.search(r"\((\d+)(?:\s*,\s*(\d+))?\)", normalized)
    precision = int(precision_match.group(1)) if precision_match else None
    scale = int(precision_match.group(2)) if precision_match and precision_match.group(2) else None

    if any(token in normalized for token in ("char", "text", "clob", "varchar", "nvarchar", "nchar", "string")):
        return "string", "STRING", None
    if "bigint" in normalized or "long" in normalized:
        return "long", "BIGINT", None
    if "tinyint(1)" in normalized or normalized in {"bit", "boolean", "bool"}:
        return "boolean", "BOOLEAN", None
    if "int" in normalized or "integer" in normalized or normalized == "tinyint" or normalized == "smallint":
        return "integer", "INT", None
    if any(token in normalized for token in ("double", "float", "real", "binary_double", "binary_float")):
        return "double", "DOUBLE", None
    if any(token in normalized for token in ("decimal", "numeric", "number", "money")):
        if scale in (None, 0) and precision is not None and precision <= 18:
            return "long", "BIGINT", None
        p = precision or 38
        s = scale if scale is not None else 10
        return "decimal", f"DECIMAL({p},{s})", None
    if "timestamp" in normalized or "datetime" in normalized:
        return "timestamp", "TIMESTAMP(3)", None
    if re.search(r"\bdate\b", normalized):
        return "date", "DATE", None
    if any(token in normalized for token in ("blob", "binary", "raw")):
        return "binary", "BYTES", None
    return "unknown", "STRING", f"未知业务库类型 {raw_type!r}，已临时映射为 STRING"


def _infer_event_time_field(columns: List[Dict[str, Any]]) -> Optional[str]:
    time_columns = [
        col for col in columns
        if col["normalized_type"] in {"timestamp", "date"} or any(hint in col["name"].lower() for hint in TIME_FIELD_HINTS)
    ]
    if not time_columns:
        return None
    for col in time_columns:
        lower_name = col["name"].lower()
        if "event" in lower_name or "biz" in lower_name or "occur" in lower_name:
            return col["name"]
    return time_columns[0]["name"]


def parse_business_ddl(raw_ddl: str, dialect: str = "mysql") -> Dict[str, Any]:
    if not raw_ddl or not raw_ddl.strip():
        raise ValueError("DDL 不能为空")

    clean_sql = _remove_sql_comments(raw_ddl)
    full_name, body = _extract_create_table(clean_sql)
    table_comment, column_comments = _parse_comment_statements(raw_ddl)
    definitions = _split_top_level(body)
    primary_keys = _parse_primary_keys(definitions)
    columns = [
        column for column in (
            _parse_column_definition(definition, column_comments, primary_keys)
            for definition in definitions
        )
        if column
    ]
    if not columns:
        raise ValueError("未解析到字段定义")

    for column in columns:
        if column["name"] in primary_keys:
            column["primary_key"] = True

    indexes = _parse_indexes(definitions, clean_sql)
    name_parts = [_strip_identifier(part) for part in full_name.split(".")]
    table_name = name_parts[-1]
    schema_name = name_parts[-2] if len(name_parts) > 1 else ""
    event_time_field = _infer_event_time_field(columns)
    risks = [column["risk"] for column in columns if column.get("risk")]
    if not event_time_field:
        risks.append("未识别到事件时间字段，生成 FlinkSQL 时会缺少 WATERMARK")

    return {
        "dialect": (dialect or "mysql").lower(),
        "schema_name": schema_name,
        "table_name": table_name,
        "display_name": table_comment or table_name,
        "description": table_comment or "",
        "raw_ddl": raw_ddl,
        "columns": columns,
        "primary_keys": primary_keys,
        "indexes": indexes,
        "event_time_field": event_time_field,
        "watermark_expression": (
            f"{event_time_field} - INTERVAL '5' SECOND" if event_time_field else ""
        ),
        "risks": risks,
    }
