import json
import re
from typing import Any, Dict, List, Optional

from database.model.business_analysis import BusinessTableAsset
from service.business_analysis.agent_rag import enhance_with_agent_rag
from service.business_analysis.ddl_parser import parse_business_ddl


CONNECTOR_TEMPLATES = {
    "kafka": {
        "connector_type": "kafka",
        "source": "builtin:kafka_json",
        "description": "默认普通 Kafka JSON 流表模板",
        "options": {
            "connector": "kafka",
            "properties.bootstrap.servers": "${bootstrap_servers}",
            "topic": "${topic}",
            "properties.group.id": "${group_id}",
            "scan.startup.mode": "latest-offset",
            "format": "json",
            "json.ignore-parse-errors": "true",
        },
    },
    "upsert-kafka": {
        "connector_type": "upsert-kafka",
        "source": "builtin:upsert_kafka_json",
        "description": "主键更新/撤回流/聚合结果默认 Upsert Kafka 模板",
        "options": {
            "connector": "upsert-kafka",
            "properties.bootstrap.servers": "${bootstrap_servers}",
            "topic": "${topic}",
            "key.format": "json",
            "value.format": "json",
        },
    },
    "hudi": {
        "connector_type": "hudi",
        "source": "builtin:hudi_cow",
        "description": "Hudi 落湖模板，仅在明确要求时使用",
        "options": {
            "connector": "hudi",
            "path": "${path}",
            "table.type": "COPY_ON_WRITE",
        },
    },
    "hive": {
        "connector_type": "hive",
        "source": "builtin:hive_catalog",
        "description": "Hive Catalog 模板，仅在明确要求时使用",
        "options": {
            "connector": "filesystem",
            "path": "${path}",
            "format": "parquet",
        },
    },
}

UPDATE_HINTS = ("upsert", "更新", "撤回", "去重", "主键", "聚合结果", "update", "retract", "deduplicate")
HUDI_HINTS = ("hudi", "入湖", "湖仓", "lake")
HIVE_HINTS = ("hive", "离线", "数仓", "warehouse")


def normalize_table_asset(asset: BusinessTableAsset | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(asset, BusinessTableAsset):
        return asset.to_dict()
    return asset


def choose_connector(requirement: str, connector_preference: Optional[str], tables: List[Dict[str, Any]]) -> tuple[str, List[str]]:
    preference = (connector_preference or "").strip().lower()
    if preference and preference != "auto":
        return preference, [f"用户指定 connector={preference}"]

    text = requirement.lower()
    reasons: List[str] = []
    if any(hint in text for hint in HUDI_HINTS):
        return "hudi", ["需求中包含 Hudi/入湖/湖仓关键词"]
    if any(hint in text for hint in HIVE_HINTS):
        return "hive", ["需求中包含 Hive/离线/数仓关键词"]
    if any(hint in text for hint in UPDATE_HINTS):
        return "upsert-kafka", ["需求中包含更新/撤回/主键/去重/聚合结果语义"]
    if any(table.get("connector_hint") == "upsert-kafka" for table in tables):
        return "upsert-kafka", ["候选表资产标记了 upsert-kafka connector_hint"]
    return "kafka", ["未指定特殊 connector，按实时默认 Kafka 模板生成"]


def score_table(requirement: str, table: Dict[str, Any]) -> float:
    text = " ".join([
        table.get("table_name") or "",
        table.get("display_name") or "",
        table.get("description") or "",
        " ".join(col.get("name", "") + " " + col.get("comment", "") for col in table.get("columns") or []),
    ]).lower()
    words = [word.lower() for word in re.findall(r"[\w\u4e00-\u9fff]+", requirement or "") if len(word) > 1]
    if not words:
        return 0.0
    hits = sum(1 for word in words if word in text)
    name_bonus = 2 if table.get("table_name", "").lower() in (requirement or "").lower() else 0
    return round((hits + name_bonus) / max(len(words), 1), 4)


def select_candidate_tables(requirement: str, tables: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    scored = []
    for table in tables:
        score = score_table(requirement, table)
        scored.append({
            "table_id": table.get("id"),
            "table_name": table.get("table_name"),
            "display_name": table.get("display_name") or table.get("table_name"),
            "description": table.get("description") or "",
            "score": score,
            "event_time_field": table.get("event_time_field"),
            "primary_keys": table.get("primary_keys") or [],
            "is_dimension": bool(table.get("is_dimension")),
        })
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]


def _sql_name(name: str) -> str:
    return re.sub(r"\W+", "_", (name or "table").strip()).strip("_").lower() or "table"


def _topic_name(prefix: str, table_name: str, suffix: str = "") -> str:
    cleaned_prefix = (prefix or "ods").strip(".")
    cleaned_table = _sql_name(table_name)
    return ".".join(part for part in [cleaned_prefix, cleaned_table + suffix] if part)


def _format_options(options: Dict[str, str], replacements: Dict[str, str]) -> str:
    lines = []
    for key, value in options.items():
        next_value = value
        for r_key, r_value in replacements.items():
            next_value = next_value.replace("${" + r_key + "}", r_value)
        lines.append(f"  '{key}' = '{next_value}'")
    return ",\n".join(lines)


def _watermark_line(table: Dict[str, Any]) -> Optional[str]:
    event_field = table.get("event_time_field")
    if not event_field:
        return None
    expression = table.get("watermark_expression") or f"{event_field} - INTERVAL '5' SECOND"
    return f"  WATERMARK FOR {event_field} AS {expression}"


def _primary_key_line(table: Dict[str, Any], connector: str) -> Optional[str]:
    primary_keys = table.get("primary_keys") or []
    if connector != "upsert-kafka" or not primary_keys:
        return None
    return f"  PRIMARY KEY ({', '.join(primary_keys)}) NOT ENFORCED"


def generate_create_table_sql(
        table: Dict[str, Any],
        connector: str,
        bootstrap_servers: str,
        topic_prefix: str,
        suffix: str = "_src") -> str:
    table_name = _sql_name(table.get("table_name") or "source")
    flink_table_name = f"{table_name}{suffix}"
    field_lines = []
    for column in table.get("columns") or []:
        field_lines.append(f"  {column['name']} {column.get('flink_type') or 'STRING'}")

    extra_lines = [
        line for line in [
            _watermark_line(table),
            _primary_key_line(table, connector),
        ] if line
    ]
    all_lines = field_lines + extra_lines
    template = CONNECTOR_TEMPLATES.get(connector, CONNECTOR_TEMPLATES["kafka"])
    option_text = _format_options(
        template["options"],
        {
            "bootstrap_servers": bootstrap_servers,
            "topic": _topic_name(topic_prefix, table_name),
            "group_id": f"helixflow-{table_name}",
            "path": f"/warehouse/{table_name}",
        },
    )
    field_text = ",\n".join(all_lines)
    return f"CREATE TABLE {flink_table_name} (\n{field_text}\n) WITH (\n{option_text}\n);"


def _guess_join(left: Dict[str, Any], right: Dict[str, Any]) -> Optional[Dict[str, str]]:
    left_cols = {col["name"].lower(): col["name"] for col in left.get("columns") or []}
    right_cols = {col["name"].lower(): col["name"] for col in right.get("columns") or []}
    for key in right.get("primary_keys") or []:
        if key.lower() in left_cols:
            return {"left": left_cols[key.lower()], "right": key}
    for name in left_cols:
        if name in right_cols and (name.endswith("_id") or name == "id"):
            return {"left": left_cols[name], "right": right_cols[name]}
    return None


def generate_insert_sql(tables: List[Dict[str, Any]], connector: str) -> str:
    if not tables:
        return "-- 未找到候选表，无法生成 INSERT SQL"
    main = tables[0]
    main_name = _sql_name(main["table_name"])
    source_alias = "t0"
    select_fields = []
    for column in (main.get("columns") or [])[:12]:
        select_fields.append(f"  {source_alias}.{column['name']} AS {column['name']}")
    if not select_fields:
        select_fields = [f"  {source_alias}.*"]

    from_clause = f"FROM {main_name}_src {source_alias}"
    join_lines = []
    risks = []
    for index, table in enumerate(tables[1:], start=1):
        join = _guess_join(main, table)
        alias = f"t{index}"
        table_name = _sql_name(table["table_name"])
        if join:
            join_lines.append(
                f"LEFT JOIN {table_name}_src {alias}\n"
                f"  ON {source_alias}.{join['left']} = {alias}.{join['right']}"
            )
        else:
            risks.append(f"-- 表 {table['table_name']} 未识别到 join key，请人工补充关联条件")

    sql = (
        f"INSERT INTO {main_name}_result\n"
        "SELECT\n"
        + ",\n".join(select_fields)
        + "\n"
        + from_clause
    )
    if join_lines:
        sql += "\n" + "\n".join(join_lines)
    sql += ";"
    if risks:
        sql += "\n\n" + "\n".join(risks)
    return sql


def build_sink_table_sql(table: Dict[str, Any], connector: str, bootstrap_servers: str, topic_prefix: str) -> str:
    sink_asset = dict(table)
    sink_asset["table_name"] = f"{table.get('table_name')}_result"
    sink_asset["event_time_field"] = None
    sink_asset["watermark_expression"] = ""
    return generate_create_table_sql(
        table=sink_asset,
        connector=connector,
        bootstrap_servers=bootstrap_servers,
        topic_prefix=topic_prefix,
        suffix="",
    )


def build_risks(tables: List[Dict[str, Any]], connector: str) -> List[Dict[str, str]]:
    risks: List[Dict[str, str]] = []
    if not tables:
        return [{"level": "error", "message": "没有候选表，无法生成可信 FlinkSQL"}]

    for table in tables:
        if not table.get("event_time_field"):
            risks.append({"level": "warning", "message": f"表 {table['table_name']} 缺少事件时间字段，未生成 WATERMARK"})
        if connector == "upsert-kafka" and not table.get("primary_keys"):
            risks.append({"level": "warning", "message": f"表 {table['table_name']} 使用 upsert-kafka 但缺少主键"})
        for column in table.get("columns") or []:
            if column.get("risk"):
                risks.append({"level": "warning", "message": f"{table['table_name']}.{column['name']}: {column['risk']}"})

    if len(tables) > 1:
        main = tables[0]
        for table in tables[1:]:
            if not _guess_join(main, table):
                risks.append({"level": "warning", "message": f"表 {main['table_name']} 与 {table['table_name']} 未识别到 join key"})
    return risks


def build_analysis_result(
        requirement: str,
        table_assets: List[BusinessTableAsset | Dict[str, Any]],
        raw_ddl: Optional[str] = None,
        dialect: str = "mysql",
        connector_preference: Optional[str] = "auto",
        bootstrap_servers: str = "localhost:9092",
        topic_prefix: str = "ods",
        use_agent_rag: bool = False,
        openai_api_key: Optional[str] = None,
        openai_api_base: str = "https://api.openai.com/v1",
        embedding_model: str = "text-embedding-3-small",
        llm_model_name: str = "gpt-4o-mini",
        milvus_uri: str = "http://127.0.0.1:19530",
        collection_name: str = "business_assets",
        rag_top_k: int = 6,
        agent_flow_id: Optional[str] = None,
        agent_flow_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    tables = [normalize_table_asset(asset) for asset in table_assets]
    parsed_table = None
    if raw_ddl and raw_ddl.strip():
        parsed_table = parse_business_ddl(raw_ddl, dialect=dialect)
        parsed_table["id"] = "ddl_preview"
        tables = [parsed_table] + tables

    candidates = select_candidate_tables(requirement, tables)
    selected_names = {candidate["table_name"] for candidate in candidates if candidate["score"] > 0}
    selected_tables = [table for table in tables if table.get("table_name") in selected_names]
    if not selected_tables and tables:
        selected_tables = tables[:1]

    connector, connector_reasons = choose_connector(requirement, connector_preference, selected_tables)
    if connector not in CONNECTOR_TEMPLATES:
        connector = "kafka"
        connector_reasons.append("未知 connector_preference，回退到 Kafka")
    template = CONNECTOR_TEMPLATES[connector]

    create_tables = [
        generate_create_table_sql(table, connector, bootstrap_servers, topic_prefix)
        for table in selected_tables
    ]
    if selected_tables:
        create_tables.append(build_sink_table_sql(selected_tables[0], connector, bootstrap_servers, topic_prefix))

    dimension_tables = [
        {
            "table_name": table["table_name"],
            "dimension_key": table.get("dimension_key") or (table.get("primary_keys") or [""])[0],
            "storage": "redis" if table.get("is_dimension") else "flink lookup / kafka compacted topic",
            "ttl_seconds": table.get("ttl_seconds") or 3600,
        }
        for table in selected_tables
        if table.get("is_dimension") or table is not selected_tables[0]
    ]
    risks = build_risks(selected_tables, connector)
    field_mapping = [
        {
            "table_name": table["table_name"],
            "fields": [
                {
                    "name": column["name"],
                    "raw_type": column.get("raw_type"),
                    "flink_type": column.get("flink_type"),
                    "comment": column.get("comment", ""),
                }
                for column in table.get("columns") or []
            ],
        }
        for table in selected_tables
    ]

    result = {
        "requirement_summary": requirement.strip() or "未填写业务需求，已基于表结构生成 FlinkSQL 草稿。",
        "source_tables": selected_tables,
        "candidate_tables": candidates,
        "selected_connector_templates": [{
            "connector_type": connector,
            "source": template["source"],
            "description": template["description"],
            "reasons": connector_reasons,
            "template_options": template["options"],
        }],
        "field_type_mapping": field_mapping,
        "flink_create_tables": create_tables,
        "flink_insert_sql": generate_insert_sql(selected_tables, connector),
        "dimension_table_plan": dimension_tables,
        "ttl_plan": [
            {
                "table_name": item["table_name"],
                "ttl_seconds": item["ttl_seconds"],
                "reason": "维表/关联表默认建议设置缓存 TTL，实际值需结合更新频率调整",
            }
            for item in dimension_tables
        ],
        "resource_plan": {
            "parallelism": max(1, min(8, len(selected_tables) * 2)),
            "taskmanager_memory": "2048m" if len(selected_tables) <= 2 else "4096m",
            "state_backend": "rocksdb",
            "checkpoint_interval": "60s",
        },
        "risks": risks,
        "assumptions": [
            "业务库 DDL 只用于字段理解，Flink connector DDL 按模板生成。",
            f"未特殊指定时使用 connector={connector}。",
            "Kafka topic、bootstrap servers、group id 为草稿值，提交前需要替换为生产环境配置。",
        ],
        "debug_payload": {
            "parsed_ddl_table": parsed_table,
            "connector_preference": connector_preference,
            "bootstrap_servers": bootstrap_servers,
            "topic_prefix": topic_prefix,
        },
    }
    return enhance_with_agent_rag(
        base_result=result,
        requirement=requirement,
        connector=connector,
        use_agent_rag=use_agent_rag,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        embedding_model=embedding_model,
        llm_model_name=llm_model_name,
        milvus_uri=milvus_uri,
        collection_name=collection_name,
        top_k=rag_top_k,
        agent_flow_id=agent_flow_id,
        agent_flow_data=agent_flow_data,
    )


def serialize_asset_for_knowledge(asset: Dict[str, Any]) -> str:
    columns = asset.get("columns") or []
    fields_text = "\n".join(
        f"- {column.get('name')} {column.get('raw_type')} -> {column.get('flink_type')}: {column.get('comment') or ''}"
        for column in columns
    )
    return (
        f"表名: {asset.get('table_name')}\n"
        f"展示名: {asset.get('display_name') or asset.get('table_name')}\n"
        f"描述: {asset.get('description') or ''}\n"
        f"方言: {asset.get('dialect') or ''}\n"
        f"主键: {', '.join(asset.get('primary_keys') or [])}\n"
        f"事件时间字段: {asset.get('event_time_field') or ''}\n"
        f"字段:\n{fields_text}\n"
    )


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
