import copy
import json
import re
from typing import Any, Dict, List, Optional

from core.builtin.knowledge import knowledge_base
from core.builtin.models import call_model
from core.state import StateField, update_state_by_relation
from service.flow_run_manager import flow_run_manager


AGENT_KNOWLEDGE_NODE = "business_knowledge_1"
AGENT_MODEL_NODE = "business_model_1"
AGENT_RESULT_KEYS = {
    "requirement_summary",
    "flink_create_tables",
    "flink_insert_sql",
    "dimension_table_plan",
    "ttl_plan",
    "resource_plan",
    "risks",
    "assumptions",
}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, indent=2)


def _compact_tables(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for table in tables:
        compact.append({
            "table_name": table.get("table_name"),
            "description": table.get("description"),
            "primary_keys": table.get("primary_keys") or [],
            "event_time_field": table.get("event_time_field"),
            "is_dimension": table.get("is_dimension"),
            "dimension_key": table.get("dimension_key"),
            "columns": [
                {
                    "name": column.get("name"),
                    "raw_type": column.get("raw_type"),
                    "flink_type": column.get("flink_type"),
                    "comment": column.get("comment"),
                    "business_meaning": column.get("business_meaning"),
                }
                for column in table.get("columns") or []
            ],
        })
    return compact


def _extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
    text = (raw_text or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _build_rag_query(requirement: str, base_result: Dict[str, Any], connector: str) -> str:
    table_names = ", ".join(table.get("table_name", "") for table in base_result.get("source_tables") or [])
    return (
        f"业务指标需求: {requirement}\n"
        f"候选表: {table_names}\n"
        f"目标 connector: {connector}\n"
        "请召回相关的表语义、字段说明、业务规则、SQL 示例和 Flink connector 模板。"
    )


def _build_generation_prompt(requirement: str, base_result: Dict[str, Any], connector: str) -> str:
    tables = _compact_tables(base_result.get("source_tables") or [])
    connector_templates = base_result.get("selected_connector_templates") or []
    deterministic_draft = {
        "flink_create_tables": base_result.get("flink_create_tables") or [],
        "flink_insert_sql": base_result.get("flink_insert_sql") or "",
        "risks": base_result.get("risks") or [],
    }
    prompt = f"""
你是 HelixFlow 的业务指标分析 Agent，正在复用 knowledge -> call_model 节点链路生成 FlinkSQL。

目标：根据业务指标语义、结构化表资产、Milvus 召回上下文和 Flink 模板，生成可以人工审核的 FlinkSQL 草稿。

硬性规则：
1. 不能只把字段原样 SELECT 出来，必须理解指标含义，识别过滤条件、窗口、分组维度、JOIN 和聚合。
2. 业务库 DDL 只用于理解字段，Flink CREATE TABLE 必须按召回模板或草稿模板改写。
3. 默认实时链路使用 Kafka；聚合更新结果、主键更新、撤回流优先使用 upsert-kafka。
4. 如果需求提到 GMV、支付成功、成功订单，应优先寻找支付状态字段，并过滤 SUCCESS/成功语义。
5. 如果需求提到每分钟、每小时等时间粒度，必须用事件时间字段生成窗口聚合。
6. 如果维度字段来自另一张表，必须根据主键或同名 key 生成 JOIN。
7. 缺少 topic、watermark、join key、维表 key、TTL 或资源信息时，不要阻塞生成，但必须写入 risks。

业务需求：
{requirement}

目标 connector：
{connector}

结构化表资产：
{_json_text(tables)}

已选 connector 模板：
{_json_text(connector_templates)}

当前规则草稿：
{_json_text(deterministic_draft)}

Milvus 召回上下文：
{{question}}

请只输出一个 JSON 对象，不要输出 Markdown。JSON 字段必须包含：
requirement_summary: 字符串
flink_create_tables: 字符串数组
flink_insert_sql: 字符串
dimension_table_plan: 数组
ttl_plan: 数组
resource_plan: 对象
risks: 数组，每项包含 level 和 message
assumptions: 字符串数组
""".strip()
    return prompt.replace("{", "{{").replace("}", "}}").replace("{{question}}", "{question}")


def _node_summary(graph_data: Dict[str, Any]) -> List[Dict[str, str]]:
    nodes = []
    for node in (graph_data or {}).get("nodes") or []:
        data = node.get("data") or {}
        nodes.append({
            "display_name": data.get("display_name") or node.get("id") or "",
            "node_type": data.get("name") or "",
        })
    return nodes


def _patch_agent_flow_data(
        graph_data: Dict[str, Any],
        generation_prompt: str,
        openai_api_key: str,
        openai_api_base: str,
        embedding_model: str,
        llm_model_name: str,
        milvus_uri: str,
        collection_name: str,
        top_k: int) -> Dict[str, Any]:
    patched = copy.deepcopy(graph_data)
    for node in patched.get("nodes") or []:
        data = node.get("data") or {}
        node_type = data.get("name")
        for field in data.get("input") or []:
            if node_type == "start" and field.get("name") in {"output", "question"}:
                field["value"] = ""

        for param in data.get("params") or []:
            name = param.get("name")
            if name == "openai_api_key":
                param["value"] = openai_api_key
            elif name == "openai_api_base":
                param["value"] = openai_api_base
            elif name == "model_name" and node_type == "knowledge":
                param["value"] = embedding_model
            elif name == "model_name" and node_type in {"call_model", "draw_image"}:
                param["value"] = llm_model_name if node_type == "call_model" else param.get("value") or "gpt-image-1"
            elif name == "vector_url":
                param["value"] = milvus_uri
            elif name == "collection_name":
                param["value"] = collection_name
            elif name == "top_k":
                param["value"] = top_k
            elif name == "prompts" and node_type == "call_model":
                param["value"] = generation_prompt
    return patched


def _run_selected_agent_flow(
        agent_flow_id: str,
        agent_flow_data: Dict[str, Any],
        query_text: str,
        generation_prompt: str,
        openai_api_key: str,
        openai_api_base: str,
        embedding_model: str,
        llm_model_name: str,
        milvus_uri: str,
        collection_name: str,
        top_k: int) -> Dict[str, Any]:
    graph_data = _patch_agent_flow_data(
        graph_data=agent_flow_data,
        generation_prompt=generation_prompt,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        embedding_model=embedding_model,
        llm_model_name=llm_model_name,
        milvus_uri=milvus_uri,
        collection_name=collection_name,
        top_k=top_k,
    )
    flow_output = flow_run_manager.run_process_compat(
        flow_id=agent_flow_id,
        graph_data=graph_data,
        inputs={"output": query_text, "question": query_text},
    )
    raw_output = ""
    if isinstance(flow_output, dict):
        raw_output = str(flow_output.get("input") or flow_output.get("answer") or next(iter(flow_output.values()), ""))
    else:
        raw_output = str(flow_output or "")
    return {
        "graph_data": graph_data,
        "flow_output": flow_output,
        "raw_output": raw_output,
        "parsed": _extract_json(raw_output),
    }


def _build_state(query_text: str) -> Dict[str, Any]:
    return {
        "messages": [],
        "fields": {
            f"{AGENT_KNOWLEDGE_NODE}/question": StateField(
                field_name=f"{AGENT_KNOWLEDGE_NODE}/question",
                field_value=query_text,
            ),
            f"{AGENT_KNOWLEDGE_NODE}/answer": StateField(
                field_name=f"{AGENT_KNOWLEDGE_NODE}/answer",
                field_value=None,
            ),
            f"{AGENT_MODEL_NODE}/question": StateField(
                field_name=f"{AGENT_MODEL_NODE}/question",
                field_value=None,
                field_relation=f"{AGENT_KNOWLEDGE_NODE}/answer",
            ),
            f"{AGENT_MODEL_NODE}/answer": StateField(
                field_name=f"{AGENT_MODEL_NODE}/answer",
                field_value=None,
            ),
        },
    }


def _knowledge_config(
        embedding_model: str,
        openai_api_key: str,
        openai_api_base: str,
        milvus_uri: str,
        collection_name: str,
        top_k: int) -> Dict[str, Any]:
    return {
        "metadata": {"langgraph_node": AGENT_KNOWLEDGE_NODE},
        "configurable": {
            f"{AGENT_KNOWLEDGE_NODE}/prompts": "",
            f"{AGENT_KNOWLEDGE_NODE}/model_name": embedding_model,
            f"{AGENT_KNOWLEDGE_NODE}/openai_api_key": openai_api_key,
            f"{AGENT_KNOWLEDGE_NODE}/openai_api_base": openai_api_base,
            f"{AGENT_KNOWLEDGE_NODE}/vector_url": milvus_uri,
            f"{AGENT_KNOWLEDGE_NODE}/collection_name": collection_name,
            f"{AGENT_KNOWLEDGE_NODE}/top_k": top_k,
        },
    }


def _model_config(prompt: str, llm_model_name: str, openai_api_key: str, openai_api_base: str) -> Dict[str, Any]:
    return {
        "metadata": {"langgraph_node": AGENT_MODEL_NODE},
        "configurable": {
            f"{AGENT_MODEL_NODE}/prompts": prompt,
            f"{AGENT_MODEL_NODE}/model_name": llm_model_name,
            f"{AGENT_MODEL_NODE}/openai_api_key": openai_api_key,
            f"{AGENT_MODEL_NODE}/openai_api_base": openai_api_base,
        },
    }


def enhance_with_agent_rag(
        base_result: Dict[str, Any],
        requirement: str,
        connector: str,
        use_agent_rag: bool = False,
        openai_api_key: Optional[str] = None,
        openai_api_base: str = "https://api.openai.com/v1",
        embedding_model: str = "text-embedding-3-small",
        llm_model_name: str = "gpt-4o-mini",
        milvus_uri: str = "http://127.0.0.1:19530",
        collection_name: str = "business_assets",
        top_k: int = 6,
        agent_flow_id: Optional[str] = None,
        agent_flow_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = copy.deepcopy(base_result)
    result["generation_mode"] = "rule_draft"
    result["agent_trace"] = {
        "enabled": bool(use_agent_rag),
        "used": False,
        "nodes": ["knowledge", "call_model"],
        "selected_flow_id": agent_flow_id,
        "message": "未启用 Agent RAG，使用规则草稿。",
    }
    if not use_agent_rag:
        return result
    if not openai_api_key:
        result["agent_trace"]["message"] = "已启用 Agent RAG，但缺少 openai_api_key，回退规则草稿。"
        result.setdefault("risks", []).append({
            "level": "warning",
            "message": "Agent RAG 未执行：缺少模型 API Key。",
        })
        return result

    query_text = _build_rag_query(requirement, result, connector)
    prompt = _build_generation_prompt(requirement, result, connector)

    if agent_flow_id and agent_flow_data:
        try:
            selected_flow = _run_selected_agent_flow(
                agent_flow_id=agent_flow_id,
                agent_flow_data=agent_flow_data,
                query_text=query_text,
                generation_prompt=prompt,
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
                embedding_model=embedding_model,
                llm_model_name=llm_model_name,
                milvus_uri=milvus_uri,
                collection_name=collection_name,
                top_k=top_k,
            )
            parsed = selected_flow["parsed"]
            result["agent_trace"] = {
                "enabled": True,
                "used": bool(parsed),
                "selected_flow_id": agent_flow_id,
                "nodes": _node_summary(selected_flow["graph_data"]),
                "milvus_uri": milvus_uri,
                "collection_name": collection_name,
                "embedding_model": embedding_model,
                "llm_model_name": llm_model_name,
                "flow_output": selected_flow["flow_output"],
                "raw_output": selected_flow["raw_output"],
                "message": "已执行所选 Agent 工作流生成结果。" if parsed else "所选 Agent 工作流未返回合法 JSON，继续尝试内置节点链。",
            }
            if parsed:
                for key in AGENT_RESULT_KEYS:
                    if key in parsed:
                        result[key] = parsed[key]
                result["generation_mode"] = "agent_flow"
                return result
            result.setdefault("risks", []).append({
                "level": "warning",
                "message": "所选 Agent 工作流未返回合法 JSON，已继续尝试内置 knowledge -> call_model 链路。",
            })
        except Exception as exc:
            result.setdefault("risks", []).append({
                "level": "warning",
                "message": f"所选 Agent 工作流执行失败，已继续尝试内置节点链：{exc}",
            })

    state = _build_state(query_text)
    try:
        state = knowledge_base(
            state,
            _knowledge_config(
                embedding_model=embedding_model,
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
                milvus_uri=milvus_uri,
                collection_name=collection_name,
                top_k=top_k,
            ),
        )
        retrieval_context = state["fields"][f"{AGENT_KNOWLEDGE_NODE}/answer"].field_value or ""
        update_state_by_relation(state)
        state = call_model(
            state,
            _model_config(
                prompt=prompt,
                llm_model_name=llm_model_name,
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
            ),
        )
        raw_output = str(state["fields"][f"{AGENT_MODEL_NODE}/answer"].field_value or "")
        parsed = _extract_json(raw_output)
    except Exception as exc:
        result["agent_trace"] = {
            "enabled": True,
            "used": False,
            "nodes": ["knowledge", "call_model"],
            "message": f"Agent RAG 执行失败，已回退规则草稿：{exc}",
        }
        result.setdefault("risks", []).append({
            "level": "warning",
            "message": f"Agent RAG 执行失败：{exc}",
        })
        return result

    result["agent_trace"] = {
        "enabled": True,
        "used": bool(parsed),
        "nodes": [
            {"display_name": AGENT_KNOWLEDGE_NODE, "node_type": "knowledge"},
            {"display_name": AGENT_MODEL_NODE, "node_type": "call_model"},
        ],
        "milvus_uri": milvus_uri,
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "llm_model_name": llm_model_name,
        "retrieval_context": retrieval_context,
        "raw_output": raw_output,
        "message": "已复用 Agent 开发板块 knowledge -> call_model 节点生成结果。" if parsed else "大模型未返回合法 JSON，已回退规则草稿。",
    }
    if not parsed:
        result.setdefault("risks", []).append({
            "level": "warning",
            "message": "Agent RAG 返回内容不是合法 JSON，已保留规则草稿和原始输出。",
        })
        return result

    for key in AGENT_RESULT_KEYS:
        if key in parsed:
            result[key] = parsed[key]
    result["generation_mode"] = "agent_rag"
    return result
