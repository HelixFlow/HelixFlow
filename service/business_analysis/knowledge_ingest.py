from typing import Any, Dict, Optional

from openai import OpenAI
from pymilvus import DataType, MilvusClient

from service.business_analysis.generator import dumps_json


DEFAULT_COLLECTION = "business_assets"
SUPPORTED_DOC_TYPES = {"table_asset", "flink_template", "business_rule", "sql_example"}


def _embed_text(api_key: str, api_base: str, model: str, text: str) -> list[float]:
    normalized_base = (api_base or "https://api.openai.com/v1").strip().rstrip("/")
    if not normalized_base.startswith(("http://", "https://")):
        raise ValueError("openai_api_base 必须以 http:// 或 https:// 开头")
    client = OpenAI(api_key=api_key.strip(), base_url=normalized_base)
    response = client.embeddings.create(model=model or "text-embedding-3-small", input=[text])
    return response.data[0].embedding


def _field_name(field: Dict[str, Any]) -> str:
    return field.get("name") or field.get("field_name") or ""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "dict"):
        return _json_safe(value.dict())
    if hasattr(value, "__iter__") and not isinstance(value, (bytes, bytearray)):
        try:
            return [_json_safe(item) for item in value]
        except TypeError:
            pass
    return str(value)


def _ensure_collection(client: MilvusClient, collection_name: str, dimension: int) -> tuple[set[str], str, bool]:
    if collection_name in client.list_collections():
        desc = client.describe_collection(collection_name)
        fields = desc.get("fields", [])
        field_names = {_field_name(field) for field in fields if _field_name(field)}
        vector_field = "vector"
        for field in fields:
            field_type = field.get("type") or field.get("data_type") or field.get("datatype")
            if field_type == DataType.FLOAT_VECTOR or str(field_type).endswith("FLOAT_VECTOR"):
                vector_field = _field_name(field)
                break
        return field_names, vector_field, bool(desc.get("enable_dynamic_field"))

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)
    schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="connector_type", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=4096)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    return {"id", "vector", "text", "doc_type", "connector_type", "metadata"}, "vector", False


def ingest_text_to_milvus(
        content: str,
        doc_type: str,
        connector_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        milvus_uri: str = "http://127.0.0.1:19530",
        collection_name: str = DEFAULT_COLLECTION,
        openai_api_key: Optional[str] = None,
        openai_api_base: str = "https://api.openai.com/v1",
        embedding_model: str = "text-embedding-3-small",
        require_vector_write: bool = False) -> Dict[str, Any]:
    text = (content or "").strip()
    if not text:
        raise ValueError("知识库内容不能为空")
    if doc_type not in SUPPORTED_DOC_TYPES:
        raise ValueError(f"不支持的 doc_type={doc_type}，可选值: {', '.join(sorted(SUPPORTED_DOC_TYPES))}")
    if doc_type == "flink_template" and not connector_type:
        raise ValueError("doc_type=flink_template 时必须提供 connector_type")
    if not openai_api_key:
        if require_vector_write:
            raise ValueError("写入 Milvus 需要 openai_api_key 生成 embedding")
        return {
            "stored": False,
            "reason": "缺少 openai_api_key，未写入 Milvus；已返回可写入文本预览",
            "collection_name": collection_name,
            "doc_type": doc_type,
            "connector_type": connector_type,
            "content_preview": text[:1000],
        }

    vector = _embed_text(openai_api_key, openai_api_base, embedding_model, text)
    client = MilvusClient(uri=milvus_uri)
    field_names, vector_field, dynamic_enabled = _ensure_collection(client, collection_name, len(vector))
    row = {
        vector_field: vector,
        "text": text[:8192],
    }
    optional_values = {
        "doc_type": doc_type,
        "connector_type": connector_type or "",
        "metadata": dumps_json(metadata or {})[:4096],
    }
    for key, value in optional_values.items():
        if key in field_names or dynamic_enabled:
            row[key] = value
    result = client.insert(collection_name=collection_name, data=[row])
    try:
        client.flush(collection_name=collection_name)
    except Exception:
        pass
    return {
        "stored": True,
        "collection_name": collection_name,
        "doc_type": doc_type,
        "connector_type": connector_type,
        "insert_result": _json_safe(result),
    }
