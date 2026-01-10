# filename: knowledge_base_milvus_fixed.py
import time
import json
import traceback
import ast
from typing import List, Dict, Any

import httpx
from openai import OpenAI
from pymilvus import MilvusClient

from core.frontend.field import FrontendField, InputField, OutputField, FrontendFieldTypes
from core.frontend.annotation import node_config
from core.state import AppState, get_field_from_state, update_state_by_relation

LOG_PREFIX = "[knowledge_base]"
def now_ms() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def mask_key(k: str) -> str:
    if not k:
        return ""
    k = k.strip()
    if len(k) <= 8:
        return "***"
    return f"{k[:3]}***{k[-3:]}"

def safe_json(data: Any, max_len: int = 2000) -> str:
    try:
        s = json.dumps(data, ensure_ascii=False, default=str)
    except Exception:
        s = str(data)
    if len(s) > max_len:
        s = s[:max_len] + f"...(truncated {len(s)-max_len} chars)"
    return s

def log(msg: str):
    print(f"{LOG_PREFIX} {now_ms()} {msg}")

def normalize_base_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    if not (u.startswith("http://") or u.startswith("https://")):
        u = "https://" + u
    return u.rstrip("/")

class DirectOpenAIEmbeddings:
    def __init__(self, model: str, api_key: str, base_url: str, extra_headers: Dict[str, str] | None = None, timeout: float = 30.0):
        self.model = (model or "").strip()
        self.api_key = (api_key or "").strip()
        self.base_url = normalize_base_url(base_url)
        self.http_client = httpx.Client(timeout=timeout)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=self.http_client,
            default_headers=extra_headers or {}
        )
        log(f"DirectOpenAIEmbeddings: init model={self.model}, base_url={self.base_url}, api_key={mask_key(self.api_key)}, headers={extra_headers or {}}")

    def embed_query(self, text: str) -> List[float]:
        t0 = time.time()
        try:
            resp = self.client.embeddings.create(model=self.model, input=text)
            vec = resp.data[0].embedding
            dim = len(vec) if vec else 0
            log(f"DirectOpenAIEmbeddings: embed_query ok, dim={dim}, elapsed_ms={(time.time()-t0)*1000:.2f}")
            return vec
        except Exception as e:
            log(f"DirectOpenAIEmbeddings: embed_query failed: {e}\n{traceback.format_exc()}")
            raise

embedding_model = FrontendField(
    name="model_name",
    display_name="model",
    field_type="input",
    type='string',
    default="text-embedding-3-small",
    required=True,
    show=True,
    description="Embedding 模型名，例如 text-embedding-3-small（第三方映射名按其文档）"
)

api_key = FrontendField(
    name="openai_api_key",
    display_name="api_key",
    field_type="input",
    type='string',
    required=True,
    show=True,
    description="请在运行时提供，不要硬编码真实密钥"
)

api_base = FrontendField(
    name="openai_api_base",
    display_name="api_base",
    field_type="input",
    type='string',
    default="https://api.chatanywhere.tech/v1",
    required=True,
    show=True,
    description="OpenAI 兼容接口的基础地址（第三方网关需支持 /v1/embeddings）"
)

vector_url = FrontendField(
    name="vector_url",
    display_name="milvus_uri",
    field_type="input",
    type='string',
    default="http://localhost:19530",
    required=True,
    show=True,
    description="Milvus 服务地址，例如 http://host.docker.internal:19530"
)

collection_name_field = FrontendField(
    name="collection_name",
    display_name="collection",
    field_type="input",
    type='string',
    default="local_server",
    required=True,
    show=True,
    description="Milvus 集合名（已建立索引并写入向量与元数据）"
)

top_k_field = FrontendField(
    name="top_k",
    display_name="top_k",
    field_type="input",
    type='number',
    default=5,
    required=True,
    show=True,
    description="返回前 K 个最相似结果"
)

prompts = FrontendField(
    name="prompts",
    display_name="prompts",
    field_type="input",
    type='textarea',
    default="请根据以下问题找到最相近的答案：{{question}}",
    required=True,
    show=True,
    description="提示模板（此算子不调用 LLM，仅用于保留与现有节点一致的参数定义）"
)

question = InputField(
    name="question",
    display_name="question",
    field_type="input",
    type='string',
    required=True,
    show=True,
    description="用户输入的问题"
)

answer = OutputField(
    name="answer",
    display_name="answer",
    type='string',
    required=True,
    show=True,
    editable=False
)

def format_results(results: List[Dict[str, Any]]) -> str:
    log(f"format_results: start, results_len={len(results) if results else 0}")
    if not results:
        return "未检索到相关结果。"
    lines: List[str] = []
    for i, r in enumerate(results, start=1):
        raw_content = r.get("text") or r.get("page_content")
        content = str(raw_content or "")
        raw_score = r.get("score", None)
        try:
            score = float(raw_score) if raw_score is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0
        metadata = r.get("metadata") or {}
        try:
            metadata_str = str(metadata)
        except Exception:
            metadata_str = "{}"
        log(f"format_results: item {i} score={score:.4f}, content_len={len(content)}, metadata_type={type(metadata).__name__}")
        lines.append(f"[{i}] score={score:.4f}\ncontent: {content}\nmetadata: {metadata_str}\n")
    joined = "\n".join(lines)
    log(f"format_results: done, output_len={len(joined)}")
    return joined

def build_embeddings(model_name: str, api_key_val: str, base_url: str) -> DirectOpenAIEmbeddings:
    base_url = normalize_base_url(base_url)
    log(f"build_embeddings: init embeddings, model={model_name}, base_url={base_url}, api_key={mask_key(api_key_val)}")
    extra_headers: Dict[str, str] = {}
    emb = DirectOpenAIEmbeddings(model=model_name, api_key=api_key_val, base_url=base_url, extra_headers=extra_headers, timeout=30.0)
    log("build_embeddings: embeddings object created (DirectOpenAIEmbeddings)")
    return emb

def search_milvus(
    uri: str,
    collection_name: str,
    query_text: str,
    embeddings: DirectOpenAIEmbeddings,
    top_k: int = 5,
    metric_type: str = "COSINE"
) -> List[Dict[str, Any]]:
    log(f"search_milvus: start, uri={uri}, collection={collection_name}, top_k={top_k}, metric_type={metric_type}")
    t0 = time.time()
    normalized: List[Dict[str, Any]] = []

    try:
        log(f"search_milvus: embedding query_text len={len(query_text)} sample='{query_text[:40]}'")
        t_e0 = time.time()
        query_vec = embeddings.embed_query(query_text)
        log(f"search_milvus: embed_query ok, dim={len(query_vec)}, elapsed_ms={(time.time() - t_e0)*1000:.2f}")
    except Exception as e:
        log(f"search_milvus: embed_query failed: {e}\n{traceback.format_exc()}")
        return normalized

    try:
        log("search_milvus: init MilvusClient")
        client = MilvusClient(uri=uri)
        try:
            collections = client.list_collections()
            log(f"search_milvus: list_collections ok -> {safe_json(collections)}")
        except Exception as e_list:
            log(f"search_milvus: list_collections failed: {e_list}\n{traceback.format_exc()}")

        # 根据索引类型设置搜索参数：HNSW 用 ef；IVF 用 nprobe
        search_params = {"metric_type": metric_type, "params": {"ef": 128}}  # 若索引是 IVF，改为 {"nprobe": 16}

        log(f"search_milvus: call client.search with output_fields=['text']")
        t_s0 = time.time()
        search_res = client.search(
            collection_name=collection_name,
            data=[query_vec],
            limit=top_k,
            search_params=search_params,
            output_fields=["text"]
        )
        log(f"search_milvus: search returned type={type(search_res).__name__}, elapsed_ms={(time.time() - t_s0)*1000:.2f}, raw={safe_json(search_res, 3000)}")
    except Exception as e:
        log(f"search_milvus: Milvus search failed: {e}\n{traceback.format_exc()}")
        return normalized

    if not search_res:
        log("search_milvus: empty search_res")
        return normalized

    first = search_res[0]
    if isinstance(first, dict):
        hits = first.get("results", []) or first.get("hits", [])
        log(f"search_milvus: hits type=dict, len={len(hits)}")
    else:
        hits = first
        try:
            l = len(hits)
        except Exception:
            l = -1
        log(f"search_milvus: hits type={type(hits).__name__}, len={l}")

    # 命中归一化（替换你的 for idx, h in enumerate(hits, start=1) 循环）
    for idx, h in enumerate(hits, start=1):
        distance = None
        content = ""

        # 情况 1：pymilvus HybridHits 中的命中对象（常见）
        # 命中对象通常有属性 distance、entity（entity.get(field) 访问字段）
        if hasattr(h, "distance") and hasattr(h, "entity"):
            try:
                distance = float(h.distance)
            except Exception:
                distance = None
            try:
                # 优先 text 字段；若你的列名不同（比如 page_content），改这里
                content = h.entity.get("text") or h.entity.get("page_content") or ""
            except Exception:
                content = ""

        # 情况 2：命中项被序列化成字符串，需要解析为 dict
        elif isinstance(h, str):
            try:
                parsed = ast.literal_eval(h)
                entity = parsed.get("entity") if isinstance(parsed, dict) else None
                # 取文本
                if isinstance(entity, dict):
                    content = entity.get("text") or entity.get("page_content") or ""
                if not content and isinstance(parsed, dict):
                    content = parsed.get("text") or parsed.get("page_content") or ""
                # 取距离
                raw_distance = parsed.get("distance") if isinstance(parsed, dict) else None
                if raw_distance is None and isinstance(parsed, dict):
                    raw_distance = parsed.get("score") or parsed.get("_score")
                try:
                    distance = float(raw_distance) if raw_distance is not None else None
                except Exception:
                    distance = None
            except Exception:
                log(f"search_milvus: hit[{idx}] parse failed, raw={h}")

        # 情况 3：顶层 dict（备用）
        elif isinstance(h, dict):
            entity = h.get("entity")
            if isinstance(entity, dict):
                content = entity.get("text") or entity.get("page_content") or ""
            if not content:
                content = h.get("text") or h.get("page_content") or ""
            raw_distance = h.get("distance")
            if raw_distance is None:
                raw_distance = h.get("score") or h.get("_score")
            try:
                distance = float(raw_distance) if raw_distance is not None else None
            except Exception:
                distance = None

        # 分数转换
        score = 0.0
        if distance is not None:
            if metric_type.upper() == "COSINE":
                score = 1.0 - distance
            elif metric_type.upper() == "L2":
                score = 1.0 / (1.0 + distance)

        normalized.append({"text": content, "metadata": {}, "score": score})
        log(f"search_milvus: normalize[{idx}] content_len={len(content)}, distance={distance}, score={score:.4f}")

    log(f"search_milvus: done, normalized_len={len(normalized)}, total_elapsed_ms={(time.time() - t0)*1000:.2f}")
    return normalized

@node_config(
    name="knowledge",
    description="knowledge base",
    inputs=[question],
    outputs=[answer],
    parameters=[prompts, embedding_model, api_key, api_base, vector_url, collection_name_field, top_k_field]
)
def knowledge_base(state: AppState, config):
    log("knowledge_base: node start")
    try:
        current_node = config["metadata"]["langgraph_node"]
        fields = get_field_from_state(state, current_node)
        log(f"knowledge_base: current_node={current_node}")

        query_text = str(fields.get("question", "")).strip()
        model_name = config["configurable"].get(f"{current_node}/model_name")
        openai_api_key = config["configurable"].get(f"{current_node}/openai_api_key")
        openai_api_base = config["configurable"].get(f"{current_node}/openai_api_base")
        milvus_uri = config["configurable"].get(f"{current_node}/vector_url")
        collection_name = config["configurable"].get(f"{current_node}/collection_name")
        top_k = config["configurable"].get(f"{current_node}/top_k", 5)

        log(f"knowledge_base: params model_name={model_name}, api_base={openai_api_base}, api_key={mask_key(openai_api_key)}, milvus_uri={milvus_uri}, collection_name={collection_name}, top_k={top_k}, question_len={len(query_text)}")

        required_keys = {
            "model_name": model_name,
            "openai_api_key": openai_api_key,
            "openai_api_base": openai_api_base,
            "vector_url": milvus_uri,
            "collection_name": collection_name,
            "question": query_text,
        }
        missing = [k for k, v in required_keys.items() if v in (None, "")]
        if missing:
            msg = f"配置缺失：{missing}"
            log(f"knowledge_base: missing required -> {missing}")
            state["fields"][f"{current_node}/answer"].field_value = msg
            update_state_by_relation(state)
            log("knowledge_base: node end (missing)")
            return state

        if not query_text:
            msg = "问题为空，无法检索。"
            log("knowledge_base: empty query_text")
            state["fields"][f"{current_node}/answer"].field_value = msg
            update_state_by_relation(state)
            log("knowledge_base: node end (empty question)")
            return state

        try:
            t0 = time.time()
            embeddings = build_embeddings(model_name, openai_api_key, openai_api_base)
            log(f"knowledge_base: build_embeddings ok, elapsed_ms={(time.time() - t0)*1000:.2f}")
        except Exception as e_emb:
            err = f"embeddings初始化失败：{e_emb}"
            log(f"knowledge_base: {err}\n{traceback.format_exc()}")
            state["fields"][f"{current_node}/answer"].field_value = f"检索失败：{e_emb}"
            update_state_by_relation(state)
            log("knowledge_base: node end (embeddings init error)")
            return state

        try:
            t0 = time.time()
            results = search_milvus(
                uri=milvus_uri,
                collection_name=collection_name,
                query_text=query_text,
                embeddings=embeddings,
                top_k=int(top_k),
                metric_type="COSINE",
            )
            log(f"knowledge_base: search_milvus ok, results_len={len(results)}, elapsed_ms={(time.time() - t0)*1000:.2f}")
        except Exception as e_search:
            err = f"Milvus检索失败：{e_search}"
            log(f"knowledge_base: {err}\n{traceback.format_exc()}")
            state["fields"][f"{current_node}/answer"].field_value = f"检索失败：{e_search}"
            update_state_by_relation(state)
            log("knowledge_base: node end (search error)")
            return state

        try:
            t0 = time.time()
            answer_text = format_results(results)
            log(f"knowledge_base: format_results ok, answer_len={len(answer_text)}, elapsed_ms={(time.time() - t0)*1000:.2f}")
        except Exception as e_fmt:
            err = f"结果格式化失败：{e_fmt}"
            log(f"knowledge_base: {err}\n{traceback.format_exc()}")
            state["fields"][f"{current_node}/answer"].field_value = f"检索失败：{e_fmt}"
            update_state_by_relation(state)
            log("knowledge_base: node end (format error)")
            return state

        state["fields"][f"{current_node}/answer"].field_value = str(answer_text)
        update_state_by_relation(state)
        log("knowledge_base: node end (success)")
        return state

    except Exception as e:
        log(f"knowledge_base: unexpected error: {e}\n{traceback.format_exc()}")
        try:
            state["fields"][f"{current_node}/answer"].field_value = f"检索失败：{e}"
            update_state_by_relation(state)
        except Exception:
            log("knowledge_base: state write failed after unexpected error")
        log("knowledge_base: node end (unexpected)")
        return state
