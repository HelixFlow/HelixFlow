import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from pymilvus import DataType, MilvusClient


DEFAULT_DOCS = [
    "HelixFlow is a visual workflow builder for LangGraph-based agent applications.",
    "HelixFlow provides a FastAPI backend for graph execution and a Umi frontend for designing and testing flows.",
    "HelixFlow supports built-in nodes such as LLM calls, conditional routing, and knowledge base retrieval.",
    "The knowledge node searches Milvus for similar text and outputs formatted retrieval results.",
    "A RAG flow can connect start, knowledge, call_model, and end nodes to answer questions using retrieved context.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Seed Milvus with text documents for HelixFlow knowledge retrieval.")
    parser.add_argument("--milvus-uri", default=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"))
    parser.add_argument("--collection", default=os.getenv("MILVUS_COLLECTION", "local_server"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE", "https://api.chatanywhere.tech/v1"))
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--source", default="manual_seed")
    parser.add_argument("--text", action="append", help="Text to insert. Can be provided multiple times.")
    parser.add_argument("--file", help="Plain text file. Blank-line separated paragraphs will be inserted.")
    parser.add_argument("--drop", action="store_true", help="Drop and recreate the collection before inserting.")
    return parser.parse_args()


def load_texts(args) -> List[str]:
    texts = []
    if args.text:
        texts.extend(args.text)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as file:
            raw = file.read()
        texts.extend(part.strip() for part in raw.split("\n\n"))
    if not texts:
        texts = DEFAULT_DOCS
    return [text.strip() for text in texts if text and text.strip()]


def embed_texts(api_key: str, api_base: str, model: str, texts: List[str]) -> List[List[float]]:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required. Export it or pass --api-key.")
    client = OpenAI(api_key=api_key, base_url=api_base.rstrip("/"))
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def _field_name(field: Dict[str, Any]) -> str:
    return field.get("name") or field.get("field_name") or ""


def _field_type(field: Dict[str, Any]) -> Any:
    return field.get("type") or field.get("data_type") or field.get("datatype")


def describe_fields(client: MilvusClient, collection_name: str) -> Tuple[set, str, bool]:
    desc = client.describe_collection(collection_name)
    fields = desc.get("fields", [])
    field_names = {_field_name(field) for field in fields if _field_name(field)}
    dynamic_enabled = bool(desc.get("enable_dynamic_field"))

    vector_field = "vector"
    for field in fields:
        field_type = _field_type(field)
        if field_type == DataType.FLOAT_VECTOR or str(field_type).endswith("FLOAT_VECTOR"):
            vector_field = _field_name(field)
            break

    return field_names, vector_field, dynamic_enabled


def ensure_collection(client: MilvusClient, collection_name: str, dimension: int, drop: bool):
    if drop and collection_name in client.list_collections():
        client.drop_collection(collection_name)

    if collection_name in client.list_collections():
        return describe_fields(client, collection_name)

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    return describe_fields(client, collection_name)


def main():
    args = parse_args()
    texts = load_texts(args)
    if not texts:
        raise ValueError("No texts to insert.")

    print(f"Embedding {len(texts)} document(s) with model={args.model}, base={args.api_base}")
    vectors = embed_texts(args.api_key, args.api_base, args.model, texts)
    if not vectors:
        raise ValueError("Embedding API returned no vectors.")

    dimension = len(vectors[0])
    print(f"Connecting Milvus at {args.milvus_uri}")
    client = MilvusClient(uri=args.milvus_uri)
    field_names, vector_field, dynamic_enabled = ensure_collection(client, args.collection, dimension, args.drop)

    if "text" not in field_names and not dynamic_enabled:
        raise ValueError(
            f"Collection {args.collection!r} has no 'text' field. "
            "The HelixFlow knowledge node reads output_fields=['text'], "
            "so recreate the collection with --drop or use a collection that has a text field."
        )

    rows = []
    for text, vector in zip(texts, vectors):
        row = {
            vector_field: vector,
            "text": text[:8192],
        }
        if "source" in field_names or dynamic_enabled:
            row["source"] = args.source
        rows.append(row)

    result = client.insert(collection_name=args.collection, data=rows)
    try:
        client.flush(collection_name=args.collection)
    except Exception:
        pass

    print(f"Inserted {len(rows)} document(s) into collection={args.collection}")
    print(result)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"seed_milvus failed: {exc}", file=sys.stderr)
        sys.exit(1)
