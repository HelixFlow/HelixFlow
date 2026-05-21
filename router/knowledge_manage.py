from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from database.base import get_table_session
from database.model.business_analysis import BusinessTableAsset
from router.base import CommonResponse
from service.business_analysis.generator import serialize_asset_for_knowledge
from service.business_analysis.knowledge_ingest import DEFAULT_COLLECTION, ingest_text_to_milvus


router = APIRouter(prefix="/knowledge", tags=["Knowledge"])


class KnowledgeIngestRequest(BaseModel):
    doc_type: str
    content: Optional[str] = None
    table_id: Optional[UUID] = None
    connector_type: Optional[str] = None
    metadata: Dict[str, Any] = {}
    milvus_uri: str = "http://127.0.0.1:19530"
    collection_name: str = DEFAULT_COLLECTION
    openai_api_key: Optional[str] = None
    openai_api_base: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    require_vector_write: bool = False


@router.post("/ingest", status_code=200)
def ingest_knowledge(*, payload: KnowledgeIngestRequest, session: Session = Depends(get_table_session)):
    try:
        content = payload.content
        metadata = dict(payload.metadata or {})
        if payload.table_id:
            asset = session.get(BusinessTableAsset, payload.table_id)
            if not asset:
                raise HTTPException(status_code=404, detail="Table asset not found")
            asset_data = asset.to_dict()
            content = serialize_asset_for_knowledge(asset_data)
            metadata = {
                **metadata,
                "table_id": str(payload.table_id),
                "table_name": asset_data.get("table_name"),
            }
        result = ingest_text_to_milvus(
            content=content or "",
            doc_type=payload.doc_type,
            connector_type=payload.connector_type,
            metadata=metadata,
            milvus_uri=payload.milvus_uri,
            collection_name=payload.collection_name,
            openai_api_key=payload.openai_api_key,
            openai_api_base=payload.openai_api_base,
            embedding_model=payload.embedding_model,
            require_vector_write=payload.require_vector_write,
        )
        return CommonResponse(data=result)
    except HTTPException:
        raise
    except ValueError as exc:
        return CommonResponse(code=400, msg=str(exc), data=None)
    except Exception as exc:
        return CommonResponse(code=500, msg=str(exc), data=None)
