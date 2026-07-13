from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from database.base import get_table_session
from database.model.business_analysis import (
    BusinessAnalysisRun,
    BusinessTableAsset,
    create_analysis_run_record,
)
from database.model.flow import Flow
from router.base import CommonResponse
from service.business_analysis.generator import build_analysis_result
from utils.json_util import json_deserialization


router = APIRouter(prefix="/business-analysis", tags=["BusinessAnalysis"])


class BusinessAnalysisRunCreate(BaseModel):
    requirement: str = ""
    raw_ddl: Optional[str] = None
    dialect: str = "mysql"
    table_ids: List[UUID] = []
    connector_preference: Optional[str] = "auto"
    bootstrap_servers: str = "localhost:9092"
    topic_prefix: str = "ods"
    use_agent_rag: bool = False
    milvus_uri: str = "http://127.0.0.1:19530"
    collection_name: str = "business_assets"
    openai_api_key: Optional[str] = None
    openai_api_base: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    llm_model_name: str = "gpt-4o-mini"
    rag_top_k: int = 6
    agent_flow_id: Optional[UUID] = None


class BusinessAnalysisSelectTables(BaseModel):
    table_ids: List[UUID]
    connector_preference: Optional[str] = "auto"
    bootstrap_servers: str = "localhost:9092"
    topic_prefix: str = "ods"
    use_agent_rag: bool = False
    milvus_uri: str = "http://127.0.0.1:19530"
    collection_name: str = "business_assets"
    openai_api_key: Optional[str] = None
    openai_api_base: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    llm_model_name: str = "gpt-4o-mini"
    rag_top_k: int = 6
    agent_flow_id: Optional[UUID] = None


def _load_assets(session: Session, table_ids: List[UUID]):
    if table_ids:
        assets = []
        for table_id in table_ids:
            asset = session.get(BusinessTableAsset, table_id)
            if not asset:
                raise HTTPException(status_code=404, detail=f"Table asset not found: {table_id}")
            assets.append(asset)
        return assets
    return session.exec(select(BusinessTableAsset).order_by(BusinessTableAsset.update_time.desc())).all()


def _safe_request_payload(payload: BaseModel):
    data = payload.dict()
    if data.get("openai_api_key"):
        data["openai_api_key"] = "********"
        data["has_openai_api_key"] = True
    return data


def _load_agent_flow_data(session: Session, flow_id: Optional[UUID]):
    if not flow_id:
        return None
    flow = session.get(Flow, flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail=f"Agent flow not found: {flow_id}")
    graph_data = json_deserialization(flow.data)
    if not graph_data:
        raise HTTPException(status_code=400, detail=f"Agent flow data is empty or invalid: {flow_id}")
    return graph_data


@router.post("/runs", status_code=201)
def create_business_analysis_run(*, payload: BusinessAnalysisRunCreate, session: Session = Depends(get_table_session)):
    try:
        assets = _load_assets(session, payload.table_ids)
        agent_flow_data = _load_agent_flow_data(session, payload.agent_flow_id)
        request_payload = _safe_request_payload(payload)
        result = build_analysis_result(
            requirement=payload.requirement,
            table_assets=assets,
            raw_ddl=payload.raw_ddl,
            dialect=payload.dialect,
            connector_preference=payload.connector_preference,
            bootstrap_servers=payload.bootstrap_servers,
            topic_prefix=payload.topic_prefix,
            use_agent_rag=payload.use_agent_rag,
            openai_api_key=payload.openai_api_key,
            openai_api_base=payload.openai_api_base,
            embedding_model=payload.embedding_model,
            llm_model_name=payload.llm_model_name,
            milvus_uri=payload.milvus_uri,
            collection_name=payload.collection_name,
            rag_top_k=payload.rag_top_k,
            agent_flow_id=str(payload.agent_flow_id) if payload.agent_flow_id else None,
            agent_flow_data=agent_flow_data,
        )
        run = create_analysis_run_record(
            requirement=payload.requirement,
            connector_preference=payload.connector_preference,
            request_payload=request_payload,
            result=result,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return CommonResponse(data=run.to_dict())
    except HTTPException:
        raise
    except ValueError as exc:
        return CommonResponse(code=400, msg=str(exc), data=None)
    except Exception as exc:
        session.rollback()
        return CommonResponse(code=500, msg=str(exc), data=None)


@router.get("/runs/{run_id}", status_code=200)
def get_business_analysis_run(*, run_id: UUID, session: Session = Depends(get_table_session)):
    run = session.get(BusinessAnalysisRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    return CommonResponse(data=run.to_dict())


@router.post("/runs/{run_id}/select-tables", status_code=200)
def select_tables_for_run(
        *,
        run_id: UUID,
        payload: BusinessAnalysisSelectTables,
        session: Session = Depends(get_table_session)):
    run = session.get(BusinessAnalysisRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    try:
        assets = _load_assets(session, payload.table_ids)
        agent_flow_data = _load_agent_flow_data(session, payload.agent_flow_id)
        result = build_analysis_result(
            requirement=run.requirement or "",
            table_assets=assets,
            raw_ddl=None,
            dialect="mysql",
            connector_preference=payload.connector_preference,
            bootstrap_servers=payload.bootstrap_servers,
            topic_prefix=payload.topic_prefix,
            use_agent_rag=payload.use_agent_rag,
            openai_api_key=payload.openai_api_key,
            openai_api_base=payload.openai_api_base,
            embedding_model=payload.embedding_model,
            llm_model_name=payload.llm_model_name,
            milvus_uri=payload.milvus_uri,
            collection_name=payload.collection_name,
            rag_top_k=payload.rag_top_k,
            agent_flow_id=str(payload.agent_flow_id) if payload.agent_flow_id else None,
            agent_flow_data=agent_flow_data,
        )
        run.connector_preference = payload.connector_preference
        run.result_json = create_analysis_run_record(
            requirement=run.requirement or "",
            connector_preference=payload.connector_preference,
            request_payload=_safe_request_payload(payload),
            result=result,
        ).result_json
        session.add(run)
        session.commit()
        session.refresh(run)
        return CommonResponse(data=run.to_dict())
    except HTTPException:
        raise
    except Exception as exc:
        session.rollback()
        return CommonResponse(code=500, msg=str(exc), data=None)
