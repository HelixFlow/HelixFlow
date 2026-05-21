from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from database.base import get_table_session
from database.model.business_analysis import (
    BusinessTableAsset,
    table_asset_from_payload,
    update_table_asset_from_payload,
)
from router.base import CommonResponse
from service.business_analysis.ddl_parser import parse_business_ddl


router = APIRouter(prefix="/assets", tags=["Assets"])


class DDLParseRequest(BaseModel):
    dialect: str = "mysql"
    raw_ddl: str


class TableAssetCreate(BaseModel):
    table_name: str
    display_name: Optional[str] = None
    dialect: Optional[str] = "mysql"
    database_name: Optional[str] = None
    schema_name: Optional[str] = None
    description: Optional[str] = None
    raw_ddl: Optional[str] = None
    columns: List[Dict[str, Any]] = []
    primary_keys: List[str] = []
    indexes: List[Dict[str, Any]] = []
    tags: List[str] = []
    event_time_field: Optional[str] = None
    watermark_expression: Optional[str] = None
    is_dimension: Optional[bool] = False
    dimension_key: Optional[str] = None
    ttl_seconds: Optional[int] = None
    connector_hint: Optional[str] = None


class TableAssetUpdate(BaseModel):
    table_name: Optional[str] = None
    display_name: Optional[str] = None
    dialect: Optional[str] = None
    database_name: Optional[str] = None
    schema_name: Optional[str] = None
    description: Optional[str] = None
    raw_ddl: Optional[str] = None
    columns: Optional[List[Dict[str, Any]]] = None
    primary_keys: Optional[List[str]] = None
    indexes: Optional[List[Dict[str, Any]]] = None
    tags: Optional[List[str]] = None
    event_time_field: Optional[str] = None
    watermark_expression: Optional[str] = None
    is_dimension: Optional[bool] = None
    dimension_key: Optional[str] = None
    ttl_seconds: Optional[int] = None
    connector_hint: Optional[str] = None


@router.post("/ddl/parse", status_code=200)
def parse_ddl(*, payload: DDLParseRequest):
    try:
        return CommonResponse(data=parse_business_ddl(payload.raw_ddl, payload.dialect))
    except ValueError as exc:
        return CommonResponse(code=400, msg=str(exc), data=None)
    except Exception as exc:
        return CommonResponse(code=500, msg=str(exc), data=None)


@router.post("/tables", status_code=201)
def create_table_asset(*, payload: TableAssetCreate, session: Session = Depends(get_table_session)):
    try:
        data = payload.dict()
        asset = table_asset_from_payload(data)
        session.add(asset)
        session.commit()
        session.refresh(asset)
        return CommonResponse(data=asset.to_dict())
    except Exception as exc:
        session.rollback()
        return CommonResponse(code=500, msg=str(exc), data=None)


@router.get("/tables", status_code=200)
def list_table_assets(
        *,
        keyword: Optional[str] = Query(default=None),
        session: Session = Depends(get_table_session)):
    statement = select(BusinessTableAsset).order_by(BusinessTableAsset.update_time.desc())
    assets = session.exec(statement).all()
    data = [asset.to_dict() for asset in assets]
    if keyword:
        normalized = keyword.lower()
        data = [
            asset for asset in data
            if normalized in (asset.get("table_name") or "").lower()
            or normalized in (asset.get("display_name") or "").lower()
            or normalized in (asset.get("description") or "").lower()
        ]
    return CommonResponse(data=data)


@router.get("/tables/{table_id}", status_code=200)
def get_table_asset(*, table_id: UUID, session: Session = Depends(get_table_session)):
    asset = session.get(BusinessTableAsset, table_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Table asset not found")
    return CommonResponse(data=asset.to_dict())


@router.patch("/tables/{table_id}", status_code=200)
def update_table_asset(*, table_id: UUID, payload: TableAssetUpdate, session: Session = Depends(get_table_session)):
    asset = session.get(BusinessTableAsset, table_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Table asset not found")
    try:
        update_table_asset_from_payload(asset, payload.dict(exclude_unset=True))
        session.add(asset)
        session.commit()
        session.refresh(asset)
        return CommonResponse(data=asset.to_dict())
    except Exception as exc:
        session.rollback()
        return CommonResponse(code=500, msg=str(exc), data=None)


@router.delete("/tables/{table_id}", status_code=200)
def delete_table_asset(*, table_id: UUID, session: Session = Depends(get_table_session)):
    asset = session.get(BusinessTableAsset, table_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Table asset not found")
    try:
        session.delete(asset)
        session.commit()
        return CommonResponse(data={"deleted": True, "table_id": str(table_id)})
    except Exception as exc:
        session.rollback()
        return CommonResponse(code=500, msg=str(exc), data=None)
