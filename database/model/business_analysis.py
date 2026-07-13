import json
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, Text
from sqlmodel import Field

from database.model.base import SQLModelSerializable
from utils.date_util import get_current_time_str


class BusinessTableAsset(SQLModelSerializable, table=True):
    __tablename__ = "business_table_asset"

    id: UUID = Field(default_factory=uuid4, primary_key=True, unique=True)
    table_name: str = Field(index=True)
    display_name: Optional[str] = Field(default=None)
    dialect: Optional[str] = Field(default="mysql", index=True)
    database_name: Optional[str] = Field(default=None)
    schema_name: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None, sa_column=Column(Text))
    raw_ddl: Optional[str] = Field(default=None, sa_column=Column(Text))
    columns_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    primary_keys_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    indexes_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    tags_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    event_time_field: Optional[str] = Field(default=None)
    watermark_expression: Optional[str] = Field(default=None)
    is_dimension: Optional[bool] = Field(default=False)
    dimension_key: Optional[str] = Field(default=None)
    ttl_seconds: Optional[int] = Field(default=None)
    connector_hint: Optional[str] = Field(default=None)
    update_time: Optional[str] = Field(default_factory=get_current_time_str, index=True)
    create_time: Optional[str] = Field(default_factory=get_current_time_str, index=True)

    def to_dict(self):
        return {
            "id": str(self.id),
            "table_name": self.table_name,
            "display_name": self.display_name,
            "dialect": self.dialect,
            "database_name": self.database_name,
            "schema_name": self.schema_name,
            "description": self.description,
            "raw_ddl": self.raw_ddl,
            "columns": _loads(self.columns_json, []),
            "primary_keys": _loads(self.primary_keys_json, []),
            "indexes": _loads(self.indexes_json, []),
            "tags": _loads(self.tags_json, []),
            "event_time_field": self.event_time_field,
            "watermark_expression": self.watermark_expression,
            "is_dimension": self.is_dimension,
            "dimension_key": self.dimension_key,
            "ttl_seconds": self.ttl_seconds,
            "connector_hint": self.connector_hint,
            "create_time": self.create_time,
            "update_time": self.update_time,
        }


class BusinessAnalysisRun(SQLModelSerializable, table=True):
    __tablename__ = "business_analysis_run"

    id: UUID = Field(default_factory=uuid4, primary_key=True, unique=True)
    status: str = Field(default="completed", index=True)
    requirement: Optional[str] = Field(default=None, sa_column=Column(Text))
    connector_preference: Optional[str] = Field(default=None)
    request_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    result_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    error: Optional[str] = Field(default=None, sa_column=Column(Text))
    update_time: Optional[str] = Field(default_factory=get_current_time_str, index=True)
    create_time: Optional[str] = Field(default_factory=get_current_time_str, index=True)

    def to_dict(self):
        return {
            "run_id": str(self.id),
            "status": self.status,
            "requirement": self.requirement,
            "connector_preference": self.connector_preference,
            "request": _loads(self.request_json, {}),
            "result": _loads(self.result_json, {}),
            "error": self.error,
            "create_time": self.create_time,
            "update_time": self.update_time,
        }


def _dumps(value) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, default=str)


def _loads(value: Optional[str], default):
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def table_asset_from_payload(payload: dict) -> BusinessTableAsset:
    now = get_current_time_str()
    return BusinessTableAsset(
        table_name=payload["table_name"],
        display_name=payload.get("display_name") or payload["table_name"],
        dialect=payload.get("dialect") or "mysql",
        database_name=payload.get("database_name"),
        schema_name=payload.get("schema_name"),
        description=payload.get("description"),
        raw_ddl=payload.get("raw_ddl"),
        columns_json=_dumps(payload.get("columns") or []),
        primary_keys_json=_dumps(payload.get("primary_keys") or []),
        indexes_json=_dumps(payload.get("indexes") or []),
        tags_json=_dumps(payload.get("tags") or []),
        event_time_field=payload.get("event_time_field"),
        watermark_expression=payload.get("watermark_expression"),
        is_dimension=bool(payload.get("is_dimension") or False),
        dimension_key=payload.get("dimension_key"),
        ttl_seconds=payload.get("ttl_seconds"),
        connector_hint=payload.get("connector_hint"),
        create_time=now,
        update_time=now,
    )


def update_table_asset_from_payload(asset: BusinessTableAsset, payload: dict) -> BusinessTableAsset:
    for key in (
        "table_name",
        "display_name",
        "dialect",
        "database_name",
        "schema_name",
        "description",
        "raw_ddl",
        "event_time_field",
        "watermark_expression",
        "is_dimension",
        "dimension_key",
        "ttl_seconds",
        "connector_hint",
    ):
        if key in payload:
            setattr(asset, key, payload[key])
    if "columns" in payload:
        asset.columns_json = _dumps(payload["columns"])
    if "primary_keys" in payload:
        asset.primary_keys_json = _dumps(payload["primary_keys"])
    if "indexes" in payload:
        asset.indexes_json = _dumps(payload["indexes"])
    if "tags" in payload:
        asset.tags_json = _dumps(payload["tags"])
    asset.update_time = get_current_time_str()
    return asset


def create_analysis_run_record(requirement: str, connector_preference: Optional[str], request_payload: dict, result: dict):
    now = get_current_time_str()
    return BusinessAnalysisRun(
        status="completed",
        requirement=requirement,
        connector_preference=connector_preference,
        request_json=_dumps(request_payload),
        result_json=_dumps(result),
        create_time=now,
        update_time=now,
    )
