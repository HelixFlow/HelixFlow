from typing import Dict, Optional
from uuid import UUID, uuid4
from pydantic import Field, validator
from sqlmodel import Field
from database.model.base import SQLModelSerializable
from utils.date_util import get_current_time_str
class FlowBase(SQLModelSerializable):
    name: str = Field(index=True)
    user_id: Optional[int] = Field(index=False)
    description: Optional[str] = Field(index=False)
    data: Optional[Dict] = Field(default=None)
    logo: Optional[str] = Field(index=False, default=None)
    status: Optional[int] = Field(index=False, default=1)
    update_time: Optional[str] = Field(default_factory=get_current_time_str, index=True)
    create_time: Optional[str] = Field(default_factory=get_current_time_str, index=True)

    @validator('data')
    def validate_json(v):
        # dict_keys(['description', 'name', 'id', 'data'])
        if not v:
            return v
        if not isinstance(v, dict):
            raise ValueError('Flow must be a valid JSON')

        # data must contain nodes and edges
        if 'nodes' not in v.keys():
            raise ValueError('Flow must have nodes')
        if 'edges' not in v.keys():
            raise ValueError('Flow must have edges')

        return v


class Flow(FlowBase, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, unique=True)
    data: Optional[str] = Field(default=None)
class FlowRead(FlowBase):
    id: UUID
class FlowCreate(SQLModelSerializable):
    name: str = Field(index=True)
    description: Optional[str] = Field(index=False)
    data: Optional[Dict] = Field(default=None)
    status: Optional[int] = Field(index=False, default=1)

class FlowUpdate(SQLModelSerializable):
    name: Optional[str] = None
    description: Optional[str] = None
    data: Optional[Dict] = None
    status: Optional[int] = None