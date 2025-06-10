import pydantic
from typing import Any
from pydantic import BaseModel
from database.model.flow import FlowRead
class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="HTTP code")
    msg: str = pydantic.Field("success", description="HTTP status message")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }

class CommonResponse(BaseResponse):
    data: Any = pydantic.Field(None, description="返回数据")
class FlowResponse(BaseResponse):
    flow: FlowRead = pydantic.Field(None, description="工作流")