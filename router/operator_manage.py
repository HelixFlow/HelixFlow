from core.initial import ALL_NODES
from fastapi import APIRouter, Depends
from core.frontend.node import StartNode, EndNode
from router.base import CommonResponse


router = APIRouter(prefix='/operators', tags=['Operators'])
@router.get("/all", status_code=200, summary="查询所有算子")
def read_operators():
    # operator from directory
    list = ALL_NODES
    # operator from default

    return CommonResponse(data=list)


