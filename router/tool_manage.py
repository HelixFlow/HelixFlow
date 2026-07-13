from fastapi import APIRouter

from core.tools import list_tools
from router.base import CommonResponse

router = APIRouter(prefix='/tools', tags=['Tools'])


@router.get('/', status_code=200)
def get_tools():
    """List registered tools available to agent nodes (name/description/args schema)."""
    return CommonResponse(code=200, msg='success', data=list_tools())
