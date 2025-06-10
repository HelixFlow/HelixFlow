from router import flow_manage
from router import operator_manage
from fastapi import APIRouter

router = APIRouter(
    prefix='/helixflow',
)
router.include_router(flow_manage)
router.include_router(operator_manage)