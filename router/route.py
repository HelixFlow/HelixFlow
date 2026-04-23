from fastapi import APIRouter

from router.flow_manage import router as flow_manage_router
from router.operator_manage import router as operator_manage_router
from router.user_manager import router as user_manager_router

router = APIRouter(
    prefix='/helixflow',
)
router.include_router(flow_manage_router)
router.include_router(operator_manage_router)
router.include_router(user_manager_router)
