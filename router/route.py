from fastapi import APIRouter

from router.asset_manage import router as asset_manage_router
from router.business_analysis import router as business_analysis_router
from router.flow_manage import router as flow_manage_router
from router.knowledge_manage import router as knowledge_manage_router
from router.operator_manage import router as operator_manage_router
from router.user_manager import router as user_manager_router

router = APIRouter(
    prefix='/helixflow',
)
router.include_router(flow_manage_router)
router.include_router(operator_manage_router)
router.include_router(user_manager_router)
router.include_router(asset_manage_router)
router.include_router(knowledge_manage_router)
router.include_router(business_analysis_router)
