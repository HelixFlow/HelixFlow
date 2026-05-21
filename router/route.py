from router import flow_manage
from router import operator_manage
from router import user_manager
from router import asset_manage
from router import business_analysis
from router import knowledge_manage
from fastapi import APIRouter

router = APIRouter(
    prefix='/helixflow',
)
router.include_router(flow_manage)
router.include_router(operator_manage)
router.include_router(user_manager)
router.include_router(asset_manage.router)
router.include_router(knowledge_manage.router)
router.include_router(business_analysis.router)
