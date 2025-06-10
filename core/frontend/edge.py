from pydantic import BaseModel
from typing import Optional
from core.frontend.node import FrontendNode
class FrontendEdge(BaseModel):
    id: str
    source: str
    sourceHandle: Optional[str] = None
    target: str
    targetHandle: Optional[str] = None