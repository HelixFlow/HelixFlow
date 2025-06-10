from pydantic import BaseModel
from typing import Any, Optional
from enum import Enum


class FrontendFieldTypes(str, Enum):
    text = 'text'
    textarea = 'textarea'
    mutiline = 'mutiline'
    code = 'code'
    select = 'select'
    number = 'number'
    radio = 'radio'
    checkbox = 'checkbox'
    orther = 'orther'


class FrontendField(BaseModel):
    field_type: str = 'str'
    required: bool = False
    show: bool = True
    value: Any = None
    name: str = ''
    display_name: Optional[str] = None
    display_type: Optional[FrontendFieldTypes] = FrontendFieldTypes.text
    description: Optional[str] = ''
    editable: Optional[bool] = True




class InputField(FrontendField):
    reference: bool = False

class OutputField(FrontendField):
    editable: bool = False
    reference: bool = False
class IfCondition(BaseModel):
    reference: Optional[str] = None
    compare: Optional[str] = None
    compare_reference: Optional[bool] = False
    compare_value: Optional[str] = None
class ConditionField(FrontendField):
    field_type: str = 'condition'
    value: IfCondition  = None