from pydantic import BaseModel, Field
from typing import Optional
from typing import List, Callable
from core.frontend.field import FrontendField, InputField, OutputField, ConditionField, IfCondition

class FrontendNode(BaseModel):
    name: str = 'node'
    description: str
    display_name: str = ''
    documentation: str = ''
    input: Optional[List[InputField]] = None
    output: Optional[List[OutputField]] = None
    params: Optional[List[FrontendField]] = None
    function: Callable = Field(exclude=True,default=None)
    class Config:
        # Exclude 'function' field by default
        fields = {
            'function': {'exclude': True}
        }

    def dict(self, **kwargs):
        exclude = kwargs.get('exclude', set())
        if isinstance(exclude, set):
            exclude.add('function')
        else:
            exclude = set()
            exclude.add('function')
        kwargs['exclude'] = exclude
        return super().dict(**kwargs)

    def json(self, **kwargs):
        exclude = kwargs.get('exclude', set())
        if isinstance(exclude, set):
            exclude.add('function')
        else:
            exclude = set()
            exclude.add('function')
        kwargs['exclude'] = exclude
        return super().json(**kwargs)

class StartNode(FrontendNode):
    name: str = 'start'
    display_name: str = 'start'
    description: str = """开始节点，用来自动对接输入"""
    input: List[InputField] = [InputField(name='output', display_name='输出内容')]
    output: List[OutputField] = None


class EndNode(FrontendNode):
    name: str = 'end'
    display_name: str = 'end'
    description: str = """结束节点，用来自动对接输出"""
    input: List[InputField] = None
    output: List[OutputField] = [OutputField(name='input', display_name='输入内容', reference=True, editable=True)]

class IfConditionNode(FrontendNode):
    name: str = 'if_condition'
    display_name: str = 'if_condition'
    description: str = """条件判断节点"""
    params: List[FrontendField] = [ConditionField(name='if', display_name='if', type='if_condition', default='True', required=True,
                                                 value=IfCondition(reference=None,compare=None,compare_reference=False,compare_value=None)),
                                  ConditionField(name='else', display_name='else', type='if_condition', default='True', required=True,
                                                 value=IfCondition(reference=None,compare=None,compare_reference=False,compare_value=None))]
    input: List[InputField] = None
    output: List[OutputField] = None