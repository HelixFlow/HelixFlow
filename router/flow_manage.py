from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, func, select

from core.frontend.graph import FrontendGraph, compile_graph
from core.state import parse_end_node_to_output, parse_input_to_state
from database.base import get_table_session
from database.model.flow import Flow, FlowCreate, FlowUpdate
from router.base import BaseResponse, CommonResponse, FlowResponse
from utils.date_util import get_current_time_str
from utils.json_util import json_deserialization, json_serialization
from utils.logger import logger






router = APIRouter(prefix='/flows', tags=['Flows'])


@router.post('/', status_code=201)
def create_flow(*,flow: FlowCreate,
                session: Session = Depends(get_table_session)):
    """Create a new flow."""
    db_flow = Flow(**flow.dict())
    flow = session.query(Flow).filter(Flow.name == db_flow.name).first()
    if flow:
        return BaseResponse(code=500, msg='Flow name already exists')
    db_flow.create_time = get_current_time_str()
    db_flow.update_time = db_flow.create_time
    db_flow.user_id = 1
    session.add(db_flow)
    session.commit()
    session.refresh(db_flow)
    return db_flow
@router.get('/{flow_id}', status_code=200)
def read_flow(*,flow_id: UUID, session: Session = Depends(get_table_session)):
    """Read a flow."""
    flow = session.get(Flow, flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail='Flow not found')
    flow.data = json_deserialization(flow.data)
    return CommonResponse(code=200, msg='success', data=flow)


@router.get('/', status_code=200)
def read_flows(*,
               session: Session = Depends(get_table_session),
               name: str = Query(default=None, description='flow name'),
               page_size: int = Query(default=None),
               page_num: int = Query(default=None),
               status: int = None):
    """Read all flows."""
    sql = select(Flow)
    count_sql = select(func.count(Flow.id))
    if name:
        sql = sql.where(Flow.name.like(f'%{name}%'))
        count_sql = count_sql.where(Flow.name.like(f'%{name}%'))
    if status:
        sql = sql.where(Flow.status == status)
        count_sql = count_sql.where(Flow.status == status)
    total_count = session.scalar(count_sql)

    sql = sql.order_by(Flow.update_time.desc())
    if page_num and page_size:
        sql = sql.offset((page_num - 1) * page_size).limit(page_size)
    flows = session.exec(sql).all()
    for flow in flows:
        flow.data = None
    return CommonResponse(code=200, msg='success', data={
        'total_count': total_count,
        'flows': flows
    })




@router.patch('/{flow_id}', status_code=200)
def update_flow(*,flow_id: UUID,
                session: Session = Depends(get_table_session),
                flow: FlowUpdate):

    """Update a flow."""
    db_flow = session.get(Flow,  flow_id)
    if not db_flow:
        return FlowResponse(code=404, msg='Flow not found')

    flow_data = flow.dict(exclude_unset=True)
    if 'name' in flow_data:
        flow = session.query(Flow).filter(Flow.name == flow_data['name']).first()
        if flow and flow.id != flow_id:
            return FlowResponse(code=500, msg='Flow name already exists')


    if 'status' in flow_data and flow_data['status'] == 2 and db_flow.status == 1:
        # 上线校验
        try:
            graph_data = json_deserialization(db_flow.data)
            if graph_data.get('nodes') == []:
                return FlowResponse(code=500, msg='Flow compile failed, nodes cannot be empty')
            compile_graph(data=graph_data)
        except Exception as exc:
            return FlowResponse(code=500, msg=f'Flow compile failed, {str(exc)}')

    res_data = {}
    for key, value in flow_data.items():
        if key == 'data':
            res_data = value
            value = json_serialization(value)
        setattr(db_flow, key, value)

    db_flow.update_time = get_current_time_str()
    session.add(db_flow)
    session.commit()
    session.refresh(db_flow)
    db_flow.data = res_data
    return FlowResponse(code=200, msg='success', data=db_flow)


@router.delete('/{flow_id}', status_code=200)
def delete_flow(*,
                session: Session = Depends(get_table_session),
                flow_id: UUID):
    """Delete a flow."""
    flow = session.get(Flow,flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail='Flow not found')

    session.delete(flow)
    session.commit()
    return {'message': 'Flow deleted successfully'}


@router.post('/process', status_code=200)
def process_flow(id: UUID,inputs: Optional[dict] = None,saver: Optional[str]="memory",
                 session: Session = Depends(get_table_session)):
    data = json_deserialization(session.get(Flow, id).data)
    graph = FrontendGraph.from_payload(data)
    state_graph = graph.compile_graph(checkpointer_type=saver)
    print(state_graph.get_graph().print_ascii())
    req = uuid4()
    logger.info(f'Processing flow {id} with request id {req}')
    # B5 fix: obtain a fresh, per-request initial state rather than reusing
    # ``graph.state`` (which was shared across concurrent requests and caused
    # field bleed-through).
    state = graph.fresh_initial_state()

    state = parse_input_to_state(inputs["inputs"], state, start_node=graph.get_start_node())

    config = graph.config
    config["configurable"]["thread_id"] = str(req)
    result = state_graph.invoke(input=state, config=config)
    result = parse_end_node_to_output(result)

    return CommonResponse(code=200, msg='success', data=result)



