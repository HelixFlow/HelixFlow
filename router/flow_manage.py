from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from router.base import FlowResponse,BaseResponse, CommonResponse
from sqlmodel import Session, select, func
from database.base import get_table_session
from database.model.flow import Flow, FlowCreate, FlowUpdate
from utils.logger import logger
from utils.json_util import json_serialization, json_deserialization
from utils.date_util import get_current_time_str
from core.frontend.graph import compile_graph
from service.flow_run_manager import flow_run_manager






router = APIRouter(prefix='/flows', tags=['Flows'])


class FlowRunCreate(BaseModel):
    id: UUID
    inputs: Optional[dict] = None
    saver: Optional[str] = "memory"


class FlowRunPatch(BaseModel):
    inputs: Optional[dict] = None
    state: Optional[dict] = None
    fields: Optional[dict] = None
    config: Optional[dict] = None
    configurable: Optional[dict] = None


def _get_flow_graph_data(session: Session, flow_id: UUID):
    flow = session.get(Flow, flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail='Flow not found')
    graph_data = json_deserialization(flow.data)
    if not graph_data:
        raise HTTPException(status_code=400, detail='Flow data is empty or invalid')
    return graph_data


def _flow_run_patch_to_dict(patch: FlowRunPatch):
    return patch.dict(exclude_unset=True)


@router.post('/', status_code=201)
def create_flow(*,flow: FlowCreate,
                session: Session = Depends(get_table_session)):
    """Create a new flow."""
    try:
        db_flow = Flow(**flow.dict())
        existed_flow = session.query(Flow).filter(Flow.name == db_flow.name).first()
        if existed_flow:
            return CommonResponse(code=500, msg='Flow name already exists', data=None)
        db_flow.create_time = get_current_time_str()
        db_flow.update_time = db_flow.create_time
        db_flow.user_id = 1
        session.add(db_flow)
        session.commit()
        session.refresh(db_flow)
        return CommonResponse(code=200, msg='success', data=db_flow)
    except Exception as exc:
        logger.exception(f'Create flow failed: {flow.dict()}')
        return CommonResponse(code=500, msg=str(exc), data=None)


@router.post('/runs', status_code=201)
def create_flow_run(*, run: FlowRunCreate, session: Session = Depends(get_table_session)):
    graph_data = _get_flow_graph_data(session, run.id)
    try:
        run_data = flow_run_manager.create_run(
            flow_id=str(run.id),
            graph_data=graph_data,
            inputs=(run.inputs or {}).get("inputs", run.inputs or {}),
        )
        return CommonResponse(code=200, msg='success', data=run_data)
    except Exception as exc:
        logger.exception(f'Create flow run failed: {run.id}')
        return CommonResponse(code=500, msg=str(exc), data=None)


@router.get('/runs/{run_id}', status_code=200)
def get_flow_run(*, run_id: str):
    try:
        return CommonResponse(code=200, msg='success', data=flow_run_manager.get_run(run_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        return CommonResponse(code=500, msg=str(exc), data=None)


@router.post('/runs/{run_id}/pause', status_code=200)
def pause_flow_run(*, run_id: str):
    try:
        return CommonResponse(code=200, msg='success', data=flow_run_manager.pause_run(run_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        return CommonResponse(code=500, msg=str(exc), data=None)


@router.patch('/runs/{run_id}', status_code=200)
def patch_flow_run(*, run_id: str, patch: FlowRunPatch):
    try:
        return CommonResponse(code=200, msg='success',
                              data=flow_run_manager.patch_run(run_id, _flow_run_patch_to_dict(patch)))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        return CommonResponse(code=400, msg=str(exc), data=None)
    except Exception as exc:
        return CommonResponse(code=500, msg=str(exc), data=None)


@router.post('/runs/{run_id}/resume', status_code=200)
def resume_flow_run(*, run_id: str, patch: Optional[FlowRunPatch] = None):
    try:
        patch_data = _flow_run_patch_to_dict(patch) if patch else None
        return CommonResponse(code=200, msg='success',
                              data=flow_run_manager.resume_run(run_id, patch_data))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        return CommonResponse(code=400, msg=str(exc), data=None)
    except Exception as exc:
        return CommonResponse(code=500, msg=str(exc), data=None)


@router.post('/runs/{run_id}/stop', status_code=200)
def stop_flow_run(*, run_id: str):
    try:
        return CommonResponse(code=200, msg='success', data=flow_run_manager.stop_run(run_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        return CommonResponse(code=500, msg=str(exc), data=None)


@router.post('/process', status_code=200)
def process_flow(id: UUID,inputs: Optional[dict] = None,saver: Optional[str]="memory",
                 session: Session = Depends(get_table_session)):
    data = _get_flow_graph_data(session, id)
    logger.info(f'Processing flow {id}')
    try:
        result = flow_run_manager.run_process_compat(
            flow_id=str(id),
            graph_data=data,
            inputs=(inputs or {}).get("inputs", inputs or {}),
        )
        return CommonResponse(code=200, msg='success', data=result)
    except Exception as exc:
        logger.exception(f'Processing flow {id} failed')
        return CommonResponse(code=500, msg=str(exc), data=None)


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
                return FlowResponse(code=500, msg=f'Flow compile failed, nodes cannot be empty')
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
