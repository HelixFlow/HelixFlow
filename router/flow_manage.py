import json
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
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
MASKED_SECRET = "********"
SECRET_KEYWORDS = ("api_key", "apikey", "access_key", "secret", "token", "password", "authorization")


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


def _serialize_flow_create_data(flow_data: dict) -> tuple[dict, dict]:
    response_data = flow_data.get('data')
    db_data = dict(flow_data)
    if response_data:
        db_data['data'] = json_serialization(response_data)
    return db_data, response_data or {}


def _mask_sensitive_payload(value):
    if isinstance(value, dict):
        masked = {}
        named_secret = any(
            secret in str(value.get(name_key, "")).lower()
            for secret in SECRET_KEYWORDS
            for name_key in ("name", "display_name")
        )
        for key, item in value.items():
            key_text = str(key).lower()
            if any(secret in key_text for secret in SECRET_KEYWORDS) or (named_secret and key_text == "value"):
                masked[key] = MASKED_SECRET if item else item
            else:
                masked[key] = _mask_sensitive_payload(item)
        return masked
    if isinstance(value, list):
        return [_mask_sensitive_payload(item) for item in value]
    return value


@router.post('/', status_code=201)
def create_flow(*,flow: FlowCreate,
                session: Session = Depends(get_table_session)):
    """Create a new flow."""
    try:
        db_data, response_data = _serialize_flow_create_data(flow.dict())
        db_flow = Flow(**db_data)
        existed_flow = session.query(Flow).filter(Flow.name == db_flow.name).first()
        if existed_flow:
            return CommonResponse(code=500, msg='Flow name already exists', data=None)
        db_flow.create_time = get_current_time_str()
        db_flow.update_time = db_flow.create_time
        db_flow.user_id = 1
        session.add(db_flow)
        session.commit()
        session.refresh(db_flow)
        result = db_flow.dict()
        result['data'] = response_data
        return CommonResponse(code=200, msg='success', data=result)
    except Exception as exc:
        logger.exception(f'Create flow failed: {_mask_sensitive_payload(flow.dict())}')
        session.rollback()
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


def _sse_frame(event: str, data) -> str:
    # 注意不能用 utils.json_util.json_serialization（它做 base64 编码，是给
    # flow.data 落库用的）；SSE data 必须是明文 JSON。
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.post('/process', status_code=200)
def process_flow(id: UUID, inputs: Optional[dict] = None, saver: Optional[str] = "memory",
                 conversation_id: Optional[str] = None, stream: bool = False,
                 recursion_limit: Optional[int] = None,
                 session: Session = Depends(get_table_session)):
    """Run a flow synchronously.

    - ``conversation_id``: reuse the same LangGraph thread across calls for
      multi-turn memory (combine with ``saver=sqlite`` to survive restarts).
    - ``stream=true``: return an SSE stream (start → node* → end).
    - ``recursion_limit``: raise LangGraph's step budget for looping flows.
    """
    data = _get_flow_graph_data(session, id)
    request_inputs = (inputs or {}).get("inputs", inputs or {})
    logger.info(f'Processing flow {id} (conversation_id={conversation_id}, stream={stream})')

    if stream:
        def event_stream():
            for event, payload in flow_run_manager.stream_sync(
                    flow_id=str(id), graph_data=data, inputs=request_inputs,
                    thread_id=conversation_id, checkpointer_type=saver or "memory",
                    recursion_limit=recursion_limit):
                yield _sse_frame(event, payload)
        return StreamingResponse(event_stream(), media_type='text/event-stream')

    try:
        if conversation_id or (saver and saver != "memory") or recursion_limit:
            result = flow_run_manager.run_sync(
                flow_id=str(id), graph_data=data, inputs=request_inputs,
                thread_id=conversation_id, checkpointer_type=saver or "memory",
                recursion_limit=recursion_limit)
        else:
            result = flow_run_manager.run_process_compat(
                flow_id=str(id),
                graph_data=data,
                inputs=request_inputs,
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
