"""FastAPI 路由层：对外暴露 chat / ingest / health 等接口。

请求/响应体全部用 Pydantic BaseModel 定义，自动生成 Swagger 文档。
业务逻辑委托给 agent.graph.run_agent 与 rag.ingest_service.ingest，
本模块只做参数校验、异常兜底与 HTTP 层适配。

聊天接口的角色来自登录 token（见 auth_routes），不再信任请求体传 role。
"""

import asyncio
import json
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import AgentResult, run_agent
from app.auth.deps import get_current_user
from app.auth.schemas import AuthUser
from app.memory.session_store import get_session_store
from app.rag.ingest_service import ingest
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["api"])


# --- 请求 / 响应模型 ---


class ChatRequest(BaseModel):
    """聊天请求体。角色由 Authorization Bearer 决定，不在此字段传入。"""

    message: str = Field(min_length=1, description="用户问题文本")
    session_id: str | None = Field(
        default=None,
        description="可选会话 ID；不传则服务端生成并在响应中回显，用于多轮记忆",
    )


class ToolCallInfo(BaseModel):
    """单次工具调用的展示信息（供前端展示思考过程）。"""

    tool_name: str = Field(description="工具名称")
    args: dict[str, Any] = Field(default_factory=dict, description="调用参数")
    success: bool = Field(description="是否执行成功")
    result: str = Field(description="结果摘要或错误信息")


class ChatResponse(BaseModel):
    """聊天响应体，字段与 AgentResult 一一对应。"""

    answer: str = Field(description="最终回答文本")
    sources: list[str] = Field(default_factory=list, description="引用的来源文件名")
    route: str = Field(default="", description="分诊路由：rag / tool / both")
    used_retrieval: bool = Field(default=False, description="是否使用了知识库检索")
    tool_calls: list[ToolCallInfo] = Field(default_factory=list, description="工具调用历史")
    degraded: bool = Field(default=False, description="回答过程是否发生过降级")
    session_id: str = Field(default="", description="会话 ID（请在后续请求中回传）")
    user_role: str = Field(default="", description="实际生效的角色（来自登录 token）")
    search_query: str = Field(default="", description="实际检索 query（可能经多轮改写）")
    request_id: str = Field(default="", description="请求追踪 ID")
    timings: dict[str, float] = Field(
        default_factory=dict,
        description="各节点耗时（毫秒）",
    )


class IngestRequest(BaseModel):
    """文档重新入库请求体。"""

    rebuild: bool = Field(default=True, description="是否先清空向量库再全量重建")
    strategy: Literal["fixed", "recursive"] | None = Field(
        default=None, description="切分策略；None 时使用配置默认值"
    )


class IngestResponse(BaseModel):
    """文档入库响应体。"""

    chunk_count: int = Field(description="写入的文本块数量")
    message: str = Field(description="结果说明")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field(description="服务状态")


def _to_chat_response(result: AgentResult) -> ChatResponse:
    """把 AgentResult 转为 API 响应模型。"""
    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        route=result.route,
        used_retrieval=result.used_retrieval,
        tool_calls=[ToolCallInfo(**call) for call in result.tool_calls],
        degraded=result.degraded,
        session_id=result.session_id,
        user_role=result.user_role,
        search_query=result.search_query,
        request_id=result.request_id,
        timings=result.timings,
    )


# --- 接口 ---


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """健康检查，便于启动自检与探活。"""
    return HealthResponse(status="ok")


def _prepare_session_for_user(
    session_id: str | None,
    hire_date: str | None,
) -> str | None:
    """准备会话 ID，并把登录用户的入职日期写入画像（若尚无）。

    Args:
        session_id: 客户端传来的会话 ID；空则新生成。
        hire_date: 花名册中的入职日期。

    Returns:
        将传给 run_agent 的 session_id；异常时仍返回原/新 ID，不阻断聊天。
    """
    sid = (session_id or "").strip() or uuid.uuid4().hex
    if not hire_date:
        return sid
    try:
        store = get_session_store()
        profile = store.get_or_create_profile(sid)
        if not profile.get("hire_date"):
            store.update_profile(sid, hire_date=hire_date)
    except Exception:
        logger.exception("写入会话入职日期失败: session_id=%r", sid[:8])
    return sid


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> ChatResponse:
    """同步聊天接口：接收用户消息，返回 Agent 回答与思考过程元数据。

    Args:
        request: 含 message 与可选 session_id 的请求体。
        current_user: 由 Bearer token 解析的登录用户（角色不可由客户端伪造）。

    Returns:
        ChatResponse：回答文本 + 来源 + 路由 + 工具调用 + 会话信息。

    Raises:
        HTTPException 401: 未登录或 token 无效。
        HTTPException 500: Agent 执行出现未预期异常。
    """
    sid = _prepare_session_for_user(request.session_id, current_user.hire_date)
    try:
        result = run_agent(
            request.message,
            session_id=sid,
            user_role=current_user.role,
        )
    except Exception as exc:
        logger.exception("聊天接口执行失败: message=%r", request.message[:50])
        raise HTTPException(status_code=500, detail=f"问答服务暂时不可用: {exc}") from exc
    return _to_chat_response(result)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> StreamingResponse:
    """伪流式聊天接口：Agent 跑完后以 SSE 按块推送回答。

    事件顺序：
    1. event=meta — sources / route / tool_calls / session_id / timings 等
    2. event=token — answer 文本小块（演示打字机效果）
    3. event=done — 流结束

    说明：真·token 流式需要改造 LangGraph generate 节点，本接口为伪流式。
    角色同样来自登录 token，不信任请求体。
    """
    user_role = current_user.role
    sid = _prepare_session_for_user(request.session_id, current_user.hire_date)

    def _sse(event: str, data: dict | str) -> str:
        payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {payload}\n\n"

    async def event_generator():
        try:
            result = await asyncio.to_thread(
                run_agent,
                request.message,
                sid,
                user_role,
            )
        except Exception as exc:
            logger.exception("流式聊天接口执行失败: message=%r", request.message[:50])
            yield _sse("error", {"detail": f"问答服务暂时不可用: {exc}"})
            yield _sse("done", {})
            return

        yield _sse(
            "meta",
            {
                "sources": result.sources,
                "route": result.route,
                "used_retrieval": result.used_retrieval,
                "tool_calls": result.tool_calls,
                "degraded": result.degraded,
                "session_id": result.session_id,
                "user_role": result.user_role,
                "search_query": result.search_query,
                "request_id": result.request_id,
                "timings": result.timings,
            },
        )

        chunk_size = 8
        answer = result.answer or ""
        for i in range(0, len(answer), chunk_size):
            yield _sse("token", {"text": answer[i : i + chunk_size]})
            await asyncio.sleep(0.02)

        yield _sse("done", {})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(request: IngestRequest) -> IngestResponse:
    """触发文档重新入库（load → split → 写入向量库 → 刷新 BM25）。

    Args:
        request: rebuild / strategy 参数。

    Returns:
        IngestResponse：写入块数与说明。

    Raises:
        HTTPException: 入库过程异常时返回 500。
    """
    try:
        chunk_count = ingest(rebuild=request.rebuild, strategy=request.strategy)
    except Exception as exc:
        logger.exception("入库接口执行失败")
        raise HTTPException(status_code=500, detail=f"文档入库失败: {exc}") from exc

    if chunk_count == 0:
        return IngestResponse(
            chunk_count=0,
            message="未写入任何文本块，请检查 data/raw_docs 目录是否有可加载的文档",
        )
    return IngestResponse(
        chunk_count=chunk_count,
        message=f"入库成功，共写入 {chunk_count} 个文本块",
    )
