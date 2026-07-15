"""LangGraph Agent 决策图：整合 RAG 检索与工具调用的核心流程。

图结构（多轮改写 + ReAct + 显式路由分诊）：

    START -> rewrite -> router -+-> (rag)  retrieve -> generate -> END
                               +-> (both) retrieve -> agent <-> execute_tools -> generate -> END
                               +-> (tool) agent <-> execute_tools -> generate -> END

架构约束：
- 状态用 state.py 的 AgentState 显式建模，节点函数只读写状态字段
- 工具列表运行时从 tools/registry.py 动态获取，本模块不 import 任何
  具体工具模块 —— 未来工具改造成 MCP Server 时本文件零改动
- 检索通过独立的 rag/retriever.py 完成，Agent 不感知向量库细节
- 跨轮记忆由 memory/session_store 管理，不使用 LangGraph Checkpointer

兜底逻辑（任何节点异常都不向上冒泡导致服务崩溃）：
- Query Rewrite 失败 -> 使用原始 question
- 路由 LLM 失败/输出不合法 -> 降级为 both
- 工具参数解析失败 -> ToolResult.error 以 ToolMessage 回喂 LLM 修正重试
- 工具轮次达上限仍有未执行的调用 -> 置 degraded，回答附不确定提示
- 检索与工具均无结果 -> 如实告知"未找到相关信息"，禁止编造
- 生成 LLM 失败 -> 返回固定道歉话术
"""

from __future__ import annotations

import json
import time
import uuid
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    FINAL_ANSWER_DEGRADED_NOTICE,
    FINAL_ANSWER_EMPTY_CONTEXT,
    FINAL_ANSWER_SYSTEM_PROMPT,
    FINAL_ANSWER_USER_PROMPT_TEMPLATE,
    FINAL_ANSWER_USER_PROMPT_WITH_HISTORY_TEMPLATE,
    RAG_QA_NO_RESULT_ANSWER,
    ROUTER_SYSTEM_PROMPT,
    ROUTER_USER_PROMPT_TEMPLATE,
    ROUTER_USER_PROMPT_WITH_HISTORY_TEMPLATE,
)
from app.agent.state import AgentState, ChatTurn, ToolCallRecord
from app.config import get_settings
from app.llm import get_chat_model
from app.memory.context_budget import (
    apply_history_budget,
    apply_retrieval_budget,
    apply_tool_results_budget,
)
from app.memory.history_summary import maybe_summarize_history
from app.memory.query_rewrite import format_history_text, rewrite_query
from app.memory.session_store import get_session_store
from app.rag.access_control import normalize_role
from app.rag.qa_chain import format_context
from app.rag.retriever import Retriever
from app.tools.base import BaseTool
from app.tools.registry import get_all_tools, get_tool
from app.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_ROUTES = {"rag", "tool", "both"}

# LLM 生成失败时的固定兜底回答（不让异常冒泡到 API 层导致 500）
_GENERATION_FAILED_ANSWER = "抱歉，系统暂时无法生成回答，请稍后重试或咨询人力资源部。"


class AgentResult(BaseModel):
    """一次 Agent 问答的结构化结果（API 层的唯一数据来源）。"""

    answer: str = Field(description="最终回答文本")
    sources: list[str] = Field(default_factory=list, description="引用的来源文件名（去重）")
    route: str = Field(default="", description="分诊路由（rag/tool/both）")
    tool_calls: list[dict] = Field(default_factory=list, description="工具调用历史")
    used_retrieval: bool = Field(default=False, description="是否使用了知识库检索")
    degraded: bool = Field(default=False, description="回答过程是否发生过降级")
    session_id: str = Field(default="", description="会话 ID")
    user_role: str = Field(default="", description="模拟角色（intern/employee/admin）")
    search_query: str = Field(default="", description="实际用于检索的 query（可能经改写）")
    request_id: str = Field(default="", description="本次请求追踪 ID")
    timings: dict[str, float] = Field(
        default_factory=dict,
        description="各节点耗时（毫秒）",
    )


def _to_llm_tool_spec(tool: BaseTool) -> dict:
    """把 BaseTool 转换为 OpenAI 工具调用协议的声明格式。

    这是工具抽象层与 LLM 协议之间唯一的适配点：消费 BaseTool 的
    name / description / input_schema 三个稳定接口。未来工具改造成
    MCP Server 时，MCP 的工具声明同样从这三个字段生成，业务零改动。

    Args:
        tool: 注册表中的工具实例。

    Returns:
        OpenAI function calling 格式的工具声明 dict。
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema.model_json_schema(),
        },
    }


@lru_cache
def _get_retriever() -> Retriever:
    """惰性构造检索器（首次调用加载 embedding 模型，lru_cache 复用）。"""
    return Retriever()


def _history_block(state: AgentState) -> str:
    """组装并按预算裁剪的历史文本。"""
    raw = format_history_text(
        list(state.get("chat_history") or []),
        summary=state.get("session_summary") or "",
    )
    return apply_history_budget(raw)


def _record_timing(state: AgentState, node: str, elapsed_ms: float) -> dict[str, float]:
    """合并节点耗时到 state 副本用的 timings dict。"""
    timings = dict(state.get("node_timings") or {})
    timings[node] = round(elapsed_ms, 2)
    return timings


def _extract_hire_date_from_tools(history: list[ToolCallRecord]) -> str | None:
    """从成功的 leave_calculator 调用参数中提取入职日期。"""
    for rec in history:
        if rec.get("tool_name") == "leave_calculator" and rec.get("success"):
            args = rec.get("args") or {}
            hire = args.get("hire_date")
            if hire:
                return str(hire)
    return None


# --- 节点函数 ---


def rewrite_node(state: AgentState) -> dict:
    """Query Rewrite 节点：多轮指代消解，写出独立 search_query。

    Args:
        state: 读 question / chat_history / session_summary。

    Returns:
        {"search_query": ..., "node_timings": ...}。
    """
    started = time.perf_counter()
    question = state["question"]
    search_query = rewrite_query(
        question,
        list(state.get("chat_history") or []),
        summary=state.get("session_summary") or "",
    )
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "search_query": search_query,
        "node_timings": _record_timing(state, "rewrite", elapsed),
    }


def route_node(state: AgentState) -> dict:
    """分诊节点：LLM 判断问题走 rag / tool / both 哪条路径。

    Args:
        state: 当前状态（读 question / chat_history）。

    Returns:
        {"route": "rag"|"tool"|"both", "node_timings": ...}。

    兜底：LLM 输出不在合法集合内或调用异常时，降级为 "both"。
    """
    started = time.perf_counter()
    question = state["question"]
    history_text = _history_block(state)
    if history_text and history_text != "（无）":
        user_content = ROUTER_USER_PROMPT_WITH_HISTORY_TEMPLATE.format(
            history=history_text, question=question
        )
    else:
        user_content = ROUTER_USER_PROMPT_TEMPLATE.format(question=question)

    try:
        response = get_chat_model().invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]
        )
        route = str(response.content).strip().lower()
    except Exception:
        logger.exception("路由 LLM 调用失败，降级为 both: question=%r", question[:50])
        elapsed = (time.perf_counter() - started) * 1000
        return {"route": "both", "node_timings": _record_timing(state, "router", elapsed)}

    if route not in _VALID_ROUTES:
        logger.warning("路由输出不合法(%r)，降级为 both", route)
        route = "both"

    elapsed = (time.perf_counter() - started) * 1000
    logger.info("分诊完成: route=%s, question=%r", route, question[:50])
    return {"route": route, "node_timings": _record_timing(state, "router", elapsed)}


def retrieve_node(state: AgentState) -> dict:
    """检索节点：使用 search_query（经改写）调用独立 Retriever。

    Args:
        state: 读 search_query / question。

    Returns:
        {"retrieved": [...], "sources": [...], ...}。
    """
    started = time.perf_counter()
    query = (state.get("search_query") or state["question"]).strip()
    try:
        results = _get_retriever().retrieve(query, user_role=state.get("user_role"))
        results = apply_retrieval_budget(results)
    except Exception:
        logger.exception("检索失败，以空结果继续: query=%r", query[:50])
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "retrieved": [],
            "sources": [],
            "degraded": True,
            "node_timings": _record_timing(state, "retrieve", elapsed),
        }

    sources = list(dict.fromkeys(r.source for r in results))
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "retrieved": results,
        "sources": sources,
        "node_timings": _record_timing(state, "retrieve", elapsed),
    }


def agent_node(state: AgentState) -> dict:
    """工具决策节点：LLM（绑定工具）决定是否调用工具及参数。"""
    started = time.perf_counter()
    messages: list[BaseMessage] = list(state["messages"])

    if not messages:
        history_text = _history_block(state)
        profile = state.get("user_profile") or {}
        profile_hint = ""
        if profile.get("hire_date"):
            profile_hint = f"\n已知员工画像：入职日期={profile['hire_date']}（缺参数时可使用，勿编造其他信息）"

        user_content = f"员工问题：{state['question']}{profile_hint}"
        if history_text and history_text != "（无）":
            user_content = f"近期对话历史：\n{history_text}\n\n{user_content}"
        if state["retrieved"]:
            user_content = (
                f"参考资料：\n{format_context(state['retrieved'])}\n\n{user_content}"
            )
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), HumanMessage(content=user_content)]

    tool_specs = [_to_llm_tool_spec(t) for t in get_all_tools()]
    llm_with_tools = get_chat_model().bind_tools(tool_specs)

    try:
        response = llm_with_tools.invoke(messages)
    except Exception:
        logger.exception("工具决策 LLM 调用失败，跳过工具循环")
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "messages": messages,
            "degraded": True,
            "node_timings": _record_timing(state, "agent", elapsed),
        }

    elapsed = (time.perf_counter() - started) * 1000
    return {
        "messages": messages + [response],
        "node_timings": _record_timing(state, "agent", elapsed),
    }


def execute_tools_node(state: AgentState) -> dict:
    """工具执行节点：执行 LLM 请求的工具调用，结果回喂消息序列。"""
    started = time.perf_counter()
    messages: list[BaseMessage] = list(state["messages"])
    history: list[ToolCallRecord] = list(state["tool_calls_history"])
    last = messages[-1]
    profile_updates: dict[str, Any] = {}

    for call in last.tool_calls:
        tool_name, args, call_id = call["name"], call["args"], call["id"]
        try:
            # 注入 user_role：供 policy_lookup 等工具做密级过滤（不由 LLM 伪造）
            result = get_tool(tool_name).invoke(
                args,
                context={"user_role": state.get("user_role")},
            )
            if result.success:
                content = json.dumps(result.data, ensure_ascii=False, default=str)
            else:
                content = f"调用出错：{result.error}"
            success = result.success
        except KeyError as exc:
            content = f"调用出错：{exc}"
            success = False

        messages.append(ToolMessage(content=content, tool_call_id=call_id))
        history.append(
            ToolCallRecord(tool_name=tool_name, args=args, success=success, result=content)
        )
        logger.info("工具执行: name=%s, success=%s", tool_name, success)

        if success and tool_name == "leave_calculator" and args.get("hire_date"):
            profile_updates["hire_date"] = str(args["hire_date"])

    elapsed = (time.perf_counter() - started) * 1000
    out: dict[str, Any] = {
        "messages": messages,
        "tool_calls_history": history,
        "current_round": state["current_round"] + 1,
        "node_timings": _record_timing(state, "execute_tools", elapsed),
    }
    if profile_updates:
        merged = dict(state.get("user_profile") or {})
        merged.update(profile_updates)
        out["user_profile"] = merged
    return out


def generate_node(state: AgentState) -> dict:
    """综合生成节点：汇总检索资料与工具结果，生成最终回答。"""
    started = time.perf_counter()
    question = state["question"]
    retrieved = apply_retrieval_budget(list(state["retrieved"] or []))
    history = state["tool_calls_history"]
    degraded = state["degraded"]

    messages = state["messages"]
    last = messages[-1] if messages else None

    if isinstance(last, AIMessage) and last.tool_calls:
        logger.warning("工具轮次达上限(%d)仍有未执行调用，降级作答", state["current_round"])
        degraded = True

    if not retrieved and not history:
        if isinstance(last, AIMessage) and not last.tool_calls and str(last.content).strip():
            elapsed = (time.perf_counter() - started) * 1000
            return {
                "final_answer": str(last.content).strip(),
                "sources": [],
                "degraded": degraded,
                "node_timings": _record_timing(state, "generate", elapsed),
            }
        logger.info("检索与工具均无结果，返回未找到话术: question=%r", question[:50])
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "final_answer": RAG_QA_NO_RESULT_ANSWER,
            "sources": [],
            "degraded": degraded,
            "node_timings": _record_timing(state, "generate", elapsed),
        }

    context = format_context(retrieved) if retrieved else FINAL_ANSWER_EMPTY_CONTEXT
    tool_results = (
        "\n".join(
            f"- 工具 {rec['tool_name']}（参数 {json.dumps(rec['args'], ensure_ascii=False)}）："
            f"{'成功，结果 ' + rec['result'] if rec['success'] else '失败，' + rec['result']}"
            for rec in history
        )
        if history
        else FINAL_ANSWER_EMPTY_CONTEXT
    )
    tool_results = apply_tool_results_budget(tool_results)
    history_text = _history_block(state)

    system_prompt = FINAL_ANSWER_SYSTEM_PROMPT + (FINAL_ANSWER_DEGRADED_NOTICE if degraded else "")
    if history_text and history_text != "（无）":
        user_prompt = FINAL_ANSWER_USER_PROMPT_WITH_HISTORY_TEMPLATE.format(
            history=history_text,
            context=context,
            tool_results=tool_results,
            question=question,
        )
    else:
        user_prompt = FINAL_ANSWER_USER_PROMPT_TEMPLATE.format(
            context=context, tool_results=tool_results, question=question
        )

    try:
        response = get_chat_model().invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        answer = str(response.content).strip()
    except Exception:
        logger.exception("生成回答 LLM 调用失败: question=%r", question[:50])
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "final_answer": _GENERATION_FAILED_ANSWER,
            "sources": state["sources"],
            "degraded": True,
            "node_timings": _record_timing(state, "generate", elapsed),
        }

    elapsed = (time.perf_counter() - started) * 1000
    logger.info("回答生成完成: question=%r, degraded=%s", question[:50], degraded)
    return {
        "final_answer": answer,
        "sources": state["sources"],
        "degraded": degraded,
        "node_timings": _record_timing(state, "generate", elapsed),
    }


# --- 条件边 ---


def _after_route(state: AgentState) -> str:
    """route 之后的分支：rag/both 先检索，tool 直接进工具循环。"""
    return "retrieve" if state["route"] in ("rag", "both") else "agent"


def _after_retrieve(state: AgentState) -> str:
    """retrieve 之后的分支：纯 rag 直接生成，both 进工具循环。"""
    return "generate" if state["route"] == "rag" else "agent"


def _after_agent(state: AgentState) -> str:
    """agent 之后的分支：有 tool_calls 且轮次未达上限则执行工具，否则生成。"""
    last = state["messages"][-1] if state["messages"] else None
    has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
    if has_tool_calls and state["current_round"] < get_settings().max_tool_rounds:
        return "execute_tools"
    return "generate"


# --- 图构建与运行入口 ---


@lru_cache
def build_agent_graph():
    """构建并编译 Agent 决策状态图（lru_cache 单例）。

    Returns:
        编译后的 LangGraph 图对象，可直接 invoke(AgentState)。
    """
    graph = StateGraph(AgentState)
    # 注：节点名不能与 AgentState 的状态键重名（LangGraph 限制）
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("router", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("agent", agent_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "router")
    graph.add_conditional_edges("router", _after_route, {"retrieve": "retrieve", "agent": "agent"})
    graph.add_conditional_edges(
        "retrieve", _after_retrieve, {"generate": "generate", "agent": "agent"}
    )
    graph.add_conditional_edges(
        "agent", _after_agent, {"execute_tools": "execute_tools", "generate": "generate"}
    )
    graph.add_edge("execute_tools", "agent")
    graph.add_edge("generate", END)

    return graph.compile()


def run_agent(
    question: str,
    session_id: str | None = None,
    user_role: str | None = None,
) -> AgentResult:
    """Agent 问答统一入口（API 层的唯一调用点）。

    Args:
        question: 用户问题文本。
        session_id: 可选会话 ID；为空时自动生成，用于跨轮记忆。
        user_role: 登录用户角色 intern/employee/admin，用于文档密级过滤。

    Returns:
        AgentResult：回答、来源、路由、工具调用、会话与耗时等。
        任何内部异常都被兜底为可读的回答文本，不向上抛。
    """
    request_id = uuid.uuid4().hex
    role = normalize_role(user_role)
    if not question or not question.strip():
        return AgentResult(
            answer="请输入有效的问题。",
            session_id=session_id or "",
            user_role=role,
            request_id=request_id,
        )

    sid = (session_id or "").strip() or uuid.uuid4().hex
    store = get_session_store()
    store.ensure_session(sid)

    # 超长会话时先尝试摘要更早轮次（失败不阻断）
    session_summary = maybe_summarize_history(store, sid)
    chat_history: list[ChatTurn] = [
        {"role": m["role"], "content": m["content"]}
        for m in store.get_history(sid)
    ]
    user_profile = store.get_or_create_profile(sid)

    initial_state: AgentState = {
        "question": question.strip(),
        "session_id": sid,
        "user_role": role,
        "chat_history": chat_history,
        "session_summary": session_summary,
        "user_profile": user_profile,
        "search_query": "",
        "route": "",
        "retrieved": [],
        "messages": [],
        "tool_calls_history": [],
        "current_round": 0,
        "degraded": False,
        "node_timings": {},
        "final_answer": "",
        "sources": [],
    }

    logger.info(
        "Agent 开始: request_id=%s session_id=%s role=%s history_turns=%d question=%r",
        request_id,
        sid[:8],
        role,
        len(chat_history) // 2,
        question[:50],
    )

    try:
        final_state = build_agent_graph().invoke(initial_state)
    except Exception:
        logger.exception(
            "Agent 图执行失败: request_id=%s question=%r", request_id, question[:50]
        )
        return AgentResult(
            answer=_GENERATION_FAILED_ANSWER,
            degraded=True,
            session_id=sid,
            user_role=role,
            request_id=request_id,
        )

    answer = final_state["final_answer"]
    # 持久化本轮对话与画像
    try:
        store.append_turn(sid, question.strip(), answer)
        hire = _extract_hire_date_from_tools(final_state["tool_calls_history"])
        profile = dict(final_state.get("user_profile") or {})
        if hire:
            profile["hire_date"] = hire
        if profile:
            store.update_profile(sid, **profile)
    except Exception:
        logger.exception("会话持久化失败: session_id=%s", sid[:8])

    timings = dict(final_state.get("node_timings") or {})
    logger.info(
        "Agent 完成: request_id=%s session_id=%s route=%s degraded=%s timings=%s",
        request_id,
        sid[:8],
        final_state["route"],
        final_state["degraded"],
        timings,
    )

    return AgentResult(
        answer=answer,
        sources=final_state["sources"],
        route=final_state["route"],
        tool_calls=[dict(rec) for rec in final_state["tool_calls_history"]],
        used_retrieval=bool(final_state["retrieved"]),
        degraded=final_state["degraded"],
        session_id=sid,
        user_role=role,
        search_query=final_state.get("search_query") or "",
        request_id=request_id,
        timings=timings,
    )
