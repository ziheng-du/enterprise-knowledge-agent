"""多轮 Query Rewrite：把带指代的追问改写成独立检索 query。

失败时降级为原始 question，不中断主流程。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    QUERY_REWRITE_USER_PROMPT_TEMPLATE,
)
from app.config import get_settings
from app.llm import get_chat_model
from app.utils.logger import get_logger

logger = get_logger(__name__)


def format_history_text(
    chat_history: list[dict[str, str]],
    summary: str = "",
    max_chars: int | None = None,
) -> str:
    """把历史消息格式化为 Prompt 可读文本。

    Args:
        chat_history: [{"role","content"}, ...]。
        summary: 更早轮次的摘要（可空）。
        max_chars: 可选字符上限；超出则从头部截断（保留最近内容）。

    Returns:
        多行文本；无内容时返回「（无）」。
    """
    parts: list[str] = []
    if summary and summary.strip():
        parts.append(f"更早对话摘要：{summary.strip()}")
    for msg in chat_history:
        role = "员工" if msg.get("role") == "user" else "助手"
        content = (msg.get("content") or "").strip()
        if content:
            parts.append(f"{role}：{content}")
    text = "\n".join(parts) if parts else "（无）"
    if max_chars is not None and len(text) > max_chars:
        text = "…\n" + text[-(max_chars - 2) :]
    return text


def rewrite_query(
    question: str,
    chat_history: list[dict[str, str]],
    summary: str = "",
) -> str:
    """根据会话历史将当前问题改写成独立检索 query。

    Args:
        question: 当前用户原问题。
        chat_history: 最近若干轮历史。
        summary: 更早对话摘要。

    Returns:
        改写后的检索 query；无历史、关闭改写或 LLM 失败时返回原问题。
    """
    settings = get_settings()
    if not settings.enable_query_rewrite:
        return question
    if not chat_history and not (summary and summary.strip()):
        return question

    history_text = format_history_text(chat_history, summary=summary)
    try:
        response = get_chat_model().invoke(
            [
                SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT),
                HumanMessage(
                    content=QUERY_REWRITE_USER_PROMPT_TEMPLATE.format(
                        history=history_text,
                        question=question,
                    )
                ),
            ]
        )
        rewritten = str(response.content).strip()
        # 边界：模型可能输出空或解释性长文；过长则降级
        if not rewritten or len(rewritten) > 500:
            logger.warning("Query Rewrite 输出异常，降级为原问题")
            return question
        logger.info("Query Rewrite: %r -> %r", question[:40], rewritten[:40])
        return rewritten
    except Exception:
        logger.exception("Query Rewrite 失败，降级为原问题: question=%r", question[:50])
        return question
