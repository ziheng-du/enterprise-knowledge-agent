"""历史摘要：会话过长时把更早轮次压成摘要，写入 SessionStore。

触发条件与「最近保留几轮原文」共用同一旋钮 max_history_turns：
消息条数超过 max_history_turns * 2 时，才对窗口外的旧消息做摘要。
摘要失败时仅记日志，调用方继续只用最近 K 轮原文。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import (
    HISTORY_SUMMARY_SYSTEM_PROMPT,
    HISTORY_SUMMARY_USER_PROMPT_TEMPLATE,
)
from app.config import get_settings
from app.llm import get_chat_model
from app.memory.query_rewrite import format_history_text
from app.memory.session_store import SessionStore
from app.utils.logger import get_logger

logger = get_logger(__name__)


def maybe_summarize_history(store: SessionStore, session_id: str) -> str:
    """若存在「最近保留窗口」之外的旧消息，则对其做摘要。

    保留窗口大小 = max_history_turns 轮 × 2（一问一答两条消息）。
    总消息不超过该窗口时不调用 LLM。

    Args:
        store: SessionStore 实例。
        session_id: 会话 ID。

    Returns:
        当前可用的摘要文本（可能为空；失败时返回已有摘要或空串）。
    """
    settings = get_settings()
    existing = store.get_summary(session_id)
    # 与注入 Prompt 的最近原文窗口对齐：只压窗口外的旧消息
    keep_n = settings.max_history_turns * 2
    all_msgs = store.get_all_messages(session_id)
    if len(all_msgs) <= keep_n:
        return existing

    older = all_msgs[:-keep_n]
    older_text = format_history_text(older, summary=existing)
    try:
        response = get_chat_model().invoke(
            [
                SystemMessage(content=HISTORY_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(
                    content=HISTORY_SUMMARY_USER_PROMPT_TEMPLATE.format(history=older_text)
                ),
            ]
        )
        summary = str(response.content).strip()
        if not summary:
            logger.warning("历史摘要为空，保留旧摘要: session_id=%s", session_id[:8])
            return existing
        store.set_summary(session_id, summary)
        logger.info("历史摘要已更新: session_id=%s, chars=%d", session_id[:8], len(summary))
        return summary
    except Exception:
        logger.exception("历史摘要失败，使用旧摘要继续: session_id=%s", session_id[:8])
        return existing
