"""上下文预算：在注入 Prompt 前按字符上限截断历史 / 检索 / 工具结果。

不改动检索模块本身；仅在 Agent 组装最终上下文时调用。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.rag.retriever import RetrievalResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BudgetedContext:
    """预算裁剪后的上下文字符串与检索结果。"""

    history_text: str
    retrieved: list[RetrievalResult]
    tool_results_text: str


def truncate_tail(text: str, max_chars: int) -> str:
    """保留文本尾部（最近内容），超限时加省略前缀。

    Args:
        text: 原始文本。
        max_chars: 最大字符数。

    Returns:
        裁剪后的文本。
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 2:
        return text[-max_chars:]
    return "…" + text[-(max_chars - 1) :]


def apply_retrieval_budget(
    retrieved: list[RetrievalResult],
    max_chars: int | None = None,
) -> list[RetrievalResult]:
    """按字符预算从前往后保留检索片段（高相关优先）。

    Args:
        retrieved: 已按分数降序的检索结果。
        max_chars: 内容总字符上限。None 时读配置。

    Returns:
        截断后的结果列表（可能为空）。
    """
    limit = max_chars if max_chars is not None else get_settings().max_retrieval_chars
    kept: list[RetrievalResult] = []
    used = 0
    for item in retrieved:
        piece_len = len(item.content)
        if kept and used + piece_len > limit:
            break
        if not kept and piece_len > limit:
            # 第一条就超限：硬截断内容，仍保留一条供引用
            truncated = item.model_copy(update={"content": item.content[:limit]})
            kept.append(truncated)
            break
        kept.append(item)
        used += piece_len
    if len(kept) < len(retrieved):
        logger.info("检索上下文超预算，保留 %d/%d 条", len(kept), len(retrieved))
    return kept


def apply_tool_results_budget(tool_results_text: str, max_chars: int | None = None) -> str:
    """截断工具结果文本。

    Args:
        tool_results_text: 已拼接的工具结果说明。
        max_chars: 字符上限。None 时读配置。

    Returns:
        裁剪后的文本。
    """
    limit = max_chars if max_chars is not None else get_settings().max_tool_result_chars
    return truncate_tail(tool_results_text, limit)


def apply_history_budget(history_text: str, max_chars: int | None = None) -> str:
    """截断历史对话文本（保留最近部分）。

    Args:
        history_text: 格式化后的历史。
        max_chars: 字符上限。None 时读配置。

    Returns:
        裁剪后的文本。
    """
    limit = max_chars if max_chars is not None else get_settings().max_history_chars
    return truncate_tail(history_text, limit)
