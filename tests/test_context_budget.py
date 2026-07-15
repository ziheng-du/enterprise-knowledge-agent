"""上下文预算纯函数单测。"""

from app.memory.context_budget import (
    apply_history_budget,
    apply_retrieval_budget,
    apply_tool_results_budget,
    truncate_tail,
)
from app.rag.retriever import RetrievalResult


class TestContextBudget:
    """截断与检索预算。"""

    def test_truncate_tail_keeps_end(self):
        text = "abcdefghij"
        assert truncate_tail(text, 5).endswith("j")
        assert len(truncate_tail(text, 5)) == 5

    def test_history_budget(self):
        long_text = "历史" * 100
        out = apply_history_budget(long_text, max_chars=20)
        assert len(out) <= 20

    def test_retrieval_budget_keeps_prefix(self):
        items = [
            RetrievalResult(content="a" * 10, source="s1", score=0.9),
            RetrievalResult(content="b" * 10, source="s2", score=0.8),
            RetrievalResult(content="c" * 10, source="s3", score=0.7),
        ]
        kept = apply_retrieval_budget(items, max_chars=25)
        assert len(kept) == 2
        assert kept[0].source == "s1"

    def test_tool_results_budget(self):
        text = "工具结果" * 50
        out = apply_tool_results_budget(text, max_chars=30)
        assert len(out) <= 30
