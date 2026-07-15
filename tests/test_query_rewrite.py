"""Query Rewrite 节点与 format 辅助函数测试（mock LLM）。"""

from unittest.mock import MagicMock, patch

from app.agent.graph import rewrite_node
from app.memory.query_rewrite import format_history_text, rewrite_query


class TestFormatHistory:
    def test_empty(self):
        assert format_history_text([]) == "（无）"

    def test_with_summary_and_turns(self):
        text = format_history_text(
            [{"role": "user", "content": "年假"}, {"role": "assistant", "content": "3天"}],
            summary="此前聊过入职",
        )
        assert "更早对话摘要" in text
        assert "年假" in text


class TestRewriteQuery:
    def test_no_history_returns_original(self):
        assert rewrite_query("报销上限", []) == "报销上限"

    def test_rewrite_success(self):
        fake = MagicMock()
        fake.invoke.return_value = MagicMock(content="办公用品报销额度上限")
        with patch("app.memory.query_rewrite.get_settings") as gs:
            gs.return_value.enable_query_rewrite = True
            with patch("app.memory.query_rewrite.get_chat_model", return_value=fake):
                out = rewrite_query(
                    "那上限呢",
                    [{"role": "user", "content": "报销制度"}, {"role": "assistant", "content": "..."}],
                )
        assert out == "办公用品报销额度上限"

    def test_rewrite_failure_falls_back(self):
        fake = MagicMock()
        fake.invoke.side_effect = RuntimeError("boom")
        with patch("app.memory.query_rewrite.get_settings") as gs:
            gs.return_value.enable_query_rewrite = True
            with patch("app.memory.query_rewrite.get_chat_model", return_value=fake):
                out = rewrite_query(
                    "那上限呢",
                    [{"role": "user", "content": "报销"}],
                )
        assert out == "那上限呢"


class TestRewriteNode:
    def test_rewrite_node_writes_search_query(self):
        state = {
            "question": "那报销呢",
            "chat_history": [{"role": "user", "content": "年假"}, {"role": "assistant", "content": "3天"}],
            "session_summary": "",
            "node_timings": {},
        }
        with patch("app.agent.graph.rewrite_query", return_value="报销制度相关规定"):
            out = rewrite_node(state)
        assert out["search_query"] == "报销制度相关规定"
        assert "rewrite" in out["node_timings"]
