"""历史摘要触发条件测试（mock LLM）。

摘要触发与 max_history_turns 对齐：消息条数 > max_history_turns×2 才调用 LLM。
"""

from unittest.mock import MagicMock, patch

from app.memory.history_summary import maybe_summarize_history
from app.memory.session_store import SessionStore


class TestHistorySummary:
    def test_within_keep_window_skips_llm(self, tmp_path):
        """消息未超过保留窗口时不调用 LLM。"""
        store = SessionStore(tmp_path / "s.db")
        store.append_turn("s1", "q1", "a1")  # 2 条；窗口=2轮→4 条
        with patch("app.memory.history_summary.get_chat_model") as mock_llm:
            with patch("app.memory.history_summary.get_settings") as gs:
                gs.return_value.max_history_turns = 2
                out = maybe_summarize_history(store, "s1")
        mock_llm.assert_not_called()
        assert out == ""

    def test_beyond_keep_window_writes_summary(self, tmp_path):
        """存在窗口外旧消息时写入摘要。"""
        store = SessionStore(tmp_path / "s.db")
        for i in range(6):
            store.append_turn("s1", f"q{i}", f"a{i}")  # 12 条；窗口=2轮→4 条
        fake = MagicMock()
        fake.invoke.return_value = MagicMock(content="摘要：讨论过多项制度")
        with patch("app.memory.history_summary.get_chat_model", return_value=fake):
            with patch("app.memory.history_summary.get_settings") as gs:
                gs.return_value.max_history_turns = 2
                out = maybe_summarize_history(store, "s1")
        assert "摘要" in out
        assert store.get_summary("s1") == out
        fake.invoke.assert_called_once()
