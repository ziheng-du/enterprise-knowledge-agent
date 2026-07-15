"""SessionStore 单元测试（临时 SQLite，不依赖 LLM）。"""

from app.memory.session_store import SessionStore


class TestSessionStore:
    """会话读写与画像更新。"""

    def test_append_and_get_history(self, tmp_path):
        store = SessionStore(tmp_path / "s.db")
        store.append_turn("s1", "年假几天", "3天")
        store.append_turn("s1", "那报销呢", "上限500")
        history = store.get_history("s1", max_turns=6)
        assert len(history) == 4
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "年假几天"
        assert history[-1]["content"] == "上限500"

    def test_history_turn_limit(self, tmp_path):
        store = SessionStore(tmp_path / "s.db")
        for i in range(5):
            store.append_turn("s1", f"q{i}", f"a{i}")
        history = store.get_history("s1", max_turns=2)
        assert len(history) == 4
        assert history[0]["content"] == "q3"
        assert history[-1]["content"] == "a4"

    def test_profile_update(self, tmp_path):
        store = SessionStore(tmp_path / "s.db")
        profile = store.update_profile("s1", hire_date="2023-07-01")
        assert profile["hire_date"] == "2023-07-01"
        again = store.get_or_create_profile("s1")
        assert again["hire_date"] == "2023-07-01"

    def test_summary_roundtrip(self, tmp_path):
        store = SessionStore(tmp_path / "s.db")
        store.set_summary("s1", "讨论过年假与报销")
        assert store.get_summary("s1") == "讨论过年假与报销"
