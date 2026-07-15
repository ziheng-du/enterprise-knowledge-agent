"""文档密级 ACL 单元测试（不依赖 Embedding / LLM）。"""

from unittest.mock import MagicMock

from app.rag.access_control import role_can_access
from app.rag.retriever import RetrievalResult, Retriever


class TestRoleCanAccess:
    def test_intern_public_only(self):
        assert role_can_access("intern", "public") is True
        assert role_can_access("intern", "internal") is False
        assert role_can_access("intern", "confidential") is False

    def test_employee_up_to_internal(self):
        assert role_can_access("employee", "public") is True
        assert role_can_access("employee", "internal") is True
        assert role_can_access("employee", "confidential") is False

    def test_admin_all(self):
        assert role_can_access("admin", "confidential") is True


class TestRetrieverAclFilter:
    def test_same_query_different_roles(self, monkeypatch):
        from app.rag import retriever as retriever_mod

        class Settings:
            enable_access_control = True
            retrieval_mode = "vector"
            retrieval_top_k = 4
            retrieval_score_threshold = 0.0

        monkeypatch.setattr(retriever_mod, "get_settings", lambda: Settings())

        results = [
            RetrievalResult(
                content="报销 30 天",
                source="报销制度.md",
                score=0.9,
                metadata={"access_level": "internal", "source": "报销制度.md"},
            ),
            RetrievalResult(
                content="高管薪酬带宽",
                source="高管薪酬指引.md",
                score=0.85,
                metadata={"access_level": "confidential", "source": "高管薪酬指引.md"},
            ),
            RetrievalResult(
                content="考勤规定",
                source="员工手册.md",
                score=0.8,
                metadata={"access_level": "public", "source": "员工手册.md"},
            ),
        ]

        r = Retriever(vector_store=MagicMock())
        monkeypatch.setattr(
            r,
            "_retrieve_vector",
            lambda query, top_k, score_threshold: list(results),
        )

        intern_hits = r.retrieve("制度", user_role="intern")
        assert {x.source for x in intern_hits} == {"员工手册.md"}

        employee_hits = r.retrieve("制度", user_role="employee")
        assert {x.source for x in employee_hits} == {"员工手册.md", "报销制度.md"}

        admin_hits = r.retrieve("制度", user_role="admin")
        assert {x.source for x in admin_hits} == {
            "员工手册.md",
            "报销制度.md",
            "高管薪酬指引.md",
        }

    def test_acl_disabled_keeps_all(self, monkeypatch):
        from app.rag import retriever as retriever_mod

        class Settings:
            enable_access_control = False
            retrieval_mode = "vector"
            retrieval_top_k = 4
            retrieval_score_threshold = 0.0

        monkeypatch.setattr(retriever_mod, "get_settings", lambda: Settings())
        results = [
            RetrievalResult(
                content="x",
                source="高管薪酬指引.md",
                score=0.9,
                metadata={"access_level": "confidential"},
            )
        ]
        r = Retriever(vector_store=MagicMock())
        monkeypatch.setattr(
            r,
            "_retrieve_vector",
            lambda query, top_k, score_threshold: list(results),
        )
        assert len(r.retrieve("薪酬", user_role="intern")) == 1
