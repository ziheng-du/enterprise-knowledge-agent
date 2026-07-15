"""BM25 分词 / RRF / Hybrid 融合离线测试（不依赖 Embedding）。"""

from langchain_core.documents import Document

from app.rag.bm25_index import Bm25Index, reciprocal_rank_fusion, tokenize_for_bm25
from app.rag.retriever import Retriever, RetrievalResult


class TestBm25Basics:
    def test_tokenize_chinese(self):
        tokens = tokenize_for_bm25("报销须在30天内提交")
        assert "报" in tokens
        assert "30" in tokens

    def test_bm25_prefers_literal_match(self):
        index = Bm25Index()
        docs = [
            Document(page_content="年假每满一年增加一天", metadata={"source": "leave.md"}),
            Document(
                page_content="报销须在发生后30天内提交申请",
                metadata={"source": "expense.md"},
            ),
        ]
        index.rebuild(docs)
        hits = index.search("报销 30天", top_k=2)
        assert hits
        assert hits[0].document.metadata["source"] == "expense.md"

    def test_rrf_merges_lists(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
        keys = [k for k, _ in fused]
        assert keys[0] == "b"


class TestHybridRetrieverMock:
    """注入假向量库，验证 hybrid 能合并 BM25 命中。"""

    def test_hybrid_includes_bm25_only_hit(self, monkeypatch):
        from app.rag import bm25_index as bm25_mod
        from app.rag import retriever as retriever_mod

        class FakeStore:
            def search_with_scores(self, query, k):
                # 向量路故意不返回报销文档
                doc = Document(page_content="考勤打卡规定", metadata={"source": "handbook.md"})
                return [(doc, 0.9)]

            def get_all_documents(self):
                return [
                    Document(page_content="考勤打卡规定", metadata={"source": "handbook.md"}),
                    Document(
                        page_content="报销须在发生后30天内提交",
                        metadata={"source": "报销制度.md"},
                    ),
                ]

        index = Bm25Index()
        index.rebuild(FakeStore().get_all_documents())
        monkeypatch.setattr(bm25_mod, "_global_bm25", index)
        monkeypatch.setattr(retriever_mod, "get_bm25_index", lambda: index)

        class Settings:
            retrieval_mode = "hybrid"
            retrieval_top_k = 4
            retrieval_score_threshold = 0.3
            hybrid_rrf_k = 60
            hybrid_bm25_top_k = 8
            enable_access_control = False

        monkeypatch.setattr(retriever_mod, "get_settings", lambda: Settings())

        r = Retriever(vector_store=FakeStore())
        results = r.retrieve("报销须在多久内提交")
        sources = {x.source for x in results}
        assert "报销制度.md" in sources
