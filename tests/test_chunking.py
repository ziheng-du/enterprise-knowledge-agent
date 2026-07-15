"""文档切分模块（app/rag/chunking.py）单元测试。

不依赖向量库与 LLM，验证两种策略均可切分且保留元数据。
"""

from langchain_core.documents import Document

from app.rag.chunking import split_documents


def _sample_docs() -> list[Document]:
    """构造一段足够长、含段落边界的中文样本文档。"""
    text = (
        "这是第一段制度说明。" * 20
        + "\n\n"
        + "这是第二段关于报销与年假的规定。" * 20
        + "\n\n"
        + "这是第三段关于考勤与加班的补充。" * 20
    )
    return [Document(page_content=text, metadata={"source": "测试文档.md"})]


class TestSplitDocuments:
    """切分策略与元数据保留。"""

    def test_empty_documents_returns_empty(self):
        assert split_documents([]) == []

    def test_fixed_strategy_produces_chunks_with_metadata(self):
        chunks = split_documents(_sample_docs(), strategy="fixed", chunk_size=200, chunk_overlap=20)
        assert len(chunks) >= 2
        assert all(c.metadata.get("source") == "测试文档.md" for c in chunks)
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_recursive_strategy_produces_chunks_with_metadata(self):
        chunks = split_documents(
            _sample_docs(), strategy="recursive", chunk_size=200, chunk_overlap=20
        )
        assert len(chunks) >= 2
        assert all(c.metadata.get("source") == "测试文档.md" for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)
