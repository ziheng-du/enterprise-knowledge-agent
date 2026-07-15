"""Chroma 向量库封装模块：提供增删查改的统一入口。

对 langchain_chroma.Chroma 做一层薄封装，收敛持久化目录、collection
名称、距离度量等初始化细节，上层（ingest 脚本 / retriever）不直接
操作 Chroma 原生 API。

距离度量说明：collection 显式使用余弦距离（hnsw:space=cosine），
配合归一化后的 embedding 向量，similarity_search_with_relevance_scores
返回的分数落在 [0, 1] 区间（1 为最相似），与配置项
retrieval_score_threshold 的语义一致。
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.rag.embedding import get_embeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Chroma 向量库管理器（增删查改）。

    可独立于 Agent 逻辑使用：ingest 脚本用它写入，retriever 用它查询。
    """

    def __init__(
        self,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
        embeddings: Embeddings | None = None,
    ):
        """初始化向量库连接。

        Args:
            persist_dir: Chroma 持久化目录。None 时读取配置。
            collection_name: collection 名称。None 时读取配置。
            embeddings: Embeddings 实例。None 时通过工厂按配置构造。
        """
        settings = get_settings()
        self._persist_dir = persist_dir or settings.vector_db_dir
        self._collection_name = collection_name or settings.chroma_collection_name

        self._store = Chroma(
            collection_name=self._collection_name,
            embedding_function=embeddings or get_embeddings(),
            persist_directory=str(self._persist_dir),
            # 显式指定余弦距离：保证 relevance score 语义为 [0,1] 越大越相似
            collection_metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "已连接 Chroma 向量库: collection=%s, dir=%s, 现有 %d 条记录",
            self._collection_name,
            self._persist_dir,
            self.count(),
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """批量写入文档块。

        Args:
            documents: 切分后的 Document 块列表。

        Returns:
            写入记录的 ID 列表。空输入直接返回空列表并 warning。
        """
        if not documents:
            logger.warning("add_documents 收到空列表，跳过写入")
            return []
        ids = self._store.add_documents(documents)
        logger.info("已写入 %d 个文本块到向量库", len(ids))
        return ids

    def delete_by_source(self, source: str) -> None:
        """删除指定来源文件的全部文本块（按 metadata.source 过滤）。

        用途：某份制度文档更新后，先删旧块再重新入库，实现单文件级更新。

        Args:
            source: 来源文件名（与入库时 metadata.source 一致）。
        """
        collection = self._store._collection
        existing = collection.get(where={"source": source})
        ids = existing.get("ids", [])
        if not ids:
            logger.warning("向量库中没有来源为 %s 的记录，无需删除", source)
            return
        collection.delete(ids=ids)
        logger.info("已删除来源 %s 的 %d 个文本块", source, len(ids))

    def clear(self) -> None:
        """清空当前 collection 的全部记录（全量重建入库前调用）。"""
        collection = self._store._collection
        existing = collection.get()
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        logger.info("已清空向量库 collection=%s（删除 %d 条记录）", self._collection_name, len(ids))

    def count(self) -> int:
        """返回当前 collection 中的记录数。"""
        return self._store._collection.count()

    def search_with_scores(self, query: str, k: int) -> list[tuple[Document, float]]:
        """相似度检索，返回文档块及其相关性分数。

        Args:
            query: 查询文本。
            k: 返回的候选数量。

        Returns:
            (Document, score) 元组列表，score 为 [0, 1] 区间的相关性
            分数（1 为最相似），按分数从高到低排序。
        """
        return self._store.similarity_search_with_relevance_scores(query, k=k)

    def get_all_documents(self) -> list[Document]:
        """导出 collection 中全部文档块（供 BM25 索引构建）。

        Returns:
            Document 列表；空库返回 []。
        """
        collection = self._store._collection
        raw = collection.get(include=["documents", "metadatas"])
        docs_text = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        results: list[Document] = []
        for i, text in enumerate(docs_text):
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            results.append(Document(page_content=text or "", metadata=dict(meta)))
        logger.info("已导出 %d 个文本块供关键词索引", len(results))
        return results
