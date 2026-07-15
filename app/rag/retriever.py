"""检索模块：面向上层的统一检索入口。

架构约束（见 .cursorrules / PROJECT_SPEC）：
- 本模块完全独立于 Agent 逻辑，不依赖 LangGraph 的上下文或状态对象，
  可以被单独实例化和测试
- 只负责"检索"，不掺杂"生成回答"的逻辑；检索为空时返回空列表，
  "知识库中未找到相关信息"的兜底话术由上层（Agent 的 Prompt 层）处理

检索模式（配置 retrieval_mode）：
- vector：纯向量相似度 + 阈值过滤
- hybrid：向量 + BM25，经 RRF 融合后再截断 top_k
"""

from __future__ import annotations

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag.access_control import role_can_access, resolve_access_level
from app.rag.bm25_index import doc_key, get_bm25_index, reciprocal_rank_fusion
from app.rag.vector_store import VectorStoreManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalResult(BaseModel):
    """单条检索结果的结构化表示。

    上层（Agent / API）通过该模型消费检索结果，
    source 与 score 可直接用于前端展示"引用来源"。
    """

    content: str = Field(description="命中的文本块内容")
    source: str = Field(description="来源文件名")
    score: float = Field(description="相关性分数，[0, 1] 区间，越大越相似")
    metadata: dict = Field(default_factory=dict, description="文本块完整元数据（页码、块序号等）")


class Retriever:
    """检索器：支持 vector / hybrid 两种模式，对外签名统一。

    阈值过滤的意义：避免把相关性很低的内容强行塞给 LLM，
    宁可返回空结果让上层如实告知"未找到"，也不给编造答案留素材。
    """

    def __init__(self, vector_store: VectorStoreManager | None = None):
        """初始化检索器。

        Args:
            vector_store: 向量库管理器。None 时新建默认实例（读取全局配置），
                测试时可注入 mock 或指向临时目录的实例。
        """
        self._vector_store = vector_store or VectorStoreManager()
        self._ensure_bm25_warm()

    def _ensure_bm25_warm(self) -> None:
        """hybrid 模式下若 BM25 为空，尝试从向量库加载语料。"""
        settings = get_settings()
        if settings.retrieval_mode != "hybrid":
            return
        index = get_bm25_index()
        if index.size > 0:
            return
        try:
            docs = self._vector_store.get_all_documents()
            if docs:
                index.rebuild(docs)
        except Exception:
            logger.exception("预热 BM25 索引失败，hybrid 将退化为偏向量结果")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        user_role: str | None = None,
    ) -> list[RetrievalResult]:
        """执行检索（按配置选择 vector 或 hybrid），并可按角色过滤密级。

        Args:
            query: 用户查询文本。
            top_k: 最终返回数量。None 时读取配置 retrieval_top_k。
            score_threshold: 向量相似度阈值（hybrid 中仅过滤向量路）。
                None 时读取配置 retrieval_score_threshold。
            user_role: 用户角色（intern/employee/admin）。启用 ACL 时用于过滤。

        Returns:
            按相关性降序的 RetrievalResult 列表；空查询或无命中时返回 []。
        """
        if not query or not query.strip():
            logger.warning("收到空查询，返回空检索结果")
            return []

        settings = get_settings()
        top_k = top_k if top_k is not None else settings.retrieval_top_k
        score_threshold = (
            score_threshold if score_threshold is not None else settings.retrieval_score_threshold
        )

        if settings.retrieval_mode == "hybrid":
            results = self._retrieve_hybrid(query, top_k=top_k, score_threshold=score_threshold)
        else:
            results = self._retrieve_vector(query, top_k=top_k, score_threshold=score_threshold)

        return self._filter_by_access(results, user_role)

    def _filter_by_access(
        self,
        results: list[RetrievalResult],
        user_role: str | None,
    ) -> list[RetrievalResult]:
        """按文档密级过滤检索结果（不依赖 LangGraph）。"""
        settings = get_settings()
        if not settings.enable_access_control:
            return results

        kept: list[RetrievalResult] = []
        denied = 0
        for item in results:
            level = item.metadata.get("access_level") or resolve_access_level(item.source)
            if role_can_access(user_role, level):
                kept.append(item)
            else:
                denied += 1
        if denied:
            logger.info(
                "ACL 过滤: role=%s 拦截 %d 条，保留 %d 条",
                user_role,
                denied,
                len(kept),
            )
        return kept

    def _retrieve_vector(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
    ) -> list[RetrievalResult]:
        """纯向量检索 + 阈值过滤。"""
        scored_docs = self._vector_store.search_with_scores(query, k=top_k)
        results = [
            RetrievalResult(
                content=doc.page_content,
                source=doc.metadata.get("source", "未知来源"),
                score=score,
                metadata=doc.metadata,
            )
            for doc, score in scored_docs
            if score >= score_threshold
        ]
        filtered_count = len(scored_docs) - len(results)
        logger.info(
            "向量检索完成: query=%r, 候选 %d 条, 阈值(%.2f)过滤 %d 条, 返回 %d 条",
            query[:50],
            len(scored_docs),
            score_threshold,
            filtered_count,
            len(results),
        )
        return results

    def _retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
    ) -> list[RetrievalResult]:
        """向量 + BM25，经 RRF 融合。

        向量路仍用 score_threshold 过滤低相关噪声；BM25 路取 top_n。
        融合分归一化到约 [0,1] 便于展示（非严格概率）。
        """
        settings = get_settings()
        self._ensure_bm25_warm()

        vector_k = max(top_k, settings.hybrid_bm25_top_k)
        scored_docs = self._vector_store.search_with_scores(query, k=vector_k)
        vector_filtered = [
            (doc, score) for doc, score in scored_docs if score >= score_threshold
        ]

        bm25_hits = get_bm25_index().search(query, top_k=settings.hybrid_bm25_top_k)

        key_to_doc: dict[str, Document] = {}
        key_to_vector_score: dict[str, float] = {}
        vector_keys: list[str] = []
        for doc, score in vector_filtered:
            key = doc_key(doc)
            key_to_doc[key] = doc
            key_to_vector_score[key] = score
            vector_keys.append(key)

        bm25_keys: list[str] = []
        for hit in bm25_hits:
            key = doc_key(hit.document)
            key_to_doc[key] = hit.document
            bm25_keys.append(key)

        if not vector_keys and not bm25_keys:
            logger.info("Hybrid 检索无命中: query=%r", query[:50])
            return []

        fused = reciprocal_rank_fusion(
            [vector_keys, bm25_keys],
            k=settings.hybrid_rrf_k,
        )
        # RRF 理论最大约 2/(k+1)，归一化到约 [0,1]
        norm = 2.0 / (settings.hybrid_rrf_k + 1)
        results: list[RetrievalResult] = []
        for key, rrf_score in fused[:top_k]:
            doc = key_to_doc[key]
            display_score = min(1.0, rrf_score / norm) if norm > 0 else rrf_score
            # 若同时有向量分，取较大展示分，避免 RRF 归一化过低难看
            if key in key_to_vector_score:
                display_score = max(display_score, key_to_vector_score[key])
            results.append(
                RetrievalResult(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "未知来源"),
                    score=float(display_score),
                    metadata=doc.metadata,
                )
            )

        logger.info(
            "Hybrid 检索完成: query=%r, 向量路 %d, BM25路 %d, 返回 %d",
            query[:50],
            len(vector_keys),
            len(bm25_keys),
            len(results),
        )
        return results
