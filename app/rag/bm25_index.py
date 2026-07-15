"""BM25 关键词检索与 RRF 融合，供 Hybrid 检索使用。

制度文档含大量专名与数字条款，纯向量检索偶发漏召回；
BM25 补充字面匹配，再与向量结果做 Reciprocal Rank Fusion。
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.utils.logger import get_logger

logger = get_logger(__name__)

# 中英数字与 CJK 连续片段
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+")


def tokenize_for_bm25(text: str) -> list[str]:
    """简易中文友好分词：英文/数字按词，中文按字 + 相邻 bigram。

    不引入 jieba，避免额外重依赖；对制度短句足够用。

    Args:
        text: 原始文本。

    Returns:
        token 列表（可能为空）。
    """
    if not text or not text.strip():
        return []
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", match):
            chars = list(match)
            tokens.extend(chars)
            tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
        else:
            tokens.append(match)
    return tokens


@dataclass
class Bm25Hit:
    """BM25 单条命中。"""

    document: Document
    score: float
    rank: int


class Bm25Index:
    """内存 BM25 索引，可在 ingest 后刷新。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None

    @property
    def size(self) -> int:
        """索引文档数。"""
        return len(self._documents)

    def rebuild(self, documents: list[Document]) -> None:
        """用全量文档重建索引。

        Args:
            documents: 文本块列表。
        """
        with self._lock:
            self._documents = list(documents)
            if not self._documents:
                self._bm25 = None
                logger.warning("BM25 索引重建为空")
                return
            corpus = [tokenize_for_bm25(d.page_content) for d in self._documents]
            # 空 token 文档用占位，避免 BM25 构造失败
            corpus = [c if c else ["_empty_"] for c in corpus]
            self._bm25 = BM25Okapi(corpus)
            logger.info("BM25 索引已重建: %d 条", len(self._documents))

    def search(self, query: str, top_k: int) -> list[Bm25Hit]:
        """BM25 检索。

        Args:
            query: 查询文本。
            top_k: 返回条数。

        Returns:
            按分数降序的 Bm25Hit 列表。
        """
        with self._lock:
            if not self._bm25 or not self._documents or not query.strip():
                return []
            tokens = tokenize_for_bm25(query)
            if not tokens:
                return []
            scores = self._bm25.get_scores(tokens)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            hits: list[Bm25Hit] = []
            # 注意：rank_bm25 在部分语料上可能给出负分，不能用 <=0 过滤，否则会空召回
            for rank, idx in enumerate(ranked[:top_k], start=1):
                hits.append(
                    Bm25Hit(document=self._documents[idx], score=float(scores[idx]), rank=rank)
                )
            return hits


_global_bm25 = Bm25Index()


def get_bm25_index() -> Bm25Index:
    """获取进程内全局 BM25 索引。"""
    return _global_bm25


def refresh_bm25_from_vector_store(documents: list[Document]) -> None:
    """用给定文档列表刷新全局 BM25（ingest 后调用）。

    Args:
        documents: 全量文本块。
    """
    get_bm25_index().rebuild(documents)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """对多路排序结果做 RRF 融合。

    Args:
        ranked_lists: 每路为文档键（如 source+content hash）的有序列表。
        k: RRF 常数。

    Returns:
        (doc_key, rrf_score) 按分数降序。
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def doc_key(doc: Document) -> str:
    """生成文档去重键（来源 + 内容前缀）。"""
    source = str(doc.metadata.get("source", ""))
    return f"{source}::{hash(doc.page_content)}"
