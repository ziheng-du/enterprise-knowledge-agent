"""文档入库服务：load → split → add 的统一编排。

供 CLI（scripts/ingest.py）与 FastAPI（api/routes.py）共用，
避免 API 层依赖 scripts 包或 sys.path hack。
"""

import time

from app.config import get_settings
from app.rag.bm25_index import refresh_bm25_from_vector_store
from app.rag.chunking import split_documents
from app.rag.document_loader import load_documents
from app.rag.vector_store import VectorStoreManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


def ingest(rebuild: bool = False, strategy: str | None = None) -> int:
    """执行文档入库流程。

    Args:
        rebuild: True 时先清空向量库再全量写入；False 时直接追加。
        strategy: 切分策略（"fixed"/"recursive"），None 时使用配置值。

    Returns:
        成功写入的文本块数量；没有加载到文档时返回 0。
    """
    settings = get_settings()
    start = time.time()

    documents = load_documents(settings.raw_docs_dir)
    if not documents:
        logger.error("没有加载到任何文档，入库终止。请检查目录: %s", settings.raw_docs_dir)
        return 0

    chunks = split_documents(documents, strategy=strategy)

    store = VectorStoreManager()
    if rebuild:
        store.clear()
    ids = store.add_documents(chunks)

    # 入库后刷新 BM25 语料，保证 hybrid 检索与向量库一致
    try:
        refresh_bm25_from_vector_store(store.get_all_documents())
    except Exception:
        logger.exception("刷新 BM25 索引失败（向量入库已成功，hybrid 可能暂不可用）")

    # 使 Agent 侧缓存的 Retriever 在下次调用时重新构造（拿到新索引预热逻辑）
    try:
        from app.agent.graph import _get_retriever

        _get_retriever.cache_clear()
    except Exception:
        logger.debug("清理 Retriever 缓存失败（可忽略）", exc_info=True)

    elapsed = time.time() - start
    logger.info(
        "入库完成: %d 个文档 -> %d 个文本块, 向量库现有 %d 条记录, 耗时 %.1f 秒",
        len(documents),
        len(ids),
        store.count(),
        elapsed,
    )
    return len(ids)
