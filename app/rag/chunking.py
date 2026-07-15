"""文档切分模块：提供两种可配置切换的切分策略。

策略说明（通过配置 CHUNKING_STRATEGY 或调用时显式传参切换，便于对比实验）：
- fixed: 固定长度切分（CharacterTextSplitter，带 overlap），实现简单，
  但可能在句子中间截断，适合作为对比基线
- recursive: 递归字符切分（RecursiveCharacterTextSplitter），按分隔符
  优先级尽量在段落/句子边界切分，中文场景下语义完整性更好（默认策略）

切分时保留原始 Document 的全部 metadata（source、page 等），
并补充 chunk_index（块在原文档中的序号），便于回答时标注引用来源。
"""

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 中文文本的分隔符优先级：段落 > 换行 > 句号 > 分号 > 逗号 > 空格 > 强制截断
_CHINESE_SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]


def get_text_splitter(
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> TextSplitter:
    """按策略构造文本切分器（工厂函数）。

    Args:
        strategy: 切分策略，"fixed" 或 "recursive"。None 时读取配置。
        chunk_size: 单块目标长度（字符数）。None 时读取配置。
        chunk_overlap: 相邻块重叠长度（字符数）。None 时读取配置。

    Returns:
        配置好的 TextSplitter 实例。

    Raises:
        ValueError: strategy 不是受支持的取值时抛出（配置层已用 Literal
            约束，这里的校验用于兜底显式传参的调用方）。
    """
    settings = get_settings()
    strategy = strategy or settings.chunking_strategy
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    if strategy == "fixed":
        # separator="" 使其退化为纯固定长度切分（不依赖任何分隔符）
        return CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if strategy == "recursive":
        return RecursiveCharacterTextSplitter(
            separators=_CHINESE_SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    raise ValueError(f"未知的切分策略: {strategy!r}（支持 'fixed' / 'recursive'）")


def split_documents(
    documents: list[Document],
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """将文档列表切分为文本块，保留并补充元数据。

    Args:
        documents: 待切分的 Document 列表（来自 document_loader）。
        strategy / chunk_size / chunk_overlap: 见 get_text_splitter，
            None 时使用全局配置。

    Returns:
        切分后的 Document 块列表。每块继承原文档 metadata，
        并额外写入 chunk_index（该块在所属原文档内的序号，从 0 起）。
    """
    if not documents:
        logger.warning("split_documents 收到空文档列表，返回空结果")
        return []

    splitter = get_text_splitter(strategy, chunk_size, chunk_overlap)

    chunks: list[Document] = []
    # 逐文档切分（而非整批调用 split_documents），以便为每块生成
    # 相对所属原文档的 chunk_index
    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_index"] = idx
        chunks.extend(doc_chunks)

    logger.info(
        "切分完成: %d 个文档 -> %d 个文本块（strategy=%s）",
        len(documents),
        len(chunks),
        strategy or get_settings().chunking_strategy,
    )
    return chunks
