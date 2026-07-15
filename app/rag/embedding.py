"""向量化封装模块：按配置提供统一的 Embeddings 实例。

通过 EMBEDDING_PROVIDER 配置切换：
- local: 本地 sentence-transformers 模型（默认 BAAI/bge-small-zh-v1.5，
  中文效果好、体积小、无需 API Key）
- api: OpenAI 兼容的在线 Embedding API（复用 .env 中的 LLM_API_KEY /
  LLM_BASE_URL）

设计说明：这里自写一个实现 langchain_core Embeddings 接口的薄适配类，
而不用 langchain-community 中已标记弃用的 HuggingFaceEmbeddings，
避免引入额外的 langchain-huggingface 依赖。
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LocalSentenceTransformerEmbeddings(Embeddings):
    """基于 sentence-transformers 的本地 Embedding 适配器。

    实现 langchain_core.embeddings.Embeddings 接口，使本地模型可以
    无缝接入 LangChain 的 VectorStore 体系。

    向量归一化说明：normalize_embeddings=True 使向量模长为 1，
    配合 Chroma 的余弦距离，相似度分数落在 [0, 1] 的可解释区间，
    与配置项 retrieval_score_threshold 的语义保持一致。
    """

    def __init__(self, model_name: str):
        """加载本地 embedding 模型。

        Args:
            model_name: HuggingFace 模型 ID（如 BAAI/bge-small-zh-v1.5）。
                首次使用会自动下载模型权重（约 100MB），
                国内网络可配置 HF_ENDPOINT=https://hf-mirror.com 加速。
        """
        # 延迟导入：sentence-transformers 依赖 torch，导入较慢，
        # 仅在真正使用 local 模式时才加载
        from sentence_transformers import SentenceTransformer

        logger.info("正在加载本地 Embedding 模型: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        logger.info("Embedding 模型加载完成: %s", model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档文本块（入库时调用）。

        Args:
            texts: 文本块内容列表。

        Returns:
            与输入等长的向量列表（已归一化）。
        """
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """向量化用户查询（检索时调用）。

        Args:
            text: 查询文本。

        Returns:
            查询向量（已归一化）。
        """
        # bge 系列模型对短查询建议加检索指令前缀以提升召回效果，
        # bge-small-zh-v1.5 官方说明该前缀为可选项，为保持入库/查询
        # 空间一致性并简化逻辑，这里统一不加前缀
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()


@lru_cache
def get_embeddings() -> Embeddings:
    """按配置构造 Embeddings 实例（工厂函数，lru_cache 避免重复加载模型）。

    Returns:
        Embeddings 实例：local 模式为本地 sentence-transformers 适配器，
        api 模式为 OpenAIEmbeddings（OpenAI 兼容接口）。

    Raises:
        ValueError: api 模式下未配置 LLM_API_KEY 时抛出（带配置指引）。
    """
    settings = get_settings()

    if settings.embedding_provider == "local":
        return LocalSentenceTransformerEmbeddings(settings.embedding_model_name)

    # api 模式：复用 LLM 的 key 与 base_url（OpenAI 兼容接口）
    from langchain_openai import OpenAIEmbeddings

    logger.info("使用在线 Embedding API: %s", settings.embedding_model_name)
    return OpenAIEmbeddings(
        model=settings.embedding_model_name,
        api_key=settings.require_llm_api_key(),
        base_url=settings.llm_base_url,
    )
