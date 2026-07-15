"""配置管理模块：全项目唯一的配置入口。

所有配置项通过 pydantic-settings 从环境变量 / `.env` 文件读取，
其他模块一律通过 `get_settings()` 获取配置，禁止散落 `os.getenv` 调用。
敏感信息（API Key、Base URL）不允许硬编码在代码中。

新增配置项时，请同步更新项目根目录的 `.env.example`。
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict  # pyright: ignore[reportMissingImports]

# 项目根目录（config.py 位于 app/ 下，向上一级即项目根）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用全局配置。

    字段按用途分组：LLM、Embedding、路径、RAG 参数、Agent 参数、日志。
    除 `llm_api_key` 外均提供安全默认值，保证在缺少 `.env` 时
    非 LLM 相关功能（如本地检索、单元测试）仍可运行。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # 忽略 .env 中与 Settings 无关的额外变量，避免误报错
        extra="ignore",
    )

    # --- LLM（OpenAI 兼容接口，如通义千问 / DeepSeek） ---
    llm_api_key: str = Field(
        default="",
        description="LLM API 密钥。必须通过环境变量或 .env 提供，禁止硬编码。",
    )
    llm_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        description="LLM API 的 Base URL（OpenAI 兼容格式）。",
    )
    llm_model_name: str = Field(
        default="deepseek-chat",
        description="LLM 模型名称。",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM 采样温度。知识问答场景取低值以保证回答稳定。",
    )
    llm_request_timeout: float = Field(
        default=60.0,
        gt=0.0,
        description="LLM API 单次请求超时时间（秒），防止网络异常时请求无限挂起。",
    )

    # --- Embedding ---
    embedding_provider: Literal["local", "api"] = Field(
        default="local",
        description="Embedding 提供方：local=本地 sentence-transformers 模型；api=在线 Embedding API。",
    )
    embedding_model_name: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        description="Embedding 模型名称（local 模式下为 HuggingFace 模型 ID）。",
    )

    # --- 路径 ---
    raw_docs_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "raw_docs",
        description="原始企业制度文档目录。",
    )
    vector_db_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "vector_db",
        description="Chroma 向量库持久化目录。",
    )
    chroma_collection_name: str = Field(
        default="enterprise_knowledge",
        description="Chroma collection 名称。",
    )

    # --- RAG 参数（阶段二使用，属于配置管理范畴，先集中定义） ---
    chunking_strategy: Literal["fixed", "recursive"] = Field(
        default="recursive",
        description="文档切分策略：fixed=固定长度切分；recursive=递归字符切分（优先段落/句子边界）。",
    )
    chunk_size: int = Field(
        default=500,
        gt=0,
        description="单个文本块的目标长度（字符数）。",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="相邻文本块之间的重叠长度（字符数）。",
    )
    retrieval_top_k: int = Field(
        default=4,
        gt=0,
        description="相似度检索返回的候选文档数量。",
    )
    retrieval_score_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="相似度分数阈值，低于该值的检索结果将被过滤，避免不相关内容进入 LLM 上下文。",
    )

    # --- Agent 参数 ---
    max_tool_rounds: int = Field(
        default=3,
        gt=0,
        description="Agent 单次会话中工具调用的最大轮次，防止死循环。",
    )

    # --- 会话记忆（短期多轮，SQLite SessionStore） ---
    session_db_path: Path = Field(
        default=PROJECT_ROOT / "data" / "sessions.db",
        description="会话记忆 SQLite 数据库路径（跨轮问答历史与用户画像）。",
    )
    max_history_turns: int = Field(
        default=6,
        gt=0,
        description=(
            "注入 Prompt 的最近完整问答轮数（一轮=用户+助手各一条）。"
            "历史摘要也按此窗口：仅当消息条数超过 max_history_turns×2 时，"
            "才对窗口外更早轮次做摘要。"
        ),
    )
    enable_query_rewrite: bool = Field(
        default=True,
        description="多轮场景下是否先把当前问题改写成独立检索 query。",
    )

    # --- 上下文预算（Phase 2） ---
    max_history_chars: int = Field(
        default=2000,
        gt=0,
        description="历史对话注入 Prompt 的最大字符数。",
    )
    max_retrieval_chars: int = Field(
        default=3000,
        gt=0,
        description="检索资料注入 Prompt 的最大字符数。",
    )
    max_tool_result_chars: int = Field(
        default=1500,
        gt=0,
        description="工具结果注入 Prompt 的最大字符数。",
    )

    # --- 检索模式（Phase 3） ---
    retrieval_mode: Literal["vector", "hybrid"] = Field(
        default="hybrid",
        description="检索模式：vector=纯向量；hybrid=BM25+向量 RRF 融合。",
    )
    hybrid_rrf_k: int = Field(
        default=60,
        gt=0,
        description="Hybrid 检索 RRF 融合常数 k（常用 60）。",
    )
    hybrid_bm25_top_k: int = Field(
        default=8,
        gt=0,
        description="BM25 侧召回的候选数量（再与向量结果融合）。",
    )

    # --- 文档密级 ACL（角色来自登录 token，非请求体伪造） ---
    enable_access_control: bool = Field(
        default=True,
        description="是否按用户角色过滤文档密级（public/internal/confidential）。",
    )
    doc_access_path: Path = Field(
        default=PROJECT_ROOT / "data" / "doc_access.json",
        description="source 文件名到 access_level 的映射 JSON。",
    )

    # --- 登录鉴权（工号+密码 → HMAC token） ---
    auth_secret_key: str = Field(
        default="dev-only-change-me-eka-auth-secret",
        description="token HMAC 签名密钥。生产环境必须通过环境变量覆盖默认值。",
    )
    auth_token_expire_hours: float = Field(
        default=24.0,
        gt=0.0,
        description="登录 token 有效小时数。",
    )
    users_file: Path = Field(
        default=PROJECT_ROOT / "data" / "users.json",
        description="演示用户花名册 JSON（工号/姓名/角色/密码哈希）。",
    )

    # --- 日志 ---
    log_level: str = Field(
        default="INFO",
        description="日志级别：DEBUG / INFO / WARNING / ERROR。",
    )

    def require_llm_api_key(self) -> str:
        """获取 LLM API Key，缺失时抛出带明确指引的错误。

        Returns:
            非空的 API Key 字符串。

        Raises:
            ValueError: 未配置 LLM_API_KEY 时抛出，提示用户如何配置，
                避免在实际调用 LLM 时才出现难以定位的鉴权失败。
        """
        if not self.llm_api_key:
            raise ValueError(
                "未配置 LLM_API_KEY。请在项目根目录创建 .env 文件"
                "（可参考 .env.example）并填入有效的 API 密钥。"
            )
        return self.llm_api_key


@lru_cache
def get_settings() -> Settings:
    """获取全局配置单例（lru_cache 缓存，避免重复解析 .env）。

    Returns:
        Settings 实例。全项目统一通过本函数获取配置。
    """
    return Settings()
