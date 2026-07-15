"""配置管理模块（app/config.py）的冒烟测试。

测试不依赖真实 .env 文件：通过 `_env_file=None` 关闭文件加载，
并用 monkeypatch 注入环境变量，保证测试在任何机器上可复现。
"""

import pytest

from app.config import PROJECT_ROOT, Settings, get_settings


def _make_settings(**overrides) -> Settings:
    """构造不读取 .env 文件的 Settings，仅受显式传参和环境变量影响。"""
    return Settings(_env_file=None, **overrides)


class TestSettingsDefaults:
    """验证各字段默认值与类型正确。"""

    def test_llm_defaults(self):
        settings = _make_settings()
        assert settings.llm_api_key == ""
        assert settings.llm_model_name == "deepseek-chat"
        assert isinstance(settings.llm_temperature, float)

    def test_embedding_defaults(self):
        settings = _make_settings()
        assert settings.embedding_provider == "local"
        assert settings.embedding_model_name == "BAAI/bge-small-zh-v1.5"

    def test_path_defaults_anchored_to_project_root(self):
        settings = _make_settings()
        assert settings.raw_docs_dir == PROJECT_ROOT / "data" / "raw_docs"
        assert settings.vector_db_dir == PROJECT_ROOT / "data" / "vector_db"

    def test_rag_and_agent_defaults(self):
        settings = _make_settings()
        assert settings.chunking_strategy == "recursive"
        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 50
        assert settings.retrieval_top_k == 4
        assert 0.0 <= settings.retrieval_score_threshold <= 1.0
        assert settings.max_tool_rounds == 3
        assert settings.log_level == "INFO"


class TestSettingsFromEnv:
    """验证环境变量能正确覆盖默认值（含类型转换）。"""

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key-123")
        monkeypatch.setenv("CHUNK_SIZE", "800")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "api")
        settings = _make_settings()
        assert settings.llm_api_key == "test-key-123"
        assert settings.chunk_size == 800
        assert settings.embedding_provider == "api"

    def test_invalid_literal_rejected(self, monkeypatch):
        # Literal 字段收到非法取值时应报校验错误，而非静默接受
        monkeypatch.setenv("CHUNKING_STRATEGY", "not-a-strategy")
        with pytest.raises(Exception):
            _make_settings()


class TestRequireLlmApiKey:
    """验证 API Key 缺失时的显式报错兜底。"""

    def test_missing_key_raises_with_guidance(self):
        settings = _make_settings(llm_api_key="")
        with pytest.raises(ValueError, match="LLM_API_KEY"):
            settings.require_llm_api_key()

    def test_present_key_returned(self):
        settings = _make_settings(llm_api_key="sk-xxx")
        assert settings.require_llm_api_key() == "sk-xxx"


def test_get_settings_is_cached_singleton():
    """get_settings 应返回同一实例（lru_cache 单例）。"""
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()
