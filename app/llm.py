"""共享 LLM 客户端工厂。

rag/qa_chain.py（纯 RAG 问答）与 agent/graph.py（Agent 决策图）
共用同一个 ChatOpenAI 配置，集中在这里构造，避免连接参数散落两处。
所有敏感配置（API Key / Base URL）来自 config.py，禁止硬编码。
"""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import get_settings


@lru_cache
def get_chat_model() -> ChatOpenAI:
    """构造 LLM 客户端（lru_cache 复用实例）。

    Returns:
        指向 OpenAI 兼容接口的 ChatOpenAI 实例，model/key/base_url/
        温度/超时均来自配置；内置 2 次自动重试对抗瞬时网络抖动。

    Raises:
        ValueError: 未配置 LLM_API_KEY 时抛出（带配置指引）。
    """
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model_name,
        api_key=settings.require_llm_api_key(),
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        timeout=settings.llm_request_timeout,
        max_retries=2,
    )
