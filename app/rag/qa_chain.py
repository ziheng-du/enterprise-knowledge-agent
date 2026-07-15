"""纯 RAG 问答链：检索 + LLM 生成回答的独立封装。

架构说明（对应阶段二"跑通纯RAG问答"的要求）：
- 本模块不依赖 LangGraph 的状态或上下文，`answer_question()` 可被
  单独调用测试；阶段四的 Agent 决策图会把它复用为"生成回答"节点
  的核心逻辑，因此接口保持干净：输入用户问题，输出回答文本 + 来源。
- 检索（Retriever）与生成（LLM 调用）在这里组合但各自独立实现，
  符合"检索与生成解耦"的架构约束。
- Prompt 模板统一来自 agent/prompts.py，本文件不硬编码提示词。
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent.prompts import (
    RAG_QA_CONTEXT_ITEM_TEMPLATE,
    RAG_QA_NO_RESULT_ANSWER,
    RAG_QA_SYSTEM_PROMPT,
    RAG_QA_USER_PROMPT_TEMPLATE,
)
from app.llm import get_chat_model
from app.rag.retriever import RetrievalResult, Retriever
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QAResult(BaseModel):
    """一次 RAG 问答的结构化结果。

    上层（阶段四 Agent / 阶段五 API）可直接消费：
    answer 用于展示回答，sources 用于展示"引用来源"标签，
    retrieved 保留完整检索明细（含分数），便于前端展示思考过程。
    """

    answer: str = Field(description="LLM 生成的回答文本")
    sources: list[str] = Field(default_factory=list, description="去重后的来源文件名列表")
    retrieved: list[RetrievalResult] = Field(
        default_factory=list, description="本次回答引用的检索结果明细"
    )


def format_context(results: list[RetrievalResult]) -> str:
    """把检索结果拼接为 Prompt 中的"参考资料"文本块。

    Args:
        results: 检索结果列表（已按分数降序）。

    Returns:
        带编号与来源标注的多段文本。
    """
    return "\n\n".join(
        RAG_QA_CONTEXT_ITEM_TEMPLATE.format(index=i, source=r.source, content=r.content)
        for i, r in enumerate(results, 1)
    )


def answer_question(question: str, retriever: Retriever | None = None) -> QAResult:
    """纯 RAG 问答入口：检索相关文档并调用 LLM 生成回答。

    Args:
        question: 用户问题文本。
        retriever: 检索器实例。None 时新建默认实例（读取全局配置），
            测试时可注入指向测试向量库的实例。

    Returns:
        QAResult（回答文本 + 来源文件名列表 + 检索明细）。

    Raises:
        RuntimeError: LLM 调用失败（网络异常、超时、鉴权失败等）时抛出，
            带明确错误信息；由上层决定重试或降级，本函数不吞掉异常。

    边界处理：
    - 空问题：直接返回提示性回答，不触发检索与 LLM 调用
    - 检索无结果：直接返回"未找到相关信息"的固定回复，不调用 LLM
      （既省一次 API 调用，也从源头杜绝模型编造）
    """
    if not question or not question.strip():
        logger.warning("收到空问题，跳过问答流程")
        return QAResult(answer="请输入有效的问题。")

    retriever = retriever or Retriever()
    results = retriever.retrieve(question)

    if not results:
        logger.info("检索无结果，返回固定的未找到回复: question=%r", question[:50])
        return QAResult(answer=RAG_QA_NO_RESULT_ANSWER)

    user_prompt = RAG_QA_USER_PROMPT_TEMPLATE.format(
        context=format_context(results),
        question=question.strip(),
    )

    try:
        response = get_chat_model().invoke(
            [
                SystemMessage(content=RAG_QA_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception as exc:
        # 网络超时、鉴权失败等一律转成带上下文的 RuntimeError 抛给上层，
        # 不在这里静默降级（降级策略属于阶段四 Agent 的职责）
        logger.exception("LLM 调用失败: question=%r", question[:50])
        raise RuntimeError(f"LLM 调用失败，请检查网络与 API 配置: {exc}") from exc

    # 来源去重并保持检索分数顺序
    sources = list(dict.fromkeys(r.source for r in results))
    answer = str(response.content).strip()
    logger.info(
        "问答完成: question=%r, 引用 %d 个片段（%d 个来源文件）",
        question[:50],
        len(results),
        len(sources),
    )
    return QAResult(answer=answer, sources=sources, retrieved=results)
