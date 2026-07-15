"""Agent 状态定义：LangGraph 状态图中流转的全部字段。

架构约束：状态用 TypedDict 显式建模，所有节点函数读写的字段
都在这里声明，禁止用普通字典隐式传递状态。

字段生命周期说明：
- question / session_id 在图入口写入后只读
- chat_history / user_profile / session_summary 在入口由 SessionStore 加载
- search_query 由 rewrite 节点写入（无改写时等于 question）
- route 由 router 节点写入
- retrieved / sources 由 retrieve 节点写入
- messages / tool_calls_history / current_round 由 agent 与
  execute_tools 节点在工具循环中累积更新
- degraded 任何节点发生降级时置 True
- node_timings 各节点累计耗时（毫秒）
- final_answer 由 generate 节点写入（图的最终输出）
"""

from typing import Any

# 注：Python < 3.12 下 pydantic（LangGraph 内部用它解析状态 schema）
# 要求使用 typing_extensions 的 TypedDict，typing.TypedDict 会报错
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from app.rag.retriever import RetrievalResult


class ToolCallRecord(TypedDict):
    """单次工具调用的历史记录（供前端展示思考过程与调试）。"""

    tool_name: str
    args: dict
    success: bool
    result: str  # 成功时为结果摘要，失败时为错误信息


class ChatTurn(TypedDict):
    """会话历史中的单条消息。"""

    role: str  # user | assistant
    content: str


class AgentState(TypedDict):
    """Agent 决策图的全局状态。

    LangGraph 每个节点函数接收当前状态、返回要更新的字段子集，
    框架负责合并；节点之间不允许通过其他渠道传递数据。
    """

    # --- 输入 / 会话 ---
    question: str  # 用户原始问题
    session_id: str  # 会话 ID（可空字符串表示无会话持久化）
    user_role: str  # 模拟角色：intern / employee / admin（文档密级过滤）
    chat_history: list[ChatTurn]  # 最近若干轮历史
    session_summary: str  # 更早轮次摘要
    user_profile: dict[str, Any]  # 轻量画像，如 hire_date
    search_query: str  # 改写后的检索 query

    # --- 路由 ---
    route: str  # 分诊结果：rag / tool / both

    # --- 检索 ---
    retrieved: list[RetrievalResult]  # 检索结果（可能为空列表）

    # --- 工具循环 ---
    messages: list[BaseMessage]  # 工具循环的对话消息
    tool_calls_history: list[ToolCallRecord]
    current_round: int

    # --- 兜底与可观测 ---
    degraded: bool
    node_timings: dict[str, float]  # 节点名 -> 耗时毫秒

    # --- 输出 ---
    final_answer: str
    sources: list[str]
