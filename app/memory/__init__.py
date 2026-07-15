"""会话记忆与上下文管理包。

SessionStore：跨轮对话历史与用户画像（产品层短期记忆）。
与 LangGraph Checkpointer 职责不同——后者管图执行断点，本包管「聊过什么」。
"""

from app.memory.session_store import SessionStore, get_session_store

__all__ = ["SessionStore", "get_session_store"]
