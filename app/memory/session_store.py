"""SQLite 会话存储：跨轮问答历史与轻量用户画像。

设计意图：
- 存的是产品层「聊过什么 / 用户已知属性」，不是 LangGraph 图执行快照。
- 每轮 Agent 仍是完整 START→END 的 invoke；本模块在 invoke 前后读写历史。
- 使用 SQLite 便于按 session_id 查询、清理，比散落 JSON 更适合作品集讲解。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _utc_now() -> str:
    """返回 UTC ISO 时间戳字符串。"""
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """按 session_id 管理消息历史、会话摘要与用户画像。"""

    def __init__(self, db_path: Path | None = None):
        """初始化并确保表结构存在。

        Args:
            db_path: SQLite 文件路径。None 时读取配置 session_db_path。
        """
        self._db_path = Path(db_path) if db_path is not None else get_settings().session_db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """打开连接；check_same_thread=False 供 FastAPI 多线程复用。"""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """创建 sessions / messages 表（幂等）。"""
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        summary TEXT NOT NULL DEFAULT '',
                        profile_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_messages_session
                        ON messages(session_id, id);
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def ensure_session(self, session_id: str) -> None:
        """确保会话行存在（首次访问时插入空画像）。

        Args:
            session_id: 会话唯一标识。
        """
        if not session_id:
            raise ValueError("session_id 不能为空")
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO sessions
                        (session_id, summary, profile_json, created_at, updated_at)
                    VALUES (?, '', '{}', ?, ?)
                    """,
                    (session_id, now, now),
                )
                conn.commit()
            finally:
                conn.close()

    def get_history(self, session_id: str, max_turns: int | None = None) -> list[dict[str, str]]:
        """读取最近若干完整问答轮次对应的消息列表。

        Args:
            session_id: 会话 ID。
            max_turns: 最大轮数（一轮=user+assistant）。None 时用配置。

        Returns:
            [{"role": "user"|"assistant", "content": "..."}, ...]，按时间正序。
        """
        self.ensure_session(session_id)
        turns = max_turns if max_turns is not None else get_settings().max_history_turns
        limit = max(turns, 1) * 2
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT role, content FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()
            finally:
                conn.close()
        # 查出来是倒序，翻转为时间正序
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def get_all_messages(self, session_id: str) -> list[dict[str, str]]:
        """读取会话全部消息（摘要压缩时使用）。

        Args:
            session_id: 会话 ID。

        Returns:
            全量消息列表，时间正序。
        """
        self.ensure_session(session_id)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT role, content FROM messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                ).fetchall()
            finally:
                conn.close()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def message_count(self, session_id: str) -> int:
        """返回会话中的消息条数。"""
        self.ensure_session(session_id)
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM messages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return int(row["c"]) if row else 0
            finally:
                conn.close()

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        """追加一轮完整问答（user + assistant）。

        Args:
            session_id: 会话 ID。
            question: 用户问题。
            answer: 助手回答。
        """
        self.ensure_session(session_id)
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (session_id, "user", question, now),
                )
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (session_id, "assistant", answer, now),
                )
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
                conn.commit()
            finally:
                conn.close()
        logger.info("会话已追加一轮: session_id=%s", session_id[:8])

    def get_summary(self, session_id: str) -> str:
        """读取会话摘要（可能为空字符串）。"""
        self.ensure_session(session_id)
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT summary FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return (row["summary"] or "") if row else ""
            finally:
                conn.close()

    def set_summary(self, session_id: str, summary: str) -> None:
        """写入会话摘要。

        Args:
            session_id: 会话 ID。
            summary: 摘要文本。
        """
        self.ensure_session(session_id)
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
                    (summary, now, session_id),
                )
                conn.commit()
            finally:
                conn.close()

    def get_or_create_profile(self, session_id: str) -> dict[str, Any]:
        """读取用户画像字典；不存在时返回空 dict。

        Args:
            session_id: 会话 ID。

        Returns:
            画像字段，如 {"hire_date": "2023-07-01"}。
        """
        self.ensure_session(session_id)
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT profile_json FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            finally:
                conn.close()
        if not row:
            return {}
        try:
            data = json.loads(row["profile_json"] or "{}")
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            logger.warning("画像 JSON 损坏，返回空画像: session_id=%s", session_id[:8])
            return {}

    def update_profile(self, session_id: str, **fields: Any) -> dict[str, Any]:
        """合并更新用户画像字段。

        Args:
            session_id: 会话 ID。
            **fields: 要合并的字段（值为 None 的键会被忽略）。

        Returns:
            更新后的完整画像。
        """
        profile = self.get_or_create_profile(session_id)
        for key, value in fields.items():
            if value is not None and value != "":
                profile[key] = value
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE sessions SET profile_json = ?, updated_at = ? WHERE session_id = ?",
                    (json.dumps(profile, ensure_ascii=False), now, session_id),
                )
                conn.commit()
            finally:
                conn.close()
        return profile


@lru_cache
def get_session_store() -> SessionStore:
    """获取全局 SessionStore 单例（按默认配置路径）。"""
    return SessionStore()
