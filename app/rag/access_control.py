"""文档密级与角色可见性（模拟 ACL，非真实鉴权）。

等级：public < internal < confidential
角色映射到「最高可读密级」：
- intern（实习生）→ public
- employee（员工，默认）→ internal
- admin（管理员）→ confidential
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from app.config import PROJECT_ROOT, get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

AccessLevel = Literal["public", "internal", "confidential"]
UserRole = Literal["intern", "employee", "admin"]

LEVEL_RANK: dict[str, int] = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
}

ROLE_MAX_LEVEL: dict[str, AccessLevel] = {
    "intern": "public",
    "employee": "internal",
    "admin": "confidential",
}

DEFAULT_ACCESS_LEVEL: AccessLevel = "internal"
DEFAULT_ROLE: UserRole = "employee"


def normalize_role(role: str | None) -> str:
    """规范化角色名；未知角色降级为 employee。"""
    if not role:
        return DEFAULT_ROLE
    key = role.strip().lower()
    if key not in ROLE_MAX_LEVEL:
        logger.warning("未知角色 %r，降级为 employee", role)
        return DEFAULT_ROLE
    return key


def normalize_access_level(level: str | None) -> str:
    """规范化文档密级；未知或空则按 internal（偏保守）。"""
    if not level:
        return DEFAULT_ACCESS_LEVEL
    key = level.strip().lower()
    if key not in LEVEL_RANK:
        logger.warning("未知密级 %r，按 internal 处理", level)
        return DEFAULT_ACCESS_LEVEL
    return key


def role_can_access(user_role: str | None, doc_level: str | None) -> bool:
    """判断角色是否可读指定密级文档。

    Args:
        user_role: intern / employee / admin。
        doc_level: public / internal / confidential。

    Returns:
        True 表示可见。
    """
    role = normalize_role(user_role)
    level = normalize_access_level(doc_level)
    max_level = ROLE_MAX_LEVEL[role]
    return LEVEL_RANK[level] <= LEVEL_RANK[max_level]


@lru_cache
def load_doc_access_map(path: str | None = None) -> dict[str, str]:
    """加载 source 文件名 -> access_level 映射。

    Args:
        path: JSON 路径字符串；None 时用配置 doc_access_path。

    Returns:
        映射字典；文件缺失时返回空 dict。
    """
    settings = get_settings()
    file_path = Path(path) if path else settings.doc_access_path
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    if not file_path.exists():
        logger.warning("密级映射文件不存在: %s", file_path)
        return {}
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("读取密级映射失败: %s", file_path)
        return {}
    if not isinstance(data, dict):
        logger.error("密级映射格式错误，期望 JSON 对象")
        return {}
    return {str(k): normalize_access_level(str(v)) for k, v in data.items()}


def resolve_access_level(source_filename: str) -> str:
    """根据文件名解析密级；未配置时返回默认 internal。"""
    mapping = load_doc_access_map()
    return mapping.get(source_filename, DEFAULT_ACCESS_LEVEL)
