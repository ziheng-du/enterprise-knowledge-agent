"""用户仓库：从本地 JSON 加载演示账号并校验密码。

可理解为「公司花名册」：工号、姓名、角色、入职日期、密码哈希。
作品集场景用 JSON 即可；未来换数据库只需替换本模块读写实现。
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 与生成 users.json 时一致的 PBKDF2 迭代次数
_PBKDF2_ITERATIONS = 100_000


@dataclass(frozen=True)
class UserRecord:
    """内存中的用户记录。"""

    employee_id: str
    name: str
    role: str
    hire_date: str | None
    password_salt: str
    password_hash: str


def _hash_password(password: str, salt_hex: str) -> str:
    """用 PBKDF2-HMAC-SHA256 计算密码哈希（十六进制）。

    Args:
        password: 明文密码。
        salt_hex: 十六进制盐值。

    Returns:
        密码派生密钥的十六进制字符串。
    """
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        _PBKDF2_ITERATIONS,
    ).hex()


def verify_password(password: str, salt_hex: str, expected_hash: str) -> bool:
    """校验明文密码是否与存储的 salt/hash 匹配。

    Args:
        password: 用户输入的明文密码。
        salt_hex: 存储的盐。
        expected_hash: 存储的哈希。

    Returns:
        匹配为 True，否则 False。空输入或格式异常视为不匹配。
    """
    if not password or not salt_hex or not expected_hash:
        return False
    try:
        actual = _hash_password(password, salt_hex)
    except (ValueError, TypeError):
        logger.warning("密码哈希计算失败：salt 格式异常")
        return False
    # 恒定时间比较，降低时序攻击面（演示级仍建议使用）
    return hmac.compare_digest(actual, expected_hash)


def _parse_user(raw: dict[str, Any]) -> UserRecord | None:
    """把 JSON 条目解析为 UserRecord；缺字段则跳过。"""
    try:
        return UserRecord(
            employee_id=str(raw["employee_id"]).strip(),
            name=str(raw["name"]).strip(),
            role=str(raw["role"]).strip().lower(),
            hire_date=(str(raw["hire_date"]).strip() if raw.get("hire_date") else None),
            password_salt=str(raw["password_salt"]).strip(),
            password_hash=str(raw["password_hash"]).strip(),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        logger.warning("跳过非法用户条目: %s error=%s", raw, exc)
        return None


def load_users(path: Path | None = None) -> dict[str, UserRecord]:
    """从 JSON 文件加载用户，以工号为键。

    Args:
        path: users.json 路径；None 时读配置 USERS_FILE。

    Returns:
        employee_id -> UserRecord。文件缺失或空列表时返回空字典。
    """
    file_path = Path(path) if path is not None else get_settings().users_file
    if not file_path.is_file():
        logger.error("用户文件不存在: %s", file_path)
        return {}

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("读取用户文件失败: %s", file_path)
        raise ValueError(f"无法读取用户文件: {file_path}") from exc

    if not isinstance(data, list):
        raise ValueError(f"用户文件格式错误，期望 JSON 数组: {file_path}")

    users: dict[str, UserRecord] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        record = _parse_user(item)
        if record is None:
            continue
        if record.employee_id in users:
            logger.warning("重复工号 %s，后者覆盖前者", record.employee_id)
        users[record.employee_id] = record

    logger.info("已加载 %d 个用户账号: %s", len(users), file_path)
    return users


@lru_cache
def get_user_store() -> dict[str, UserRecord]:
    """缓存的用户表（进程内单例）。测试时可 clear_user_store_cache。"""
    return load_users()


def clear_user_store_cache() -> None:
    """清除用户表缓存（供测试替换 users 文件后重载）。"""
    get_user_store.cache_clear()


def get_user_by_id(employee_id: str) -> UserRecord | None:
    """按工号查找用户。

    Args:
        employee_id: 工号。

    Returns:
        找到返回 UserRecord，否则 None。
    """
    key = (employee_id or "").strip()
    if not key:
        return None
    return get_user_store().get(key)


def authenticate(employee_id: str, password: str) -> UserRecord | None:
    """用工号+密码鉴权。

    Args:
        employee_id: 工号。
        password: 明文密码。

    Returns:
        成功返回 UserRecord；工号不存在或密码错误返回 None（不区分原因，防枚举）。
    """
    user = get_user_by_id(employee_id)
    if user is None:
        return None
    if not verify_password(password, user.password_salt, user.password_hash):
        return None
    return user
