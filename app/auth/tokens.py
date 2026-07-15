"""HMAC 签名 token：登录后发给客户端的「通行证」。

格式（URL-safe base64）：
  payload_b64.signature_b64
payload 为 JSON：employee_id / role / name / hire_date / exp

不引入 PyJWT，用标准库 hmac + hashlib，满足作品集演示与工程边界清晰。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _b64url_encode(raw: bytes) -> str:
    """URL-safe base64，去掉尾部 =。"""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    """还原 URL-safe base64（补齐 padding）。"""
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode((text + pad).encode("ascii"))


def _sign(payload_b64: str, secret: str) -> str:
    """对 payload 段做 HMAC-SHA256，返回 base64url 签名。"""
    digest = hmac.new(
        secret.encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(digest)


def create_token(
    *,
    employee_id: str,
    role: str,
    name: str,
    hire_date: str | None = None,
    expire_hours: float | None = None,
    secret: str | None = None,
) -> str:
    """签发登录 token。

    Args:
        employee_id: 工号。
        role: 角色 intern/employee/admin。
        name: 姓名（写入 token，便于 /me 少查一次库；仍以用户库为准刷新）。
        hire_date: 可选入职日期。
        expire_hours: 有效小时数；None 用配置。
        secret: 签名密钥；None 用配置 AUTH_SECRET_KEY。

    Returns:
        形如 payload.signature 的 token 字符串。
    """
    settings = get_settings()
    key = secret if secret is not None else settings.auth_secret_key
    hours = expire_hours if expire_hours is not None else settings.auth_token_expire_hours
    now = int(time.time())
    payload: dict[str, Any] = {
        "employee_id": employee_id,
        "role": role,
        "name": name,
        "hire_date": hire_date,
        "iat": now,
        "exp": now + int(hours * 3600),
    }
    payload_b64 = _b64url_encode(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    signature = _sign(payload_b64, key)
    return f"{payload_b64}.{signature}"


def verify_token(token: str, secret: str | None = None) -> dict[str, Any] | None:
    """校验 token 签名与过期时间。

    Args:
        token: 客户端传来的 Bearer token。
        secret: 签名密钥；None 用配置。

    Returns:
        合法则返回 payload 字典；非法/过期返回 None。
    """
    if not token or "." not in token:
        return None

    settings = get_settings()
    key = secret if secret is not None else settings.auth_secret_key

    try:
        payload_b64, signature = token.rsplit(".", 1)
    except ValueError:
        return None

    expected = _sign(payload_b64, key)
    if not hmac.compare_digest(signature, expected):
        logger.info("token 签名校验失败")
        return None

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        logger.info("token payload 解析失败")
        return None

    if not isinstance(payload, dict):
        return None

    exp = payload.get("exp")
    if not isinstance(exp, (int, float)) or time.time() > float(exp):
        logger.info("token 已过期: employee_id=%s", payload.get("employee_id"))
        return None

    employee_id = payload.get("employee_id")
    role = payload.get("role")
    if not employee_id or not role:
        return None

    return payload
