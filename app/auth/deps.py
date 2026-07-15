"""FastAPI 依赖：从 Authorization Bearer 解析当前登录用户。

可以理解为「门禁刷卡」：请求进来时先查通行证，认出是谁再放行业务接口。
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.schemas import AuthUser
from app.auth.tokens import verify_token
from app.auth.users import get_user_by_id
from app.rag.access_control import normalize_role
from app.utils.logger import get_logger

logger = get_logger(__name__)

# auto_error=False：缺 header 时我们自己返回统一的 401 文案
_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthUser:
    """从 Bearer token 解析并返回当前用户。

    Args:
        credentials: FastAPI 注入的 Authorization 头解析结果。

    Returns:
        AuthUser：工号、姓名、角色、入职日期。

    Raises:
        HTTPException 401: 未登录、token 无效/过期、或用户已不在花名册。
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录或缺少 Authorization Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已失效，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        )

    employee_id = str(payload["employee_id"])
    # 以用户库为准刷新姓名/角色，避免 token 内陈旧字段；库中无此人则拒绝
    record = get_user_by_id(employee_id)
    if record is None:
        logger.warning("token 对应用户已不存在: %s", employee_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在或已被移除，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        )

    role = normalize_role(record.role)
    return AuthUser(
        employee_id=record.employee_id,
        name=record.name,
        role=role,  # type: ignore[arg-type]
        hire_date=record.hire_date,
    )
