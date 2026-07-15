"""登录鉴权路由：login / me。

与业务 chat 路由分离，便于讲解「门禁」与「办事窗口」的边界。
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.deps import get_current_user
from app.auth.schemas import AuthUser, LoginRequest, LoginResponse, UserInfo
from app.auth.tokens import create_token
from app.auth.users import authenticate
from app.rag.access_control import normalize_role
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest) -> LoginResponse:
    """用工号与密码登录，返回 Bearer token 与用户信息。

    Args:
        request: employee_id + password。

    Returns:
        LoginResponse：token 与姓名/角色等。

    Raises:
        HTTPException 401: 工号或密码错误（文案统一，避免暴露工号是否存在）。
    """
    user = authenticate(request.employee_id.strip(), request.password)
    if user is None:
        logger.info("登录失败: employee_id=%r", request.employee_id.strip()[:32])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="工号或密码错误",
        )

    role = normalize_role(user.role)
    token = create_token(
        employee_id=user.employee_id,
        role=role,
        name=user.name,
        hire_date=user.hire_date,
    )
    logger.info("登录成功: employee_id=%s role=%s", user.employee_id, role)
    return LoginResponse(
        token=token,
        employee_id=user.employee_id,
        name=user.name,
        role=role,  # type: ignore[arg-type]
        hire_date=user.hire_date,
    )


@router.get("/me", response_model=UserInfo)
def me(current_user: AuthUser = Depends(get_current_user)) -> UserInfo:
    """返回当前登录用户信息（刷新页面时恢复页头展示）。

    Args:
        current_user: 由 Bearer token 解析出的用户。

    Returns:
        UserInfo：工号、姓名、角色、入职日期。
    """
    return UserInfo(
        employee_id=current_user.employee_id,
        name=current_user.name,
        role=current_user.role,
        hire_date=current_user.hire_date,
    )
