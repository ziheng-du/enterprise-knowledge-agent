"""登录鉴权包：工号密码登录、token 签发校验、FastAPI 依赖注入。

设计意图：权限由服务端会话（token）决定，不信任客户端传的 role。
与工具层「context 注入 user_role」同一思路——客户端/LLM 都不能伪造身份。
"""

from app.auth.deps import get_current_user
from app.auth.schemas import AuthUser, LoginRequest, LoginResponse, UserInfo

__all__ = [
    "AuthUser",
    "LoginRequest",
    "LoginResponse",
    "UserInfo",
    "get_current_user",
]
