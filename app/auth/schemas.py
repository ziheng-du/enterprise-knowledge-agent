"""鉴权相关的 Pydantic 请求/响应模型。"""

from typing import Literal

from pydantic import BaseModel, Field

UserRole = Literal["intern", "employee", "admin"]


class LoginRequest(BaseModel):
    """登录请求：工号 + 密码。"""

    employee_id: str = Field(min_length=1, description="员工工号")
    password: str = Field(min_length=1, description="登录密码")


class UserInfo(BaseModel):
    """对外展示的用户信息（不含密码哈希）。"""

    employee_id: str = Field(description="员工工号")
    name: str = Field(description="姓名")
    role: UserRole = Field(description="角色：intern / employee / admin")
    hire_date: str | None = Field(default=None, description="入职日期 YYYY-MM-DD")


class LoginResponse(BaseModel):
    """登录成功响应：token + 用户信息。"""

    token: str = Field(description="Bearer token，后续请求放在 Authorization 头")
    employee_id: str = Field(description="员工工号")
    name: str = Field(description="姓名")
    role: UserRole = Field(description="角色")
    hire_date: str | None = Field(default=None, description="入职日期")


class AuthUser(BaseModel):
    """已通过鉴权的当前用户（供 Depends 注入）。"""

    employee_id: str
    name: str
    role: UserRole
    hire_date: str | None = None
