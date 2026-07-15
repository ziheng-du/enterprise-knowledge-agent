"""登录鉴权单元测试：密码校验、token、login/me API。"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.auth.tokens import create_token, verify_token
from app.auth.users import authenticate, clear_user_store_cache, verify_password
from app.main import app

client = TestClient(app)

# 与 data/users.json 演示账号一致
DEMO_PASSWORD = "demo123"
EMPLOYEE_ID = "E1001"
INTERN_ID = "I2001"
ADMIN_ID = "A9001"


def _login(employee_id: str = EMPLOYEE_ID, password: str = DEMO_PASSWORD) -> str:
    """登录并返回 token。"""
    resp = client.post(
        "/api/auth/login",
        json={"employee_id": employee_id, "password": password},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["token"]


class TestPasswordAndUsers:
    """用户库与密码哈希。"""

    def setup_method(self):
        clear_user_store_cache()

    def test_authenticate_success(self):
        user = authenticate(EMPLOYEE_ID, DEMO_PASSWORD)
        assert user is not None
        assert user.name == "张小明"
        assert user.role == "employee"

    def test_authenticate_wrong_password(self):
        assert authenticate(EMPLOYEE_ID, "wrong") is None

    def test_authenticate_unknown_user(self):
        assert authenticate("NOPE", DEMO_PASSWORD) is None

    def test_verify_password_rejects_bad_salt(self):
        assert verify_password("x", "not-hex", "abcd") is False


class TestTokens:
    """HMAC token 签发与校验。"""

    def test_create_and_verify(self):
        token = create_token(
            employee_id=EMPLOYEE_ID,
            role="employee",
            name="张小明",
            hire_date="2023-07-01",
            secret="test-secret",
            expire_hours=1,
        )
        payload = verify_token(token, secret="test-secret")
        assert payload is not None
        assert payload["employee_id"] == EMPLOYEE_ID
        assert payload["role"] == "employee"

    def test_tampered_token_rejected(self):
        token = create_token(
            employee_id=EMPLOYEE_ID,
            role="employee",
            name="张小明",
            secret="test-secret",
        )
        bad = token[:-4] + "xxxx"
        assert verify_token(bad, secret="test-secret") is None

    def test_expired_token_rejected(self):
        token = create_token(
            employee_id=EMPLOYEE_ID,
            role="employee",
            name="张小明",
            secret="test-secret",
            expire_hours=-0.001,
        )
        time.sleep(0.05)
        assert verify_token(token, secret="test-secret") is None

    def test_wrong_secret_rejected(self):
        token = create_token(
            employee_id=EMPLOYEE_ID,
            role="employee",
            name="张小明",
            secret="secret-a",
        )
        assert verify_token(token, secret="secret-b") is None


class TestAuthAPI:
    """HTTP 登录与 /me。"""

    def setup_method(self):
        clear_user_store_cache()

    def test_login_success(self):
        resp = client.post(
            "/api/auth/login",
            json={"employee_id": ADMIN_ID, "password": DEMO_PASSWORD},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["employee_id"] == ADMIN_ID
        assert data["name"] == "王管理"
        assert data["role"] == "admin"
        assert data["token"]

    def test_login_failure(self):
        resp = client.post(
            "/api/auth/login",
            json={"employee_id": EMPLOYEE_ID, "password": "bad"},
        )
        assert resp.status_code == 401
        assert "工号或密码" in resp.json()["detail"]

    def test_me_with_token(self):
        token = _login(INTERN_ID)
        resp = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["employee_id"] == INTERN_ID
        assert data["role"] == "intern"
        assert data["name"] == "李实习"

    def test_me_without_token(self):
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401
