"""FastAPI 路由层（app/api/routes.py）单元测试。

用 TestClient + mock run_agent，不依赖真实 LLM / 向量库。
聊天接口需登录 Bearer token；角色来自用户库，不信任请求体。
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.agent.graph import AgentResult
from app.auth.users import clear_user_store_cache
from app.main import app

client = TestClient(app)

DEMO_PASSWORD = "demo123"
EMPLOYEE_ID = "E1001"
INTERN_ID = "I2001"


def _auth_header(employee_id: str = EMPLOYEE_ID) -> dict[str, str]:
    """登录指定演示账号并返回 Authorization 头。"""
    clear_user_store_cache()
    resp = client.post(
        "/api/auth/login",
        json={"employee_id": employee_id, "password": DEMO_PASSWORD},
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


class TestHealthAndValidation:
    """健康检查与请求校验。"""

    def test_health_ok(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_chat_without_auth_returns_401(self):
        resp = client.post("/api/chat", json={"message": "你好"})
        assert resp.status_code == 401

    def test_chat_empty_message_returns_422(self):
        resp = client.post(
            "/api/chat",
            json={"message": ""},
            headers=_auth_header(),
        )
        assert resp.status_code == 422


class TestChatWithMockAgent:
    """mock run_agent 后验证响应字段映射与角色注入。"""

    def test_chat_maps_agent_result(self):
        fake = AgentResult(
            answer="年假为3天",
            sources=["请假与年假政策.md"],
            route="tool",
            tool_calls=[
                {
                    "tool_name": "leave_calculator",
                    "args": {"hire_date": "2023-07-01"},
                    "success": True,
                    "result": '{"annual_leave_days": 3}',
                }
            ],
            used_retrieval=False,
            degraded=False,
            session_id="abc123",
            user_role="employee",
            search_query="年假天数计算",
            request_id="req1",
            timings={"router": 12.5, "generate": 80.0},
        )
        with patch("app.api.routes.run_agent", return_value=fake) as mocked:
            resp = client.post(
                "/api/chat",
                json={
                    "message": "我2023年入职年假几天",
                    "session_id": "abc123",
                },
                headers=_auth_header(EMPLOYEE_ID),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "年假为3天"
        assert data["route"] == "tool"
        assert data["used_retrieval"] is False
        assert data["sources"] == ["请假与年假政策.md"]
        assert data["tool_calls"][0]["tool_name"] == "leave_calculator"
        assert data["degraded"] is False
        assert data["session_id"] == "abc123"
        assert data["user_role"] == "employee"
        assert data["search_query"] == "年假天数计算"
        assert data["request_id"] == "req1"
        assert data["timings"]["router"] == 12.5
        mocked.assert_called_once()
        call_kwargs = mocked.call_args
        assert call_kwargs.kwargs["user_role"] == "employee"
        assert call_kwargs.kwargs["session_id"] == "abc123"

    def test_chat_role_comes_from_token_not_body(self):
        """请求体即便带 role 字段也应被忽略；实习生登录则注入 intern。"""
        fake = AgentResult(
            answer="ok",
            session_id="s1",
            request_id="r1",
            user_role="intern",
        )
        with patch("app.api.routes.run_agent", return_value=fake) as mocked:
            resp = client.post(
                "/api/chat",
                json={
                    "message": "你好",
                    "role": "admin",  # 已从 ChatRequest 删除；多余字段默认忽略
                },
                headers=_auth_header(INTERN_ID),
            )
        assert resp.status_code == 200
        mocked.assert_called_once()
        assert mocked.call_args.kwargs["user_role"] == "intern"

    def test_chat_without_session_id_generates_one(self):
        fake = AgentResult(
            answer="ok",
            session_id="generated",
            request_id="r2",
            user_role="employee",
        )
        with patch("app.api.routes.run_agent", return_value=fake) as mocked:
            resp = client.post(
                "/api/chat",
                json={"message": "你好"},
                headers=_auth_header(EMPLOYEE_ID),
            )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "generated"
        mocked.assert_called_once()
        # 路由层会预生成 session_id 再传给 run_agent
        assert mocked.call_args.kwargs["session_id"]
        assert mocked.call_args.kwargs["user_role"] == "employee"
