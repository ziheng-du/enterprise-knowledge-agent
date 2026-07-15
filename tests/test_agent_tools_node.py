"""Agent 工具执行节点（execute_tools_node）离线单元测试。

不调用 LLM，直接构造带 tool_calls 的假状态，验证：
成功执行、参数错误回喂、未知工具名兜底。
"""

from langchain_core.messages import AIMessage, ToolMessage

from app.agent.graph import execute_tools_node


def _fake_state(tool_calls: list[dict]) -> dict:
    """构造仅含工具执行节点所需字段的最小状态。"""
    return {
        "question": "q",
        "session_id": "",
        "user_role": "employee",
        "chat_history": [],
        "session_summary": "",
        "user_profile": {},
        "search_query": "q",
        "route": "tool",
        "retrieved": [],
        "messages": [AIMessage(content="", tool_calls=tool_calls)],
        "tool_calls_history": [],
        "current_round": 0,
        "degraded": False,
        "node_timings": {},
        "final_answer": "",
        "sources": [],
    }


class TestExecuteToolsNode:
    """execute_tools_node 三类行为。"""

    def test_successful_leave_calculator(self):
        out = execute_tools_node(
            _fake_state(
                [{"name": "leave_calculator", "args": {"hire_date": "2023-07-01"}, "id": "c1"}]
            )
        )
        assert out["current_round"] == 1
        assert out["tool_calls_history"][0]["success"] is True
        assert "annual_leave_days" in out["tool_calls_history"][0]["result"]
        assert isinstance(out["messages"][-1], ToolMessage)

    def test_invalid_args_feeds_error_back(self):
        out = execute_tools_node(
            _fake_state(
                [{"name": "leave_calculator", "args": {"hire_date": "不是日期"}, "id": "c2"}]
            )
        )
        record = out["tool_calls_history"][0]
        assert record["success"] is False
        assert "参数校验失败" in record["result"]
        assert isinstance(out["messages"][-1], ToolMessage)

    def test_unknown_tool_does_not_raise(self):
        out = execute_tools_node(
            _fake_state([{"name": "no_such_tool", "args": {}, "id": "c3"}])
        )
        record = out["tool_calls_history"][0]
        assert record["success"] is False
        assert "不存在" in record["result"]
        assert "leave_calculator" in record["result"]
