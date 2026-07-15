"""工具层（app/tools/）的单元测试。

覆盖：年假计算规则与边界、制度参数查询命中/未命中、
参数校验失败的结构化返回、注册表的取用行为。
所有测试不依赖网络与 LLM，可离线运行。
"""

from datetime import date, timedelta

import pytest

from app.tools.leave_calculator import ANNUAL_LEAVE_DAYS_CAP, LeaveCalculatorTool
from app.tools.policy_lookup import PolicyLookupTool
from app.tools.registry import get_all_tools, get_tool


def _date_years_ago(years: int, extra_days: int = 1) -> str:
    """构造 N 年零几天前的日期字符串（保证已满 N 整年）。"""
    return (date.today() - timedelta(days=365 * years + extra_days + years // 4)).isoformat()


class TestLeaveCalculator:
    """年假计算器：规则正确性与边界处理。"""

    tool = LeaveCalculatorTool()

    def test_three_years_service(self):
        result = self.tool.invoke({"hire_date": _date_years_ago(3)})
        assert result.success
        assert result.data["years_of_service"] == 3
        assert result.data["annual_leave_days"] == 3

    def test_cap_at_15_days(self):
        result = self.tool.invoke({"hire_date": _date_years_ago(20)})
        assert result.success
        assert result.data["annual_leave_days"] == ANNUAL_LEAVE_DAYS_CAP

    def test_less_than_one_year(self):
        hire_date = (date.today() - timedelta(days=100)).isoformat()
        result = self.tool.invoke({"hire_date": hire_date})
        assert result.success
        assert result.data["annual_leave_days"] == 0

    def test_future_hire_date_returns_error(self):
        future = (date.today() + timedelta(days=30)).isoformat()
        result = self.tool.invoke({"hire_date": future})
        assert not result.success
        assert "晚于今天" in result.error

    def test_invalid_date_string_returns_validation_error(self):
        result = self.tool.invoke({"hire_date": "不是日期"})
        assert not result.success
        assert "参数校验失败" in result.error

    def test_missing_param_returns_validation_error(self):
        result = self.tool.invoke({})
        assert not result.success
        assert "hire_date" in result.error


class TestPolicyLookup:
    """制度参数查询：命中、未命中与参数校验。"""

    tool = PolicyLookupTool()

    def test_hit_office_supplies(self):
        result = self.tool.invoke(
            {"keyword": "办公用品"},
            context={"user_role": "employee"},
        )
        assert result.success
        assert result.data["found"]
        values = {m["key"]: m["value"] for m in result.data["matches"]}
        assert values["office_supplies_limit"] == 500

    def test_hit_hotel_standard_multiple_matches(self):
        # "住宿标准"应同时命中一线/二线城市条目，全部返回由 LLM 挑选
        result = self.tool.invoke(
            {"keyword": "住宿标准"},
            context={"user_role": "employee"},
        )
        assert result.success
        assert result.data["found"]
        assert len(result.data["matches"]) >= 2

    def test_intern_cannot_bypass_acl_via_tool(self):
        """实习生无权查看 internal 报销参数，不能靠工具绕过文档 ACL。"""
        result = self.tool.invoke(
            {"keyword": "办公用品"},
            context={"user_role": "intern"},
        )
        assert result.success
        assert not result.data["found"]
        assert result.data.get("filtered_by_acl") is True
        assert "权限" in result.data["message"]

    def test_miss_returns_explicit_message(self):
        # 未命中是正常业务结果：success=True 但 found=False 且带提示
        result = self.tool.invoke(
            {"keyword": "食堂菜单"},
            context={"user_role": "admin"},
        )
        assert result.success
        assert not result.data["found"]
        assert "未找到" in result.data["message"]

    def test_empty_keyword_returns_validation_error(self):
        result = self.tool.invoke({"keyword": ""})
        assert not result.success
        assert "参数校验失败" in result.error


class TestRegistry:
    """注册表：按名取用、未知名称报错、清单完整性。"""

    def test_get_tool_by_name(self):
        tool = get_tool("leave_calculator")
        assert tool.name == "leave_calculator"

    def test_unknown_tool_raises_with_available_list(self):
        with pytest.raises(KeyError, match="leave_calculator"):
            get_tool("no_such_tool")

    def test_all_tools_registered_with_unique_names(self):
        tools = get_all_tools()
        names = [t.name for t in tools]
        assert sorted(names) == ["leave_calculator", "policy_lookup"]
        # 每个工具都必须具备统一接口要素（MCP 适配依赖这些字段）
        for tool in tools:
            assert tool.description
            assert tool.input_schema is not None


class TestRegistryDrivenInvoke:
    """模拟 Agent 的用法：从 registry 取工具再 invoke，全程不 import 具体工具。"""

    def test_invoke_via_registry(self):
        result = get_tool("policy_lookup").invoke(
            {"keyword": "餐补"},
            context={"user_role": "employee"},
        )
        assert result.success
        assert result.data["found"]
