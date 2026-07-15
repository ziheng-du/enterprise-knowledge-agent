"""工具1：年假天数计算器。

年假规则为模拟设定（与 data/raw_docs/请假与年假政策.md 保持一致）：
- 员工自入职之日起满 1 年后开始享有带薪年假
- 每满 1 年工龄增加 1 天年假，上限 15 天
- 不满 1 年为 0 天

设计说明：本工具继承 BaseTool 统一接口（name/description/input_schema/run），
Agent 通过 registry 动态获取，未来可无改动地包一层 MCP 协议适配对外暴露。
"""

from datetime import date

from pydantic import BaseModel, Field

from app.tools.base import BaseTool

# 模拟设定的年假规则参数
ANNUAL_LEAVE_DAYS_CAP = 15
ANNUAL_LEAVE_RULE_TEXT = "每满 1 年工龄增加 1 天年假，上限 15 天；不满 1 年为 0 天（模拟设定）"


class LeaveCalculatorInput(BaseModel):
    """年假计算器的输入参数。"""

    hire_date: date = Field(
        description="员工入职日期，ISO 格式字符串，如 2023-07-01",
    )


def _full_years_between(start: date, end: date) -> int:
    """计算两个日期之间的整年数（按周年日计算，未到周年日不满一年）。

    例如 2023-07-01 入职，到 2026-06-30 为 2 整年，到 2026-07-01 为 3 整年。
    """
    years = end.year - start.year
    # 若今年的周年日还没到，整年数减 1
    if (end.month, end.day) < (start.month, start.day):
        years -= 1
    return years


class LeaveCalculatorTool(BaseTool):
    """年假天数计算工具。"""

    name = "leave_calculator"
    description = (
        "根据员工入职日期计算当前可用的年假天数。"
        "适用于'我入职X年了，年假有几天/还剩多少天'这类需要计算的问题。"
        "输入：hire_date（入职日期，如 2023-07-01）。"
        "返回：工龄整年数与对应的年假天数。"
    )
    input_schema = LeaveCalculatorInput

    def run(self, params: LeaveCalculatorInput) -> dict:
        """计算年假天数。

        Args:
            params: 校验后的输入参数（含 hire_date）。

        Returns:
            dict：hire_date（入职日期）、years_of_service（整年工龄）、
            annual_leave_days（年假天数）、rule（规则说明，方便 LLM
            在回答中向用户解释计算依据）。

        Raises:
            ValueError: 入职日期晚于今天时抛出（业务上不合法），
                由基类 invoke() 统一转为 ToolResult.error。
        """
        today = date.today()
        if params.hire_date > today:
            raise ValueError(f"入职日期 {params.hire_date} 晚于今天 {today}，无法计算工龄")

        years = _full_years_between(params.hire_date, today)
        leave_days = min(years, ANNUAL_LEAVE_DAYS_CAP)

        return {
            "hire_date": params.hire_date.isoformat(),
            "years_of_service": years,
            "annual_leave_days": leave_days,
            "rule": ANNUAL_LEAVE_RULE_TEXT,
        }
