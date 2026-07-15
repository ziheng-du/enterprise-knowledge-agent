"""工具注册表：统一管理所有可用工具。

设计意图（架构硬性约束）：
- 这里是全项目唯一的"可用工具清单"，采用显式注册（而非装饰器自动扫描），
  新增/下线工具只改这一个文件，一目了然
- Agent 决策逻辑（agent/graph.py）只 import 本模块，通过 get_all_tools()
  动态获取工具列表，禁止直接 import 具体工具模块 —— 未来某个工具改造成
  MCP Server 时，只需在这里把本地实例替换为 MCP 客户端适配器实例，
  Agent 侧代码零改动
"""

from app.tools.base import BaseTool
from app.tools.leave_calculator import LeaveCalculatorTool
from app.tools.policy_lookup import PolicyLookupTool

# 显式注册的工具清单（name -> 工具实例）。
# 工具实例是无状态的，模块加载时构造一次全局复用。
_TOOLS: dict[str, BaseTool] = {
    tool.name: tool
    for tool in [
        LeaveCalculatorTool(),
        PolicyLookupTool(),
    ]
}


def get_tool(name: str) -> BaseTool:
    """按名称获取工具实例。

    Args:
        name: 工具名（BaseTool.name）。

    Returns:
        对应的工具实例。

    Raises:
        KeyError: 工具名不存在时抛出，错误信息附带可用工具列表，
            方便 Agent 层记录日志排查（LLM 可能生成不存在的工具名）。
    """
    if name not in _TOOLS:
        raise KeyError(f"工具 {name!r} 不存在，可用工具: {sorted(_TOOLS)}")
    return _TOOLS[name]


def get_all_tools() -> list[BaseTool]:
    """获取全部已注册工具。

    Returns:
        工具实例列表，供 Agent 动态生成 LLM 的工具描述
        （阶段四会把 name/description/input_schema 转成工具调用协议格式）。
    """
    return list(_TOOLS.values())
