"""Agent 决策图演示脚本：手动验证完整问答链路（支持多轮 session）。

用法（在项目根目录执行，需已配置 .env 的 LLM_API_KEY）：
    python scripts/agent_demo.py "我2023年7月入职，年假有几天"
    python scripts/agent_demo.py "那报销上限呢" --session-id <上一步输出的 session_id>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.graph import run_agent
from app.utils.logger import setup_logging


def main() -> None:
    """解析命令行参数，执行 Agent 问答并打印全过程信息。"""
    parser = argparse.ArgumentParser(description="Agent 决策图问答演示")
    parser.add_argument("question", help="用户问题，如：我2023年7月入职，年假有几天")
    parser.add_argument(
        "--session-id",
        default=None,
        help="可选会话 ID；多轮追问时传入上一轮输出的 session_id",
    )
    parser.add_argument(
        "--role",
        default="employee",
        choices=["intern", "employee", "admin"],
        help="模拟角色（影响文档密级可见性）",
    )
    args = parser.parse_args()

    setup_logging()
    result = run_agent(args.question, session_id=args.session_id, user_role=args.role)

    print("\n" + "=" * 60)
    print(f"问题:     {args.question}")
    print(f"会话 ID:  {result.session_id}")
    print(f"角色:     {result.user_role}")
    print(f"请求 ID:  {result.request_id}")
    print(f"检索 query: {result.search_query or '（同问题）'}")
    print(f"分诊路由: {result.route or '（未经过路由）'}")
    print(f"使用检索: {'是' if result.used_retrieval else '否'}")
    print(f"发生降级: {'是' if result.degraded else '否'}")
    if result.timings:
        timing_str = ", ".join(f"{k}={v:.0f}ms" for k, v in result.timings.items())
        print(f"节点耗时: {timing_str}")
    if result.tool_calls:
        print("工具调用轨迹:")
        for i, rec in enumerate(result.tool_calls, 1):
            status = "成功" if rec["success"] else "失败"
            print(f"  [{i}] {rec['tool_name']}({rec['args']}) -> {status}: {rec['result'][:100]}")
    else:
        print("工具调用轨迹: （无）")
    if result.sources:
        print(f"引用来源: {', '.join(result.sources)}")
    print("-" * 60)
    print(f"回答:\n{result.answer}")
    print("=" * 60)


if __name__ == "__main__":
    main()
