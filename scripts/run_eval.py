"""黄金集评测脚本：量化路由 / 工具 / 拒答 / 关键词命中，并可导出 Markdown 报告。

用法（项目根目录）：
    python scripts/run_eval.py --offline
    python scripts/run_eval.py --output docs/eval_report.md
    python scripts/run_eval.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.graph import run_agent
from app.utils.logger import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = PROJECT_ROOT / "data" / "eval" / "golden_set.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "eval_report.md"


def load_cases(path: Path) -> list[dict]:
    """加载 jsonl 黄金集。"""
    cases: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def validate_structure(cases: list[dict]) -> list[str]:
    """离线结构校验，返回错误列表（空表示通过）。"""
    errors: list[str] = []
    required_ids: set[str] = set()
    for i, case in enumerate(cases):
        cid = case.get("id") or f"line_{i}"
        if cid in required_ids:
            errors.append(f"重复 id: {cid}")
        required_ids.add(cid)
        ctype = case.get("type")
        if ctype == "offline":
            continue
        if ctype == "multi_turn":
            if not case.get("turns"):
                errors.append(f"{cid}: multi_turn 缺少 turns")
            continue
        if not case.get("question"):
            errors.append(f"{cid}: 缺少 question")
        if ctype not in {"rag", "tool", "both", "refuse", "multi_turn", "offline"}:
            errors.append(f"{cid}: 未知 type={ctype}")
    return errors


def _check_keywords(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    return all(k in answer for k in keywords)


def eval_single(case: dict) -> dict:
    """评测单条（非 multi_turn）。"""
    result = run_agent(case["question"])
    ok = True
    reasons: list[str] = []

    expect_route = case.get("expect_route") or []
    if expect_route and result.route not in expect_route:
        ok = False
        reasons.append(f"route={result.route} not in {expect_route}")

    expect_tool = case.get("expect_tool")
    if expect_tool:
        names = [t.get("tool_name") for t in result.tool_calls]
        if expect_tool not in names:
            ok = False
            reasons.append(f"missing tool {expect_tool}, got {names}")

    if case.get("expect_refuse"):
        if "未找到" not in result.answer:
            ok = False
            reasons.append("expected refuse phrase 未找到")

    if not _check_keywords(result.answer, case.get("expect_keywords") or []):
        ok = False
        reasons.append(f"missing keywords {case.get('expect_keywords')}")

    sources_any = case.get("expect_sources_any") or []
    if sources_any:
        joined = " ".join(result.sources)
        if not any(s in joined for s in sources_any):
            if not case.get("expect_refuse"):
                ok = False
                reasons.append(f"sources {result.sources} miss any of {sources_any}")

    return {
        "id": case["id"],
        "type": case.get("type", "unknown"),
        "ok": ok,
        "route": result.route,
        "reasons": reasons,
        "timings": result.timings,
        "request_id": result.request_id,
    }


def eval_multi_turn(case: dict) -> dict:
    """多轮题：同一 session_id 连续提问。"""
    session_id = f"eval-{case['id']}"
    reasons: list[str] = []
    ok = True
    last_route = ""
    for turn in case.get("turns") or []:
        result = run_agent(turn["question"], session_id=session_id)
        last_route = result.route
        if not _check_keywords(result.answer, turn.get("expect_keywords") or []):
            ok = False
            reasons.append(f"turn {turn['question'][:20]} missing keywords")
    return {
        "id": case["id"],
        "type": "multi_turn",
        "ok": ok,
        "route": last_route,
        "reasons": reasons,
        "timings": {},
        "request_id": "",
    }


def summarize(results: list[dict]) -> dict:
    """按 type 汇总通过率。"""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_type[row.get("type", "unknown")].append(row)
    groups = {}
    for ctype, rows in sorted(by_type.items()):
        p = sum(1 for r in rows if r["ok"])
        groups[ctype] = {"passed": p, "total": len(rows)}
    passed = sum(1 for r in results if r["ok"])
    return {
        "passed": passed,
        "total": len(results),
        "by_type": groups,
        "failures": [
            {"id": r["id"], "type": r.get("type"), "reasons": r["reasons"]}
            for r in results
            if not r["ok"]
        ],
    }


def render_report(summary: dict, results: list[dict], *, offline: bool = False) -> str:
    """生成 Markdown 评测报告文本。"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# 黄金集评测报告",
        "",
        f"生成时间：{now}",
        "",
        "## 如何复现",
        "",
        "```bash",
        "python scripts/run_eval.py --offline          # 无 Key：仅结构校验",
        "python scripts/run_eval.py --output docs/eval_report.md  # 有 Key：全量并写报告",
        "```",
        "",
    ]
    if offline:
        lines.extend(
            [
                "## 本次运行",
                "",
                "本次为 `--offline` 结构校验，**未调用 LLM**，无在线通过率。",
                "配置 `LLM_API_KEY` 后执行上方全量命令即可刷新本报告中的指标表。",
                "",
                "## 指标占位（需本地有 Key 复现）",
                "",
                "| 维度 | 通过 | 总数 | 通过率 |",
                "|------|------|------|--------|",
                "| overall | - | - | 需本地复现 |",
                "| rag / tool / both / refuse / multi_turn | - | - | 需本地复现 |",
                "",
            ]
        )
        return "\n".join(lines)

    total = summary["total"]
    passed = summary["passed"]
    rate = (passed / total * 100) if total else 0.0
    lines.extend(
        [
            "## 总体结果",
            "",
            f"- 通过：**{passed}/{total}**（{rate:.1f}%）",
            "",
            "## 按题型分组",
            "",
            "| type | 通过 | 总数 | 通过率 |",
            "|------|------|------|--------|",
        ]
    )
    for ctype, g in summary["by_type"].items():
        g_rate = (g["passed"] / g["total"] * 100) if g["total"] else 0.0
        lines.append(f"| {ctype} | {g['passed']} | {g['total']} | {g_rate:.1f}% |")
    lines.append("")

    failures = summary.get("failures") or []
    lines.extend(["## 失败用例", ""])
    if not failures:
        lines.append("无失败用例。")
        lines.append("")
    else:
        for fail in failures:
            reasons = "; ".join(fail.get("reasons") or []) or "（无详情）"
            lines.append(f"- `{fail['id']}`（{fail.get('type')}）：{reasons}")
        lines.append("")

    lines.extend(
        [
            "## 逐条结果",
            "",
            "| id | type | 结果 | route |",
            "|----|------|------|-------|",
        ]
    )
    for row in results:
        status = "PASS" if row["ok"] else "FAIL"
        lines.append(
            f"| {row['id']} | {row.get('type', '')} | {status} | {row.get('route', '')} |"
        )
    lines.append("")
    lines.extend(
        [
            "## 指标解读（面试可用）",
            "",
            "- **rag**：制度问答是否命中正确文档/关键词。",
            "- **tool**：是否调用了预期工具（如年假计算）。",
            "- **refuse**：知识库外问题是否如实「未找到」而非编造。",
            "- **multi_turn**：同 session 追问是否仍答对关键信息。",
            "- 路由允许一定容差（如 rag/both），因分诊 LLM 非确定性。",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    """写入报告文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"报告已写入: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="黄金集评测")
    parser.add_argument("--offline", action="store_true", help="仅校验题集结构，不调 LLM")
    parser.add_argument("--limit", type=int, default=0, help="限制在线评测条数（0=全部）")
    parser.add_argument("--path", type=Path, default=GOLDEN_PATH, help="黄金集 jsonl 路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Markdown 报告输出路径（默认不写文件；建议 {DEFAULT_REPORT_PATH}）",
    )
    args = parser.parse_args()

    setup_logging()
    cases = load_cases(args.path)
    struct_errors = validate_structure(cases)
    if struct_errors:
        print("结构校验失败:")
        for e in struct_errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"结构校验通过，共 {len(cases)} 条")

    if args.offline:
        print("(--offline) 跳过在线评测")
        if args.output:
            write_report(args.output, render_report({}, [], offline=True))
        return

    online = [c for c in cases if c.get("type") != "offline"]
    if args.limit > 0:
        online = online[: args.limit]

    results = []
    for case in online:
        if case.get("type") == "multi_turn":
            row = eval_multi_turn(case)
        else:
            row = eval_single(case)
        results.append(row)
        status = "PASS" if row["ok"] else "FAIL"
        print(f"[{status}] {row['id']} route={row['route']} {'; '.join(row['reasons'])}")

    summary = summarize(results)
    passed, total = summary["passed"], summary["total"]
    print("-" * 40)
    print(f"汇总: {passed}/{total} 通过 ({(passed / total * 100) if total else 0:.1f}%)")
    for ctype, g in summary["by_type"].items():
        print(f"  {ctype}: {g['passed']}/{g['total']}")

    if args.output:
        write_report(args.output, render_report(summary, results, offline=False))

    sys.exit(0 if passed == total else 2)


if __name__ == "__main__":
    main()
