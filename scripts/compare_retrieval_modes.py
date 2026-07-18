"""纯向量 vs Hybrid 检索对照实验（不调用 LLM）。

在同一向量库、同一 top_k / 阈值下，对固定查询集分别跑 vector 与 hybrid，
统计 top-1 / Hit@K 是否命中期望来源，并打印 Markdown 表格（可重定向写入 docs）。

用法（项目根目录）：
    python scripts/compare_retrieval_modes.py
    python scripts/compare_retrieval_modes.py --output docs/retrieval_comparison.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.rag.bm25_index import get_bm25_index
from app.rag.retriever import Retriever
from app.utils.logger import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "retrieval_comparison.md"


@dataclass(frozen=True)
class Case:
    """单条对照查询。"""

    query: str
    expect_sources_any: tuple[str, ...]
    expect_keywords_any: tuple[str, ...] = ()
    note: str = ""


# 偏制度问答 + 专名/数字敏感查询；期望来源用文件名子串匹配
# 扩充语料后追加易混数字、废止旧版、联查与噪声场景（见 docs/corpus_expansion.md）
CASES: list[Case] = [
    Case("报销要在多久内提交", ("报销",), ("30天", "30 天"), "数字时限"),
    Case(
        "发生后30天内必须提交的费用申请是什么",
        ("报销",),
        ("30天", "30 天"),
        "数字+专名，Hybrid 敏感",
    ),
    Case(
        "费用发生后45天内还能报销吗",
        ("废止", "报销", "FAQ"),
        ("45", "废止"),
        "旧版45天 vs 现行；期望命中废止说明或现行报销",
    ),
    Case(
        "调休要在多少天内休完",
        ("调休", "加班"),
        ("30天", "30 天"),
        "30天易混：调休时限",
    ),
    Case(
        "入职满多少天有补充医疗",
        ("福利", "保险"),
        ("30天", "30 天"),
        "30天易混：补充医疗起缴",
    ),
    Case(
        "迟到超过30分钟怎么处理",
        ("员工手册", "考勤"),
        ("30分钟", "30 分钟"),
        "30分钟 vs 30天",
    ),
    Case(
        "差旅住宿标准和报销提交时限分别是什么",
        ("联查", "差旅", "报销"),
        ("500", "30"),
        "Completeness 联查",
    ),
    Case("差旅住宿标准是什么", ("差旅",)),
    Case("差旅可以坐什么交通工具", ("差旅",)),
    Case("公司考勤制度怎么规定的", ("员工手册", "考勤"), ("考勤",)),
    Case("新员工入职需要办理哪些手续", ("员工手册", "入职"), ("入职",)),
    Case("年假申请流程是什么", ("年假", "请假"), ("年假",)),
    Case("年假可以跨年结转吗", ("年假", "请假"), ("结转", "跨年")),
    Case("入职3年年假几天", ("请假", "年假"), ("年假",), "数字+政策"),
    Case("报销需要哪些审批环节", ("报销",), ("审批",)),
    Case("报销需要提供发票吗", ("报销",), ("发票",)),
    Case("正式员工每月远程办公最多几天", ("远程",), ("8",), "远程额度"),
    Case("企业账号密码最少要几位", ("信息安全", "账号"), ("12",), "口令长度"),
    Case("客户数据泄露要在多久内上报", ("客户数据", "隐私"), ("24",), "上报时限"),
    Case("单笔采购多少钱必须招标", ("采购",), ("5000",), "confidential 采购门槛"),
    Case(
        "团建花絮里说报销很宽松到底按几天算",
        ("报销", "FAQ", "废止", "花絮"),
        ("30",),
        "噪声：非正式可召回，期望仍能落到报销相关",
    ),
]


def _source_hit(source: str, expect: tuple[str, ...]) -> bool:
    return any(token in source for token in expect)


def _keyword_hit(content: str, expect: tuple[str, ...]) -> bool:
    if not expect:
        return True
    return any(token in content for token in expect)


def _first_hit_rank(results: list, expect: tuple[str, ...]) -> int | None:
    for i, item in enumerate(results, start=1):
        if _source_hit(item.source, expect):
            return i
    return None


def _fmt_top1(results: list) -> str:
    if not results:
        return "（空）"
    top = results[0]
    return f"{top.score:.4f} / {top.source}"


def _unique_sources(results: list) -> list[str]:
    seen: list[str] = []
    for item in results:
        if item.source not in seen:
            seen.append(item.source)
    return seen


def run_comparison(top_k: int, threshold: float) -> str:
    """执行对照并返回 Markdown 正文。"""
    settings = get_settings()
    retriever = Retriever()

    # 无论当前 RETRIEVAL_MODE 为何，都预热 BM25，保证 hybrid 路可用
    docs = retriever._vector_store.get_all_documents()
    if docs:
        get_bm25_index().rebuild(docs)

    rows: list[dict] = []
    for case in CASES:
        vector = retriever._retrieve_vector(case.query, top_k=top_k, score_threshold=threshold)
        hybrid = retriever._retrieve_hybrid(case.query, top_k=top_k, score_threshold=threshold)

        v_rank = _first_hit_rank(vector, case.expect_sources_any)
        h_rank = _first_hit_rank(hybrid, case.expect_sources_any)
        v_top1 = bool(vector) and _source_hit(vector[0].source, case.expect_sources_any)
        h_top1 = bool(hybrid) and _source_hit(hybrid[0].source, case.expect_sources_any)
        v_kw = bool(vector) and _keyword_hit(vector[0].content, case.expect_keywords_any)
        h_kw = bool(hybrid) and _keyword_hit(hybrid[0].content, case.expect_keywords_any)
        v_sources = _unique_sources(vector)
        h_sources = _unique_sources(hybrid)

        rows.append(
            {
                "query": case.query,
                "note": case.note,
                "expect": "/".join(case.expect_sources_any),
                "vector_top1": _fmt_top1(vector),
                "hybrid_top1": _fmt_top1(hybrid),
                "vector_top1_ok": v_top1,
                "hybrid_top1_ok": h_top1,
                "vector_hit_at_k": v_rank is not None,
                "hybrid_hit_at_k": h_rank is not None,
                "vector_kw_ok": v_kw,
                "hybrid_kw_ok": h_kw,
                "has_kw": bool(case.expect_keywords_any),
                "vector_rank": v_rank,
                "hybrid_rank": h_rank,
                "vector_sources": "、".join(v_sources) if v_sources else "（空）",
                "hybrid_sources": "、".join(h_sources) if h_sources else "（空）",
                "vector_unique_n": len(v_sources),
                "hybrid_unique_n": len(h_sources),
            }
        )

    n = len(rows)
    v_top1_ok = sum(1 for r in rows if r["vector_top1_ok"])
    h_top1_ok = sum(1 for r in rows if r["hybrid_top1_ok"])
    v_hit = sum(1 for r in rows if r["vector_hit_at_k"])
    h_hit = sum(1 for r in rows if r["hybrid_hit_at_k"])
    kw_rows = [r for r in rows if r["has_kw"]]
    kn = len(kw_rows)
    v_kw_ok = sum(1 for r in kw_rows if r["vector_kw_ok"])
    h_kw_ok = sum(1 for r in kw_rows if r["hybrid_kw_ok"])
    avg_v_unique = sum(r["vector_unique_n"] for r in rows) / n
    avg_h_unique = sum(r["hybrid_unique_n"] for r in rows) / n

    # Hybrid 相对纯向量：top-1 纠正 / Hit@K 纠正
    top1_fixed = sum(
        1 for r in rows if (not r["vector_top1_ok"]) and r["hybrid_top1_ok"]
    )
    top1_regressed = sum(
        1 for r in rows if r["vector_top1_ok"] and (not r["hybrid_top1_ok"])
    )
    hit_fixed = sum(
        1 for r in rows if (not r["vector_hit_at_k"]) and r["hybrid_hit_at_k"]
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# 纯向量 vs Hybrid 检索对照",
        "",
        f"生成时间：{now}",
        "",
        "## 如何复现",
        "",
        "```bash",
        "python scripts/compare_retrieval_modes.py",
        "python scripts/compare_retrieval_modes.py --output docs/retrieval_comparison.md",
        "```",
        "",
        "## 实验设置",
        "",
        "- 对比对象：`vector`（纯向量 + 阈值）vs `hybrid`（向量 + BM25，经 RRF 融合）",
        "- 不调用 LLM；直接调用检索器两路实现，避免 Agent 路由噪声",
        "- 未套用文档密级 ACL（两路一致，聚焦召回差异）",
        f"- `top_k={top_k}`，`score_threshold={threshold}`（向量路过滤；与配置默认一致时见下）",
        f"- 当前配置默认：`RETRIEVAL_TOP_K={settings.retrieval_top_k}`，"
        f"`RETRIEVAL_SCORE_THRESHOLD={settings.retrieval_score_threshold}`，"
        f"`HYBRID_RRF_K={settings.hybrid_rrf_k}`，`HYBRID_BM25_TOP_K={settings.hybrid_bm25_top_k}`",
        f"- 向量库文档块数：{len(docs)}；Embedding / 切分策略以当前入库结果为准",
        "- 判定：",
        "  - **来源正确**：期望子串出现在命中文件名（如「报销」→「报销制度.md」）",
        "  - **关键词命中**：top-1 正文包含约定关键片段（如「30天」「发票」）",
        "",
        "## 汇总指标",
        "",
        f"| 指标 | 纯向量 | Hybrid |",
        f"|------|--------|--------|",
        f"| Top-1 来源正确 | {v_top1_ok}/{n}（{v_top1_ok / n * 100:.1f}%） | "
        f"{h_top1_ok}/{n}（{h_top1_ok / n * 100:.1f}%） |",
        f"| Hit@{top_k} 来源命中 | {v_hit}/{n}（{v_hit / n * 100:.1f}%） | "
        f"{h_hit}/{n}（{h_hit / n * 100:.1f}%） |",
        f"| Top-1 关键词命中 | {v_kw_ok}/{kn}（{v_kw_ok / kn * 100:.1f}%） | "
        f"{h_kw_ok}/{kn}（{h_kw_ok / kn * 100:.1f}%） |",
        f"| Top-{top_k} 平均独特来源数 | {avg_v_unique:.2f} | {avg_h_unique:.2f} |",
        "",
        f"- Hybrid 相对纯向量：**Top-1 纠正 {top1_fixed} 条**，**Top-1 回退 {top1_regressed} 条**；"
        f"**Hit@{top_k} 额外召回 {hit_fixed} 条**。",
        f"- Top-{top_k} 独特来源数：Hybrid 平均 {avg_h_unique:.2f}，纯向量平均 {avg_v_unique:.2f}"
        "（更高通常表示关键词路拉入了更多不同文档，需结合是否引入干扰解读）。",
        "",
        "## 逐条结果（来源命中）",
        "",
        "| 查询 | 期望来源 | 纯向量 top-1（分/源） | Hybrid top-1（分/源） | "
        f"V Top-1 | H Top-1 | V Hit@{top_k} | H Hit@{top_k} | 备注 |",
        "|------|----------|----------------------|----------------------|"
        "--------|--------|----------|----------|------|",
    ]

    def yn(ok: bool) -> str:
        return "Y" if ok else "N"

    for r in rows:
        lines.append(
            f"| {r['query']} | {r['expect']} | {r['vector_top1']} | {r['hybrid_top1']} | "
            f"{yn(r['vector_top1_ok'])} | {yn(r['hybrid_top1_ok'])} | "
            f"{yn(r['vector_hit_at_k'])} | {yn(r['hybrid_hit_at_k'])} | {r['note']} |"
        )

    lines.extend(
        [
            "",
            "## 逐条结果（关键词 + Top-K 来源列表）",
            "",
            f"| 查询 | V 关键词 | H 关键词 | 纯向量 Top-{top_k} 来源 | Hybrid Top-{top_k} 来源 |",
            "|------|----------|----------|------------------------|------------------------|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['query']} | {yn(r['vector_kw_ok']) if r['has_kw'] else '-'} | "
            f"{yn(r['hybrid_kw_ok']) if r['has_kw'] else '-'} | "
            f"{r['vector_sources']} | {r['hybrid_sources']} |"
        )

    lines.extend(
        [
            "",
            "## 解读（简历/面试可用）",
            "",
            "- 分数列在两种模式下**不可直接横向比大小**：纯向量分为余弦相似度；"
            "Hybrid 展示分为 RRF 归一化与向量分的合成，主要用于排序，不是同一量纲。"
            "两路都命中且 BM25/向量 top-1 一致时，Hybrid 展示分常被归一化到约 1.0。",
            "- 在当前语料规模下，**来源级 Hit@K 仍可能打满**；"
            "更应看关键词是否进 top-1 正文，以及 Hybrid 是否因字面「30」等把无关制度拉进 Top-K。",
            "- 若来源命中率持平，简历宜写「与纯向量对照实验，小语料下来源 Hit 持平，"
            "Hybrid 作专名/数字兜底默认策略」，**不要编造「提升 XX%」**。",
            "",
            "## 结论",
            "",
        ]
    )

    if h_top1_ok > v_top1_ok or h_hit > v_hit:
        lines.append(
            f"在本查询集上，Hybrid 的 Top-1 正确率 "
            f"（{h_top1_ok}/{n}）与 Hit@{top_k}（{h_hit}/{n}）不低于纯向量"
            f"（{v_top1_ok}/{n}、{v_hit}/{n}），"
            f"其中 Top-1 纠正 {top1_fixed} 条、Hit@{top_k} 额外召回 {hit_fixed} 条；"
            "默认继续采用 `RETRIEVAL_MODE=hybrid`，可用 `vector` 做对照基线。"
        )
    elif h_top1_ok == v_top1_ok and h_hit == v_hit:
        lines.append(
            f"在本查询集（{n} 条）上，两路 **Top-1 来源**与 **Hit@{top_k}** 均为 "
            f"{v_top1_ok}/{n}、{v_hit}/{n}（持平）；"
            f"Top-1 关键词命中为 纯向量 {v_kw_ok}/{kn}、Hybrid {h_kw_ok}/{kn}。"
            f"小语料下语义召回已较强，来源级指标触顶；"
            f"Hybrid Top-{top_k} 平均独特来源数（{avg_h_unique:.2f}）高于纯向量"
            f"（{avg_v_unique:.2f}），体现关键词路会拉入更多文档（含潜在干扰）。"
            "默认仍采用 `RETRIEVAL_MODE=hybrid` 作为专名/数字兜底，"
            "可用 `vector` 随时对照。"
        )
    else:
        lines.append(
            f"在本查询集上 Hybrid 未优于纯向量（Top-1 {h_top1_ok}/{n} vs {v_top1_ok}/{n}）；"
            "需结合逐条结果排查 BM25 分词或融合参数，勿在简历中写「全面提升」。"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="纯向量 vs Hybrid 检索对照")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"写入 Markdown 报告路径（默认 {DEFAULT_OUTPUT}）",
    )
    parser.add_argument("--top-k", type=int, default=None, help="覆盖 retrieval_top_k")
    parser.add_argument(
        "--threshold", type=float, default=None, help="覆盖 retrieval_score_threshold"
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="只打印到标准输出，不写文件",
    )
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    top_k = args.top_k if args.top_k is not None else settings.retrieval_top_k
    threshold = (
        args.threshold
        if args.threshold is not None
        else settings.retrieval_score_threshold
    )

    report = run_comparison(top_k=top_k, threshold=threshold)
    print(report)

    if not args.stdout_only:
        out = args.output or DEFAULT_OUTPUT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\n已写入: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
