"""run_eval 报告渲染与汇总的离线单测（不调用 LLM）。"""

from scripts.run_eval import render_report, summarize


def test_summarize_by_type():
    results = [
        {"id": "a", "type": "rag", "ok": True, "reasons": []},
        {"id": "b", "type": "rag", "ok": False, "reasons": ["x"]},
        {"id": "c", "type": "refuse", "ok": True, "reasons": []},
    ]
    summary = summarize(results)
    assert summary["passed"] == 2
    assert summary["total"] == 3
    assert summary["by_type"]["rag"] == {"passed": 1, "total": 2}
    assert len(summary["failures"]) == 1


def test_render_report_offline_mentions_reproduce():
    text = render_report({}, [], offline=True)
    assert "需本地复现" in text
    assert "--offline" in text


def test_render_report_online_has_table():
    results = [
        {"id": "a", "type": "rag", "ok": True, "route": "rag", "reasons": []},
        {"id": "b", "type": "tool", "ok": False, "route": "tool", "reasons": ["missing tool"]},
    ]
    summary = summarize(results)
    text = render_report(summary, results, offline=False)
    assert "1/2" in text or "**1/2**" in text
    assert "missing tool" in text
    assert "| rag |" in text
