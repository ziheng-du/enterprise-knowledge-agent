"""黄金集离线结构校验（不调用 LLM，不依赖 scripts 包导入）。"""

import json
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent.parent / "data" / "eval" / "golden_set.jsonl"


def test_golden_set_structure():
    cases = []
    with GOLDEN.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    assert len(cases) >= 20
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        if case.get("type") == "offline":
            continue
        if case.get("type") == "multi_turn":
            assert case.get("turns")
            continue
        assert case.get("question")
        assert case.get("type") in {"rag", "tool", "both", "refuse", "multi_turn"}
