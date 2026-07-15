"""检索演示脚本：验证"纯检索"链路，不调用 LLM。

输入一个问题，输出命中的文档片段、相似度分数和来源文件，
用于人工评估检索质量与阈值过滤效果（阶段二验收工具）。

用法（在项目根目录执行）：
    python scripts/search_demo.py "报销多久内要提交"
    python scripts/search_demo.py "入职3年年假几天" --top-k 5 --threshold 0.2
"""

import argparse
import sys
from pathlib import Path

# 保证以 `python scripts/search_demo.py` 方式运行时能找到 app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.retriever import Retriever
from app.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """解析命令行参数，执行检索并打印结果。"""
    parser = argparse.ArgumentParser(description="向量检索演示（不调用 LLM）")
    parser.add_argument("query", help="查询问题，如：报销多久内要提交")
    parser.add_argument("--top-k", type=int, default=None, help="候选数量，默认读配置")
    parser.add_argument(
        "--threshold", type=float, default=None, help="相似度阈值（0-1），默认读配置"
    )
    args = parser.parse_args()

    setup_logging()

    retriever = Retriever()
    results = retriever.retrieve(args.query, top_k=args.top_k, score_threshold=args.threshold)

    # 演示脚本的结果输出属于程序功能输出（非调试信息），使用 print 合理
    print(f"\n查询: {args.query}")
    if not results:
        print("未检索到高于阈值的相关内容（知识库中可能没有相关信息）。")
        return

    print(f"命中 {len(results)} 条结果:\n" + "=" * 60)
    for i, r in enumerate(results, 1):
        print(f"[{i}] 分数: {r.score:.4f} | 来源: {r.source} | 块序号: {r.metadata.get('chunk_index')}")
        print(f"    {r.content.strip()[:200]}")
        print("-" * 60)


if __name__ == "__main__":
    main()
