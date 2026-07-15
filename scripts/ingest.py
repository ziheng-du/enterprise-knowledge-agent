"""文档入库脚本：将 data/raw_docs 中的文档批量处理并写入 Chroma 向量库。

实际入库逻辑在 app.rag.ingest_service.ingest，本脚本只负责 CLI 参数解析。

用法（在项目根目录执行）：
    python scripts/ingest.py                    # 增量入库（追加写入）
    python scripts/ingest.py --rebuild          # 清空后全量重建
    python scripts/ingest.py --strategy fixed   # 临时指定切分策略（覆盖配置）
"""

import argparse
import sys
from pathlib import Path

# 保证以 `python scripts/ingest.py` 方式运行时能找到 app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.ingest_service import ingest
from app.utils.logger import setup_logging


def main() -> None:
    """解析命令行参数并执行入库。"""
    parser = argparse.ArgumentParser(description="将 raw_docs 中的文档批量写入 Chroma 向量库")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="先清空向量库再全量重建（默认为追加写入）",
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "recursive"],
        default=None,
        help="切分策略，默认使用 .env 中的 CHUNKING_STRATEGY",
    )
    args = parser.parse_args()

    setup_logging()
    ingest(rebuild=args.rebuild, strategy=args.strategy)


if __name__ == "__main__":
    main()
