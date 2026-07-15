"""日志基础设施：全项目统一的 logging 初始化与 logger 获取入口。

项目规范：禁止使用 print 做调试输出，所有模块通过 `get_logger(__name__)`
获取 logger，日志级别由配置（Settings.log_level）统一控制。
"""

import logging
import sys

from app.config import get_settings

# 统一日志格式：时间 | 级别 | 模块名 | 消息
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 模块级标记，防止重复调用 setup_logging 时叠加多个 handler 导致日志重复输出
_configured = False


def setup_logging(level: str | None = None) -> None:
    """初始化根 logger：设置级别、格式，并输出到控制台（stdout）。

    应在应用入口（如 FastAPI 启动、ingest 脚本）处调用一次；
    重复调用是安全的（幂等），不会叠加 handler。

    Args:
        level: 日志级别字符串（如 "DEBUG"/"INFO"）。传 None 时从
            Settings.log_level 读取。非法级别字符串会降级为 INFO 并告警。
    """
    global _configured
    if _configured:
        return

    level_name = (level or get_settings().log_level).upper()
    # 边界情况：配置了非法级别字符串时降级为 INFO，避免启动直接崩溃。
    # 注：不用 logging.getLevelNamesMapping()，它是 Python 3.11+ API，项目需兼容 3.10
    resolved_level = getattr(logging, level_name, None)
    is_valid_level = isinstance(resolved_level, int)
    if not is_valid_level:
        resolved_level = logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)
    root_logger.addHandler(handler)

    _configured = True
    if not is_valid_level:
        root_logger.warning("非法日志级别 %r，已降级为 INFO。", level_name)


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的 logger（通常传入模块的 __name__）。

    首次调用时会自动完成 logging 初始化，保证即使调用方忘记
    显式执行 setup_logging，日志也有统一格式而非丢失。

    Args:
        name: logger 名称，约定为调用方模块的 __name__。

    Returns:
        配置好的 logging.Logger 实例。
    """
    setup_logging()
    return logging.getLogger(name)
