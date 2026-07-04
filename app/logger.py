"""日志配置模块。

提供统一的日志记录功能，支持控制台和文件双输出，
日志级别和文件路径通过配置中心管理。
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from app.config import settings


def setup_logger(name: str = "rag_system") -> logging.Logger:
    """配置并返回应用日志器。

    同时输出到控制台和文件，文件日志自动轮转。
    """
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL.upper())

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---- 控制台 Handler ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings.LOG_LEVEL.upper())
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ---- 文件 Handler ----
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / Path(settings.LOG_FILE).name

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_path),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(settings.LOG_LEVEL.upper())
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()
