"""日志配置模块。

提供统一的日志记录功能，支持控制台和文件双输出，
日志级别和文件路径通过配置中心管理。
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

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


# ==================== LangChain LLM 调用日志 ====================


class LLMCallbackHandler(BaseCallbackHandler):
    """LangChain 回调处理器，记录每次 LLM 调用的输入和输出。"""

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        """LLM 开始调用时触发（非流式 prompt 场景）。"""
        logger.info("=" * 60)
        logger.info(">>> LLM 请求 [非流式]")
        for i, p in enumerate(prompts):
            logger.info("--- Prompt[%d] ---\n%s", i, p)

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs) -> None:
        """聊天模型开始调用时触发。"""
        logger.info("=" * 60)
        logger.info(">>> LLM 请求 [聊天模式]")
        for i, msg_list in enumerate(messages):
            for j, msg in enumerate(msg_list):
                role = msg.type or "unknown"
                content = msg.content
                # 截断过长内容防止刷屏
                if len(str(content)) > 2000:
                    content = str(content)[:2000] + "\n... (truncated)"
                logger.info("--- Message[%d][%s] ---\n%s", i, role, content)

    def on_llm_end(self, response, **kwargs) -> None:
        """LLM 调用完成时触发。"""
        try:
            # 处理不同响应格式
            if hasattr(response, "generations"):
                for i, gen_list in enumerate(response.generations):
                    for j, gen in enumerate(gen_list):
                        text = gen.text if hasattr(gen, "text") else str(gen)
                        if len(text) > 2000:
                            text = text[:2000] + "\n... (truncated)"
                        logger.info("--- Response[%d][%d] ---\n%s", i, j, text)
            elif hasattr(response, "message"):
                content = response.message.content
                if len(content) > 2000:
                    content = content[:2000] + "\n... (truncated)"
                logger.info("--- Response ---\n%s", content)
            else:
                logger.info("--- Response ---\n%s", str(response)[:2000])
        except Exception as e:
            logger.debug("LLM 响应日志解析失败: %s", e)
        logger.info("=" * 60)

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """LLM 调用出错时触发。"""
        logger.error("!!! LLM 错误: %s", error)
