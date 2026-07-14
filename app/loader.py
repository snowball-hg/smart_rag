"""文档处理入口模块。

统一使用 DocumentProcessor 管线处理文档，
不再保留降级到基础加载器的路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.config import settings
from app.logger import logger


def process_file(
    file_path: str | Path,
    category: Optional[str] = None,
    enable_ocr: Optional[bool] = None,
    enable_cleaning: Optional[bool] = None,
) -> list[Document]:
    """使用 DocumentProcessor 管线处理文档。

    Args:
        file_path: 上传文件路径。
        category: 文档分类。
        enable_ocr: 是否启用 OCR。
        enable_cleaning: 是否启用文本清洗。

    Returns:
        处理后的文档块列表。

    Raises:
        RuntimeError: 管线处理失败时抛出。
    """
    from app.processors.pipeline import document_processor

    docs = document_processor.process(
        file_path=Path(file_path),
        category=category,
        enable_ocr=enable_ocr if enable_ocr is not None else settings.PROCESSOR_ENABLE_OCR,
        enable_cleaning=enable_cleaning if enable_cleaning is not None else settings.PROCESSOR_ENABLE_CLEANING,
    )

    if not docs:
        raise RuntimeError(f"文档处理失败: {file_path.name}")

    logger.info("文档处理完成: %s → %d 块", file_path.name, len(docs))
    return docs
