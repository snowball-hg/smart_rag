"""文件接入路由模块。

基于 Unstructured 支持多格式文件接入，作为后备方案保留原有加载器。
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.logger import logger


def is_likely_scanned_pdf(file_path: str | Path) -> bool:
    """快速判断 PDF 是否为扫描件（无文本层）。

    读取 PDF 前几 KB，如果没有可提取的文本内容则判定为扫描件。
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() != ".pdf":
        return False
    try:
        import PyPDF2
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages[:3]:
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    return False
            return True
    except ImportError:
        logger.warning("PyPDF2 未安装，无法检测是否为扫描件，默认按非扫描件处理")
        return False
    except Exception as e:
        logger.warning("检测 PDF 文本层失败: %s，按非扫描件处理", e)
        return False


def detect_file_type(file_path: str | Path) -> dict:
    """检测文件类型和基本属性。

    Args:
        file_path: 文件路径。

    Returns:
        包含文件类型信息的字典（ext, mime_type, category 等）。
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    name = file_path.name.lower()

    # 按扩展名分类
    category_map = {
        ".pdf": "pdf",
        ".txt": "text",
        ".md": "markdown",
        ".mdx": "markdown",
        ".doc": "word",
        ".docx": "word",
        ".ppt": "presentation",
        ".pptx": "presentation",
        ".xls": "excel",
        ".xlsx": "excel",
        ".html": "html",
        ".htm": "html",
        ".csv": "csv",
        ".json": "json",
        ".xml": "xml",
        ".eml": "email",
        ".msg": "email",
    }

    category = category_map.get(ext, "unknown")
    is_scanned = False
    if category == "pdf":
        is_scanned = is_likely_scanned_pdf(file_path)

    return {
        "ext": ext,
        "category": category,
        "is_scanned": is_scanned,
        "file_name": file_path.name,
        "file_size": file_path.stat().st_size,
    }


def load_with_unstructured(
    file_path: str | Path,
    strategy: str = "auto",
) -> list[Document]:
    """使用 Unstructured 加载文档。

    Args:
        file_path: 文件路径。
        strategy: 解析策略，可选 "auto", "fast", "ocr_only", "hi_res"。

    Returns:
        Document 列表。
    """
    file_path = Path(file_path)
    try:
        from unstructured.partition.auto import partition

        elements = partition(
            filename=str(file_path),
            strategy=strategy,
            include_page_breaks=True,
        )

        docs = []
        for el in elements:
            text = str(el)
            if not text.strip():
                continue
            metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
            }
            # 提取结构化元数据
            if hasattr(el, "metadata"):
                if hasattr(el.metadata, "page_number"):
                    metadata["page"] = el.metadata.page_number
                if hasattr(el.metadata, "category"):
                    metadata["element_type"] = el.metadata.category
                if hasattr(el.metadata, "emphasized_text_contents"):
                    metadata["emphasized_texts"] = el.metadata.emphasized_text_contents

            doc = Document(page_content=text, metadata=metadata)
            docs.append(doc)

        logger.info(
            "Unstructured 解析完成: %s (%d 元素)",
            file_path.name, len(docs),
        )
        return docs

    except ImportError:
        logger.warning(
            "unstructured 未安装，回退到默认加载器"
        )
        raise
    except Exception as e:
        logger.error("Unstructured 解析失败 [%s]: %s", file_path.name, e)
        raise
