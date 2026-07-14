"""文件类型检测模块。

提供文件类型分类和扫描件检测功能。
"""

from __future__ import annotations

from pathlib import Path

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
        logger.warning("PyPDF2 未安装，默认按非扫描件处理")
        return False
    except Exception as e:
        logger.warning("检测 PDF 文本层失败: %s，按非扫描件处理", e)
        return False


def detect_file_type(file_path: str | Path) -> dict:
    """检测文件类型和基本属性。

    Args:
        file_path: 文件路径。

    Returns:
        包含文件类型信息的字典（ext, category, is_scanned 等）。
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

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
