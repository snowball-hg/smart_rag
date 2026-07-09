"""PDF 高级处理器模块。

基于 Docling 实现版面分析（Layout Analysis）、标题层级识别、
表格/图片区域检测、OCR（EasyOCR 后端，中文+GPU），
输出结构化的 Markdown 内容。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.logger import logger

# 模块级 Docling converter 单例（避免每次调用重新加载模型权重）
# 分两份缓存：有 OCR / 无 OCR
_docling_converter: dict[bool, object] = {}

# EasyOCR 模型下载代理（下载完成后可移除）
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")


def _get_docling_converter(use_ocr: bool = False):
    """获取或初始化 Docling DocumentConverter 实例（单例）。

    按 use_ocr 分别缓存，避免重复初始化和模型加载。
    EasyOCR 后端支持中文（简/繁）+ GPU 自动检测。
    """
    global _docling_converter
    if use_ocr not in _docling_converter:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            RapidOcrOptions,
        )

        opts = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
        )
        # 配置 RapidOCR：中文 + onnxruntime 后端（跨平台，无需 CUDA）
        opts.ocr_options = RapidOcrOptions(
            lang=["chinese"],
            backend="onnxruntime",
        )
        _docling_converter[use_ocr] = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
            }
        )
    return _docling_converter[use_ocr]


def process_with_docling(
    file_path: str | Path,
    use_ocr: bool = False,
    max_pages: Optional[int] = None,
) -> list[Document]:
    """使用 Docling 解析 PDF，保留版面结构和语义信息。

    Docling 能够自动识别：
    - 标题层级（Heading 1/2/3...）
    - 段落（Paragraph）
    - 列表（List）
    - 表格（Table）— 保留行列结构
    - 图片区域
    - 页眉页脚

    Args:
        file_path: PDF 文件路径。
        use_ocr: 是否启用 OCR（对扫描件有效）。
        max_pages: 最大处理页数，None 表示全部。

    Returns:
        按版面元素划分的 Document 列表。
    """
    file_path = Path(file_path)
    try:
        from docling.datamodel.base_models import InputFormat

        converter = _get_docling_converter(use_ocr=use_ocr)
        result = converter.convert(str(file_path))

        docling_doc = result.document

        # 导出为 Markdown 格式（保留标题层级、表格等结构）
        md_content = docling_doc.export_to_markdown()

        # 按 Docling 的 layout 分块
        docs = []
        for i, (item, level) in enumerate(docling_doc.iterate_items()):
            text = getattr(item, "text", None) or getattr(item, "orig", "") or ""
            text = text.strip()
            if not text:
                continue

            # 判断元素类型
            element_type = _get_element_type(item, level)

            metadata = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": ".pdf",
                "page": getattr(item, "page_no", None) or (i // 10),
                "element_type": element_type,
                "heading_level": level if level > 0 else None,
            }

            doc = Document(page_content=text, metadata=metadata)
            docs.append(doc)

        # 同时保留整篇 Markdown（用于上下文感知切分）
        md_doc = Document(
            page_content=md_content,
            metadata={
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": ".pdf",
                "element_type": "full_markdown",
                "is_full_text": True,
            },
        )

        logger.info(
            "Docling 解析完成: %s (%d 元素, OCR=%s)",
            file_path.name, len(docs), use_ocr,
        )
        return [md_doc] + docs

    except ImportError:
        logger.warning("docling 未安装，无法使用版面分析")
        raise
    except Exception as e:
        logger.error("Docling 解析失败 [%s]: %s", file_path.name, e)
        raise


def _get_element_type(item, level: int) -> str:
    """推断 Docling 元素的类型。"""
    if level == 1:
        return "heading"
    elif level >= 2:
        return "sub_heading"
    # 根据 item 文本特征判断
    text = (item.text or item.orig or "").strip()
    if text.startswith("|") and text.endswith("|"):
        return "table"
    if any(text.startswith(p) for p in ["- ", "* ", "+ "]):
        return "list_item"
    if len(text) < 30 and text.isupper():
        return "heading"
    return "paragraph"
