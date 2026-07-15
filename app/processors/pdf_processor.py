"""PDF 高级处理器模块。

基于 Docling 实现版面分析（Layout Analysis）、标题层级识别、
表格/图片区域检测、OCR（RapidOCR 后端，中文+跨平台 ONNX Runtime），
输出结构化的 Markdown 内容。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.config import settings
from app.logger import logger

# 模块级 Docling converter 单例（避免每次调用重新加载模型权重）
# 分两份缓存：有 OCR / 无 OCR
_docling_converter: dict[bool, object] = {}


def _get_docling_converter(use_ocr: bool = False):
    """获取或初始化 Docling DocumentConverter 实例（单例）。

    按 use_ocr 分别缓存，避免重复初始化和模型加载。
    RapidOCR 后端支持中文（简/繁）+ 跨平台 ONNX Runtime 推理。
    """
    if use_ocr not in _docling_converter:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            RapidOcrOptions,
        )

        opts = PdfPipelineOptions(
            do_ocr=use_ocr,
            do_table_structure=True,
        )
        if use_ocr:
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
    """使用 Docling 解析 PDF，通过 HybridChunker 进行结构+Token 感知切片。

    Docling 能够自动识别：
    - 标题层级（Heading 1/2/3...）
    - 段落（Paragraph）
    - 列表（List）
    - 表格（Table）— 保留行列结构
    - 图片区域
    - 页眉页脚

    HybridChunker 在文档层级结构基础上叠加 tokenizer 感知：
    - 超大章节按 token 上限自动细分
    - 过小段落自动合并到相邻块
    - 输出大小稳定的 chunk，对齐 embedding 模型

    Args:
        file_path: PDF 文件路径。
        use_ocr: 是否启用 OCR（对扫描件有效）。
        max_pages: 最大处理页数，None 表示全部。

    Returns:
        按文档层级结构切片后的 Document 列表。
    """
    file_path = Path(file_path)
    try:
        converter = _get_docling_converter(use_ocr=use_ocr)
        result = converter.convert(str(file_path))

        docling_doc = result.document

        # 使用 HybridChunker 进行结构+Token 感知切片
        from docling.chunking import HybridChunker

        chunker = HybridChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            merge_peers=True,
            use_hierarchy=True,
        )
        chunks = list(chunker.chunk(docling_doc))

        docs = []
        for chunk in chunks:
            heading_path = " > ".join(chunk.meta.headings) if chunk.meta.headings else ""

            # 从 doc_items 的 prov 中提取页码（替代已移除的 page_numbers 属性）
            page_nums = sorted(set(
                prov.page_no
                for doc_item in chunk.meta.doc_items
                for prov in doc_item.prov
            ))

            metadata: dict = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": ".pdf",
                "page_numbers": page_nums,
                "heading_path": heading_path,
            }

            doc = Document(page_content=chunk.text, metadata=metadata)
            docs.append(doc)

        logger.info(
            "Docling 解析完成: %s (%d HybridChunks, OCR=%s)",
            file_path.name, len(docs), use_ocr,
        )
        return docs

    except ImportError:
        logger.warning("docling 未安装，无法使用版面分析")
        raise
    except Exception as e:
        logger.error("Docling 解析失败 [%s]: %s", file_path.name, e)
        raise
