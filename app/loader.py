"""文档加载与文本分块模块。

支持 PDF、TXT、Markdown 等常见格式的文档解析。
通过 DocumentProcessor 管线提供高级功能（OCR、版面分析、智能切分等）。

两种使用方式：
1. 直接使用 process_file() — 保持向后兼容，自动使用启用的处理器
2. 通过 DocumentProcessor 使用 — 精细控制各环节
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.base import BaseLoader

from app.config import settings
from app.logger import logger


# 支持的文件类型及其对应的加载器（兜底使用）
LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".mdx": UnstructuredMarkdownLoader,
}


def get_loader(file_path: str | Path) -> Optional[BaseLoader]:
    """根据文件扩展名返回对应的文档加载器（兜底用）。"""
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADER_MAP.get(ext)

    if loader_cls is None:
        logger.warning("不支持的文件类型: %s", ext)
        return None

    try:
        if loader_cls == TextLoader:
            return TextLoader(str(file_path), encoding="utf-8")
        return loader_cls(str(file_path))
    except Exception as e:
        logger.error("创建加载器失败 [%s]: %s", file_path, e)
        return None


def load_document(file_path: str | Path) -> list[Document]:
    """加载单个文档并返回 Document 列表（兜底用）。"""
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("文件不存在: %s", file_path)
        return []

    loader = get_loader(file_path)
    if loader is None:
        return []

    try:
        docs = loader.load()
        logger.info("成功加载文档: %s (%d 页)", file_path.name, len(docs))
        return docs
    except Exception as e:
        logger.error("加载文档失败 [%s]: %s", file_path.name, e)
        return []


def split_documents(docs: list[Document]) -> list[Document]:
    """使用 RecursiveCharacterTextSplitter 对文档进行分块（兜底用）。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(docs)
    logger.info(
        "文档分块完成: %d 块 (size=%d, overlap=%d)",
        len(chunks),
        settings.CHUNK_SIZE,
        settings.CHUNK_OVERLAP,
    )
    return chunks


def process_file(
    file_path: str | Path,
    category: Optional[str] = None,
    enable_ocr: Optional[bool] = None,
    enable_cleaning: Optional[bool] = None,
    enable_smart_chunk: Optional[bool] = None,
) -> list[Document]:
    """完整的文档处理入口（兼容原接口，自动使用高级管线）。

    流程：
    1. 尝试使用 DocumentProcessor 管线（启用高级功能）
    2. 如果依赖缺失，自动降级到基础流程
    3. 为每个块添加统一元数据

    Args:
        file_path: 上传文件路径。
        category: 文档分类（regulation/safety/manual/report/general）。
        enable_ocr: 是否启用 OCR 识别扫描件。
        enable_cleaning: 是否启用文本清洗。
        enable_smart_chunk: 是否启用智能切分。

    Returns:
        处理后的文档块列表，带有完整元数据。
    """
    file_path = Path(file_path)

    # ----- 尝试使用高级管线 -----
    try:
        from app.processors.pipeline import document_processor

        docs = document_processor.process(
            file_path=file_path,
            category=category,
            enable_ocr=enable_ocr if enable_ocr is not None else settings.PROCESSOR_ENABLE_OCR,
            enable_cleaning=enable_cleaning if enable_cleaning is not None else settings.PROCESSOR_ENABLE_CLEANING,
            enable_smart_chunk=enable_smart_chunk if enable_smart_chunk is not None else settings.PROCESSOR_ENABLE_SMART_CHUNK,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        if docs:
            return docs

    except ImportError as e:
        logger.info("高级处理器依赖缺失 (%s)，降级到基础流程", e)
    except Exception as e:
        logger.warning("高级管线处理失败 (%s)，降级到基础流程", e)

    # ----- 降级：使用原始基础流程 -----
    logger.info("使用基础流程处理: %s", file_path.name)
    raw_docs = load_document(file_path)
    if not raw_docs:
        return []

    chunks = split_documents(raw_docs)

    doc_id = _generate_doc_id(file_path)
    upload_time = datetime.now(timezone.utc).isoformat()

    for idx, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "doc_id": doc_id,
                "doc_name": file_path.name,
                "chunk_index": idx,
                "upload_time": upload_time,
                "chunk_id": f"{doc_id}_{idx}",
            }
        )
        chunk.metadata.setdefault("author", "")
        chunk.metadata.setdefault("source", str(file_path))
        if category:
            chunk.metadata["category"] = category

    logger.info(
        "基础流程完成: %s (doc_id=%s, chunks=%d)",
        file_path.name, doc_id, len(chunks),
    )
    return chunks


def _generate_doc_id(file_path: Path) -> str:
    """基于文件路径和修改时间生成文档唯一标识。"""
    stat = file_path.stat()
    content = f"{file_path.absolute()}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(content.encode()).hexdigest()[:12]
