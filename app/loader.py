"""文档加载与文本分块模块。

支持 PDF、TXT、Markdown 等常见格式的文档解析，
并使用 RecursiveCharacterTextSplitter 进行智能分块。
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

# 支持的文件类型及其对应的加载器
LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".mdx": UnstructuredMarkdownLoader,
}


def get_loader(file_path: str | Path) -> Optional[BaseLoader]:
    """根据文件扩展名返回对应的文档加载器。

    Args:
        file_path: 文件路径。

    Returns:
        文档加载器实例，如果文件类型不支持则返回 None。
    """
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
    """加载单个文档并返回 Document 列表。

    Args:
        file_path: 文件路径。

    Returns:
        文档对象列表。如果加载失败则返回空列表。
    """
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
    """使用 RecursiveCharacterTextSplitter 对文档进行分块。

    Args:
        docs: 原始文档列表。

    Returns:
        分块后的文档列表。
    """
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


def process_file(file_path: str | Path) -> list[Document]:
    """完整的文档处理流程：加载 -> 分块 -> 添加元数据。

    整个流程：
    1. 根据文件扩展名选择合适的加载器
    2. 加载文档内容
    3. 将文档切分成小块
    4. 为每个块添加统一的元数据（doc_id, doc_name, chunk_index, upload_time 等）

    Args:
        file_path: 上传文件路径。

    Returns:
        处理后的文档块列表，带有完整元数据。
    """
    file_path = Path(file_path)
    raw_docs = load_document(file_path)
    if not raw_docs:
        return []

    chunks = split_documents(raw_docs)

    # 生成文档唯一标识（基于文件内容哈希）
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

        # 确保 metadata 中包含 Milvus schema 必需的字段
        chunk.metadata.setdefault("author", "")
        chunk.metadata.setdefault("source", str(file_path))

    logger.info(
        "文档处理完成: %s (doc_id=%s, chunks=%d)", file_path.name, doc_id, len(chunks)
    )
    return chunks


def _generate_doc_id(file_path: Path) -> str:
    """基于文件路径和修改时间生成文档唯一标识。"""
    stat = file_path.stat()
    content = f"{file_path.absolute()}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(content.encode()).hexdigest()[:12]
