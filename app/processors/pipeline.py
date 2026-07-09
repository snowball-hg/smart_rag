"""文档预处理管线编排器。

整合所有处理器模块，提供统一的 DocumentProcessor 入口。
支持按配置启停各环节，向后兼容现有 loader.py。
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.config import settings
from app.logger import logger
from app.processors import data_cleaner, smart_chunker
from app.processors.file_router import detect_file_type


class DocumentProcessor:
    """文档预处理管线，编排完整的接入→分析→清洗→切分流程。

    工作流程：
    ```
    文件接入 → 文件类型检测
              ├── PDF → Docling 版面分析
              │        ├── 有文本层 → 纯版面分析
              │        └── 扫描件   → EasyOCR + 版面分析
              ├── Office/HTML    → Unstructured 解析
              └── TXT/MD        → 直接读取
    清洗     → 去噪 → 去重 → 术语标准化
    切分     → 按标题结构 / 动态粒度
    元数据   → 注入 doc_id / heading_path / category 等
    ```
    """

    def __init__(self):
        self._docling_available = False
        self._unstructured_available = False

        self._check_dependencies()

    def _check_dependencies(self):
        """检查可选依赖的可用性。"""
        try:
            import docling  # noqa
            self._docling_available = True
        except ImportError:
            logger.info("Docling 未安装，PDF 将使用基础 PyPDFLoader")

        try:
            import unstructured  # noqa
            self._unstructured_available = True
        except ImportError:
            logger.info("Unstructured 未安装，仅支持 PDF/TXT/MD 格式")

    def process(
        self,
        file_path: str | Path,
        category: Optional[str] = None,
        enable_ocr: bool = True,
        enable_cleaning: bool = True,
        enable_dedup: bool = True,
        enable_smart_chunk: bool = True,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> list[Document]:
        """完整文档预处理管线。

        Args:
            file_path: 文件路径。
            category: 文档分类（regulation/safety/manual/report/general）。
            enable_ocr: 是否启用 OCR（对扫描件有效）。
            enable_cleaning: 是否启用文本清洗。
            enable_dedup: 是否启用去重。
            enable_smart_chunk: 是否启用智能切分。
            chunk_size: 自定义块大小。
            chunk_overlap: 自定义重叠大小。

        Returns:
            处理后的 Document 列表。
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error("文件不存在: %s", file_path)
            return []

        # Step 1: 文件类型检测
        file_info = detect_file_type(file_path)
        logger.info(
            "处理文件: %s (type=%s, scanned=%s, size=%dKB)",
            file_path.name,
            file_info["category"],
            file_info["is_scanned"],
            file_info["file_size"] // 1024,
        )

        # Step 2: 文档解析
        raw_docs = self._parse_document(file_path, file_info, enable_ocr)
        if not raw_docs:
            logger.warning("文档解析结果为空: %s", file_path.name)
            return []

        # Step 3: 清洗
        # Docling 产出结构化的 Markdown 文本，无需 OCR 纠错或断行合并
        if enable_cleaning:
            raw_docs = data_cleaner.clean_document_chunks(
                raw_docs,
                merge_broken_lines=False,
                is_ocr=False,
            )
        else:
            logger.info("文本清洗已禁用")

        # Step 4: 智能切分
        if enable_smart_chunk:
            raw_docs = smart_chunker.chunk_documents(
                raw_docs,
                doc_type=category,
                chunk_size=chunk_size or settings.CHUNK_SIZE,
                chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
            )
        else:
            # 基础切分
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size or settings.CHUNK_SIZE,
                chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
                separators=["\n\n", "。", "！", "？", "；", "\n", "，", " ", ""],
                length_function=len,
            )
            raw_docs = splitter.split_documents(raw_docs)

        # Step 5: 注入元数据
        final_docs = self._enrich_metadata(raw_docs, file_path, category)

        logger.info(
            "管线处理完成: %s → %d 块 (cleaning=%s, smart_chunk=%s, dedup=%s)",
            file_path.name,
            len(final_docs),
            enable_cleaning,
            enable_smart_chunk,
            enable_dedup,
        )
        return final_docs

    def _parse_document(
        self,
        file_path: Path,
        file_info: dict,
        enable_ocr: bool,
    ) -> list[Document]:
        """根据文件类型选择解析策略。"""
        category = file_info["category"]
        ext = file_info["ext"]

        # PDF：首选 Docling，扫描件用 OCR
        if category == "pdf":
            return self._parse_pdf(file_path, file_info, enable_ocr)

        # 非 PDF 多格式：用 Unstructured
        if self._unstructured_available and ext not in {".txt", ".md", ".mdx"}:
            try:
                from app.processors.file_router import load_with_unstructured
                return load_with_unstructured(file_path)
            except Exception:
                logger.warning("Unstructured 解析失败，回退到默认加载器")

        # 兜底：使用 loader.py 的原始加载方式
        return self._fallback_load(file_path)

    def _parse_pdf(
        self,
        file_path: Path,
        file_info: dict,
        enable_ocr: bool,
    ) -> list[Document]:
        """PDF 解析策略选择。

        所有 PDF 统一走 Docling 管线，扫描件自动启用 EasyOCR 后端。
        """
        if self._docling_available:
            try:
                from app.processors.pdf_processor import process_with_docling

                # 扫描件 → 启用 OCR（EasyOCR 中文+GPU）；有文本层 → 纯版面分析
                use_ocr = enable_ocr and file_info["is_scanned"]
                docs = process_with_docling(file_path, use_ocr=use_ocr)
                if docs:
                    # 标记是否为扫描件（供下游清洗判断）
                    for d in docs:
                        if not d.metadata.get("is_full_text"):
                            d.metadata["is_scanned"] = file_info["is_scanned"]
                    # 合并 full_markdown 和元素级文档
                    element_docs = [d for d in docs if not d.metadata.get("is_full_text")]
                    return element_docs or docs
            except Exception as e:
                logger.warning("Docling 解析失败: %s，回退到基础加载", e)

        # 兜底：基础 PyPDFLoader
        return self._fallback_load(file_path)

    def _fallback_load(self, file_path: Path) -> list[Document]:
        """使用 loader.py 的原始加载方式作为兜底。"""
        from app.loader import load_document
        return load_document(file_path)

    def _enrich_metadata(
        self,
        docs: list[Document],
        file_path: Path,
        category: Optional[str] = None,
    ) -> list[Document]:
        """为文档块注入标准化元数据。

        包括 doc_id、heading_path、category、chunk_index 等。
        """
        doc_id = self._generate_doc_id(file_path)
        upload_time = datetime.now(timezone.utc).isoformat()

        for idx, doc in enumerate(docs):
            doc.metadata.update({
                "doc_id": doc_id,
                "doc_name": file_path.name,
                "chunk_index": idx,
                "chunk_id": f"{doc_id}_{idx}",
                "upload_time": upload_time,
                "source": str(file_path),
            })

            if category:
                doc.metadata["category"] = category

            # 确保 Milvus schema 必需的字段存在
            doc.metadata.setdefault("author", "")

        return docs

    def _generate_doc_id(self, file_path: Path) -> str:
        """基于文件路径和修改时间生成唯一文档标识。"""
        stat = file_path.stat()
        content = f"{file_path.absolute()}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


# 全局单例
document_processor = DocumentProcessor()
