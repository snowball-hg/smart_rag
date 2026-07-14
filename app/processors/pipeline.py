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
from app.processors import data_cleaner
from app.processors.file_router import detect_file_type


class DocumentProcessor:
    """文档预处理管线，编排完整的接入→解析(HybridChunker)→清洗→元数据注入流程。

    工作流程：
    ```
    文件接入 → 文件类型检测
              ├── PDF → Docling 版面分析 + HybridChunker 切片
              │        ├── 有文本层 → 纯版面分析
              │        └── 扫描件   → RapidOCR + 版面分析
              ├── Office/HTML    → Unstructured 解析
              └── TXT/MD        → 直接读取
    清洗     → 去噪 → 去重 → 术语标准化
    元数据   → 注入 doc_id / heading_path / category 等
    ```
    """

    def __init__(self):
        self._docling_available = False

        self._check_dependencies()

    def _check_dependencies(self):
        """检查可选依赖的可用性。"""
        try:
            import docling  # noqa
            self._docling_available = True
        except ImportError:
            logger.info("Docling 未安装，无法处理文档")

    def process(
        self,
        file_path: str | Path,
        category: Optional[str] = None,
        enable_ocr: bool = True,
        enable_cleaning: bool = True,
        enable_dedup: bool = True,
    ) -> list[Document]:
        """完整文档预处理管线。

        Args:
            file_path: 文件路径。
            category: 文档分类（regulation/safety/manual/report/general）。
            enable_ocr: 是否启用 OCR（对扫描件有效）。
            enable_cleaning: 是否启用文本清洗。
            enable_dedup: 是否启用去重。

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

        # Step 2: 文档解析 + HybridChunker 切片
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

        # Step 4: 注入元数据
        final_docs = self._enrich_metadata(raw_docs, file_path, category)

        logger.info(
            "管线处理完成: %s → %d 块 (cleaning=%s, dedup=%s)",
            file_path.name,
            len(final_docs),
            enable_cleaning,
            enable_dedup,
        )
        return final_docs

    def _parse_document(
        self,
        file_path: Path,
        file_info: dict,
        enable_ocr: bool,
    ) -> list[Document]:
        """统一使用 Docling 解析所有文档类型，返回 HybridChunker 切片结果。"""
        if not self._docling_available:
            raise RuntimeError("Docling 未安装，无法处理文档")

        # 图片和 PDF 可能需要 OCR
        use_ocr = enable_ocr and file_info.get("is_scanned", False)
        from app.processors.pdf_processor import process_with_docling
        docs = process_with_docling(file_path, use_ocr=use_ocr)
        if not docs:
            raise RuntimeError(f"Docling 解析结果为空: {file_path.name}")

        for d in docs:
            d.metadata["is_scanned"] = file_info.get("is_scanned", False)

        return docs

    def _parse_pdf(
        self,
        file_path: Path,
        file_info: dict,
        enable_ocr: bool,
    ) -> list[Document]:
        """PDF 解析策略选择。

        所有 PDF 统一走 Docling 管线，扫描件自动启用 RapidOCR 后端。
        """
        if not self._docling_available:
            raise RuntimeError("Docling 未安装，无法处理 PDF")

        try:
            from app.processors.pdf_processor import process_with_docling
            use_ocr = enable_ocr and file_info["is_scanned"]
            docs = process_with_docling(file_path, use_ocr=use_ocr)
            if docs:
                for d in docs:
                    d.metadata["is_scanned"] = file_info["is_scanned"]
                return docs
        except Exception as e:
            logger.error("Docling 解析失败: %s", e)
            raise

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
