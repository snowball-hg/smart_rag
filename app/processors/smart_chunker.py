"""智能文档切分模块。

支持语义感知切分（按标题/段落/列表结构）和动态粒度调整，
替代原始的固定长度一刀切策略。Docling 产出的 Markdown 格式文本
可直接按标题层级进行结构化切分。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.logger import logger


# 按文档类型的推荐切分配置
CHUNK_CONFIGS: dict[str, dict] = {
    # 规程类（运行规程、操作规程、检修规程）— 条款式，粒度细
    "regulation": {
        "chunk_size": 400,
        "chunk_overlap": 80,
        "description": "规程类（细粒度，确保每条款独立）",
    },
    # 安全规范 — 条款式，中等粒度
    "safety": {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "description": "安全规范（中等粒度，保留条款上下文）",
    },
    # 设备说明书 — 段落式，中等偏粗
    "manual": {
        "chunk_size": 700,
        "chunk_overlap": 150,
        "description": "设备说明书（段落式，保留功能描述完整性）",
    },
    # 报告/新闻 — 叙事式，粗粒度
    "report": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "description": "报告/新闻（叙事式，保留完整叙事）",
    },
    # 通用/默认
    "general": {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "description": "通用文档",
    },
}


def classify_document(
    doc_name: str,
    content: str,
    category: Optional[str] = None,
) -> str:
    """根据文档名称和内容特征判断文档类型。

    Args:
        doc_name: 文档名称。
        content: 文档内容预览（前 500 字符）。
        category: 上传时指定的分类标签。

    Returns:
        文档类型标识（regulation / safety / manual / report / general）。
    """
    if category and category in CHUNK_CONFIGS:
        return category

    name_lower = doc_name.lower()
    content_lower = content[:500].lower()

    # 关键词匹配
    regulations_kw = ["规程", "规定", "规则", "条例", "规章", "制度"]
    safety_kw = ["安全", "消防", "危险", "防护", "应急", "事故"]
    manual_kw = ["说明书", "手册", "指南", "介绍", "参数", "规格", "型号"]
    report_kw = ["报告", "汇报", "总结", "分析", "评估", "通告"]

    # 文档名优先
    for kw in regulations_kw:
        if kw in name_lower:
            return "regulation"
    for kw in safety_kw:
        if kw in name_lower:
            return "safety"
    for kw in manual_kw:
        if kw in name_lower:
            return "manual"
    for kw in report_kw:
        if kw in name_lower:
            return "report"

    # 内容兜底
    score = {"regulation": 0, "safety": 0, "manual": 0, "report": 0}
    for kw in regulations_kw:
        if kw in content_lower:
            score["regulation"] += 1
    for kw in safety_kw:
        if kw in content_lower:
            score["safety"] += 1
    for kw in manual_kw:
        if kw in content_lower:
            score["manual"] += 1
    for kw in report_kw:
        if kw in content_lower:
            score["report"] += 1

    best = max(score, key=score.get)
    if score[best] > 0:
        return best

    return "general"


def _merge_tables_with_context(docs: list[Document]) -> list[Document]:
    """将表格 Document 合并到前面的文本 Document 中。

    Docling 产出的表格是独立元素，合并到前一个非表格文段，
    保持"描述文字 + 表格"的语义完整性。

    Args:
        docs: 输入文档列表（保持原始页序）。

    Returns:
        表格已合并的文档列表。
    """
    if not docs:
        return []

    merged = []
    for doc in docs:
        element_type = doc.metadata.get("element_type", "")
        if element_type == "table" and merged:
            merged[-1].page_content += "\n\n" + doc.page_content
        else:
            merged.append(doc)

    logger.info("表格合并: %d → %d 块", len(docs), len(merged))
    return merged


def chunk_by_markdown_headings(markdown_text: str) -> list[tuple[str, str, int]]:
    """按 Markdown 标题层级切分。

    将 ## / ### 等标题作为分块边界，
    确保标题和正文绑定在一起。

    Args:
        markdown_text: Markdown 格式文本。

    Returns:
        (标题路径, 块文本, 标题层级) 列表。
    """
    if not markdown_text:
        return []

    lines = markdown_text.split("\n")
    chunks: list[tuple[str, str, int]] = []

    # 提取文档第一个 H1 作为文档标题
    doc_title = ""
    for line in lines:
        if line.startswith("# ") or line.startswith("#\t"):
            doc_title = line.lstrip("# ").strip()
            break

    # 按标题行拆分
    heading_paths: list[str] = []
    current_chunk: list[str] = []
    current_level = 0

    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            # 保存当前块
            if current_chunk:
                text = "\n".join(current_chunk).strip()
                if text:
                    path_str = " > ".join(p for p in heading_paths if p)
                    chunks.append((path_str, text, current_level))
                current_chunk = []

            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            # 更新标题路径
            while len(heading_paths) >= level:
                heading_paths.pop()
            heading_paths.append(title)

            current_level = level
            current_chunk.append(line)
        else:
            current_chunk.append(line)

    # 最后一块
    if current_chunk:
        text = "\n".join(current_chunk).strip()
        if text:
            path_str = " > ".join(p for p in heading_paths if p)
            chunks.append((path_str, text, current_level))

    return chunks


def _merge_small_chunks(
    docs: list[Document], min_chunk_size: int = 100
) -> list[Document]:
    """将过小的文档块合并到前一块中，减少碎片。

    Args:
        docs: 输入文档块列表。
        min_chunk_size: 最小块大小，小于此值的块将被合并。

    Returns:
        合并后的文档块列表。
    """
    if not docs:
        return []

    merged = []
    for doc in docs:
        if merged and len(doc.page_content) < min_chunk_size:
            # 合并到前一块
            merged[-1].page_content += "\n" + doc.page_content
            # 合并元数据中的 heading_path
            old_heading = doc.metadata.get("heading_path", "")
            if old_heading and old_heading not in merged[-1].metadata.get("heading_path", ""):
                merged[-1].metadata["heading_path"] = (
                    merged[-1].metadata.get("heading_path", "") + " > " + old_heading
                ).strip(" > ")
        else:
            merged.append(doc)

    if len(merged) != len(docs):
        logger.info("小块合并: %d → %d 块 (min_size=%d)", len(docs), len(merged), min_chunk_size)

    return merged


def chunk_documents(
    docs: list[Document],
    doc_type: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Document]:
    """智能文档切分主入口。

    策略：
    1. 如果文档包含 Markdown 标题，按标题结构切分
    2. 否则按文档类型选择粒度，用 RecursiveCharacterTextSplitter 切分
    3. 保留上下文重叠
    4. 合并过小的碎片块

    Args:
        docs: 输入文档列表。
        doc_type: 文档类型（regulation/safety/manual/report/general）。
        chunk_size: 自定义块大小（覆盖自动判定）。
        chunk_overlap: 自定义重叠大小。

    Returns:
        切分后的文档块列表。
    """
    if not docs:
        return []

    # 将表格与前面的描述文本合并
    docs = _merge_tables_with_context(docs)

    # 合并所有文档为一个文本块（保留原始元数据）
    merged_text = []
    source_names = set()
    for doc in docs:
        merged_text.append(doc.page_content)
        if "file_name" in doc.metadata:
            source_names.add(doc.metadata["file_name"])

    full_text = "\n\n".join(merged_text)

    # 判断文档类型
    sample_name = next(iter(source_names), "unknown") if source_names else "unknown"

    if doc_type and doc_type in CHUNK_CONFIGS:
        detected_type = doc_type
    else:
        detected_type = classify_document(sample_name, full_text)

    config = CHUNK_CONFIGS.get(detected_type, CHUNK_CONFIGS["general"])
    final_chunk_size = chunk_size or config["chunk_size"]
    final_chunk_overlap = chunk_overlap or config["chunk_overlap"]

    logger.info(
        "智能切分: type=%s, size=%d, overlap=%d (%s)",
        detected_type,
        final_chunk_size,
        final_chunk_overlap,
        config["description"],
    )

    # 策略 1：尝试按 Markdown 标题切分
    heading_chunks = chunk_by_markdown_headings(full_text)

    if len(heading_chunks) > 3:
        # 标题切分成功
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=final_chunk_size,
            chunk_overlap=final_chunk_overlap,
            separators=["\n\n", "。", "！", "？", "；", "\n", "，", " ", ""],
            length_function=len,
        )

        result_docs = []
        for heading_path, section_text, heading_level in heading_chunks:
            if len(section_text) > final_chunk_size * 1.5:
                # 长章节内部再切分
                sub_chunks = splitter.split_text(section_text)
                for i, sub_text in enumerate(sub_chunks):
                    doc = Document(
                        page_content=sub_text,
                        metadata={
                            "heading_path": heading_path,
                            "heading_level": heading_level,
                            "chunk_sub_index": i,
                        },
                    )
                    result_docs.append(doc)
            else:
                doc = Document(
                    page_content=section_text,
                    metadata={
                        "heading_path": heading_path,
                        "heading_level": heading_level,
                    },
                )
                result_docs.append(doc)

        logger.info("标题感知切分完成: %d 个章节 → %d 块", len(heading_chunks), len(result_docs))
        return _merge_small_chunks(result_docs)

    # 策略 2：按递归字符切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=final_chunk_size,
        chunk_overlap=final_chunk_overlap,
        separators=["\n\n", "。", "！", "？", "；", "\n", "，", " ", ""],
        length_function=len,
    )

    result_docs = splitter.split_documents(docs)
    result_docs = _merge_small_chunks(result_docs)
    logger.info(
        "递归切分完成: %d 块 (type=%s, size=%d)",
        len(result_docs),
        detected_type,
        final_chunk_size,
    )

    return result_docs
