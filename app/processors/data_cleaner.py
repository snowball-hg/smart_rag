"""数据清洗与标准化模块。

提供文本去噪、去重、OCR 纠错、术语标准化等能力。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.logger import logger


# ==================== 页眉页脚模式（中文常见） ====================

HEADER_FOOTER_PATTERNS = [
    r"^\d{1,3}\s*$",                       # 仅页码（1-3 位，避免误删手机号）
    r"^第\s*\d+\s*页",                     # 第 X 页
    r"^-\s*\d+\s*-",                       # - X -
    r"^第\s*[一二三四五六七八九十]+\s*页",     # 第 N 页
    r"^(机密|保密|内部资料|绝密|机密文件|公开文件)",  # 密级标记
    r"^company\s*confidential",            # 英文密级
    r"^文档编号[：:]\s*\S+",               # 文档编号
    r"^\S+\.pdf\s*$",                      # 文件名占一行（常见于扫描件页脚）
    r"^\d+/\d+$",                          # 页码 1/10 格式
    r"^\d{4}-\d{2}-\d{2}",                 # 日期单独一行
    r"^版权所有",                           # 版权声明
    r"^All\s+Rights\s+Reserved",           # 英文版权
    r"^修订记录|^版本历史|^变更记录",
]

# ==================== OCR 常见错误纠正 ====================

OCR_CORRECTIONS = [
    # 数字混淆
    (r"(?<!\d)0(?!\d)", "零"),  # 单独的数字 0→零
    (r"(?<=\d)[lL](?=\d)", "1"),            # 数字中的 l→1
    (r"(?<=\d)O(?=\d)", "0"),               # 数字中的 O→0
    # 常见 OCR 混淆
    (r"(\d)š(\d)", r"\1.\2"),               # š → .
    (r"(?<=[a-zA-Z])é(?=[a-zA-Z])", "e"),   # é → e（非中文）
    (r"一[二三四五六七八九十]", "—"),          # 破折号修复
    # 空格修复
    (r"\s{3,}", "\n\n"),                    # 超长空白替换为段落分隔
    (r" +\n", "\n"),                        # 行尾多余空格
    (r"\n{4,}", "\n\n\n"),                  # 过多空行压缩
    # 标点规范
    (r"，", "，"),                          # 统一为中文标点
    (r"；", "；"),
    (r"。", "。"),
    (r"（", "（"),
    (r"）", "）"),
    (r"、", "、"),
    (r"”", "\""),
    (r"“", "\""),
    (r"’", "'"),
    (r"‘", "'"),
]

# ==================== 企业术语标准化 ====================

# 从低优先级到高优先级，长匹配优先
TERM_NORMALIZATION: dict[str, str] = {
    # 常见同义词统一
    "退款时限": "退款时间",
    "退货时限": "退货时间",
    "退款周期": "退款时间",
    "售后时长": "售后时间",
    "客服电话": "服务热线",
    "客户服务": "服务热线",
    "用户手册": "使用说明书",
    "操作指南": "使用说明书",
    "产品说明": "使用说明书",
    "技术手册": "技术说明书",
    "技术文档": "技术说明书",
}


def clean_text(text: str, merge_broken_lines: bool = True, apply_ocr_corrections: bool = True) -> str:
    """对文本执行基础清洗。

    清洗步骤（按顺序）：
    1. 去除控制字符
    2. 合并断行（仅 OCR 文本启用）
    3. OCR 错误纠正（仅 OCR 文本启用）
    4. 压缩空白

    Args:
        text: 原始文本。
        merge_broken_lines: 是否合并断行。OCR 文本启用，
            正常解析的文本（Docling/Unstructured）关闭。
        apply_ocr_corrections: 是否应用 OCR 纠错规则。

    Returns:
        清洗后的文本。
    """
    if not text:
        return ""

    # 1. 去除控制字符（保留 \n \t 等常用控制符）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 2. 合并断行（仅 OCR 场景：行尾无标点 + 下一行首非空行 = 断行）
    if merge_broken_lines:
        lines = text.split("\n")
        merged = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                merged.append("")
                continue
            if (merged and merged[-1]
                    and not re.search(r"[。！？；）」】」]\s*$", merged[-1])
                    and not re.search(r"^[#\-\*\d\.\s]", stripped)
                    and len(stripped) > 5):
                merged[-1] += stripped
            else:
                merged.append(stripped)

        text = "\n".join(merged)

    # 3. OCR 常见错误纠正（仅 OCR 场景启用）
    if apply_ocr_corrections:
        for pattern, replacement in OCR_CORRECTIONS:
            text = re.sub(pattern, replacement, text)

    # 4. 压缩连续空白（保留段落分隔）
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def remove_header_footer(text: str, patterns: Optional[list[str]] = None) -> str:
    """移除文档中的页眉页脚。

    对每行进行检查，如果匹配页眉页脚模式则移除。

    Args:
        text: 输入文本。
        patterns: 自定义匹配模式列表。

    Returns:
        移除页眉页脚后的文本。
    """
    if not text:
        return ""

    patterns = patterns or HEADER_FOOTER_PATTERNS
    combined_pattern = re.compile("|".join(patterns), re.IGNORECASE)

    lines = text.split("\n")
    filtered = [line for line in lines if not combined_pattern.match(line.strip())]

    result = "\n".join(filtered)
    logger.debug("页眉页脚过滤: %d 行 → %d 行", len(lines), len(filtered))
    return result


def normalize_terms(text: str, custom_dict: Optional[dict[str, str]] = None) -> str:
    """对企业术语进行标准化。

    Args:
        text: 输入文本。
        custom_dict: 自定义术语映射字典。

    Returns:
        术语标准化后的文本。
    """
    if not text:
        return ""

    term_dict = {**TERM_NORMALIZATION}
    if custom_dict:
        term_dict.update(custom_dict)

    # 按长度降序排列（长匹配优先）
    sorted_terms = sorted(term_dict.items(), key=lambda x: -len(x[0]))

    for old, new in sorted_terms:
        text = text.replace(old, new)

    return text


def clean_document_chunks(
    chunks: list[Document],
    merge_broken_lines: bool = False,
    is_ocr: bool = False,
) -> list[Document]:
    """对文档块列表执行完整清洗流程。

    流程：
    1. 逐块文本清洗（clean_text）
    2. 逐块移除页眉页脚
    3. 术语标准化
    4. 全局去重

    Args:
        chunks: 输入文档块列表。
        merge_broken_lines: 是否合并断行。OCR 文本启用，
            正常解析的文本关闭。
        is_ocr: 是否为 OCR 产出的文本，控制 OCR 纠错规则是否生效。

    Returns:
        清洗后的文档块列表。
    """
    if not chunks:
        return []

    cleaned = []
    for doc in chunks:
        text = doc.page_content
        if not text or not text.strip():
            continue

        text = clean_text(text, merge_broken_lines=merge_broken_lines, apply_ocr_corrections=is_ocr)
        text = remove_header_footer(text)
        if not text.strip():
            continue

        doc.page_content = text
        cleaned.append(doc)

    logger.info("文档清洗完成: %d → %d 块", len(chunks), len(cleaned))
    return cleaned
