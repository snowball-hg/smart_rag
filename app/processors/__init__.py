"""文档预处理管线包。

提供从文件接入 → 版面分析 → OCR → 表格提取 → 清洗 → 智能切分的完整预处理流程。
各组件可独立启用/禁用，通过 config 控制。
"""

from __future__ import annotations

from app.processors.pipeline import DocumentProcessor

__all__ = ["DocumentProcessor"]
