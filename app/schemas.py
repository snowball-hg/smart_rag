"""Pydantic 数据模型定义。

定义所有 API 请求和响应的数据模型，
提供自动的请求体验证和响应序列化。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ==================== 请求模型 ====================


class QueryRequest(BaseModel):
    """纯问答请求（无会话记忆）。"""

    question: str = Field(..., min_length=1, description="用户问题")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="检索返回的文档块数量")


class ChatRequest(BaseModel):
    """对话请求（带会话记忆）。"""

    question: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field(..., min_length=1, description="会话标识，用于区分不同对话")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="检索返回的文档块数量")


class UploadRequest(BaseModel):
    """文件上传请求（扩展参数）。"""

    category: Optional[str] = Field(
        None,
        description="文档分类，可选值: regulation/safety/manual/report/general",
    )
    enable_ocr: Optional[bool] = Field(
        None, description="是否启用 OCR 识别扫描件"
    )
    enable_cleaning: Optional[bool] = Field(
        None, description="是否启用文本清洗"
    )
    enable_smart_chunk: Optional[bool] = Field(
        None, description="是否启用智能切分"
    )


# ==================== 响应模型 ====================


class SourceDocument(BaseModel):
    """引用来源信息。"""

    doc_name: str = Field(..., description="文档名称")
    chunk_index: int = Field(..., description="块序号")
    content: str = Field(..., description="块内容摘要")
    category: Optional[str] = Field(None, description="文档分类")
    heading_path: Optional[str] = Field(None, description="标题路径")


class QueryResponse(BaseModel):
    """问答响应。"""

    answer: str = Field(..., description="LLM 生成的回答")
    sources: list[SourceDocument] = Field(
        default_factory=list, description="引用来源列表"
    )


class ChatResponse(BaseModel):
    """对话响应。"""

    answer: str = Field(..., description="LLM 生成的回答")
    session_id: str = Field(..., description="会话标识")
    sources: list[SourceDocument] = Field(
        default_factory=list, description="引用来源列表"
    )


class UploadResponse(BaseModel):
    """文件上传响应。"""

    doc_id: str = Field(..., description="文档唯一标识")
    doc_name: str = Field(..., description="文档名称")
    chunk_count: int = Field(..., description="分块数量")
    message: str = Field(..., description="处理结果信息")
    processors_used: list[str] = Field(
        default_factory=list, description="本次使用的处理器"
    )


class DeleteResponse(BaseModel):
    """删除操作响应。"""

    message: str = Field(..., description="操作结果信息")
    deleted_count: int = Field(..., description="删除的记录数")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field(..., description="服务状态")
    llm_configured: bool = Field(..., description="LLM 是否已配置 API Key")
    milvus_connected: bool = Field(..., description="Milvus 是否连接成功")
    vector_count: int = Field(..., description="当前向量库中的文档块数量")
    bm25_enabled: bool = Field(default=True, description="Milvus 内置 BM25 是否启用")
    retrieval_mode: str = Field(default="hybrid", description="当前检索模式")
    timestamp: str = Field(..., description="当前时间戳")


# ==================== 对话历史模型 ====================


class SessionInfo(BaseModel):
    """会话信息。"""

    id: str = Field(..., description="会话 ID")
    title: str = Field(..., description="会话标题")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="最后更新时间")


class SessionListResponse(BaseModel):
    """会话列表响应。"""

    sessions: list[SessionInfo] = Field(..., description="会话列表")


class RenameSessionRequest(BaseModel):
    """重命名会话请求。"""

    title: str = Field(..., min_length=1, max_length=100, description="新会话标题")


class MessageInfo(BaseModel):
    """消息信息。"""

    id: int = Field(..., description="消息 ID")
    session_id: str = Field(..., description="会话 ID")
    role: str = Field(..., description="角色 (user/assistant)")
    content: str = Field(..., description="消息内容")
    sources: list = Field(default_factory=list, description="来源信息")
    created_at: str = Field(..., description="创建时间")


class MessageListResponse(BaseModel):
    """消息列表响应。"""

    messages: list[MessageInfo] = Field(..., description="消息列表")
