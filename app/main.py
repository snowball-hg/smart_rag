"""FastAPI 应用主文件。

定义所有 RESTful API 端点，包括：
- 文件上传与索引
- 纯问答（RAG，无会话）
- 对话式交互（Agent，带记忆）
- 文档删除管理
- 健康检查
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI

from app.agent import RAGAgent
from app.config import settings
from app.loader import process_file
from app.logger import logger
from app.retriever import RAGRetriever
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    UploadResponse,
)
from app.vector_store import vector_store_manager


# ==================== 全局状态 ====================

_llm: Optional[ChatOpenAI] = None
_rag_retriever: Optional[RAGRetriever] = None
_rag_agent: Optional[RAGAgent] = None
_llm_configured: bool = False


def _init_llm() -> Optional[ChatOpenAI]:
    """初始化 DeepSeek LLM 实例。

    如果 API Key 未配置，返回 None，后续端点会返回 503。
    """
    api_key = settings.DEEPSEEK_API_KEY
    if not api_key or api_key == "sk-your-deepseek-api-key-here":
        logger.warning("DEEPSEEK_API_KEY 未配置，LLM 功能不可用")
        return None

    return ChatOpenAI(
        model=settings.DEEPSEEK_MODEL_NAME,
        api_key=api_key,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=0.3,
        max_tokens=4096,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理，替代已弃用的 on_event 装饰器。"""
    # 启动时执行
    global _llm, _rag_retriever, _rag_agent, _llm_configured

    logger.info("=" * 50)
    logger.info("RAG Agent 服务启动中...")
    logger.info("LLM 模型: %s", settings.DEEPSEEK_MODEL_NAME)
    logger.info("Milvus 地址: %s:%s", settings.MILVUS_HOST, settings.MILVUS_PORT)
    logger.info("Embedding 提供商: %s", settings.EMBEDDING_PROVIDER)
    logger.info("分块配置: size=%d, overlap=%d", settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    logger.info("=" * 50)

    # 尝试初始化 LLM
    _llm = _init_llm()
    _llm_configured = _llm is not None

    if _llm_configured:
        try:
            _rag_retriever = RAGRetriever(_llm)
            _rag_agent = RAGAgent(_llm)
            logger.info("LLM 及 RAG 组件初始化完成")
        except Exception as e:
            logger.error("RAG 组件初始化失败: %s", e)
    else:
        logger.warning("LLM 未配置，请先设置 DEEPSEEK_API_KEY")

    # 确保上传目录存在
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    yield

    # 关闭时执行（可选）
    logger.info("RAG Agent 服务关闭中...")

# ==================== 应用初始化 ====================

app = FastAPI(
    title="RAG Agent 服务",
    description="基于 LangChain + Milvus + FastAPI 的企业级检索增强生成系统",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 配置，允许跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API 端点 ====================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点。

    返回服务状态、LLM 配置状态、Milvus 连接状态和向量数量。
    """
    try:
        vector_count = vector_store_manager.get_collection_stats()
        milvus_ok = True
    except Exception as e:
        logger.warning("健康检查 - Milvus 连接异常: %s", e)
        vector_count = 0
        milvus_ok = False

    return HealthResponse(
        status="healthy" if _llm_configured and milvus_ok else "degraded",
        llm_configured=_llm_configured,
        milvus_connected=milvus_ok,
        vector_count=vector_count,
        timestamp=__import__("datetime").datetime.now().isoformat(),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """上传文件并索引到知识库。

    支持的格式：PDF、TXT、Markdown（.md/.mdx）。
    上传后自动执行：加载 -> 分块 -> 向量化 -> 存入 Milvus。
    """
    # ---- 验证文件类型 ----
    allowed_extensions = {".pdf", ".txt", ".md", ".mdx"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}。支持: {', '.join(allowed_extensions)}",
        )

    # ---- 保存上传文件 ----
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # 生成唯一文件名，防止冲突
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = upload_dir / unique_name

    try:
        content = await file.read()
        file_path.write_bytes(content)
        logger.info("文件已保存: %s (%d bytes)", file_path, len(content))
    except Exception as e:
        logger.error("文件保存失败: %s", e)
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    # ---- 处理文档（加载 -> 分块 -> 添加元数据） ----
    try:
        chunks = process_file(file_path)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="文件内容为空或无法解析",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("文档处理失败: %s", e)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

    # ---- 存入向量库 ----
    try:
        ids = vector_store_manager.add_documents(chunks)
    except Exception as e:
        logger.error("向量存储失败: %s", e)
        raise HTTPException(status_code=500, detail=f"向量存储失败: {str(e)}")

    # ---- 获取 doc_id ----
    doc_id = chunks[0].metadata["doc_id"]

    return UploadResponse(
        doc_id=doc_id,
        doc_name=file.filename,
        chunk_count=len(chunks),
        message=f"文档 '{file.filename}' 处理完成，共 {len(chunks)} 个文档块",
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """纯问答模式（RAG，不保留会话记忆）。

    从知识库检索相关文档，结合 LLM 生成回答。
    """
    if not _llm_configured or _rag_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="LLM 未配置，请先设置 DEEPSEEK_API_KEY",
        )

    try:
        result = _rag_retriever.query(
            question=request.question, top_k=request.top_k
        )
        return QueryResponse(
            answer=result["answer"],
            sources=[
                SourceDocument(**s) for s in result["sources"]
            ],
        )
    except Exception as e:
        logger.error("查询失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """对话式交互（Agent 模式，带多轮记忆）。

    Agent 可自主决定是否调用检索工具，并记忆上下文。
    """
    if not _llm_configured or _rag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="LLM 未配置，请先设置 DEEPSEEK_API_KEY",
        )

    try:
        result = _rag_agent.chat(
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k,
        )
        return ChatResponse(
            answer=result["answer"],
            session_id=result["session_id"],
            sources=[
                SourceDocument(**s) for s in result["sources"]
            ],
        )
    except Exception as e:
        logger.error("对话失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"对话处理失败: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式对话（SSE），逐 token 返回 AI 生成内容。

    使用 Server-Sent Events 协议，事件格式：
    - data: {"type": "token", "content": "..."}   文本片段
    - data: {"type": "done", "session_id": "...", "sources": [...]}  完成
    - data: {"type": "error", "content": "..."}   错误
    """
    if not _llm_configured or _rag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="LLM 未配置，请先设置 DEEPSEEK_API_KEY",
        )

    async def event_generator():
        async for event in _rag_agent.chat_stream(
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """删除指定文档的所有数据。

    根据 doc_id 从 Milvus 中删除对应的所有文档块。
    """
    try:
        deleted = vector_store_manager.delete_by_doc_id(doc_id)
        return DeleteResponse(
            message=f"文档 {doc_id} 已删除",
            deleted_count=deleted,
        )
    except Exception as e:
        logger.error("删除文档失败: %s", e)
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")


@app.delete("/documents", response_model=DeleteResponse)
async def delete_all_documents():
    """清空知识库中的所有数据。

    谨慎操作！会删除 Milvus 集合中的所有文档块。
    """
    try:
        vector_store_manager.delete_collection()
        return DeleteResponse(
            message="已清空所有文档数据",
            deleted_count=-1,
        )
    except Exception as e:
        logger.error("清空数据失败: %s", e)
        raise HTTPException(status_code=500, detail=f"清空数据失败: {str(e)}")
