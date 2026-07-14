"""FastAPI 应用主文件。

定义所有 RESTful API 端点，包括：
- 文件上传与索引（支持文档分类和预处理开关）
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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI

from app.agent import RAGAgent
from app.chat_history import (
    add_message,
    create_session,
    delete_session,
    get_messages,
    get_session,
    list_sessions,
    rename_session,
)
from app.config import settings
from app.loader import process_file
from app.logger import logger
from app.retriever import RAGRetriever
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    HealthResponse,
    MessageInfo,
    MessageListResponse,
    QueryRequest,
    QueryResponse,
    RenameSessionRequest,
    SessionInfo,
    SessionListResponse,
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
    """初始化 DeepSeek LLM 实例。"""
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
    """应用生命周期管理。"""
    global _llm, _rag_retriever, _rag_agent, _llm_configured

    logger.info("=" * 50)
    logger.info("RAG Agent 服务启动中...")
    logger.info("LLM 模型: %s", settings.DEEPSEEK_MODEL_NAME)
    logger.info("Milvus 地址: %s:%s", settings.MILVUS_HOST, settings.MILVUS_PORT)
    logger.info("Embedding 提供商: %s", settings.EMBEDDING_PROVIDER)
    logger.info(
        "处理器配置: OCR=%s, 清洗=%s, 智能切分=%s",
        settings.PROCESSOR_ENABLE_OCR,
        settings.PROCESSOR_ENABLE_CLEANING,
        settings.PROCESSOR_ENABLE_SMART_CHUNK,
    )
    logger.info("分块配置: size=%d, overlap=%d", settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    logger.info(
        "检索配置: 模式=%s, BM25=%s, 重排序=%s",
        settings.RETRIEVAL_MODE,
        settings.BM25_ENABLED,
        settings.RERANK_ENABLED,
    )
    logger.info("=" * 50)

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

    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    yield

    logger.info("RAG Agent 服务关闭中...")

# ==================== 应用初始化 ====================

app = FastAPI(
    title="RAG Agent 服务",
    description="基于 LangChain + Milvus + FastAPI 的企业级检索增强生成系统",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS 配置
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
    """健康检查端点。"""
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
        bm25_enabled=settings.BM25_ENABLED,
        retrieval_mode=settings.RETRIEVAL_MODE,
        timestamp=__import__("datetime").datetime.now().isoformat(),
    )


VALID_CATEGORIES = {"regulation", "safety", "manual", "report", "general"}


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    category: Optional[str] = Form(None),
    enable_ocr: Optional[bool] = Form(None),
    enable_cleaning: Optional[bool] = Form(None),
    enable_smart_chunk: Optional[bool] = Form(None),
):
    """上传文件并索引到知识库。

    支持的格式：PDF、TXT、MD、Word、PPT、Excel、HTML、CSV 等。
    通过 Form 参数控制预处理行为：
    - category: 文档分类（regulation/safety/manual/report/general）
    - enable_ocr: 是否启用 OCR 识别扫描件
    - enable_cleaning: 是否启用文本清洗
    - enable_smart_chunk: 是否启用智能切分

    不传参数时使用配置文件中的默认值。
    """
    # ---- 验证文件类型 ----
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.SUPPORTED_FILE_TYPES:
        # 保持向后兼容
        allowed_basic = {".pdf", ".txt", ".md", ".mdx"}
        if ext not in allowed_basic:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"不支持的文件类型: {ext}。"
                    f"基础支持: {', '.join(allowed_basic)}；"
                    f"安装 Unstructured 后可支持: Word/PPT/Excel/HTML/CSV/Email 等"
                ),
            )

    # ---- 验证分类 ----
    if category and category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"无效的分类: {category}。可选值: {', '.join(VALID_CATEGORIES)}",
        )

    # ---- 保存上传文件 ----
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = upload_dir / unique_name

    try:
        content = await file.read()
        file_path.write_bytes(content)
        logger.info("文件已保存: %s (%d bytes)", file_path, len(content))
    except Exception as e:
        logger.error("文件保存失败: %s", e)
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    # ---- 处理文档（使用高级管线或基础流程） ----
    try:
        chunks = process_file(
            file_path,
            category=category,
            enable_ocr=enable_ocr,
            enable_cleaning=enable_cleaning,
            enable_smart_chunk=enable_smart_chunk,
        )
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

    doc_id = chunks[0].metadata["doc_id"]

    # ---- 记录使用的处理器 ----
    processors_used = []
    if "heading_path" in chunks[0].metadata:
        processors_used.append("smart_chunk")
    if any(doc.metadata.get("element_type") == "ocr_text" for doc in chunks):
        processors_used.append("ocr")
    if category and "category" in chunks[0].metadata:
        processors_used.append(f"category_{category}")

    return UploadResponse(
        doc_id=doc_id,
        doc_name=file.filename,
        chunk_count=len(chunks),
        message=(
            f"文档 '{file.filename}' 处理完成，共 {len(chunks)} 个文档块"
            + (f"（分类: {category}）" if category else "")
        ),
        processors_used=processors_used,
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """纯问答模式（RAG，不保留会话记忆）。"""
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
    """对话式交互（Agent 模式，带多轮记忆）。"""
    if not _llm_configured or _rag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="LLM 未配置，请先设置 DEEPSEEK_API_KEY",
        )

    try:
        # 确保会话存在
        create_session(request.session_id)

        # 加载历史消息（当前消息还没保存，这是之前的所有对话）
        history = get_messages(request.session_id)

        # 保存用户消息
        add_message(request.session_id, "user", request.question)

        # 将历史消息传给 Agent，让它拥有完整的对话上下文
        result = _rag_agent.chat(
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k,
            history_messages=history,
        )

        # 保存 AI 回复
        add_message(
            request.session_id,
            "assistant",
            result["answer"],
            sources=result["sources"],
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
    """流式对话（SSE），逐 token 返回 AI 生成内容。"""
    if not _llm_configured or _rag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="LLM 未配置，请先设置 DEEPSEEK_API_KEY",
        )

    # 确保会话存在
    create_session(request.session_id)

    # 加载历史消息（当前消息还没保存，这是之前的所有对话）
    history = get_messages(request.session_id)

    # 保存用户消息
    add_message(request.session_id, "user", request.question)

    async def event_generator():
        full_answer = ""

        async for event in _rag_agent.chat_stream(
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k,
            history_messages=history,
        ):
            # 累计完整回复
            if event.get("type") == "token":
                full_answer += event["content"]
            elif event.get("type") == "done":
                # 保存 AI 回复到数据库
                add_message(
                    request.session_id,
                    "assistant",
                    full_answer,
                    sources=event.get("sources", []),
                )

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
    """删除指定文档的所有数据。"""
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
    """清空知识库中的所有数据。"""
    try:
        vector_store_manager.delete_collection()
        return DeleteResponse(
            message="已清空所有文档数据",
            deleted_count=-1,
        )
    except Exception as e:
        logger.error("清空数据失败: %s", e)
        raise HTTPException(status_code=500, detail=f"清空数据失败: {str(e)}")


# ==================== 会话管理 API ====================


@app.get("/sessions", response_model=SessionListResponse)
async def get_sessions():
    """获取所有会话列表（按更新时间倒序）。"""
    try:
        sessions = list_sessions()
        return SessionListResponse(
            sessions=[SessionInfo(**s) for s in sessions]
        )
    except Exception as e:
        logger.error("获取会话列表失败: %s", e)
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")


@app.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
async def get_session_messages(session_id: str):
    """获取指定会话的所有消息。"""
    try:
        session = get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        messages = get_messages(session_id)
        return MessageListResponse(
            messages=[MessageInfo(**m) for m in messages]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("获取消息失败: %s", e)
        raise HTTPException(status_code=500, detail=f"获取消息失败: {str(e)}")


@app.patch("/sessions/{session_id}")
async def rename_session_endpoint(session_id: str, request: RenameSessionRequest):
    """重命名会话。"""
    try:
        success = rename_session(session_id, request.title)
        if not success:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"message": "重命名成功", "session_id": session_id, "title": request.title}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("重命名会话失败: %s", e)
        raise HTTPException(status_code=500, detail=f"重命名会话失败: {str(e)}")


@app.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    """删除会话及其所有消息。"""
    try:
        success = delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"message": "会话已删除", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("删除会话失败: %s", e)
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")
