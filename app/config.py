"""应用配置管理模块。

从 .env 文件和环境变量加载所有配置，
提供统一的配置访问接口，方便后续扩展。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# 加载 .env 文件（优先加载项目根目录的 .env）
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)


class Settings:
    """应用全局配置，所有配置项集中管理。"""

    # ---- DeepSeek LLM ----
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL_NAME: str = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-v4-flash")

    # 前端可选的模型列表
    AVAILABLE_MODELS: list[str] = [
        "qwen3.7-plus",
    ]

    # ---- Milvus ----
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "rag_documents")
    MILVUS_URI: str = f"http://{MILVUS_HOST}:{MILVUS_PORT}"

    # ---- Embedding ----
    EMBEDDING_PROVIDER: Literal["local", "qwen"] = os.getenv("EMBEDDING_PROVIDER", "local")  # type: ignore[assignment]
    LOCAL_EMBEDDING_MODEL: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"
    )
    LOCAL_EMBEDDING_MODEL_PATH: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL_PATH", r"D:\project\AI\smart_rag\models"
    )
    QWEN_EMBEDDING_API_KEY: str = os.getenv("QWEN_EMBEDDING_API_KEY", "")
    QWEN_EMBEDDING_MODEL: str = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")

    # ---- Chunking ----
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1024"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "256"))
    # ---- Processor ----
    PROCESSOR_ENABLE_OCR: bool = os.getenv("PROCESSOR_ENABLE_OCR", "true").lower() == "true"
    PROCESSOR_ENABLE_CLEANING: bool = os.getenv("PROCESSOR_ENABLE_CLEANING", "true").lower() == "true"

    # ---- Retrieval ----
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # ---- BM25 关键词检索 ----
    BM25_ENABLED: bool = os.getenv("BM25_ENABLED", "true").lower() == "true"
    RETRIEVAL_MODE: str = os.getenv("RETRIEVAL_MODE", "hybrid")  # vector, bm25, hybrid

    # ---- Rerank ----
    RERANK_ENABLED: bool = os.getenv("RERANK_ENABLED", "true").lower() == "true"
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    RERANK_CANDIDATE_K: int = int(os.getenv("RERANK_CANDIDATE_K", "20"))
    RERANK_MODEL_PATH: str = os.getenv(
        "RERANK_MODEL_PATH", r"D:\project\AI\smart_rag\models"
    )

    # ---- Logging ----
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/rag_system.log")
    LOG_DIR: str = os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else "logs"

    # ---- Upload ----
    UPLOAD_DIR: str = "uploads"

    # ---- 支持的文档类型 ----
    SUPPORTED_FILE_TYPES: list[str] = [
        ".pdf", ".txt", ".md", ".mdx",
        ".doc", ".docx", ".ppt", ".pptx",
        ".xls", ".xlsx",
        ".html", ".htm",
        ".csv", ".json", ".xml",
        ".eml", ".msg",
    ]


settings = Settings()
