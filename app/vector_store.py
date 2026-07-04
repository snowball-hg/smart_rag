"""向量存储封装模块。

基于 LangChain Milvus 集成，提供文档向量化、存储、
检索的统一接口。支持本地 Sentence-Transformers 和
千问 Embeddings API 两种嵌入方式。
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import MilvusClient, connections as milvus_connections

import torch

from app.config import settings
from app.logger import logger


def _create_embeddings() -> Embeddings:
    """创建 Embeddings 模型实例。

    根据配置选择本地 Sentence-Transformers 或千问 Embeddings API。
    本地模型默认使用 BAAI/bge-large-zh-v1.5（1024 维，中文优化）。

    Returns:
        Embeddings 实例。
    """
    if settings.EMBEDDING_PROVIDER == "qwen":
        return _create_qwen_embeddings()

    return _create_local_embeddings()


def _resolve_model_path(model_name: str) -> str:
    """解析模型路径，按优先级查找：
    1. 项目本地 models/ 目录
    2. ModelScope 缓存/下载
    3. HuggingFace 直接加载

    Args:
        model_name: 模型名称（如 Qwen/Qwen3-Embedding-0.6B）。

    Returns:
        可用的模型本地路径。
    """
    from pathlib import Path

    # 项目根目录下的 models 目录
    local_dir = Path(settings.LOCAL_EMBEDDING_MODEL_PATH) / model_name
    if local_dir.exists():
        logger.info("使用项目本地模型: %s", local_dir)
        return str(local_dir)

    # 尝试从 ModelScope 下载
    if "Qwen" in model_name or model_name.lower().startswith("qwen"):
        try:
            from modelscope import snapshot_download
            logger.info("从 ModelScope 下载模型: %s", model_name)
            local_path = snapshot_download(model_name, local_dir=str(settings.LOCAL_EMBEDDING_MODEL_PATH))
            logger.info("模型已下载到: %s", local_path)
            return local_path
        except ImportError:
            logger.warning("modelscope 未安装，回退到 HuggingFace 下载")
        except Exception as e:
            logger.warning("ModelScope 下载失败 (%s)，回退到 HuggingFace", e)

    return model_name


def _create_local_embeddings() -> HuggingFaceEmbeddings:
    """创建本地 Sentence-Transformers Embeddings。"""
    model_name = settings.LOCAL_EMBEDDING_MODEL
    logger.info("初始化本地 Embeddings 模型: %s", model_name)

    model_path = _resolve_model_path(model_name)

    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )


def _create_qwen_embeddings() -> Embeddings:
    """创建千问 Embeddings API 实例。

    注意: 使用千问 Embeddings 需要配置 QWEN_EMBEDDING_API_KEY。
    如果未配置，将回退到本地模型并发出警告。
    """
    if not settings.QWEN_EMBEDDING_API_KEY:
        logger.warning(
            "QWEN_EMBEDDING_API_KEY 未配置，回退到本地 Embeddings 模型"
        )
        return _create_local_embeddings()

    try:
        from langchain_community.embeddings import DashScopeEmbeddings

        logger.info("初始化千问 Embeddings API: %s", settings.QWEN_EMBEDDING_MODEL)
        return DashScopeEmbeddings(
            model=settings.QWEN_EMBEDDING_MODEL,
            dashscope_api_key=settings.QWEN_EMBEDDING_API_KEY,
        )
    except ImportError:
        logger.warning("DashScope 包未安装，回退到本地 Embeddings 模型")
        return _create_local_embeddings()


class VectorStoreManager:
    """向量存储管理器，封装 Milvus 的增删查操作。"""

    def __init__(self):
        self._embeddings: Optional[Embeddings] = None
        self._vector_store: Optional[Milvus] = None
        self._collection_initialized = False

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = _create_embeddings()
        return self._embeddings

    @property
    def vector_store(self) -> Milvus:
        if self._vector_store is None:
            self._vector_store = self._init_vector_store()
        return self._vector_store

    def _init_vector_store(self) -> Milvus:
        """初始化 Milvus 向量存储。

        如果集合不存在则自动创建，否则连接已有集合。
        """
        logger.info(
            "连接 Milvus: %s/%s",
            settings.MILVUS_URI,
            settings.MILVUS_COLLECTION,
        )

        try:
            # 预创建 MilvusClient 获取 alias，提前注册 ORM 连接
            # 否则 langchain_milvus 内部 Collection(using=self.alias) 会找不到连接
            temp_client = MilvusClient(
                host=settings.MILVUS_HOST, port=settings.MILVUS_PORT
            )
            alias = temp_client._using
            milvus_connections.connect(
                alias=alias,
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
            )

            vector_store = Milvus(
                embedding_function=self.embeddings,
                collection_name=settings.MILVUS_COLLECTION,
                connection_args={
                    "host": settings.MILVUS_HOST,
                    "port": settings.MILVUS_PORT,
                },
                auto_id=True,
                drop_old=False,
            )
            logger.info(
                "Milvus 连接成功, 集合: %s", settings.MILVUS_COLLECTION
            )
            self._collection_initialized = True
            return vector_store
        except Exception as e:
            logger.error("Milvus 连接失败: %s", e)
            raise

    def add_documents(self, documents: list[Document]) -> list[str]:
        """将文档块向量化并存入 Milvus。

        Args:
            documents: 待存储的文档块列表。

        Returns:
            存储后的 Milvus ID 列表。
        """
        if not documents:
            logger.warning("没有文档需要存储")
            return []

        try:
            ids = self.vector_store.add_documents(documents)
            logger.info("成功存入 %d 个文档块到 Milvus", len(documents))
            return ids
        except Exception as e:
            logger.error("文档存储失败: %s", e)
            raise

    def similarity_search(
        self, query: str, k: int | None = None
    ) -> list[Document]:
        """执行向量相似性检索。

        Args:
            query: 查询文本。
            k: 返回的文档块数量，默认使用配置中的 TOP_K。

        Returns:
            检索到的文档块列表。
        """
        k = k or settings.TOP_K
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.debug("向量检索完成: query='%s...', k=%d, results=%d", query[:50], k, len(docs))
            return docs
        except Exception as e:
            logger.error("向量检索失败: %s", e)
            return []

    def as_retriever(self, k: int | None = None) -> VectorStoreRetriever:
        """获取检索器实例，供 Agent 和 RAG 链使用。

        Args:
            k: 检索返回的文档块数。

        Returns:
            VectorStoreRetriever 实例。
        """
        return self.vector_store.as_retriever(
            search_kwargs={"k": k or settings.TOP_K}
        )

    def delete_collection(self) -> None:
        """清空当前集合中的所有数据。"""
        try:
            self.vector_store.col.drop()
            # 重新初始化以创建空集合
            self._vector_store = None
            self._collection_initialized = False
            self._vector_store = self._init_vector_store()
            logger.warning("已清空集合: %s", settings.MILVUS_COLLECTION)
        except Exception as e:
            logger.error("清空集合失败: %s", e)
            raise

    def get_collection_stats(self) -> int:
        """获取集合中的文档块数量。"""
        try:
            return self.vector_store.collection.num_entities
        except Exception:
            return 0

    def delete_by_doc_id(self, doc_id: str) -> int:
        """删除指定文档的所有块。

        Args:
            doc_id: 文档唯一标识。

        Returns:
            删除的记录数。
        """
        try:
            collection = self.vector_store.collection
            expr = f'doc_id == "{doc_id}"'
            result = collection.delete(expr)
            logger.info("删除文档 %s: %s", doc_id, result)
            return len(result) if result else 0
        except Exception as e:
            logger.error("删除文档失败 %s: %s", doc_id, e)
            raise


# 全局单例
vector_store_manager = VectorStoreManager()
