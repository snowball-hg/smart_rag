"""向量存储封装模块。

基于 LangChain Milvus 集成，提供文档向量化、存储、
检索的统一接口。支持本地 Sentence-Transformers 和
千问 Embeddings API 两种嵌入方式。

利用 Milvus 内置 BM25 Function 实现全文检索，
通过 HybridSearch 实现向量 + 关键词混合检索。
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import (
    MilvusClient,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    connections as milvus_connections,
)

import torch

from app.config import settings
from app.logger import logger


def _create_embeddings() -> Embeddings:
    """创建 Embeddings 模型实例。"""
    if settings.EMBEDDING_PROVIDER == "qwen":
        return _create_qwen_embeddings()
    return _create_local_embeddings()


def _resolve_model_path(model_name: str) -> str:
    """解析模型路径，按优先级查找：
    1. 项目本地 models/ 目录
    2. ModelScope 缓存/下载
    3. HuggingFace 直接加载
    """
    from pathlib import Path

    local_dir = Path(settings.LOCAL_EMBEDDING_MODEL_PATH) / model_name
    if local_dir.exists():
        logger.info("使用项目本地模型: %s", local_dir)
        return str(local_dir)

    if "Qwen" in model_name or model_name.lower().startswith("qwen"):
        try:
            from modelscope import snapshot_download
            logger.info("从 ModelScope 下载模型: %s", model_name)
            local_path = snapshot_download(
                model_name, local_dir=str(settings.LOCAL_EMBEDDING_MODEL_PATH)
            )
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
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def _create_qwen_embeddings() -> Embeddings:
    """创建千问 Embeddings API 实例。"""
    if not settings.QWEN_EMBEDDING_API_KEY:
        logger.warning("QWEN_EMBEDDING_API_KEY 未配置，回退到本地 Embeddings 模型")
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
    """向量存储管理器，封装 Milvus 的增删查与混合检索操作。

    集合使用显式 schema 创建，包含：
    - `vector`: 稠密向量场（用于语义检索）
    - `text`: 文本字段
    - `sparse_vector`: 稀疏向量场（BM25 Function 自动生成，用于关键词检索）
    - 动态字段: doc_id / doc_name / chunk_index 等元数据
    """

    # BM25 Function 名称常量
    _BM25_FN_NAME = "bm25_fn"
    _SPARSE_FIELD = "sparse_vector"
    _BM25_METRIC = "BM25"

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

    @property
    def collection(self) -> Collection:
        """获取底层的 pymilvus Collection 实例。"""
        return self.vector_store.col

    # ==================== 集合初始化 ====================

    def _init_vector_store(self) -> Milvus:
        """初始化 Milvus 向量存储。

        如果集合不存在则创建（含 BM25 Function），否则连接已有集合。
        开启动态字段，元数据无需预先定义。
        """
        logger.info(
            "连接 Milvus: %s/%s",
            settings.MILVUS_URI,
            settings.MILVUS_COLLECTION,
        )

        try:
            temp_client = MilvusClient(
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
            )
            alias = temp_client._using
            milvus_connections.connect(
                alias=alias,
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
            )

            from pymilvus import utility
            collection_name = settings.MILVUS_COLLECTION

            if not utility.has_collection(collection_name, using=alias):
                self._create_collection(collection_name, alias)
            else:
                # 检查是否为旧集合（缺少 sparse_vector 字段）
                collection = Collection(collection_name, using=alias)
                existing_fields = [f.name for f in collection.schema.fields]
                if self._SPARSE_FIELD not in existing_fields:
                    logger.warning(
                        "集合 %s 缺少 sparse_vector 字段（旧版 schema），正在重建...",
                        collection_name,
                    )
                    collection.drop()
                    self._create_collection(collection_name, alias)

            vector_store = Milvus(
                embedding_function=self.embeddings,
                collection_name=collection_name,
                connection_args={
                    "host": settings.MILVUS_HOST,
                    "port": settings.MILVUS_PORT,
                },
                auto_id=True,
                drop_old=False,
                enable_dynamic_field=True,
            )
            logger.info("Milvus 连接成功, 集合: %s", collection_name)
            self._collection_initialized = True
            return vector_store
        except Exception as e:
            logger.error("Milvus 连接失败: %s", e)
            raise

    def _create_collection(self, collection_name: str, alias: str) -> None:
        """创建含 BM25 Function 的集合。

        Schema:
          - pk: INT64 (主键, 自增)
          - vector: FLOAT_VECTOR (1024 维, 稠密向量)
          - text: VARCHAR(65535) (文档文本)
          - sparse_vector: SPARSE_FLOAT_VECTOR (BM25 Function 自动填充)

        BM25 Function 将 text 字段内容自动转为稀疏向量，
        实现 Milvus 内置的关键词全文检索。
        """
        dim = 1024
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
            ),
            FieldSchema(
                name=self._SPARSE_FIELD,
                dtype=DataType.SPARSE_FLOAT_VECTOR,
            ),
        ]

        bm25_function = Function(
            name=self._BM25_FN_NAME,
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=[self._SPARSE_FIELD],
        )

        schema = CollectionSchema(
            fields,
            description="RAG document collection with BM25 FTS",
            enable_dynamic_field=True,
            functions=[bm25_function],
        )

        collection = Collection(collection_name, schema=schema, using=alias)

        # 创建向量索引（IVF_FLAT，LangChain Milvus 连接时需要索引才能正常查询）
        vector_index = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index("vector", vector_index)

        # 为稀疏向量创建索引（BM25 使用 SPARSE_INVERTED_INDEX）
        sparse_index = {
            "metric_type": self._BM25_METRIC,
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
        }
        collection.create_index(self._SPARSE_FIELD, sparse_index)
        collection.load()

        logger.info(
            "集合已创建: %s (dim=%d, BM25_FTS=True, dynamic_field=True)",
            collection_name,
            dim,
        )

    # ==================== 文档增删 ====================

    def add_documents(self, documents: list[Document]) -> list[str]:
        """将文档块向量化并存入 Milvus。

        BM25 Function 会自动从 text 字段生成稀疏向量，
        无需额外操作。

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

    def delete_by_doc_id(self, doc_id: str) -> int:
        """按 doc_id 删除文档（Milvus 自带 BM25 索引同步删除）。

        Args:
            doc_id: 文档唯一标识。

        Returns:
            删除的记录数。
        """
        try:
            collection = self.collection
            expr = f'doc_id == "{doc_id}"'
            result = collection.delete(expr)
            deleted = len(result) if result else 0
            logger.info("Milvus 删除文档 %s: %s 个块", doc_id, deleted)
            return deleted
        except Exception as e:
            logger.error("Milvus 删除文档失败 %s: %s", doc_id, e)
            raise

    def delete_collection(self) -> None:
        """清空当前集合（包括 BM25 索引）。"""
        try:
            self.collection.drop()
            self._vector_store = None
            self._collection_initialized = False
            self._vector_store = self._init_vector_store()
            logger.warning("已清空集合: %s", settings.MILVUS_COLLECTION)
        except Exception as e:
            logger.error("清空集合失败: %s", e)
            raise

    # ==================== 检索 ====================

    def similarity_search(
        self, query: str, k: int | None = None
    ) -> list[Document]:
        """执行向量相似性检索（语义检索）。

        Args:
            query: 查询文本。
            k: 返回的文档块数量，默认使用配置中的 TOP_K。

        Returns:
            检索到的文档块列表。
        """
        k = k or settings.TOP_K
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.debug(
                "向量检索完成: query='%s...', k=%d, results=%d",
                query[:50], k, len(docs),
            )
            return docs
        except Exception as e:
            logger.error("向量检索失败: %s", e)
            return []

    def bm25_search(self, query: str, k: int = 10) -> list[Document]:
        """使用 Milvus 内置 BM25 全文检索引擎进行关键词检索。

        依赖集合中的 BM25 Function 自动生成的 sparse_vector 字段。

        Args:
            query: 关键词查询文本。
            k: 返回的文档块数量。

        Returns:
            检索到的文档块列表。
        """
        collection = self.collection
        try:
            search_params = {"metric_type": self._BM25_METRIC}
            results = collection.search(
                data=[query],
                anns_field=self._SPARSE_FIELD,
                param=search_params,
                limit=k,
                output_fields=["text", "doc_id", "doc_name", "chunk_index", "chunk_id"],
            )

            docs = []
            for hits in results:
                for hit in hits:
                    fields = hit.entity.fields if hasattr(hit.entity, "fields") else hit.entity
                    metadata = {
                        "doc_id": fields.get("doc_id", ""),
                        "doc_name": fields.get("doc_name", ""),
                        "chunk_index": fields.get("chunk_index", 0),
                        "chunk_id": fields.get("chunk_id", ""),
                        "score": hit.score,
                    }
                    doc = Document(
                        page_content=fields.get("text", ""),
                        metadata=metadata,
                    )
                    docs.append(doc)

            logger.debug(
                "BM25 检索完成: query='%s...', k=%d, results=%d",
                query[:50], k, len(docs),
            )
            return docs
        except Exception as e:
            logger.warning("BM25 检索失败（Milvus FTS 可能未就绪）: %s", e)
            return []

    def hybrid_search(
        self, query: str, top_k: int, vector_k: int | None = None, bm25_k: int | None = None
    ) -> list[Document]:
        """混合检索：向量语义检索 + Milvus 内置 BM25 关键词检索。

        两种检索结果通过 RRF（Reciprocal Rank Fusion）融合排序。

        Args:
            query: 查询文本。
            top_k: 最终返回的文档块数量。
            vector_k: 向量检索取多少候选，默认 top_k * 2。
            bm25_k: BM25 检索取多少候选，默认 max(top_k * 2, 20)。

        Returns:
            融合排序后的文档块列表。
        """
        v_k = vector_k or top_k * 2
        b_k = bm25_k or max(top_k * 2, 20)

        # 1. 向量检索
        vector_docs = self.similarity_search(query, k=v_k)

        # 2. BM25 关键词检索（Milvus 内置 FTS）
        bm25_docs = self.bm25_search(query, k=b_k) if settings.BM25_ENABLED else []

        if not vector_docs and not bm25_docs:
            return []

        # 3. RRF 融合
        rrf_k = 60
        score_map: dict[str, dict] = {}

        for rank, doc in enumerate(vector_docs):
            cid = doc.metadata.get("chunk_id", str(id(doc)))
            score_map[cid] = {"doc": doc, "score": 1.0 / (rrf_k + rank + 1)}

        for rank, doc in enumerate(bm25_docs):
            cid = doc.metadata.get("chunk_id", str(id(doc)))
            if cid in score_map:
                score_map[cid]["score"] += 1.0 / (rrf_k + rank + 1)
            else:
                score_map[cid] = {"doc": doc, "score": 1.0 / (rrf_k + rank + 1)}

        sorted_items = sorted(
            score_map.values(), key=lambda x: x["score"], reverse=True
        )
        logger.info(
            "混合检索: vector=%d, bm25=%d, fusion=%d",
            len(vector_docs),
            len(bm25_docs),
            len(sorted_items),
        )
        return [item["doc"] for item in sorted_items[:top_k]]

    def get_adjacent_chunks(
        self, doc_id: str, chunk_index: int, window: int = 1
    ) -> list[Document]:
        """获取指定文档块的前后相邻块。

        通过 Milvus query 按 doc_id + chunk_index 范围精确查询，
        不涉及向量检索，用于检索阶段的上下文扩展。

        Args:
            doc_id: 文档唯一标识。
            chunk_index: 当前块序号。
            window: 前后取多少块，默认各取 1 块。

        Returns:
            相邻文档块列表。
        """
        start = max(0, chunk_index - window)
        end = chunk_index + window
        expr = (
            f'doc_id == "{doc_id}"'
            f" and chunk_index >= {start}"
            f" and chunk_index <= {end}"
        )

        try:
            results = self.collection.query(
                expr=expr,
                output_fields=["text", "doc_id", "doc_name", "chunk_index", "chunk_id"],
            )
            docs = []
            for r in results:
                doc = Document(
                    page_content=r.get("text", ""),
                    metadata={
                        "doc_id": r.get("doc_id", ""),
                        "doc_name": r.get("doc_name", ""),
                        "chunk_index": r.get("chunk_index", 0),
                        "chunk_id": r.get("chunk_id", ""),
                    },
                )
                docs.append(doc)
            return docs
        except Exception as e:
            logger.warning("获取相邻块失败: %s", e)
            return []

    def as_retriever(self, k: int | None = None) -> VectorStoreRetriever:
        """获取检索器实例，供 Agent 和 RAG 链使用。"""
        return self.vector_store.as_retriever(
            search_kwargs={"k": k or settings.TOP_K}
        )

    def get_collection_stats(self) -> int:
        """获取集合中的文档块数量。"""
        try:
            return self.collection.num_entities
        except Exception:
            return 0


# 全局单例
vector_store_manager = VectorStoreManager()
