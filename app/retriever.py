"""检索器与 Prompt 模板模块。

提供 RAG（检索增强生成）的核心组件：
- 基于向量检索的上下文获取
- 专为 RAG 优化的 Prompt 模板
- RAG 问答链（检索 + LLM 生成）
"""

from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from app.config import settings
from app.logger import logger
from app.reranker import reranker
from app.vector_store import vector_store_manager

# RAG 专用 Prompt 模板
# 要求 LLM 基于检索到的上下文回答问题，并注明信息来源
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个专业的文档智能助手。请根据以下检索到的文档内容回答用户的问题。\n\n"
        "要求：\n"
        "1. 只基于提供的文档内容回答，不要编造信息。\n"
        "2. 如果文档内容不足以回答问题，请明确告知用户。\n"
        "3. 在回答中引用相关的文档名称和块编号（如：据《xxx.pdf》第3节所述...）。\n"
        "4. 使用中文回答，保持专业、清晰、简洁。\n\n"
        "检索到的文档内容：\n{context}",
    ),
    ("human", "{question}"),
])


def format_docs(docs) -> str:
    """将检索到的文档列表格式化为 Prompt 中的上下文文本。

    每个文档块包含内容、来源文档名和块序号。
    """
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("doc_name", "未知文档")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        formatted.append(
            f"[来源 {i + 1}] {source} (第{chunk_idx}块)\n"
            f"{doc.page_content}\n"
        )
    return "\n---\n".join(formatted)


def format_sources(docs) -> list[dict]:
    """将检索到的文档格式化为 API 响应中的来源列表。"""
    sources = []
    seen = set()
    for doc in docs:
        doc_name = doc.metadata.get("doc_name", "未知文档")
        chunk_index = doc.metadata.get("chunk_index", 0)
        key = f"{doc_name}_{chunk_index}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "doc_name": doc_name,
                "chunk_index": chunk_index,
                "content": doc.page_content[:200],  # 截取前 200 字符
            })
    return sources


class RAGRetriever:
    """RAG 检索器，封装检索 + LLM 生成的全流程。"""

    def __init__(self, llm):
        """初始化 RAG 检索器。

        Args:
            llm: 语言模型实例（需兼容 LangChain 的 BaseLanguageModel）。
        """
        self._llm = llm
        self._rag_chain = self._build_rag_chain()

    def _build_rag_chain(self):
        """构建 RAG 生成链：Prompt -> LLM -> 解析。

        检索和重排序由 query() 统一完成，链只负责生成回答。
        """
        def _format_with_question(inputs: dict) -> dict:
            return {"context": format_docs(inputs["docs"]), "question": inputs["question"]}

        return (
            RunnableLambda(_format_with_question)
            | RAG_PROMPT
            | self._llm
            | StrOutputParser()
        )

    def _hybrid_search(self, query: str, top_k: int) -> list[Document]:
        """混合检索：向量检索 + Milvus 内置 BM25 全文检索，使用 RRF 融合。

        委托给 vector_store_manager.hybrid_search() 完成。
        """
        return vector_store_manager.hybrid_search(query, top_k=top_k)

    def _expand_context(self, docs: list[Document], window: int = 1) -> list[Document]:
        """为检索到的文档块扩展相邻上下文。

        如果某块来自一个较大的文档，将其前后的邻居块也纳入上下文，
        避免因分块切碎导致信息不完整。

        Args:
            docs: 检索到的文档块列表。
            window: 前后扩展的块数。

        Returns:
            扩展后的文档块列表（已去重，保持原始排序优先）。
        """
        seen_ids = set()
        expanded = []

        for doc in docs:
            doc_id = doc.metadata.get("doc_id")
            chunk_idx = doc.metadata.get("chunk_index")

            if doc_id is not None and chunk_idx is not None:
                neighbors = vector_store_manager.get_adjacent_chunks(
                    doc_id, chunk_idx, window=window
                )
                for nd in neighbors:
                    cid = nd.metadata.get("chunk_id")
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        expanded.append(nd)

            cid = doc.metadata.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                expanded.append(doc)

        return expanded

    def query(self, question: str, top_k: Optional[int] = None) -> dict:
        """执行一次 RAG 查询（支持混合检索 + 上下文扩展）。

        检索模式由 settings.RETRIEVAL_MODE 控制：
        - "vector": 仅向量检索（原行为）
        - "bm25":   仅 Milvus 内置 BM25 关键词检索
        - "hybrid": 向量 + BM25 混合检索

        Args:
            question: 用户问题。
            top_k: 检索文档块数量，默认使用配置值。

        Returns:
            包含 answer 和 sources 的字典。
        """
        final_k = top_k or settings.TOP_K

        # 1. 根据模式选择检索方式
        mode = settings.RETRIEVAL_MODE
        if mode == "bm25":
            retrieved_docs = vector_store_manager.bm25_search(question, k=final_k)
        elif mode == "hybrid" and settings.BM25_ENABLED:
            retrieved_docs = self._hybrid_search(question, top_k=final_k)
        else:
            candidate_k = settings.RERANK_CANDIDATE_K if settings.RERANK_ENABLED else final_k
            retrieved_docs = vector_store_manager.similarity_search(question, k=candidate_k)

        if not retrieved_docs:
            logger.warning("未检索到相关文档: %s", question[:50])
            answer = self._llm.invoke(
                f"用户问：{question}\n\n注意：没有检索到相关文档内容，请告知用户目前知识库中没有相关信息。"
            )
            return {
                "answer": answer.content if hasattr(answer, "content") else str(answer),
                "sources": [],
            }

        # 2. 重排序（仅对 vector 和 hybrid 模式生效）
        if mode != "bm25" and settings.RERANK_ENABLED and len(retrieved_docs) > 1:
            try:
                retrieved_docs = reranker.rerank(question, retrieved_docs, top_k=final_k)
            except Exception as e:
                logger.warning("重排序失败，使用原始检索结果: %s", e)

        # 3. 上下文扩展：为每个命中的块带上前后邻居
        retrieved_docs = self._expand_context(retrieved_docs, window=1)

        logger.info(
            "RAG 查询: mode=%s, question='%s...', top_k=%d, expanded=%d",
            mode,
            question[:50],
            final_k,
            len(retrieved_docs),
        )

        # 4. 链生成回答
        answer = self._rag_chain.invoke({"docs": retrieved_docs, "question": question})
        sources = format_sources(retrieved_docs)

        return {"answer": answer, "sources": sources}
