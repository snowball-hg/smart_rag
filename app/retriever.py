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
from langchain_core.runnables import RunnablePassthrough

from app.config import settings
from app.logger import logger
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
        self._retriever = vector_store_manager.as_retriever()
        self._rag_chain = self._build_rag_chain()

    def _build_rag_chain(self):
        """构建 RAG 链：检索 -> Prompt -> LLM -> 解析。

        使用 LCEL（LangChain Expression Language）构建。
        """
        return (
            {"context": self._retriever | format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | self._llm
            | StrOutputParser()
        )

    def query(self, question: str, top_k: Optional[int] = None) -> dict:
        """执行一次 RAG 查询。

        Args:
            question: 用户问题。
            top_k: 检索文档块数量，默认使用配置值。

        Returns:
            包含 answer 和 sources 的字典。
        """
        if top_k is not None:
            self._retriever.search_kwargs["k"] = top_k

        # 先检索文档块用于来源展示
        retrieved_docs = vector_store_manager.similarity_search(
            question, k=top_k or settings.TOP_K
        )

        if not retrieved_docs:
            logger.warning("未检索到相关文档: %s", question[:50])
            # 即使没有检索结果也要让 LLM 回答
            answer = self._llm.invoke(
                f"用户问：{question}\n\n注意：没有检索到相关文档内容，请告知用户目前知识库中没有相关信息。"
            )
            return {
                "answer": answer.content if hasattr(answer, "content") else str(answer),
                "sources": [],
            }

        # 执行 RAG 链
        logger.info(
            "RAG 查询: question='%s...', top_k=%d, retrieved=%d",
            question[:50],
            top_k or settings.TOP_K,
            len(retrieved_docs),
        )

        answer = self._rag_chain.invoke(question)
        sources = format_sources(retrieved_docs)

        return {"answer": answer, "sources": sources}
