"""Agent 创建与执行模块。

基于 LangGraph 构建具有工具调用能力的 ReAct Agent，
通过消息列表维护多轮对话历史，支持自主检索决策。
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from app.config import settings
from app.logger import logger
from app.reranker import reranker
from app.vector_store import vector_store_manager

# Agent 系统 Prompt
AGENT_SYSTEM_PROMPT = (
    "你是一个智能文档助手，可以访问企业的知识库。"
    "你有以下能力：\n"
    "1. 检索知识库中的文档内容来回答问题\n"
    "2. 基于检索结果进行推理和总结\n"
    "3. 记住对话历史，进行多轮交流\n\n"
    "使用规则：\n"
    "- 当用户提问时，如果问题需要查阅文档，请调用检索工具。\n"
    "- 如果问题需要查阅文档，调用一次检索工具即可，不要多次检索。\n"
    "- 如果用户只是日常对话（打招呼、闲聊等），可以直接回复。\n"
    "- 检索后请引用信息来源（文档名称和块编号）。\n"
    "- 如果检索结果不充分，请告知用户并询问更多细节。\n"
    "- 始终保持专业、友好、清晰的回答风格。\n"
    "- 注意：检索工具会自动优化你的搜索关键词，你只需提供原始问题即可。"
)

# 查询重写 Prompt
QUERY_REWRITE_PROMPT = (
    "你是一个查询优化助手。你的任务是将用户的自然语言问题改写为"
    "更利于知识库检索的搜索关键词。\n\n"
    "规则：\n"
    "1. 提取关键实体和概念，去除语气词、冗余修饰\n"
    "2. 如果问题包含代词（它、这、那个等），结合上下文明确指代\n"
    "3. 保持核心语义不变，输出简洁的搜索关键词\n"
    "4. 直接输出重写后的查询，不要解释\n\n"
    "5. 进行知识库查询时，把用户的原始问题原封不动地传给知识库查询工具，不要修改\n"
    "用户问题：{question}\n\n"
    "重写后的查询："
)


def _query_rewrite(question: str, llm) -> str:
    """使用 LLM 将用户原始问题重写为更利于检索的关键词/短语。

    Args:
        question: 用户原始问题。
        llm: 语言模型实例。

    Returns:
        重写后的搜索查询。
    """
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("human", QUERY_REWRITE_PROMPT),
        ])
        chain = prompt | llm | StrOutputParser()
        rewritten = chain.invoke({"question": question}).strip()
        if rewritten:
            logger.info("查询重写: '%s...' → '%s'", question[:30], rewritten[:60])
            return rewritten
    except Exception as e:
        logger.warning("查询重写失败，使用原始查询: %s", e)
    return question


def _create_retrieval_tool(llm, top_k_override: Optional[int] = None) -> BaseTool:
    """创建知识库检索工具（含查询重写）。

    该工具供 Agent 在需要时调用，先重写用户查询，
    再从 Milvus 检索相关文档块。

    Args:
        llm: 语言模型实例，用于查询重写。
        top_k_override: 覆盖默认的 Top-K 值。

    Returns:
        配置好的 LangChain BaseTool。
    """

    @tool
    def knowledge_retrieval(query: str) -> str:
        """从企业知识库中检索与用户问题相关的文档内容。

        当用户询问关于文档内容、知识相关的问题时，应该使用此工具。
        输入应为用户的搜索问题或关键词。

        Args:
            query: 搜索关键词或问题。

        Returns:
            检索到的文档内容文本。
        """
        # 1. 查询重写
        rewritten_query = _query_rewrite(query, llm)

        k = top_k_override or settings.TOP_K
        mode = settings.RETRIEVAL_MODE

        # 2. 用重写后的查询检索
        if mode == "bm25":
            docs = vector_store_manager.bm25_search(rewritten_query, k=k)
        elif mode == "hybrid" and settings.BM25_ENABLED:
            docs = vector_store_manager.hybrid_search(rewritten_query, top_k=k)
        else:
            candidate_k = settings.RERANK_CANDIDATE_K if settings.RERANK_ENABLED else k
            docs = vector_store_manager.similarity_search(rewritten_query, k=candidate_k)

        if not docs:
            logger.info("重写查询无结果，尝试原始查询: '%s'", query[:60])
            if mode == "bm25":
                docs = vector_store_manager.bm25_search(query, k=k)
            elif mode == "hybrid" and settings.BM25_ENABLED:
                docs = vector_store_manager.hybrid_search(query, top_k=k)
            else:
                docs = vector_store_manager.similarity_search(query, k=k)

        if not docs:
            return "知识库中没有找到相关信息。"

        if mode != "bm25" and settings.RERANK_ENABLED and len(docs) > 1:
            try:
                docs = reranker.rerank(rewritten_query, docs, top_k=k)
            except Exception as e:
                logger.warning("Agent 重排序失败，使用原始检索结果: %s", e)

        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("doc_name", "未知文档")
            idx = doc.metadata.get("chunk_index", "?")
            results.append(
                f"[{i + 1}] 来源：{source}（第{idx}块）\n内容：{doc.page_content}"
            )
        return "\n\n---\n\n".join(results)

    return knowledge_retrieval


class RAGAgent:
    """RAG Agent，支持多轮对话和工具调用。

    使用 LangGraph 的 create_react_agent 创建 Agent，
    通过消息列表维护每个会话的多轮对话历史。
    """

    def __init__(self, llm):
        """初始化 Agent。

        Args:
            llm: 语言模型实例（需支持 Tool Calling）。
        """
        self._llm = llm
        # 默认工具实例（传入 LLM 用于查询重写）
        self._default_tool = _create_retrieval_tool(llm)
        self._agent = self._build_agent([self._default_tool])
        # 不同 top_k 对应的 Agent 缓存
        self._agent_cache: dict[int, CompiledStateGraph] = {}

    def _build_agent(self, tools: list) -> CompiledStateGraph:
        """构建 LangGraph React Agent。

        Args:
            tools: Agent 可用的工具列表。

        Returns:
            CompiledStateGraph 实例。
        """
        return create_agent(
            model=self._llm,
            tools=tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )


    def _parse_sources_from_tool_calls(
        self, messages: list
    ) -> list[dict]:
        """从消息列表中解析检索工具返回的来源信息。

        工具返回格式示例：
            [1] 来源：doc_name（第0块）
            内容：page_content
        """
        sources = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "knowledge_retrieval":
                content = msg.content
                if content and "知识库中没有找到" not in content:
                    for line in content.split("\n"):
                        # 匹配 "[N] 来源：doc_name（第X块）"
                        if "来源：" in line and "（" in line:
                            parts = line.split("来源：")
                            if len(parts) >= 2:
                                rest = parts[1]  # "doc_name（第X块）"
                                name_parts = rest.split("（")
                                doc_name = name_parts[0].strip()
                                chunk_info = name_parts[1].replace("）", "").strip() if len(name_parts) >= 2 else ""
                                # 从 "第3块" 中提取数字 3
                                try:
                                    chunk_index = int(''.join(filter(str.isdigit, chunk_info)))
                                except ValueError:
                                    chunk_index = 0
                                if doc_name and (doc_name, chunk_index) not in [(s["doc_name"], s["chunk_index"]) for s in sources]:
                                    sources.append({
                                        "doc_name": doc_name,
                                        "chunk_index": chunk_index,
                                        "content": "",
                                    })
        return sources

    @staticmethod
    def _build_message_list(
        question: str,
        history_messages: Optional[list[dict]] = None,
    ) -> list:
        """构建传递给 Agent 的完整消息列表，包含历史对话 + 当前问题。

        Args:
            question: 当前用户问题。
            history_messages: 历史消息列表（来自 SQLite 的 dict）。

        Returns:
            LangChain BaseMessage 列表。
        """
        messages: list = []
        if history_messages:
            for msg in history_messages:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        # 当前问题
        messages.append(HumanMessage(content=question))
        return messages

    def chat(
        self,
        question: str,
        session_id: str,
        top_k: Optional[int] = None,
        history_messages: Optional[list[dict]] = None,
    ) -> dict:
        """执行一次带记忆的对话。

        Args:
            question: 用户输入。
            session_id: 会话标识（仅用于日志）。
            top_k: 检索文档块数量（可选）。
            history_messages: 历史消息列表（来自 SQLite），用于恢复上下文。

        Returns:
            包含 answer 和 sources 的字典。
        """

        # 从缓存获取或构建指定 top_k 的 Agent
        if top_k is not None:
            if top_k not in self._agent_cache:
                self._agent_cache[top_k] = self._build_agent([_create_retrieval_tool(self._llm, top_k)])
            agent = self._agent_cache[top_k]
        else:
            agent = self._agent

        try:
            logger.info(
                "Agent 对话: session=%s, question='%s...' (history=%d条)",
                session_id,
                question[:50],
                len(history_messages) if history_messages else 0,
            )

            # 构建完整消息列表（历史 + 当前）
            input_messages = self._build_message_list(question, history_messages)
            # 每个请求用唯一 thread_id，避免 InMemorySaver 混淆状态
            request_id = f"{session_id}_{uuid.uuid4().hex[:8]}"

            # 执行 Agent
            result = agent.invoke(
                {"messages": input_messages},
                {"configurable": {"thread_id": request_id}},
            )

            # 提取回答
            result_messages = result.get("messages", [])
            answer = ""
            for msg in reversed(result_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    answer = msg.content
                    break

            # 解析来源
            sources = self._parse_sources_from_tool_calls(result_messages)

            return {
                "answer": answer,
                "session_id": session_id,
                "sources": sources,
            }

        except Exception as e:
            logger.error("Agent 执行失败: %s", e, exc_info=True)
            return {
                "answer": f"抱歉，处理您的问题时出现错误：{str(e)}",
                "session_id": session_id,
                "sources": [],
            }

    async def chat_stream(
        self,
        question: str,
        session_id: str,
        top_k: Optional[int] = None,
        history_messages: Optional[list[dict]] = None,
    ):
        """流式对话，通过异步生成器逐 token 产出 SSE 事件。

        Args:
            question: 用户输入。
            session_id: 会话标识。
            top_k: 检索文档块数量（可选）。
            history_messages: 历史消息列表（来自 SQLite），用于恢复上下文。

        Yields:
            dict: 包含 type 和对应数据的字典。
                  - {"type": "token", "content": str}   AI 生成的文本片段
                  - {"type": "done", "session_id": str, "sources": list}  完成事件
                  - {"type": "error", "content": str}  错误事件
        """


        # 从缓存获取或构建指定 top_k 的 Agent
        if top_k is not None:
            if top_k not in self._agent_cache:
                self._agent_cache[top_k] = self._build_agent([_create_retrieval_tool(self._llm, top_k)])
            agent = self._agent_cache[top_k]
        else:
            agent = self._agent


        logger.info(
            "Agent 流式对话: session=%s, question='%s...' (history=%d条)",
            session_id,
            question[:50],
            len(history_messages) if history_messages else 0,
        )

        try:
            # 构建完整消息列表（历史 + 当前）
            input_messages = self._build_message_list(question, history_messages)
            # 每个请求用唯一 thread_id，避免 InMemorySaver 混淆状态
            request_id = f"{session_id}_{uuid.uuid4().hex[:8]}"

            tool_contents: dict[str, str] = {}
            full_answer = ""

            async for msg_chunk, metadata in agent.astream(
                {"messages": input_messages},
                {"configurable": {"thread_id": request_id}},
                stream_mode="messages",
            ):
                # 记录日志方便调试
                node = metadata.get("langgraph_node", "unknown") if isinstance(metadata, dict) else "unknown"
                logger.debug(
                    "Stream chunk: node=%s, type=%s, content=%s...",
                    node,
                    type(msg_chunk).__name__,
                    str(getattr(msg_chunk, "content", ""))[:80] if hasattr(msg_chunk, "content") else "N/A",
                )

                # AI token —— 任何 AIMessageChunk 都直接产出，不依赖 node 名称
                if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
                    full_answer += msg_chunk.content
                    yield {"type": "token", "content": msg_chunk.content}
                    continue

                # 工具调用结果（ToolMessageChunk 等）
                if hasattr(msg_chunk, "content") and msg_chunk.content:
                    name = getattr(msg_chunk, "name", "knowledge_retrieval")
                    if name not in tool_contents:
                        tool_contents[name] = ""
                    tool_contents[name] += msg_chunk.content

            tool_messages = []
            for name, content in tool_contents.items():
                tool_messages.append(
                    ToolMessage(content=content, name=name, tool_call_id="")
                )
            sources = self._parse_sources_from_tool_calls(tool_messages)


            yield {
                "type": "done",
                "session_id": session_id,
                "sources": sources,
            }

        except Exception as e:
            logger.error("Agent 流式执行失败: %s", e, exc_info=True)
            yield {"type": "error", "content": str(e)}

    def clear_memory(self, session_id: str) -> None:
        """清除指定会话的记忆。"""
        if session_id in self._chat_histories:
            del self._chat_histories[session_id]
            logger.info("已清除会话记忆: %s", session_id)
