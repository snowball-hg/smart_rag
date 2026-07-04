# 源码快速梳理指南

## 阅读顺序建议

如果你是第一次接触这个项目，建议按以下顺序阅读源码，从外到内、从入口到细节：

```
① config.py       →  了解系统有哪些配置项
② schemas.py      →  了解 API 收发什么数据
③ loader.py       →  文档怎么被处理的
④ vector_store.py →  向量怎么存、怎么查
⑤ retriever.py    →  RAG 问答流的完整链路
⑥ agent.py        →  Agent 如何自主决策和调用工具
⑦ main.py         →  所有 API 端点如何串联
```

---

## 一、项目架构总览

```
┌─────────────────────────────────────────────────────┐
│                    客户端 (curl/前端)                  │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP (RESTful API)
                      ▼
┌─────────────────────────────────────────────────────┐
│                FastAPI 服务层 (main.py)                │
│  ┌──────────┬──────────┬──────────┬────────────────┐ │
│  │ /upload  │ /query   │  /chat   │  /health       │ │
│  │          │          │          │  /documents/*  │ │
│  └────┬─────┴────┬─────┴────┬─────┴────────────────┘ │
└───────┼──────────┼──────────┼────────────────────────┘
        │          │          │
        ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ loader   │ │ retriever│ │  agent   │
│ 拆文档    │ │  RAG 链  │ │  智能代理  │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
     ▼            ▼            ▼
┌─────────────────────────────────────────────────────┐
│              vector_store.py (向量引擎)                │
│         ┌─────────────────┐   ┌─────────────────┐   │
│         │  Milvus 向量库   │   │  Embedding 模型  │   │
│         └─────────────────┘   └─────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 数据流向

**上传流程**：`HTTP 文件 → main.py → loader.py (分块) → vector_store.py (向量化+存储)`

**问答流程**：`问题 → retriever.py (向量检索+LLM回答) → 返回结果`

**对话流程**：`问题 → agent.py (Agent决策→可能检索→LLM回答+记忆) → 返回结果`

---

## 二、核心模块详解

### 1. [config.py](file:///d:/project/AI/smart_rag/app/config.py) — 配置中心

**作用**：集中管理所有配置项，从 `.env` 文件中读取。

**关键设计**：

```python
class Settings:
    DEEPSEEK_API_KEY: str        # LLM 密钥
    MILVUS_HOST: str             # Milvus 地址
    EMBEDDING_PROVIDER: str      # "local" 或 "qwen"，方便切换
    CHUNK_SIZE: int              # 分块大小
    TOP_K: int                   # 检索数量
```

> **扩展点**：如果要增加新的配置项，只需在此类中添加一个字段，并在 `.env.example` 中添加对应的注释即可。

### 2. [schemas.py](file:///d:/project/AI/smart_rag/app/schemas.py) — 数据契约

**作用**：定义 API 请求和响应的数据结构，让 FastAPI 自动完成请求校验和响应序列化。

**核心模型**：

| 模型 | 用途 | 关键字段 |
|------|------|----------|
| `QueryRequest` | 纯问答请求 | `question`, `top_k?` |
| `ChatRequest` | 对话请求 | `question`, `session_id`, `top_k?` |
| `UploadResponse` | 上传响应 | `doc_id`, `chunk_count` |
| `SourceDocument` | 引用来源 | `doc_name`, `chunk_index`, `content` |
| `HealthResponse` | 健康检查 | `status`, `llm_configured`, `milvus_connected` |

> **设计意图**：`session_id` 放在请求体中而非路径参数，方便客户端管理多个并发会话。

### 3. [loader.py](file:///d:/project/AI/smart_rag/app/loader.py) — 文档处理管线

**作用**：将上传的文件转换为带元数据的文档块列表。

**核心流程**：

```
文件路径 → get_loader() → load_document() → split_documents() → process_file()
                    ↑              ↑                ↑
             根据扩展名选加载器    加载原始内容     RecursiveCharacterTextSplitter 分块
```

**关键函数**：

```python
def process_file(file_path: str | Path) -> list[Document]:
    """完整流程：加载 → 分块 → 添加元数据 (doc_id, doc_name, chunk_index, upload_time)"""
```

**文件类型支持映射**（`LOADER_MAP`）：

```python
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md":  UnstructuredMarkdownLoader,
    ".mdx": UnstructuredMarkdownLoader,
}
```

> **扩展点**：如果要支持 `.docx`、`.csv` 等格式，在 `LOADER_MAP` 中添加对应的加载器即可。LangChain 社区提供了大量现成的 DocumentLoader。

### 4. [vector_store.py](file:///d:/project/AI/smart_rag/app/vector_store.py) — 向量引擎

**作用**：封装 Embedding 模型初始化和 Milvus 的增删查操作。

**两套 Embedding 机制**：

```python
def _create_embeddings() -> Embeddings:
    if settings.EMBEDDING_PROVIDER == "qwen":
        return _create_qwen_embeddings()   # 千问 API
    return _create_local_embeddings()      # 本地 Sentence-Transformers
```

> **local 模式**：使用 `HuggingFaceEmbeddings`，默认模型 `BAAI/bge-large-zh-v1.5`
> **qwen 模式**：使用 `DashScopeEmbeddings`，需配置 API Key
> 如果 qwen 模式未配置 API Key，自动回退到 local 模式。

**核心方法**（`VectorStoreManager` 类）：

| 方法 | 作用 | 备注 |
|------|------|------|
| `add_documents(docs)` | 存入文档块 | 自动向量化并写入 Milvus |
| `similarity_search(query, k)` | 向量检索 | 返回最相似的 k 个文档块 |
| `as_retriever(k)` | 获取检索器 | 供 RAG 链和 Agent 使用 |
| `delete_by_doc_id(doc_id)` | 按文档 ID 删除 | 删除一个文档的所有块 |
| `delete_collection()` | 清空全部 | 删除集合后重新创建 |

> **单例模式**：`vector_store_manager = VectorStoreManager()` 在模块级别实例化，全局共享。

### 5. [retriever.py](file:///d:/project/AI/smart_rag/app/retriever.py) — RAG 问答链

**作用**：将「检索 + Prompt 组装 + LLM 生成」串联成一条完整的 RAG 链。

**核心组件**：

```python
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的文档智能助手。请根据以下检索到的文档内容回答问题...\n检索到的文档内容：\n{context}"),
    ("human", "{question}"),
])
```

**RAG 链构建**（LCEL 语法）：

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)
```

这个链的执行流程：`问题 → 检索文档 → 格式化为上下文 → 填入 Prompt → LLM 生成 → 解析为文本`

> **format_docs()**：将 Document 列表格式化为带来源标记的文本，供 LLM 参考。
> **format_sources()**：将 Document 列表格式化为 API 响应的来源列表。

### 6. [agent.py](file:///d:/project/AI/smart_rag/app/agent.py) — 智能代理

**作用**：构建具有自主决策能力的 Agent，支持多轮对话记忆。

**Agent 系统 Prompt**：

```python
AGENT_SYSTEM_PROMPT = (
    "你是一个智能文档助手...有以下能力："
    "1. 检索知识库中的文档内容来回答问题"
    "2. 基于检索结果进行推理和总结"
    "3. 记住对话历史，进行多轮交流"
    "...当用户提问需要查阅文档时，请调用检索工具..."
)
```

**关键设计**：

```python
class RAGAgent:
    def __init__(self, llm):
        self._memories: dict[str, ConversationBufferMemory]  # 每个 session_id 一个记忆

    def chat(self, question, session_id, top_k=None) -> dict:
        memory = self._get_memory(session_id)
        result = self._agent_executor.invoke({"input": question}, config={"memory": memory})
```

> **多轮记忆实现**：用字典 `{session_id: ConversationBufferMemory}` 管理多个会话。每个会话独立记忆。
> **工具调用**：使用 `create_tool_calling_agent`，依赖 LLM 的原生 Tool Calling 能力。

**已有工具**：

| 工具名 | 功能 | 触发条件 |
|--------|------|----------|
| `knowledge_retrieval` | 从 Milvus 检索文档 | Agent 判断需要查询知识库时 |

> **扩展点**：如需添加新工具，在 `_create_retrieval_tool()` 方法旁定义新的 Tool，然后在 `_setup_agent()` 的 `tools` 列表中加入即可。

### 7. [main.py](file:///d:/project/AI/smart_rag/app/main.py) — 服务入口

**作用**：FastAPI 应用，定义所有 HTTP 端点，串联各模块。

**端点一览**：

| 端点 | 方法 | 功能 | 调用哪个模块 |
|------|------|------|------------|
| `/health` | GET | 健康检查 | 直接查询 VectorStoreManager |
| `/upload` | POST | 上传文档 | loader.py + vector_store.py |
| `/query` | POST | RAG 问答 | retriever.py |
| `/chat` | POST | 对话 Agent | agent.py |
| `/documents/{doc_id}` | DELETE | 删除文档 | vector_store.py |
| `/documents` | DELETE | 清空所有 | vector_store.py |

**启动生命周期**：

```python
@app.on_event("startup")
async def startup_event():
    # 1. 初始化 LLM (如果配置了 API Key)
    # 2. 初始化 RAGRetriever (RAG 链)
    # 3. 初始化 RAGAgent (对话代理)
    # 4. 创建上传目录
```

> **降级策略**：如果 LLM 未配置，`/query` 和 `/chat` 返回 503，但 `/health` 和 `/upload` 仍可正常使用。

---

## 三、扩展指南

### 3.1 切换 LLM 模型

修改 [config.py](file:///d:/project/AI/smart_rag/app/config.py) 中的 `DEEPSEEK_MODEL_NAME` 即可：

```env
DEEPSEEK_MODEL_NAME=deepseek-v4-pro    # 切换到 Pro 版本
```

如果要切换到本地模型（如 Ollama 部署的 Qwen），在 `_init_llm()` 中添加分支：

```python
# main.py 中
if LLM_PROVIDER == "ollama":
    from langchain_ollama import ChatOllama
    return ChatOllama(model="qwen2.5:7b")
```

### 3.2 切换 Embedding 模型

修改 `.env`：

```env
# 切换到千问 API
EMBEDDING_PROVIDER=qwen
QWEN_EMBEDDING_API_KEY=你的千问key

# 或切换到其他本地模型
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=shibing624/text2vec-base-chinese
```

### 3.3 添加新工具

在 [agent.py](file:///d:/project/AI/smart_rag/app/agent.py) 中添加：

```python
def _create_my_tool() -> Tool:
    def my_function(input: str) -> str:
        # 你的工具逻辑
        return "结果"

    return Tool(
        name="my_tool",
        description="描述工具用途，Agent 据此决定是否调用",
        func=my_function,
    )

# 然后在 _setup_agent() 的 tools 列表中加入
tools = [_create_retrieval_tool(), _create_my_tool()]
```

### 3.4 支持流式输出

将 `/query` 和 `/chat` 端点改为：

```python
from fastapi.responses import StreamingResponse
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    # 使用 streaming LLM
    streaming_llm = ChatOpenAI(..., streaming=True)
    # 返回 StreamingResponse
```

---

## 四、模块依赖关系图

```
main.py
  ├── config.py        (配置读取)
  ├── logger.py        (日志)
  ├── schemas.py       (数据模型)
  ├── loader.py        (文档处理)
  │     └── config.py
  ├── vector_store.py  (向量存储)
  │     ├── config.py
  │     └── logger.py
  ├── retriever.py     (RAG 问答)
  │     ├── config.py
  │     ├── logger.py
  │     └── vector_store.py
  └── agent.py         (Agent 代理)
        ├── config.py
        ├── logger.py
        └── vector_store.py
```

> 所有模块都依赖 `config.py`，修改配置可影响全局行为。
> `vector_store.py` 是底层依赖，`retriever.py` 和 `agent.py` 都依赖它。
> `loader.py` 独立于其他业务模块，只依赖配置。
