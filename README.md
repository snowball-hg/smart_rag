<div align="center">

# Smart RAG

**企业级检索增强生成（RAG）Agent 系统**

基于 LangChain + Milvus + FastAPI 构建，支持文档智能处理、混合检索、重排序与多轮对话。

</div>

---

## 特性

- **智能文档处理** — PDF 版面分析（Docling）、扫描件 OCR（RapidOCR）、智能标题感知切分，支持 PDF/Office/HTML/TXT/MD
- **多模式检索** — 向量语义检索、BM25 关键词检索、RRF 混合检索融合，灵活切换
- **重排序** — CrossEncoder 对检索结果二次精排，提升答案准确性
- **Agent 对话** — LangGraph ReAct Agent，自主判断是否需要检索，带多轮对话记忆
- **查询重写** — Agent 模式下自动将口语化问题改写为利于检索的关键词
- **流式响应** — SSE 协议支持 token 流式输出，前端实时展示
- **优雅降级** — 高级处理器依赖缺失时自动回退到基础流程

---

## 快速开始

### 前置依赖

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip
- Milvus 向量数据库（Docker 启动）
- DeepSeek API Key

### 1. 启动 Milvus

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 DeepSeek API Key：

```ini
DEEPSEEK_API_KEY=sk-your-key-here
```

### 3. 安装依赖

```bash
uv sync
```

### 4. 启动服务

```bash
uv run python run.py
```

服务启动后访问 `http://localhost:8100/docs` 查看 Swagger API 文档。

---

## API 概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/upload` | POST | 上传文档并构建索引 |
| `/query` | POST | RAG 问答（无记忆） |
| `/chat` | POST | Agent 对话（带多轮记忆） |
| `/chat/stream` | POST | 流式对话（SSE） |
| `/documents/{doc_id}` | DELETE | 删除指定文档 |
| `/documents` | DELETE | 清空所有数据 |

### 示例：上传文档并问答

```bash
# 上传
curl -X POST http://localhost:8100/upload \
  -F "file=@document.pdf"

# 问答
curl -X POST http://localhost:8100/query \
  -H "Content-Type: application/json" \
  -d '{"question": "文档的核心内容是什么？"}'

# 对话（带记忆）
curl -X POST http://localhost:8100/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "刚才提到的技术细节是什么？", "session_id": "session-001"}'
```

---

## 系统架构

```
┌──────────┐    ┌─────────────────────────────────────┐    ┌──────────┐
│ Frontend │───▶│           FastAPI Backend            │───▶│  Milvus  │
│ (Vue 3)  │    │                                     │    │ (向量库)  │
└──────────┘    │  ┌───────┐  ┌────────┐  ┌────────┐ │    └──────────┘
                │  │Loader │─▶│Retriever│─▶│  Agent │ │    ┌──────────┐
                │  └───┬───┘  └────────┘  └────────┘ │───▶│ DeepSeek │
                │      │                              │    │  (LLM)   │
                │  ┌───▼──────────────────┐           │    └──────────┘
                │  │ Document Processor   │           │
                │  │ 文件检测→解析→清洗→切分│           │
                │  └──────────────────────┘           │
                └─────────────────────────────────────┘
```

### 核心流程

**文档摄入**：上传 → 文件类型检测 → 解析（Docling/OCR）→ 清洗 → 智能切分 → Embedding → 写入 Milvus

**RAG 问答**：问题 → 向量/BM25/混合检索 → RRF 融合 → 重排序 → 上下文扩展 → LLM 生成

**Agent 对话**：查询重写 → LangGraph ReAct 循环 → 检索工具调用 → 多轮记忆管理

---

## 配置参考

核心配置项（通过 `.env` 或环境变量设置）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DEEPSEEK_API_KEY` | — | **必填**，DeepSeek API Key |
| `DEEPSEEK_MODEL_NAME` | `deepseek-v4-flash` | LLM 模型名称 |
| `MILVUS_HOST` | `localhost` | Milvus 地址 |
| `MILVUS_PORT` | `19530` | Milvus 端口 |
| `EMBEDDING_PROVIDER` | `local` | `local`（本地模型）/ `qwen`（千问 API） |
| `LOCAL_EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | 本地 Embedding 模型 |
| `RETRIEVAL_MODE` | `hybrid` | 检索模式：`vector` / `bm25` / `hybrid` |
| `TOP_K` | `5` | 检索返回的文档块数 |
| `CHUNK_SIZE` | `1000` | 文档分块大小（字符） |
| `CHUNK_OVERLAP` | `200` | 分块重叠大小（字符） |

---

## 项目结构

```
smart_rag/
├── app/                        # 后端核心代码
│   ├── main.py                 # FastAPI 应用入口及 API 路由
│   ├── config.py               # 全局配置中心
│   ├── schemas.py              # 请求/响应数据模型
│   ├── loader.py               # 文档加载与分块
│   ├── vector_store.py         # Milvus 向量存储封装
│   ├── retriever.py            # RAG 检索与生成
│   ├── reranker.py             # CrossEncoder 重排序
│   ├── agent.py                # LangGraph ReAct Agent
│   └── processors/             # 文档预处理管线
│       ├── pipeline.py         # 管线编排器
│       ├── file_router.py      # 文件类型检测
│       ├── pdf_processor.py    # PDF 版面分析 & OCR
│       ├── smart_chunker.py    # 智能文档切分
│       └── data_cleaner.py     # 数据清洗
├── front/                      # Vue 3 前端
├── docs/                       # 文档
├── run.py                      # 启动脚本
├── .env.example                # 环境变量模板
└── pyproject.toml              # 项目元数据
```

---

## 技术栈

- **框架**：FastAPI, LangChain, LangGraph
- **向量库**：Milvus（稠密向量 IVF_FLAT + 稀疏向量 BM25）
- **LLM**：DeepSeek（OpenAI 兼容接口）
- **Embedding**：BAAI/bge-large-zh-v1.5 / Qwen3-Embedding（本地），或 千问 API
- **文档处理**：Docling, RapidOCR, Unstructured
- **重排序**：BAAI/bge-reranker-v2-m3
- **前端**：Vue 3, Vite

---

## 许可证

MIT
