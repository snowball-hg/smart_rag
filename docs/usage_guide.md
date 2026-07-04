# RAG Agent 系统使用指南

## 目录

1. [环境准备](#1-环境准备)
2. [配置说明](#2-配置说明)
3. [启动服务](#3-启动服务)
4. [API 使用详解](#4-api-使用详解)
5. [常见场景](#5-常见场景)
6. [故障排查](#6-故障排查)

---

## 1. 环境准备

### 1.1 确保 Milvus 已启动

本系统依赖 Milvus 向量数据库。如果你还没有启动 Milvus，通过 Docker 启动：

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest
```

验证 Milvus 是否正常运行：

```bash
curl http://localhost:9091/health
# 返回 {"status":"ok"} 即表示正常
```

### 1.2 安装项目依赖

项目使用 `uv` 管理依赖，执行：

```bash
cd d:\project\AI\smart_rag
uv sync
```

> 如果安装速度慢，可配置国内镜像源：在 `pyproject.toml` 中将 `index-url` 改为 `https://mirrors.aliyun.com/pypi/simple/`。

---

## 2. 配置说明

项目根目录下的 `.env` 文件是唯一需要你手动编辑的配置文件。核心配置项：

```ini
# ===== 必填：DeepSeek API Key =====
DEEPSEEK_API_KEY=sk-你的key         # ← 替换为你的真实 Key

# ===== Milvus 连接（一般不用改） =====
MILVUS_HOST=localhost
MILVUS_PORT=19530

# ===== Embedding 模型（默认用本地模型） =====
EMBEDDING_PROVIDER=local             # 可选: local / qwen
LOCAL_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5

# ===== 文档分块参数 =====
CHUNK_SIZE=1000                      # 每块最大字符数
CHUNK_OVERLAP=200                    # 相邻块重叠字符数

# ===== 检索参数 =====
TOP_K=5                              # 每次检索返回的文档块数
```

> **首次使用只需配置 `DEEPSEEK_API_KEY` 即可启动。** 其他参数都有默认值，后续可按需调整。

### 配置项速查表

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DEEPSEEK_API_KEY` | 空 | **必填**，DeepSeek 平台申请的 API Key |
| `DEEPSEEK_MODEL_NAME` | `deepseek-v4-flash` | 模型名称，一般不动 |
| `MILVUS_HOST` | `localhost` | Milvus 服务器地址 |
| `MILVUS_PORT` | `19530` | Milvus gRPC 端口 |
| `MILVUS_COLLECTION` | `rag_documents` | 向量集合名称 |
| `EMBEDDING_PROVIDER` | `local` | 嵌入模型类型：`local`(本地) / `qwen`(千问 API) |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-large-zh-v1.5` | 本地嵌入模型名称 |
| `CHUNK_SIZE` | `1000` | 文档分块大小（字符数） |
| `CHUNK_OVERLAP` | `200` | 分块重叠大小（字符数） |
| `TOP_K` | `5` | 每次检索返回的文档块数量 |
| `LOG_LEVEL` | `INFO` | 日志级别：DEBUG / INFO / WARNING / ERROR |

---

## 3. 启动服务

### 开发模式启动（推荐）

```bash
cd d:\project\AI\smart_rag
uv run python run.py
```

看到如下日志即启动成功：

```
[2026-06-30 17:40:00] INFO     rag_system - app/main.py:xx - RAG Agent 服务启动中...
[2026-06-30 17:40:00] INFO     rag_system - app/main.py:xx - LLM 模型: deepseek-v4-flash
[2026-06-30 17:40:00] INFO     rag_system - app/main.py:xx - Milvus 地址: localhost:19530
```

### 生产模式启动

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 访问 API 文档

启动后浏览器打开：**[http://localhost:8000/docs](http://localhost:8000/docs)**

这是 Swagger UI 交互式文档页面，你可以直接在页面上测试所有 API。

---

## 4. API 使用详解

### 4.1 健康检查

```bash
curl http://localhost:8000/health
```

响应示例：

```json
{
  "status": "healthy",
  "llm_configured": true,
  "milvus_connected": true,
  "vector_count": 0,
  "timestamp": "2026-06-30T17:40:00"
}
```

> `status` 字段：`healthy` 表示一切正常；`degraded` 表示 LLM 或 Milvus 连接异常。

---

### 4.2 上传文档

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/document.pdf"
```

支持的文件格式：**PDF、TXT、Markdown（.md / .mdx）**

响应示例：

```json
{
  "doc_id": "a1b2c3d4e5f6",
  "doc_name": "document.pdf",
  "chunk_count": 12,
  "message": "文档 'document.pdf' 处理完成，共 12 个文档块"
}
```

> `doc_id` 是文档的唯一标识，后续删除操作需要用到。

**上传多个文件**：需要多次调用 `/upload`，每次上传一个文件。

---

### 4.3 纯问答（RAG，无会话记忆）

每次提问独立，不保留上下文：

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "文档中提到了哪些关键技术？",
    "top_k": 5
  }'
```

`top_k` 是可选的，默认为配置中的 `TOP_K` 值。

响应示例：

```json
{
  "answer": "根据文档内容，主要涉及以下关键技术：...",
  "sources": [
    {
      "doc_name": "document.pdf",
      "chunk_index": 0,
      "content": "文档块内容的摘要..."
    }
  ]
}
```

---

### 4.4 对话（Agent 模式，带多轮记忆）

带会话上下文，Agent 可自主决定是否检索：

```bash
# 第一轮对话
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "你好，请介绍一下你自己",
    "session_id": "my-session-001"
  }'

# 第二轮对话（同一个 session_id，Agent 记得上一轮的内容）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "刚才说到的技术能详细解释一下吗？",
    "session_id": "my-session-001"
  }'
```

> **`session_id` 的作用**：同一个 `session_id` 的多轮对话会共享记忆。不同 `session_id` 的对话互相隔离。你可以为每个用户或每次会话分配不同的 `session_id`。

---

### 4.5 删除指定文档

```bash
curl -X DELETE http://localhost:8000/documents/{doc_id}
```

将 `{doc_id}` 替换为上传播返回的 `doc_id`。

响应示例：

```json
{
  "message": "文档 a1b2c3d4e5f6 已删除",
  "deleted_count": 12
}
```

---

### 4.6 清空所有数据

```bash
curl -X DELETE http://localhost:8000/documents
```

> ⚠️ **谨慎操作！** 此操作将删除向量库中的所有文档块，不可恢复。

---

## 5. 常见场景

### 场景一：先上传文档，然后问答

```bash
# 1. 上传文档
curl -X POST http://localhost:8000/upload -F "file=@report.pdf"

# 2. 保存返回的 doc_id（例如: "abc123"）

# 3. 提问
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "这份报告的核心结论是什么？"}'

# 4. 追问（用 chat 端点带记忆）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "这些数据的来源可靠吗？", "session_id": "my-session"}'

# 5. 不需要这份文档了，删除它
curl -X DELETE http://localhost:8000/documents/abc123
```

### 场景二：日常对话 + 知识库检索

```bash
# Agent 会判断是否需要检索知识库
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "今天有什么新知识可以分享？", "session_id": "daily-1"}'

# 闲聊场景，Agent 不会检索
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "你好，今天天气真好", "session_id": "daily-1"}'
```

---

## 6. 故障排查

### 6.1 服务启动失败

| 现象 | 原因 | 解决方法 |
|------|------|----------|
| `LLM 未配置` | `DEEPSEEK_API_KEY` 未填或填了默认值 | 编辑 `.env` 填入真实 API Key |
| `Milvus 连接失败` | Milvus 未启动或地址不对 | 检查 `docker ps` 确认 Milvus 在运行 |
| `端口被占用` | 8000 端口已被其他程序使用 | 修改 `run.py` 中的 port，或用 `--port` 指定 |

### 6.2 上传报错

| 现象 | 原因 | 解决方法 |
|------|------|----------|
| `不支持的文件类型` | 上传了 PDF/TXT/MD 以外的格式 | 目前仅支持 PDF、TXT、Markdown |
| `文件内容为空或无法解析` | 文件损坏或内容为空 | 检查源文件是否正常 |

### 6.3 问答效果不好

| 现象 | 可能原因 | 解决方法 |
|------|----------|----------|
| 回答与文档无关 | Top-K 太小，没检索到相关内容 | 调大 `TOP_K`（`.env` 中） |
| 回答太简短 | 分块粒度太大 | 调小 `CHUNK_SIZE`（`.env` 中） |
| 回答编造信息 | LLM 幻觉 | 在 Prompt 中加强约束（`retriever.py` 中的 `RAG_PROMPT`） |

### 6.4 查看日志

日志文件位于 `logs/rag_system.log`，包含详细的运行信息。设置为 `LOG_LEVEL=DEBUG` 可以获得更详细的调试信息。
