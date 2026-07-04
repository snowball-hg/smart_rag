<template>
  <div class="chat-page">
    <h1 class="page-title">智能对话</h1>
    <p class="page-desc">与 AI 代理进行多轮对话，代理可自主检索知识库</p>

    <div class="chat-card">
      <!-- 会话管理 -->
      <div class="session-bar">
        <span class="session-label">当前会话: {{ sessionId }}</span>
        <div class="session-actions">
          <label class="stream-toggle" title="流式模式下 AI 逐 token 输出，非流式模式下等待完整结果">
            <input type="checkbox" v-model="streamMode" />
            <span class="toggle-track"><span class="toggle-thumb"></span></span>
            <span class="toggle-label">流式</span>
          </label>
          <button class="btn btn-sm" @click="newSession">新建会话</button>
        </div>
      </div>

      <!-- 消息列表 -->
      <div class="message-list" ref="messageList">
        <div v-if="messages.length === 0" class="empty-chat">
          <div class="empty-icon">💬</div>
          <p>开始一段新的对话吧！</p>
        </div>
        <div
          v-for="(msg, i) in messages"
          :key="i"
          class="message"
          :class="msg.role"
        >
          <div class="msg-avatar">{{ msg.role === 'user' ? '👤' : '🤖' }}</div>
          <div class="msg-content">
            <div class="msg-text" v-html="renderMarkdown(msg.content)"></div>
            <div v-if="msg.sources && msg.sources.length > 0" class="msg-sources">
              <details>
                <summary>参考来源 ({{ msg.sources.length }})</summary>
                <div class="source-item" v-for="(src, j) in msg.sources" :key="j">
                  <span class="source-name">{{ src.doc_name }}</span> 块 #{{ src.chunk_index }}
                </div>
              </details>
            </div>
          </div>
        </div>
        <div v-if="loading" class="message assistant">
          <div class="msg-avatar">🤖</div>
          <div class="msg-content">
            <div class="msg-text typing">思考中...</div>
          </div>
        </div>
      </div>

      <!-- 输入区 -->
      <div class="input-area">
        <input
          v-model="question"
          class="chat-input"
          placeholder="输入你的消息..."
          @keyup.enter="sendMessage"
        />
        <button class="btn btn-primary" :disabled="!question.trim() || loading" @click="sendMessage">
          发送
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { marked } from 'marked'
import { chatWithAgent, chatStream } from '../api/index.js'

function renderMarkdown(text) {
  if (!text) return ''
  return marked.parse(text, { breaks: true, gfm: true })
}

const sessionId = ref('session-' + Date.now())
const question = ref('')
const loading = ref(false)
const messages = ref([])
const messageList = ref(null)
const streamMode = ref(false)

function newSession() {
  sessionId.value = 'session-' + Date.now()
  messages.value = []
}

async function sendMessage() {
  if (!question.value.trim() || loading.value) return
  const q = question.value
  messages.value.push({ role: 'user', content: q })
  question.value = ''
  loading.value = true
  scrollToBottom()

  if (streamMode.value) {
    // 流式模式
    const msgIdx = messages.value.length
    messages.value.push({ role: 'assistant', content: '' })
    scrollToBottom()

    try {
      const res = await chatStream(q, sessionId.value)
      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: `请求失败 (${res.status})` }))
        throw new Error(errData.detail)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.type === 'token') {
                messages.value[msgIdx].content += data.content
                scrollToBottom()
              } else if (data.type === 'done') {
                if (data.session_id) sessionId.value = data.session_id
                messages.value[msgIdx].sources = data.sources
              } else if (data.type === 'error') {
                messages.value[msgIdx].content = data.content
              }
            } catch { /* 忽略解析失败的帧 */ }
          }
        }
      }
    } catch (e) {
      const lastMsg = messages.value[messages.value.length - 1]
      if (lastMsg && lastMsg.role === 'assistant' && !lastMsg.content) {
        lastMsg.content = e.message || '请求失败，请重试。'
      } else {
        messages.value.push({
          role: 'assistant',
          content: e.message || '请求失败，请重试。',
        })
      }
    } finally {
      loading.value = false
      scrollToBottom()
    }
  } else {
    // 非流式模式
    try {
      const res = await chatWithAgent(q, sessionId.value)
      messages.value.push({
        role: 'assistant',
        content: res.data.answer,
        sources: res.data.sources,
      })
    } catch (e) {
      messages.value.push({
        role: 'assistant',
        content: e.response?.data?.detail || '请求失败，请重试。',
      })
    } finally {
      loading.value = false
      scrollToBottom()
    }
  }
}

function scrollToBottom() {
  nextTick(() => {
    if (messageList.value) {
      messageList.value.scrollTop = messageList.value.scrollHeight
    }
  })
}
</script>

<style scoped>
.chat-page {
  max-width: 800px;
  height: calc(100vh - 48px);
  display: flex;
  flex-direction: column;
}

.page-title {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 8px;
}

.page-desc {
  color: #888;
  margin-bottom: 16px;
}

.chat-card {
  flex: 1;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.session-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid #f0f0f0;
  background: #fafafa;
}

.session-label {
  font-size: 13px;
  color: #666;
}

.btn {
  padding: 6px 16px;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  background: #fff;
  transition: all 0.2s;
}

.btn:hover {
  border-color: #1890ff;
  color: #1890ff;
}

.btn-primary {
  background: #1890ff;
  color: #fff;
  border-color: #1890ff;
}

.btn-primary:hover {
  background: #40a9ff;
  color: #fff;
}

.btn-primary:disabled {
  background: #91d5ff;
  border-color: #91d5ff;
  cursor: not-allowed;
}

.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.empty-chat {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #ccc;
}

.empty-icon {
  font-size: 64px;
  margin-bottom: 16px;
}

.message {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}

.message.user {
  flex-direction: row-reverse;
}

.msg-avatar {
  font-size: 28px;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.msg-content {
  max-width: 70%;
}

.user .msg-content {
  text-align: right;
}

.msg-text {
  padding: 10px 14px;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

.assistant .msg-text {
  background: #f0f0f0;
  color: #333;
}

.user .msg-text {
  background: #1890ff;
  color: #fff;
}

.typing {
  animation: pulse 1.2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.msg-sources {
  margin-top: 8px;
  text-align: left;
}

.msg-sources details {
  font-size: 12px;
  color: #999;
  background: #fafafa;
  border-radius: 4px;
  padding: 4px 8px;
}

.msg-sources summary {
  cursor: pointer;
}

.source-item {
  padding: 2px 0;
  color: #666;
}

/* Markdown 样式 */
.msg-text :deep(pre) {
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 12px 16px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 13px;
  line-height: 1.5;
  margin: 8px 0;
}

.msg-text :deep(code) {
  background: #e8e8e8;
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 13px;
  font-family: 'Consolas', 'Monaco', monospace;
}

.msg-text :deep(pre code) {
  background: none;
  padding: 0;
  border-radius: 0;
}

.msg-text :deep(p) {
  margin: 4px 0;
}

.msg-text :deep(ul),
.msg-text :deep(ol) {
  padding-left: 20px;
  margin: 4px 0;
}

.msg-text :deep(li) {
  margin: 2px 0;
}

.msg-text :deep(h1),
.msg-text :deep(h2),
.msg-text :deep(h3),
.msg-text :deep(h4) {
  margin: 8px 0 4px;
  font-weight: 600;
}

.msg-text :deep(h1) { font-size: 18px; }
.msg-text :deep(h2) { font-size: 16px; }
.msg-text :deep(h3) { font-size: 15px; }

.msg-text :deep(blockquote) {
  border-left: 3px solid #1890ff;
  padding: 4px 12px;
  margin: 8px 0;
  color: #666;
  background: #f8f9fa;
  border-radius: 0 4px 4px 0;
}

.msg-text :deep(table) {
  border-collapse: collapse;
  margin: 8px 0;
  font-size: 13px;
}

.msg-text :deep(th),
.msg-text :deep(td) {
  border: 1px solid #ddd;
  padding: 6px 10px;
  text-align: left;
}

.msg-text :deep(th) {
  background: #f0f0f0;
  font-weight: 600;
}

.msg-text :deep(hr) {
  border: none;
  border-top: 1px solid #ddd;
  margin: 12px 0;
}

.msg-text :deep(a) {
  color: #1890ff;
  text-decoration: none;
}

.msg-text :deep(a:hover) {
  text-decoration: underline;
}

.user .msg-text :deep(code) {
  background: rgba(255,255,255,0.2);
}

.user .msg-text :deep(a) {
  color: #fff;
  text-decoration: underline;
}

.source-name {
  color: #1890ff;
  font-weight: 500;
}

.input-area {
  display: flex;
  gap: 12px;
  padding: 12px 16px;
  border-top: 1px solid #f0f0f0;
}

.chat-input {
  flex: 1;
  padding: 10px 16px;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.chat-input:focus {
  border-color: #1890ff;
  box-shadow: 0 0 0 2px rgba(24,144,255,0.2);
}

.session-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.stream-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  user-select: none;
  font-size: 13px;
  color: #666;
}

.stream-toggle input {
  display: none;
}

.toggle-track {
  width: 36px;
  height: 20px;
  background: #d9d9d9;
  border-radius: 10px;
  position: relative;
  transition: background 0.2s;
}

.toggle-thumb {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 16px;
  height: 16px;
  background: #fff;
  border-radius: 50%;
  transition: transform 0.2s;
  box-shadow: 0 1px 2px rgba(0,0,0,0.15);
}

.stream-toggle input:checked + .toggle-track {
  background: #1890ff;
}

.stream-toggle input:checked + .toggle-track .toggle-thumb {
  transform: translateX(16px);
}

.toggle-label {
  font-size: 13px;
}
</style>
