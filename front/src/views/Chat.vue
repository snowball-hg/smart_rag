<template>
  <div class="chat-page">
    <!-- 历史会话侧边栏 -->
    <div class="session-sidebar" :class="{ collapsed: sidebarCollapsed }">
      <div class="sidebar-header">
        <button class="btn-new" @click="createNewSession">
          <span class="btn-icon">+</span>
          <span v-if="!sidebarCollapsed">新建对话</span>
        </button>
        <button class="btn-toggle" @click="sidebarCollapsed = !sidebarCollapsed" :title="sidebarCollapsed ? '展开会话列表' : '收起会话列表'">
          {{ sidebarCollapsed ? '☰' : '◀' }}
        </button>
      </div>
      <div class="session-list" v-if="!sidebarCollapsed">
        <div
          v-for="sess in sessions"
          :key="sess.id"
          class="session-item"
          :class="{ active: sess.id === currentSessionId }"
          @click="switchSession(sess.id)"
        >
          <div class="session-info">
            <div class="session-title">{{ sess.title }}</div>
          </div>
          <div class="session-actions" v-if="sess.id === currentSessionId">
            <button class="btn-icon-only" title="重命名" @click.stop="startRename(sess)">✏️</button>
            <button class="btn-icon-only" title="删除" @click.stop="confirmDelete(sess)">🗑️</button>
          </div>
        </div>
        <div v-if="sessions.length === 0" class="empty-sessions">
          暂无历史对话
        </div>
      </div>
    </div>

    <!-- 主聊天区域 -->
    <div class="chat-main">
      <div class="chat-card">
        <!-- 会话栏 -->
        <div class="session-bar">
          <span class="session-label">
            当前会话: {{ currentSessionTitle || '新对话' }}
          </span>
          <div class="session-actions-bar">
            <label class="stream-toggle" title="流式模式下 AI 逐 token 输出，非流式模式下等待完整结果">
              <input type="checkbox" v-model="streamMode" />
              <span class="toggle-track"><span class="toggle-thumb"></span></span>
              <span class="toggle-label">流式</span>
            </label>
            <button class="btn btn-sm" @click="createNewSession">新建会话</button>
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
            :key="msg.id || i"
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

    <!-- 重命名对话框 -->
    <div v-if="renamingSession" class="modal-overlay" @click.self="renamingSession = null">
      <div class="modal-dialog">
        <h3>重命名会话</h3>
        <input
          v-model="renameTitle"
          class="modal-input"
          @keyup.enter="doRename"
          ref="renameInput"
        />
        <div class="modal-actions">
          <button class="btn" @click="renamingSession = null">取消</button>
          <button class="btn btn-primary" @click="doRename">确定</button>
        </div>
      </div>
    </div>

    <!-- 删除确认对话框 -->
    <div v-if="deletingSession" class="modal-overlay" @click.self="deletingSession = null">
      <div class="modal-dialog">
        <h3>删除会话</h3>
        <p>确定要删除「{{ deletingSession.title }}」吗？此操作不可恢复。</p>
        <div class="modal-actions">
          <button class="btn" @click="deletingSession = null">取消</button>
          <button class="btn btn-danger" @click="doDelete">删除</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch } from 'vue'
import { marked } from 'marked'
import {
  chatWithAgent,
  chatStream,
  getSessions,
  getSessionMessages,
  renameSession,
  deleteSession,
} from '../api/index.js'

function renderMarkdown(text) {
  if (!text) return ''
  return marked.parse(text, { breaks: true, gfm: true })
}

// 状态
const sessions = ref([])
const currentSessionId = ref('')
const currentSessionTitle = ref('')
const question = ref('')
const loading = ref(false)
const messages = ref([])
const messageList = ref(null)
const streamMode = ref(false)
const sidebarCollapsed = ref(false)

// 重命名
const renamingSession = ref(null)
const renameTitle = ref('')
const renameInput = ref(null)

// 删除
const deletingSession = ref(null)

// 监听重命名弹出，自动聚焦
watch(renamingSession, (val) => {
  if (val) {
    nextTick(() => {
      renameInput.value?.focus()
      renameInput.value?.select()
    })
  }
})

onMounted(() => {
  loadSessions()
})

async function loadSessions() {
  try {
    const res = await getSessions()
    sessions.value = res.data.sessions || []
  } catch (e) {
    console.error('加载会话列表失败:', e)
  }
}

async function loadMessages(sessionId) {
  try {
    const res = await getSessionMessages(sessionId)
    messages.value = res.data.messages || []
  } catch (e) {
    console.error('加载消息失败:', e)
    messages.value = []
  }
  scrollToBottom()
}

async function switchSession(sessionId) {
  currentSessionId.value = sessionId
  loading.value = false
  question.value = ''

  // 从会话列表更新标题
  const sess = sessions.value.find(s => s.id === sessionId)
  currentSessionTitle.value = sess?.title || '新对话'

  await loadMessages(sessionId)

  // 滚动到最新消息（消息可能来自历史，等渲染完再滚）
  await nextTick()
  scrollToBottom()
}

function createNewSession() {
  currentSessionId.value = 'session-' + Date.now()
  currentSessionTitle.value = '新对话'
  messages.value = []
  loading.value = false
  question.value = ''
  scrollToBottom()
}

function startRename(sess) {
  renamingSession.value = sess
  renameTitle.value = sess.title
}

async function doRename() {
  if (!renamingSession.value || !renameTitle.value.trim()) return
  try {
    await renameSession(renamingSession.value.id, renameTitle.value.trim())
    renamingSession.value.title = renameTitle.value.trim()
    if (currentSessionId.value === renamingSession.value.id) {
      currentSessionTitle.value = renameTitle.value.trim()
    }
  } catch (e) {
    console.error('重命名失败:', e)
  }
  renamingSession.value = null
}

function confirmDelete(sess) {
  deletingSession.value = sess
}

async function doDelete() {
  if (!deletingSession.value) return
  try {
    await deleteSession(deletingSession.value.id)
    sessions.value = sessions.value.filter(s => s.id !== deletingSession.value.id)
    if (currentSessionId.value === deletingSession.value.id) {
      createNewSession()
    }
  } catch (e) {
    console.error('删除失败:', e)
  }
  deletingSession.value = null
}

async function sendMessage() {
  if (!question.value.trim() || loading.value) return
  if (!currentSessionId.value) {
    createNewSession()
  }

  const q = question.value
  const sessionId = currentSessionId.value

  // 添加用户消息到界面
  messages.value.push({ role: 'user', content: q, id: 'temp-' + Date.now() })
  question.value = ''
  loading.value = true
  scrollToBottom()

  if (streamMode.value) {
    // 流式模式
    const msgIdx = messages.value.length
    messages.value.push({ role: 'assistant', content: '', id: 'temp-' + (Date.now() + 1) })
    scrollToBottom()

    try {
      const res = await chatStream(q, sessionId)
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
                // 服务端已保存消息，只需更新会话列表
                messages.value[msgIdx].sources = data.sources
                await loadSessions()
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
          id: 'temp-' + Date.now(),
        })
      }
    } finally {
      loading.value = false
      scrollToBottom()
    }
  } else {
    // 非流式模式
    try {
      const res = await chatWithAgent(q, sessionId)
      messages.value.push({
        role: 'assistant',
        content: res.data.answer,
        sources: res.data.sources,
        id: 'temp-' + (Date.now() + 1),
      })
      // 刷新会话列表（标题可能已自动生成）
      await loadSessions()
    } catch (e) {
      messages.value.push({
        role: 'assistant',
        content: e.response?.data?.detail || '请求失败，请重试。',
        id: 'temp-' + Date.now(),
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
  display: flex;
  height: calc(100vh - 48px);
  gap: 0;
  margin: -24px;
}

/* ===== 会话侧边栏 ===== */
.session-sidebar {
  width: 280px;
  background: #fff;
  border-right: 1px solid #f0f0f0;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  transition: width 0.2s;
  overflow: hidden;
}

.session-sidebar.collapsed {
  width: 48px;
}

.sidebar-header {
  display: flex;
  align-items: center;
  padding: 12px;
  border-bottom: 1px solid #f0f0f0;
  gap: 8px;
}

/* 折叠时只显示折叠按钮 */
.session-sidebar.collapsed .sidebar-header {
  justify-content: center;
  padding: 12px 4px;
}

.session-sidebar.collapsed .btn-new {
  display: none;
}

.session-sidebar.collapsed .btn-toggle {
  font-size: 18px;
  padding: 6px 8px;
}

.btn-icon {
  font-size: 16px;
  font-weight: bold;
}

.session-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.session-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  margin-bottom: 4px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.15s;
}

.session-item:hover {
  background: #f5f5f5;
}

.session-item.active {
  background: #e6f7ff;
  border: 1px solid #91d5ff;
}

.session-info {
  flex: 1;
  min-width: 0;
  overflow: hidden;
}

.session-title {
  font-size: 13px;
  color: #333;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.session-actions {
  display: flex;
  gap: 2px;
  flex-shrink: 0;
  opacity: 0;
  transition: opacity 0.15s;
}

.session-item:hover .session-actions,
.session-item.active .session-actions {
  opacity: 1;
}

.btn-icon-only {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 13px;
  padding: 2px 4px;
  border-radius: 4px;
  line-height: 1;
}

.btn-icon-only:hover {
  background: rgba(0, 0, 0, 0.06);
}

.empty-sessions {
  text-align: center;
  color: #bbb;
  font-size: 13px;
  padding: 32px 0;
}

/* ===== 主聊天区域 ===== */
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 24px;
  min-width: 0;
}

.chat-card {
  flex: 1;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
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
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-right: 12px;
}

.session-actions-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}

.btn {
  padding: 6px 16px;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  background: #fff;
  transition: all 0.2s;
  white-space: nowrap;
}

.btn:hover {
  border-color: #1890ff;
  color: #1890ff;
}

.btn-sm {
  padding: 4px 12px;
  font-size: 12px;
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

.btn-danger {
  background: #ff4d4f;
  color: #fff;
  border-color: #ff4d4f;
}

.btn-danger:hover {
  background: #ff7875;
  color: #fff;
}

/* ===== 侧边栏专用按钮（定义在 .btn 之后以确保覆盖） ===== */
.btn-new {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 8px 12px;
  background: #1890ff;
  color: #fff;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  transition: background 0.2s;
  white-space: nowrap;
}

.btn-new:hover {
  background: #40a9ff;
  color: #fff;
}

.btn-toggle {
  padding: 8px 10px;
  background: none;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
  color: #666;
  flex-shrink: 0;
}

.btn-toggle:hover {
  border-color: #1890ff;
  color: #1890ff;
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
  background: rgba(255, 255, 255, 0.2);
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
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2);
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
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15);
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

/* ===== 对话框 ===== */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-dialog {
  background: #fff;
  border-radius: 8px;
  padding: 24px;
  min-width: 360px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
}

.modal-dialog h3 {
  font-size: 16px;
  margin-bottom: 12px;
}

.modal-dialog p {
  font-size: 14px;
  color: #666;
  margin-bottom: 16px;
}

.modal-input {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  font-size: 14px;
  outline: none;
  margin-bottom: 16px;
  box-sizing: border-box;
}

.modal-input:focus {
  border-color: #1890ff;
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2);
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
</style>
