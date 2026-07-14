import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

export default api

/** 健康检查 */
export function getHealth() {
  return api.get('/health')
}

/** 上传文件 */
export function uploadFile(file) {
  const form = new FormData()
  form.append('file', file)
  return api.post('/upload', form)
}

/** 纯问答 */
export function queryDocuments(question, top_k = null) {
  return api.post('/query', { question, top_k })
}

/** 对话交互（非流式） */
export function chatWithAgent(question, session_id, top_k = null) {
  return api.post('/chat', { question, session_id, top_k })
}

/** 流式对话（SSE） */
export function chatStream(question, session_id, top_k = null) {
  const body = { question, session_id }
  if (top_k !== null) body.top_k = top_k
  return fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

/** 删除文档 */
export function deleteDocument(doc_id) {
  return api.delete(`/documents/${doc_id}`)
}

/** 清空所有文档 */
export function deleteAllDocuments() {
  return api.delete('/documents')
}

// ==================== 对话历史 API ====================

/** 获取会话列表 */
export function getSessions() {
  return api.get('/sessions')
}

/** 获取会话的消息列表 */
export function getSessionMessages(session_id) {
  return api.get(`/sessions/${session_id}/messages`)
}

/** 重命名会话 */
export function renameSession(session_id, title) {
  return api.patch(`/sessions/${session_id}`, { title })
}

/** 删除会话 */
export function deleteSession(session_id) {
  return api.delete(`/sessions/${session_id}`)
}
