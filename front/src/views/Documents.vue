<template>
  <div class="documents-page">
    <div class="page-header">
      <h1 class="page-title">知识库管理</h1>
      <button class="btn btn-primary" @click="loadDocuments" :disabled="loading">
        {{ loading ? '加载中...' : '刷新' }}
      </button>
    </div>

    <div v-if="loading && documents.length === 0" class="loading">加载中...</div>
    <div v-else-if="error" class="error">{{ error }}</div>
    <template v-else>
      <div class="toolbar">
        <input
          v-model="searchQuery"
          class="search-input"
          placeholder="搜索文档名称..."
          @input="filterDocuments"
        />
        <span class="doc-count">共 {{ filteredDocs.length }} 个文档</span>
      </div>

      <div v-if="filteredDocs.length === 0" class="empty">
        <p>知识库中暂无文档</p>
        <router-link to="/upload" class="btn btn-primary">上传文档</router-link>
      </div>

      <div v-else class="doc-list">
        <div
          v-for="doc in filteredDocs"
          :key="doc.doc_name"
          class="doc-card"
          :class="{ expanded: expandedDoc === doc.doc_name }"
        >
          <div class="doc-header" @click="toggleExpand(doc.doc_name)">
            <div class="doc-info">
              <span class="doc-icon">📄</span>
              <span class="doc-name">{{ doc.doc_name }}</span>
              <span class="doc-meta">{{ doc.chunk_count }} 块</span>
              <span v-if="doc.upload_time" class="doc-meta time">{{ doc.upload_time }}</span>
            </div>
            <div class="doc-actions">
              <button class="btn-icon-only" title="预览文档块" @click.stop="previewDoc(doc.doc_name)">📖</button>
              <button class="btn-icon-only" title="删除文档" @click.stop="confirmDelete(doc)">🗑️</button>
              <span class="expand-icon">{{ expandedDoc === doc.doc_name ? '▼' : '▶' }}</span>
            </div>
          </div>

          <div v-if="expandedDoc === doc.doc_name" class="doc-chunks">
            <div v-if="chunksLoading" class="loading">加载块中...</div>
            <div v-else-if="chunks.length === 0" class="empty">暂无块内容</div>
            <div v-else class="chunk-list">
              <div v-for="chunk in chunks" :key="chunk.chunk_index" class="chunk-item">
                <div class="chunk-header">块 #{{ chunk.chunk_index }}</div>
                <div class="chunk-content">{{ truncate(chunk.content, 300) }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </template>

    <!-- 删除确认 -->
    <div v-if="deletingDoc" class="modal-overlay" @click.self="deletingDoc = null">
      <div class="modal-dialog">
        <h3>删除文档</h3>
        <p>确定要删除「{{ deletingDoc.doc_name }}」吗？（共 {{ deletingDoc.chunk_count }} 块）此操作不可恢复。</p>
        <div class="modal-actions">
          <button class="btn" @click="deletingDoc = null">取消</button>
          <button class="btn btn-danger" @click="doDelete" :disabled="deleting">删除</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { getDocuments, getDocumentChunks, deleteDocument, deleteAllDocuments } from '../api/index.js'

const documents = ref([])
const filteredDocs = ref([])
const loading = ref(false)
const error = ref('')
const searchQuery = ref('')
const expandedDoc = ref('')
const chunks = ref([])
const chunksLoading = ref(false)
const deletingDoc = ref(null)
const deleting = ref(false)

onMounted(() => {
  loadDocuments()
})

async function loadDocuments() {
  loading.value = true
  error.value = ''
  try {
    const res = await getDocuments()
    documents.value = res.data.documents || []
    filterDocuments()
  } catch (e) {
    error.value = '加载文档列表失败: ' + (e.response?.data?.detail || e.message)
  } finally {
    loading.value = false
  }
}

function filterDocuments() {
  const q = searchQuery.value.trim().toLowerCase()
  if (!q) {
    filteredDocs.value = documents.value
  } else {
    filteredDocs.value = documents.value.filter(d => d.doc_name.toLowerCase().includes(q))
  }
}

function truncate(text, max) {
  if (!text) return ''
  return text.length > max ? text.slice(0, max) + '...' : text
}

async function toggleExpand(docName) {
  if (expandedDoc.value === docName) {
    expandedDoc.value = ''
    chunks.value = []
    return
  }
  expandedDoc.value = docName
  await previewDoc(docName)
}

async function previewDoc(docName) {
  chunksLoading.value = true
  try {
    const res = await getDocumentChunks(docName)
    chunks.value = res.data.chunks || []
  } catch (e) {
    chunks.value = []
  } finally {
    chunksLoading.value = false
  }
}

function confirmDelete(doc) {
  deletingDoc.value = doc
}

async function doDelete() {
  if (!deletingDoc.value) return
  deleting.value = true
  try {
    const docName = deletingDoc.value.doc_name
    const docId = deletingDoc.value.doc_id
    await deleteDocument(docId)
    documents.value = documents.value.filter(d => d.doc_name !== docName)
    filterDocuments()
    if (expandedDoc.value === docName) {
      expandedDoc.value = ''
      chunks.value = []
    }
  } catch (e) {
    alert('删除失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    deleting.value = false
    deletingDoc.value = null
  }
}
</script>

<style scoped>
.documents-page {
  max-width: 960px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.search-input {
  flex: 1;
  padding: 8px 14px;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.search-input:focus {
  border-color: #1890ff;
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2);
}

.doc-count {
  font-size: 13px;
  color: #999;
  white-space: nowrap;
}

.doc-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.doc-card {
  background: #fff;
  border: 1px solid #f0f0f0;
  border-radius: 8px;
  overflow: hidden;
  transition: box-shadow 0.2s;
}

.doc-card:hover {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.doc-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 16px;
  cursor: pointer;
  user-select: none;
}

.doc-info {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.doc-icon {
  font-size: 20px;
}

.doc-name {
  font-size: 14px;
  font-weight: 500;
  color: #333;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 300px;
}

.doc-meta {
  font-size: 12px;
  color: #999;
  background: #f5f5f5;
  padding: 2px 8px;
  border-radius: 4px;
}

.doc-meta.time {
  background: none;
  color: #bbb;
}

.doc-actions {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
}

.expand-icon {
  font-size: 12px;
  color: #999;
  margin-left: 4px;
}

.btn-icon-only {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 16px;
  padding: 4px 6px;
  border-radius: 4px;
  transition: background 0.15s;
}

.btn-icon-only:hover {
  background: rgba(0, 0, 0, 0.06);
}

.doc-chunks {
  border-top: 1px solid #f0f0f0;
  padding: 12px 16px;
  max-height: 400px;
  overflow-y: auto;
}

.chunk-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chunk-item {
  background: #fafafa;
  border-radius: 6px;
  padding: 10px 12px;
}

.chunk-header {
  font-size: 12px;
  color: #1890ff;
  font-weight: 500;
  margin-bottom: 4px;
}

.chunk-content {
  font-size: 13px;
  color: #555;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

.loading, .error, .empty {
  text-align: center;
  padding: 40px;
  color: #999;
  font-size: 14px;
}

.error { color: #ff4d4f; }

.empty p { margin-bottom: 12px; }
</style>