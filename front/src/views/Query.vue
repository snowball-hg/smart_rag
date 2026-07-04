<template>
  <div class="query-page">
    <h1 class="page-title">知识问答</h1>
    <p class="page-desc">基于已索引的知识库内容进行单次问答（无对话记忆）</p>

    <div class="query-card">
      <div class="input-row">
        <input
          v-model="question"
          class="query-input"
          placeholder="请输入你的问题..."
          @keyup.enter="handleQuery"
        />
        <button class="btn btn-primary" :disabled="!question.trim() || loading" @click="handleQuery">
          {{ loading ? '查询中...' : '提问' }}
        </button>
      </div>

      <div v-if="error" class="result error">{{ error }}</div>

      <div v-if="result" class="result success">
        <div class="answer-section">
          <h3>回答</h3>
          <div class="answer-content">{{ result.answer }}</div>
        </div>
        <div v-if="result.sources && result.sources.length > 0" class="sources-section">
          <h3>参考来源 ({{ result.sources.length }})</h3>
          <div class="source-item" v-for="(src, i) in result.sources" :key="i">
            <div class="source-header">
              <span class="source-name">{{ src.doc_name }}</span>
              <span class="source-chunk">块 #{{ src.chunk_index }}</span>
            </div>
            <div class="source-content">{{ src.content }}</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { queryDocuments } from '../api/index.js'

const question = ref('')
const loading = ref(false)
const result = ref(null)
const error = ref('')

async function handleQuery() {
  if (!question.value.trim()) return
  loading.value = true
  result.value = null
  error.value = ''
  try {
    const res = await queryDocuments(question.value)
    result.value = res.data
  } catch (e) {
    error.value = e.response?.data?.detail || '查询失败，请重试。'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.query-page {
  max-width: 800px;
}

.page-title {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 8px;
}

.page-desc {
  color: #888;
  margin-bottom: 24px;
}

.query-card {
  background: #fff;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.input-row {
  display: flex;
  gap: 12px;
}

.query-input {
  flex: 1;
  padding: 10px 16px;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  font-size: 15px;
  outline: none;
  transition: border-color 0.2s;
}

.query-input:focus {
  border-color: #1890ff;
  box-shadow: 0 0 0 2px rgba(24,144,255,0.2);
}

.btn {
  padding: 10px 24px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
}

.btn-primary {
  background: #1890ff;
  color: #fff;
}

.btn-primary:hover {
  background: #40a9ff;
}

.btn-primary:disabled {
  background: #91d5ff;
  cursor: not-allowed;
}

.result {
  margin-top: 16px;
  padding: 16px;
  border-radius: 6px;
}

.result.success {
  background: #f6ffed;
  border: 1px solid #b7eb8f;
  color: #389e0d;
}

.result.error {
  background: #fff2f0;
  border: 1px solid #ffccc7;
  color: #cf1322;
}

.answer-section {
  margin-bottom: 20px;
}

.answer-section h3,
.sources-section h3 {
  font-size: 16px;
  margin-bottom: 12px;
  color: #333;
}

.answer-content {
  font-size: 15px;
  line-height: 1.7;
  color: #333;
  white-space: pre-wrap;
}

.source-item {
  background: #fff;
  border: 1px solid #e8e8e8;
  border-radius: 6px;
  padding: 12px;
  margin-bottom: 8px;
}

.source-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 6px;
  font-size: 13px;
}

.source-name {
  font-weight: 600;
  color: #1890ff;
}

.source-chunk {
  color: #999;
}

.source-content {
  font-size: 13px;
  color: #666;
  line-height: 1.5;
}
</style>
