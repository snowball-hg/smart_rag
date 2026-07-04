<template>
  <div class="upload-page">
    <h1 class="page-title">上传文档</h1>
    <p class="page-desc">支持 PDF、TXT、Markdown（.md/.mdx）格式</p>

    <div class="upload-card">
      <div
        class="drop-zone"
        :class="{ 'drag-over': dragging }"
        @dragover.prevent="dragging = true"
        @dragleave="dragging = false"
        @drop.prevent="handleDrop"
        @click="$refs.fileInput.click()"
      >
        <div class="drop-icon">📄</div>
        <p class="drop-text">拖拽文件到此处，或点击选择文件</p>
        <input
          ref="fileInput"
          type="file"
          accept=".pdf,.txt,.md,.mdx"
          style="display: none"
          @change="handleFileSelect"
        />
      </div>

      <div v-if="selectedFile" class="file-info">
        <span>已选择: {{ selectedFile.name }}</span>
        <button class="btn btn-primary" :disabled="uploading" @click="handleUpload">
          {{ uploading ? '上传中...' : '上传并索引' }}
        </button>
      </div>

      <div v-if="uploadResult" class="result" :class="{ success: !uploadError, error: uploadError }">
        <p>{{ uploadResult }}</p>
        <div v-if="resultData && !uploadError" class="result-details">
          <p>文档 ID: {{ resultData.doc_id }}</p>
          <p>文档块数: {{ resultData.chunk_count }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { uploadFile } from '../api/index.js'

const dragging = ref(false)
const selectedFile = ref(null)
const uploading = ref(false)
const uploadResult = ref('')
const uploadError = ref(false)
const resultData = ref(null)
const fileInput = ref(null)

const allowed = ['.pdf', '.txt', '.md', '.mdx']

function validateFile(file) {
  const ext = '.' + file.name.split('.').pop().toLowerCase()
  return allowed.includes(ext)
}

function handleDrop(e) {
  dragging.value = false
  const file = e.dataTransfer.files[0]
  if (!validateFile(file)) {
    uploadResult.value = '不支持的文件格式，请上传 PDF、TXT 或 Markdown 文件。'
    uploadError.value = true
    return
  }
  selectedFile.value = file
  uploadResult.value = ''
  uploadError.value = false
  resultData.value = null
}

function handleFileSelect(e) {
  const file = e.target.files[0]
  if (!file) return
  selectedFile.value = file
  uploadResult.value = ''
  uploadError.value = false
  resultData.value = null
}

async function handleUpload() {
  if (!selectedFile.value) return
  uploading.value = true
  uploadResult.value = ''
  uploadError.value = false
  try {
    const res = await uploadFile(selectedFile.value)
    resultData.value = res.data
    uploadResult.value = res.data.message
    uploadError.value = false
    selectedFile.value = null
  } catch (e) {
    uploadResult.value = e.response?.data?.detail || '上传失败，请重试。'
    uploadError.value = true
  } finally {
    uploading.value = false
  }
}
</script>

<style scoped>
.upload-page {
  max-width: 700px;
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

.upload-card {
  background: #fff;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.drop-zone {
  border: 2px dashed #d9d9d9;
  border-radius: 8px;
  padding: 60px 20px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
}

.drop-zone:hover,
.drop-zone.drag-over {
  border-color: #1890ff;
  background: #e6f7ff;
}

.drop-icon {
  font-size: 48px;
  margin-bottom: 12px;
}

.drop-text {
  font-size: 16px;
  color: #999;
}

.file-info {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 16px;
  padding: 12px 16px;
  background: #fafafa;
  border-radius: 6px;
  border: 1px solid #e8e8e8;
}

.btn {
  padding: 8px 20px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
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
  padding: 12px 16px;
  border-radius: 6px;
  font-size: 14px;
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

.result-details {
  margin-top: 8px;
  font-size: 13px;
  opacity: 0.8;
}
</style>
