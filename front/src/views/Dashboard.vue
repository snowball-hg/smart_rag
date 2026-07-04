<template>
  <div class="dashboard">
    <h1 class="page-title">系统概览</h1>

    <div v-if="loading" class="loading">加载中...</div>
    <div v-else-if="error" class="error">{{ error }}</div>
    <template v-else>
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-value" :class="{ healthy: health.status === 'healthy', degraded: health.status === 'degraded' }">
            {{ health.status === 'healthy' ? '正常' : '异常' }}
          </div>
          <div class="stat-label">服务状态</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" :class="health.llm_configured ? 'healthy' : 'degraded'">
            {{ health.llm_configured ? '已配置' : '未配置' }}
          </div>
          <div class="stat-label">LLM 状态</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" :class="health.milvus_connected ? 'healthy' : 'degraded'">
            {{ health.milvus_connected ? '已连接' : '未连接' }}
          </div>
          <div class="stat-label">Milvus 状态</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ health.vector_count }}</div>
          <div class="stat-label">文档块数量</div>
        </div>
      </div>

      <div class="action-cards">
        <router-link to="/upload" class="action-card">
          <div class="action-icon">📄</div>
          <div class="action-title">上传文档</div>
          <div class="action-desc">上传 PDF、TXT、Markdown 文件到知识库</div>
        </router-link>
        <router-link to="/query" class="action-card">
          <div class="action-icon">🔍</div>
          <div class="action-title">知识问答</div>
          <div class="action-desc">基于知识库内容进行单次问答</div>
        </router-link>
        <router-link to="/chat" class="action-card">
          <div class="action-icon">💬</div>
          <div class="action-title">智能对话</div>
          <div class="action-desc">与 AI 代理进行多轮对话</div>
        </router-link>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { getHealth } from '../api/index.js'

const health = ref({})
const loading = ref(true)
const error = ref('')

onMounted(async () => {
  try {
    const res = await getHealth()
    health.value = res.data
  } catch (e) {
    error.value = '无法连接到后端服务，请确保服务已启动。'
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.dashboard {
  max-width: 1000px;
}

.page-title {
  font-size: 24px;
  margin-bottom: 24px;
  font-weight: 600;
}

.loading, .error {
  padding: 40px;
  text-align: center;
  font-size: 16px;
  color: #999;
}

.error {
  color: #ff4d4f;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 32px;
}

.stat-card {
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  text-align: center;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  margin-bottom: 8px;
}

.stat-value.healthy {
  color: #52c41a;
}

.stat-value.degraded {
  color: #faad14;
}

.stat-label {
  font-size: 14px;
  color: #666;
}

.action-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}

.action-card {
  background: #fff;
  border-radius: 8px;
  padding: 24px;
  text-decoration: none;
  color: inherit;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  transition: transform 0.2s, box-shadow 0.2s;
}

.action-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.action-icon {
  font-size: 36px;
  margin-bottom: 12px;
}

.action-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 8px;
}

.action-desc {
  font-size: 13px;
  color: #888;
  line-height: 1.5;
}
</style>
