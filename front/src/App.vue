<template>
  <div class="app-container">
    <nav class="sidebar">
      <div class="sidebar-header">
        <h2>RAG 知识库</h2>
      </div>
      <ul class="nav-list">
        <li>
          <router-link to="/" class="nav-item" active-class="active">
            <span class="nav-icon">📊</span>
            <span class="nav-label">仪表盘</span>
          </router-link>
        </li>
        <li>
          <router-link to="/upload" class="nav-item" active-class="active">
            <span class="nav-icon">📄</span>
            <span class="nav-label">上传文档</span>
          </router-link>
        </li>
        <li>
          <router-link to="/query" class="nav-item" active-class="active">
            <span class="nav-icon">🔍</span>
            <span class="nav-label">知识问答</span>
          </router-link>
        </li>
        <li>
          <router-link to="/chat" class="nav-item" active-class="active">
            <span class="nav-icon">💬</span>
            <span class="nav-label">智能对话</span>
          </router-link>
        </li>
        <li>
          <router-link to="/documents" class="nav-item" active-class="active">
            <span class="nav-icon">📚</span>
            <span class="nav-label">知识库管理</span>
          </router-link>
        </li>
      </ul>
    </nav>
    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: #f0f2f5;
  color: #333;
}

.app-container {
  display: flex;
  min-height: 100vh;
}

.sidebar {
  width: 220px;
  background: #001529;
  color: #fff;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}

.sidebar-header {
  padding: 20px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}

.sidebar-header h2 {
  font-size: 18px;
  font-weight: 600;
  color: #fff;
}

.nav-list {
  list-style: none;
  padding: 8px 0;
}

.nav-item {
  display: flex;
  align-items: center;
  padding: 12px 20px;
  color: rgba(255,255,255,0.65);
  text-decoration: none;
  transition: all 0.2s;
  gap: 10px;
}

.nav-item:hover {
  color: #fff;
  background: rgba(255,255,255,0.08);
}

.nav-item.active {
  color: #fff;
  background: #1890ff;
}

.nav-icon {
  font-size: 16px;
  width: 20px;
  text-align: center;
}

.nav-label {
  font-size: 14px;
}

.main-content {
  flex: 1;
  padding: 24px;
  overflow-y: auto;
}
</style>
