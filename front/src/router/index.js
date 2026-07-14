import { createRouter, createWebHistory } from 'vue-router'
import Dashboard from '../views/Dashboard.vue'
import Upload from '../views/Upload.vue'
import Query from '../views/Query.vue'
import Chat from '../views/Chat.vue'
import Documents from '../views/Documents.vue'

const routes = [
  { path: '/', name: 'dashboard', component: Dashboard, meta: { title: '仪表盘' } },
  { path: '/upload', name: 'upload', component: Upload, meta: { title: '上传文档' } },
  { path: '/query', name: 'query', component: Query, meta: { title: '知识问答' } },
  { path: '/chat', name: 'chat', component: Chat, meta: { title: '智能对话' } },
  { path: '/documents', name: 'documents', component: Documents, meta: { title: '知识库管理' } },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
