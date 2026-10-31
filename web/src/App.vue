<script setup lang="ts">
import {ref} from 'vue'
import {marked} from 'marked'
import DOMPurify from 'dompurify'

// ---------- 类型 ----------
interface Citation {
  index: number
  title: string
  source: string
  snippet?: string
}

interface Message {
  id: number
  role: 'user' | 'assistant'
  text: string
  citations?: Citation[]       // 只有助手消息有
}

const API_BASE = 'http://127.0.0.1:8000'
const activeCitation = ref<Citation | null>(null)
const input = ref('')
const loading = ref(false)
// 记录「正在流式写入的那条助手消息 id」，用来决定用纯文本还是 Markdown 渲染
const streamingId = ref<number | null>(null)
const messages = ref<Message[]>([
  {id: 1, role: 'user', text: '连接池耗尽怎么办'},
  {id: 2, role: 'assistant', text: '支付网关连接池耗尽时，先降载再定位根因，最后才考虑扩容。'},
])


let nextId = 100

function newId(): number {
  return nextId++
}

// ---------- 发送 ----------
async function send() {
  if (!input.value.trim() || loading.value) return

  const question = input.value.trim()
  messages.value.push({id: newId(), role: 'user', text: question})
  input.value = ''

  loading.value = true
  try {
    await streamAnswer(question)
  } catch (err) {
    console.error('请求失败', err)
  } finally {
    loading.value = false
  }
}

// ---------- 接收 SSE 流 ----------
async function streamAnswer(question: string) {
  // 先插一条空的助手消息，流式内容往里追加
  const assistantId = newId()
  streamingId.value = assistantId
  messages.value.push({id: assistantId, role: 'assistant', text: ''})

  try {
    const response = await fetch(`${API_BASE}/v1/ask/stream`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question}),
    })

    if (!response.ok || !response.body) {
      setText(assistantId, `请求失败：HTTP ${response.status}`)
      return
    }

    const reader = response.body.getReader()   // 一块一块读响应体
    const decoder = new TextDecoder()          // 字节 → 字符串
    let buffer = ''                            // 拼不完整的碎片

    while (true) {
      const {done, value} = await reader.read()
      if (done) break

      // stream: true 让解码器记住被切断的半个中文字
      buffer += decoder.decode(value, {stream: true})

      // SSE 用 \n\n 分隔事件；最后一段可能不完整，留下次拼
      const parts = buffer.split('\n\n')
      buffer = parts.pop() ?? ''

      for (const part of parts) {
        handleEvent(part, assistantId)
      }
    }

    if (buffer.trim()) handleEvent(buffer, assistantId)
  } finally {
    // 流结束：清掉标记，模板改为用完整 Markdown 渲染
    streamingId.value = null
  }
}

// ---------- 解析单个事件 ----------
function handleEvent(raw: string, assistantId: number) {
  const line = raw.trim()
  if (!line.startsWith('data: ')) return

  let event: any
  try {
    event = JSON.parse(line.slice(6))
  } catch {
    return                       // 单条坏了不影响整体
  }

  if (event.type === 'token') {
    appendText(assistantId, event.text)
  } else if (event.type === 'sources') {
    const msg = messages.value.find((m) => m.id === assistantId)
    if (msg) msg.citations = event.citations
  } else if (event.type === 'error') {
    appendText(assistantId, `\n[出错] ${event.message}`)
  } else if (event.type === 'done') {
    console.log('完成，后端耗时', event.latency_ms, 'ms')
  }
}

// ---------- 工具 ----------
function appendText(id: number, chunk: string) {
  const msg = messages.value.find((m) => m.id === id)
  if (msg) msg.text += chunk
}

function setText(id: number, text: string) {
  const msg = messages.value.find((m) => m.id === id)
  if (msg) msg.text = text
}

// Markdown 渲染：marked 负责解析，DOMPurify 负责防 XSS。
// 必须过滤：LLM 输出可能被知识库文档内容污染（如文档里藏了 <script>），
// v-html 会真的执行它。marked 自身不做任何消毒。
function renderMarkdown(text: string): string {
  if (!text) return ''
  let html: string
  try {
    html = marked.parse(text, { async: false }) as string
    // 流式输出时 Markdown 常常只写了一半（如 **1. 紧急 还没闭合），
    // 直接解析会把后续内容全吞掉。所以流式中先按纯文本显示，
    // 等 done 事件到达后再切到渲染模式。
    return DOMPurify.sanitize(html, {
      ALLOWED_TAGS: [
        'p', 'br', 'strong', 'em', 'code', 'pre', 'ul', 'ol', 'li',
        'h1', 'h2', 'h3', 'h4', 'blockquote', 'a', 'table',
        'thead', 'tbody', 'tr', 'th', 'td',
      ],
      ALLOWED_ATTR: ['href', 'title'],
    })
  } catch {
    return escapeHtml(text)
  }
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>')
}

// 流式进行中：用纯文本（半截 Markdown 会渲染错乱）
// 流式结束后：才用完整 Markdown 渲染
function isMessageStreaming(message: Message): boolean {
  return loading.value && message.role === 'assistant' && message.id === streamingId.value
}

function showCitation(c: Citation) {
  activeCitation.value = c
}
</script>

<template>
  <div class="chat">
    <!-- 消息列表：一份就够，引用卡片放在助手消息里面 -->
    <div
        v-for="message in messages"
        :key="message.id"
        class="message"
        :class="message.role"
    >
      <!-- 流式中用纯文本（半截 Markdown 会渲染错乱），结束后才上 Markdown -->
      <span v-if="isMessageStreaming(message)" class="plain">{{ message.text || '…' }}</span>
      <span v-else v-html="renderMarkdown(message.text || '…')"></span>

      <!-- 引用卡片：只有助手消息且有引用时显示 -->
      <div v-if="message.citations?.length" class="citations">
        <div class="citations-title">参考来源</div>
        <a v-for="c in message.citations" :key="c.index" class="citation" @click="showCitation(c)">
          [{{ c.index }}] {{ c.title }}
        </a>
      </div>
    </div>

    <div class="input-area">
      <input
          v-model="input"
          placeholder="输入你的问题…"
          :disabled="loading"
          @keyup.enter="send"
      />
      <button :disabled="loading" @click="send">
        {{ loading ? '思考中…' : '发送' }}
      </button>
    </div>
  </div>
  <div v-if="activeCitation" class="modal" @click="activeCitation = null">
    <div class="modal-body" @click.stop>
      <h3>[{{ activeCitation.index }}] {{ activeCitation.title }}</h3>
      <p class="modal-source">{{ activeCitation.source }}</p>
      <p class="modal-snippet">{{ activeCitation.snippet }}</p>
      <button @click="activeCitation = null">关闭</button>
    </div>
  </div>
</template>

<style scoped>
.chat {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  font-family: system-ui, -apple-system, 'Microsoft YaHei', sans-serif;
}

.message {
  padding: 10px 14px;
  margin-bottom: 10px;
  border-radius: 10px;
  line-height: 1.6;
  max-width: 75%;
  word-break: break-word;
}

/* 流式进行中用纯文本，需要保留 \n 换行 */
.plain {
  white-space: pre-wrap;
}

/* Markdown 渲染后的元素样式：收紧间距，否则列表和段落会撑得很开 */
.message :deep(p) {
  margin: 0 0 8px;
}
.message :deep(p:last-child) {
  margin-bottom: 0;
}
.message :deep(strong) {
  font-weight: 600;
}
.message :deep(ul),
.message :deep(ol) {
  margin: 6px 0;
  padding-left: 22px;
}
.message :deep(li) {
  margin: 3px 0;
}
.message :deep(code) {
  background: rgba(0, 0, 0, 0.06);
  padding: 1px 5px;
  border-radius: 4px;
  font-family: Consolas, Monaco, monospace;
  font-size: 0.92em;
}
.message :deep(pre) {
  background: #f6f8fa;
  padding: 10px 12px;
  border-radius: 6px;
  overflow-x: auto;
}
.message :deep(pre code) {
  background: none;
  padding: 0;
}
.message :deep(h1),
.message :deep(h2),
.message :deep(h3),
.message :deep(h4) {
  margin: 12px 0 6px;
  font-size: 1em;
  font-weight: 600;
}
.message :deep(table) {
  border-collapse: collapse;
  margin: 8px 0;
  font-size: 0.95em;
}
.message :deep(th),
.message :deep(td) {
  border: 1px solid #ddd;
  padding: 4px 8px;
}

.user {
  margin-left: auto;
  background: #52c41a;
  color: #fff;
}

.assistant {
  margin-right: auto;
  background: #e6f4ff;
  color: #222;
}

/* ---------- 引用卡片 ---------- */
.citations {
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px dashed #ccc;
  font-size: 12px;
}

.citations-title {
  color: #888;
  margin-bottom: 6px;
}

.citation {
  display: block;
  color: #1677ff;
  text-decoration: none;
  margin-bottom: 4px;
  cursor: pointer;
}

.citation:hover {
  text-decoration: underline;
}

/* ---------- 输入区 ---------- */
.input-area {
  display: flex;
  gap: 8px;
  margin-top: 16px;
}

.input-area input {
  flex: 1;
  padding: 10px 14px;
  border: 1px solid #ddd;
  border-radius: 8px;
  font-size: 14px;
}

.input-area button {
  padding: 10px 24px;
  background: #1677ff;
  color: #fff;
  border: none;
  border-radius: 8px;
  cursor: pointer;
}

.input-area button:disabled {
  background: #bfbfbf;
  cursor: not-allowed;
}

.modal {
  position: fixed;           /* 固定定位，覆盖全屏 */
  inset: 0;                  /* top/right/bottom/left 都是 0 */
  background: rgba(0, 0, 0, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.modal-body {
  background: #fff;
  padding: 24px;
  border-radius: 12px;
  max-width: 600px;
  max-height: 70vh;
  overflow-y: auto;
}

.modal-source {
  color: #888;
  font-size: 12px;
  margin: 4px 0 12px;
}

.modal-snippet {
  line-height: 1.7;
  white-space: pre-wrap;
}

.modal button {
  margin-top: 16px;
  padding: 8px 20px;
  background: #1677ff;
  color: #fff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}
</style>
