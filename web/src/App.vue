<script setup lang="ts">
import { ref } from 'vue'

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

const messages = ref<Message[]>([
  { id: 1, role: 'user', text: '连接池耗尽怎么办' },
  { id: 2, role: 'assistant', text: '支付网关连接池耗尽时，先降载再定位根因，最后才考虑扩容。' },
])

const input = ref('')
const loading = ref(false)

let nextId = 100
function newId(): number {
  return nextId++
}

// ---------- 发送 ----------
async function send() {
  if (!input.value.trim() || loading.value) return

  const question = input.value.trim()
  messages.value.push({ id: newId(), role: 'user', text: question })
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
  messages.value.push({ id: assistantId, role: 'assistant', text: '' })

  const response = await fetch(`${API_BASE}/v1/ask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })

  if (!response.ok || !response.body) {
    setText(assistantId, `请求失败：HTTP ${response.status}`)
    return
  }

  const reader = response.body.getReader()   // 一块一块读响应体
  const decoder = new TextDecoder()          // 字节 → 字符串
  let buffer = ''                            // 拼不完整的碎片

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    // stream: true 让解码器记住被切断的半个中文字
    buffer += decoder.decode(value, { stream: true })

    // SSE 用 \n\n 分隔事件；最后一段可能不完整，留下次拼
    const parts = buffer.split('\n\n')
    buffer = parts.pop() ?? ''

    for (const part of parts) {
      handleEvent(part, assistantId)
    }
  }

  if (buffer.trim()) handleEvent(buffer, assistantId)
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
      {{ message.text || '…' }}

      <!-- 引用卡片：只有助手消息且有引用时显示 -->
      <div v-if="message.citations?.length" class="citations">
        <div class="citations-title">参考来源</div>
        <a
          v-for="c in message.citations"
          :key="c.index"
          :href="`#${c.source}`"
          class="citation"
          :title="c.snippet"
        >
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
  white-space: pre-wrap;      /* 保留 \n 换行 */
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
</style>
