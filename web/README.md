# OpsPilot Web 前端

OpsPilot 运维知识助手的聊天界面。Vue 3 + TypeScript + Vite。

## 功能

- **SSE 流式输出**：回答逐字显示，首字延迟约 0.7 秒
- **引用溯源卡片**：回答下方展示参考来源，悬停显示原文摘要
- 多轮对话
- 发送中禁用输入，避免重复提交

## 启动

```bash
pnpm install
pnpm dev          # http://localhost:5173
```

后端需要同时运行（默认 `http://127.0.0.1:8000`）：

```bash
cd ..
.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

后端地址写在 `src/App.vue` 的 `API_BASE`，端口不同时改这一处。

## 关键技术点

### 1. 为什么用 fetch 而不是 EventSource

`EventSource` 只支持 GET，无法发送 JSON body。所以用 `fetch` + `ReadableStream`：

```typescript
const reader = response.body.getReader()
const decoder = new TextDecoder()

while (true) {
  const { done, value } = await reader.read()
  if (done) break
  buffer += decoder.decode(value, { stream: true })
  // 按 \n\n 切分事件，最后一段可能不完整，留到下次
}
```

### 2. 两个容易踩的坑

**`TextDecoder` 必须传 `{ stream: true }`**

一个中文字符占 3 字节，网络可能在中间切断。不加 `stream: true` 时，
解码器会把半个字符当损坏数据输出，导致**随机出现乱码**——
而且只在切分位置不巧时才复现，极难排查。

**必须自己处理粘包**

一次 `read()` 可能拿到半条消息、一条半消息或三条消息。
SSE 用 `\n\n` 分隔事件，所以要 buffer 累积 + `split('\n\n')` + 把最后一段
（可能不完整）`pop()` 回 buffer。

### 3. 事件类型

后端按 `meta → sources → token* → done` 顺序推送：

| 事件 | 用途 |
|---|---|
| `meta` | 收到问题，开始处理 |
| `sources` | 检索完成，返回引用列表（先于正文，让用户提前看到来源） |
| `token` | 逐片文本，追加到助手消息 |
| `error` | 出错信息 |
| `done` | 结束，带 `latency_ms` |

流式文字要**累积到同一条消息**，不能每片新建消息——所以请求发起时
先插入一条空的助手消息，后续 `msg.text += chunk` 往里追加。

## IDE 配置

[VS Code](https://code.visualstudio.com/) + [Vue (Official)](https://marketplace.visualstudio.com/items?itemName=Vue.volar)（并关闭 Vetur）。

TypeScript 默认无法处理 `.vue` 文件的类型，所以类型检查用 `vue-tsc` 而不是 `tsc`。

## 待完成

- [ ] 自动滚动到底部
- [ ] 点击引用弹出原文
- [ ] 会话历史持久化（localStorage）
- [ ] Markdown 渲染（LLM 输出含 `**粗体**` 等标记，目前原样显示）
- [ ] 新对话按钮
