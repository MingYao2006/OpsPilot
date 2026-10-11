D1 —— M3 收官
主轨 stub 镣铐测试 ×2（TestDraftTicket：抽取成功无兜底标记 / 垃圾回落兜底有标记）
chat 车间换体面文案（离线模板应答 + LLM 在线闲聊回复）
README 更新：/v1/agent、/v1/ask、/v1/query 三链接分工 + 用例数
在线验收：uvicorn 真服务手打 /v1/agent 三问句全通
✅ 硬标准：全套测试无红、真 LLM 链路 HTTP 摸通
D2 —— M4：参数入图
AgentState 补 service/top_k 通道；search 升级为带参注入；路由喂 payload
新增用例：service 过滤后 citations 全属该服务
✅ 硬标准：带参请求端到端生效，旧测试零红
D3 —— M4：工单闭环
/v1/tickets 列表 + /v1/tickets/{id} 详情路由（薄路由三件套：路由+main 挂载+测试）
✅ 硬标准：建单→列表→详情全通
D4 —— M4：真 Function Calling
扩 LLMClient：chat 加 tools 通道（OpenAI 兼容格式）
create_ticket 从 JSON-prompt 抽取切换为真 FC tool-call，_TICKET_DRAFT_PROMPT 退役为 fallback
stub 测试同步改造回归
✅ 硬标准：真 LLM 手打建单走 tool_calls 字段返回
D5 —— M4：检索自愈 + 多轮 lite
BM25 重启重建方案定稿并实现（倾向 lifespan 内随 load_directory 自动重建）
多轮 lite：内存 session dict + 最近 N 轮上下文拼进提示词，AgentRequest 加 session_id
✅ 硬标准：重启后检索不空手；同一 session 追问指代能接住
D6 —— 工程加固
pgvector 真库连通实弹（docker-compose up → 入库 → 检索 → 与 memory 后端结果对照）
JWT 真 token 鉴权脱离"测试模式"；限流中间件（/v1/agent 每 key 每分钟上限）
✅ 硬标准：不带 token 401、超频 429、pgvector 模式全套测试绿
D7 —— 质量闭环 + 上线冲刺
评测集扩到 20 题，eval.py 跑回归 + 阴性对照
Docker 部署演练（Dockerfile 构建 → compose 全栈起 → healthz 探针绿）
CI 挂 pytest（GitHub Actions 或本地钩子）
README 终稿 + 七天总复盘 + M5 展望
✅ 硬标准：照 README 从零部署一次成功；评测数字写进文档