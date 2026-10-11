"""D1 在线验收：对 /v1/agent 打三类问句 + 健康探针。"""
import httpx

BASE = "http://127.0.0.1:8000"

print("healthz:", httpx.get(f"{BASE}/healthz", timeout=10).json())

for q in ("熔断错误率阈值是多少", "支付网关挂了，帮我建工单", "今天天气怎么样"):
    r = httpx.post(f"{BASE}/v1/agent", json={"question": q}, timeout=90)
    b = r.json()
    print(f"[{r.status_code}] {q}")
    print(f"   intent={b['intent']} | refused={b['refused']} | tools={b['tools_used']} | latency={b['latency_ms']}ms")
    print(f"   steps={b['steps']}")
    print(f"   answer={b['answer'][:100]}")
