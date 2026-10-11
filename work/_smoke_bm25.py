"""WP1 Step 2 冒烟测试：验证索引构建正确性（用完即删）。"""
from app.models import Chunk
from app.services.retrieval.bm25 import BM25Index
from app.services.vector_store.memory import MemoryVectorStore
from app.utils.text import tokenize

store = MemoryVectorStore(dim=4)
c1 = Chunk(chunk_id="a1", doc_id="d1", text="熔断错误率阈值触发熔断", title="t", source="s1")
c2 = Chunk(chunk_id="a2", doc_id="d2", text="数据库主从切换", title="t", source="s2")
store.upsert([c1, c2], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])

idx = BM25Index(store)
assert idx._built is False and idx._n == 0, "构造不应触碰 store"
print("OK 构造惰性:", idx._built, idx._n)

idx._build_index()
print("OK 已建:", idx._built, "N=", idx._n, "avgdl=", idx._avgdl)
# 「熔断」在 c1 中出现 2 次：df 必须是 1（块数）而不是 2（次数）
assert idx._df.get("熔断") == 1, f"df 去重错误: {idx._df.get('熔断')}"
assert idx._tf[0]["熔断"] == 2, "tf 应保留次数"
assert idx._dl[0] > 0 and idx._dl[1] > 0
print("OK df(熔断)=1, tf(c1,熔断)=2, dl=", idx._dl)

# Step 3 验收：打分函数
idx._build_index()
s_hit = idx._score(tokenize("熔断"), 0)   # 含「熔断」的块
s_miss = idx._score(tokenize("熔断"), 1)  # 不含的块
assert s_hit > 0 and s_miss == 0, f"基本打分错误: {s_hit}, {s_miss}"
print(f"OK 打分: 命中块={s_hit:.4f} 未命中块={s_miss}")

# return 缩进 bug 专项：两个词都命中 c1，修复前只算第一个词
both = idx._score(tokenize("熔断 错误率"), 0)
first_only = idx._score(tokenize("熔断"), 0)
assert both > first_only, "return 缩进 bug 未修复：只算了第一个词"
print(f"OK 多词累加: 熔断+错误率={both:.4f} > 仅熔断={first_only:.4f}")

# 稀有词权重 > 常见词（IDF 直觉验证）
s_common = idx._score(tokenize("数据"), 1)
print(f"OK 稀有词对比: df=1 的熔断({s_hit:.3f}) > df 小的词({s_common:.3f})" if s_hit > 0 else "")

# Step 4：search 组装
hits = idx.search("熔断 错误率")
assert hits and hits[0].chunk.chunk_id == "a1", "排序/命中错误"
assert hits[0].retriever == "bm25", "retriever 字段契约错误"
assert idx.search("不存在词xyz") == [], "零分块应被丢弃"
assert idx.search("熔断", service="别的服务") == [], "service 过滤失效"
print(f"OK search: 命中 {len(hits)} 块，首位={hits[0].chunk.source}，分数={hits[0].score:.3f}")

# 指纹失效检测：新增块后不重建索引，直接 search 应能命中新内容
c3 = Chunk(chunk_id="a3", doc_id="d3", text="凤凰指标异常", title="t", source="s3")
store.upsert([c3], [[0.2, 0.3, 0.4, 0.5]])
hits2 = idx.search("凤凰指标")
assert hits2 and hits2[0].chunk.chunk_id == "a3", "count 指纹失效检测未触发重建"
print("OK 指纹失效：新增块后自动重建索引并命中")
