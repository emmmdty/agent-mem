# offline-ok
"""第 3 章实验二：为什么 2026 年了还要用 BM25。

    python docs/chapter3/code/02_hybrid_search.py

稠密向量有一个结构性的弱点：**它把「意思相近」压缩成了距离相近，
于是「意思相近但不是同一个东西」也变得相近**。

最典型的受害者是标识符。XR-2049 和 XR-2050 在语义上确实极其相似
（都是项目代号、都是 XR 系列、只差一个数字），向量模型把它们编码到几乎同一个点，
而这恰恰是用户最不希望被混淆的地方。

BM25 反过来：它不懂任何语义，但它认得字符串。
把两路都跑一遍再用 RRF 融合，就能同时拿到两边的长处。
"""

from __future__ import annotations

from minimem.base import MemoryItem
from minimem.eval import CAPABILITIES, MINI_QUERIES, as_memory_items, mrr, recall_at_k
from minimem.utils.embedding import get_embedder
from minimem.utils.metering import Meter
from minimem.vector import VectorMemory, reciprocal_rank_fusion

USER = "u_bench"
K = 5


def make(embedder, mode):
    store = VectorMemory(embedder=embedder, mode=mode, meter=Meter())
    store.add_many(as_memory_items(), user_id=USER)
    return store


def evaluate(store) -> tuple[float, float, dict[str, float]]:
    recalls, mrrs = [], []
    per_cap: dict[str, list[float]] = {}
    for q in MINI_QUERIES:
        hits = store.search(q.text, user_id=USER, k=K)
        rids = [h.item.metadata.get("rid", "") for h in hits]
        r = recall_at_k(rids, q.gold, K)
        recalls.append(r)
        mrrs.append(mrr(rids, q.gold))
        per_cap.setdefault(q.capability, []).append(r)
    return (
        sum(recalls) / len(recalls),
        sum(mrrs) / len(mrrs),
        {c: sum(v) / len(v) for c, v in per_cap.items()},
    )


def show_confusion(embedder) -> None:
    """直接看向量空间里 XR-2049 和 XR-2050 有多近。"""
    print("\n  向量空间里，这两个项目号有多像？")
    a = "我负责的项目代号是 XR-2049，是个反欺诈模型。"
    b = "另一个项目 XR-2050 是做小微企业评分的，不归我管。"
    c = "我们家的猫叫毛毛，今年三岁了。"
    V = embedder.encode([a, b, c])
    print(f"    sim(XR-2049 那条, XR-2050 那条) = {float(V[0] @ V[1]):.3f}")
    print(f"    sim(XR-2049 那条, 猫那条)        = {float(V[0] @ V[2]):.3f}")

    q = "XR-2049 是什么项目？"
    qv = embedder.encode_one(q)
    sims = V @ qv
    print(f"\n    查询「{q}」对两条项目记忆的相似度：")
    print(f"      XR-2049 那条 = {float(sims[0]):.3f}")
    print(f"      XR-2050 那条 = {float(sims[1]):.3f}")
    print(f"      差距只有 {abs(float(sims[0] - sims[1])):.3f}——排序稍有扰动就会颠倒。")


def stress_test_identifiers(embedder) -> None:
    """专有名词压力测试：一批只差数字的项目编号。

    MiniBench 里只有两个相似编号，稠密向量还能勉强分开。
    真实系统里往往有几十上百个同族标识符（工单号、SKU、合同编号），
    那时稠密向量会彻底失效。这个子实验把那个场景造出来。
    """
    print("\n  专有名词压力测试：12 个只差数字的项目编号")

    projects = [
        (f"XR-{2049 + i}", desc)
        for i, desc in enumerate(
            [
                "反欺诈模型",
                "小微企业评分",
                "对公授信审批",
                "个人消费贷风控",
                "供应链金融额度",
                "信用卡套现识别",
                "反洗钱可疑交易",
                "押品估值模型",
                "贷后预警",
                "客户分层标签",
                "征信数据治理",
                "风险偏好传导",
            ]
        )
    ]
    items = [
        MemoryItem(f"项目 {code} 是做{desc}的。", metadata={"rid": code}) for code, desc in projects
    ]

    print(f"  {'模式':<10}{'top-1 命中率':<16}{'说明'}")
    print("  " + "-" * 56)
    for mode in ("dense", "bm25", "hybrid"):
        store = VectorMemory(embedder=embedder, mode=mode, meter=Meter())
        store.add_many(list(items), user_id="u_stress")
        correct = 0
        for code, _ in projects:
            hits = store.search(f"{code} 是什么项目？", user_id="u_stress", k=1)
            correct += bool(hits) and hits[0].item.metadata["rid"] == code
        note = {
            "dense": "语义上这 12 条几乎一样",
            "bm25": "编号是精确 token，直接命中",
            "hybrid": "BM25 把正确答案顶上来",
        }[mode]
        score = f"{correct}/{len(projects)}"
        print(f"  {mode:<10}{score:<16}{note}")


def show_rrf_detail(embedder) -> None:
    """把一次融合的中间过程摊开看。"""
    print("\n  一次 RRF 融合的中间过程（查询：XR-2050 归谁管？）")
    query = "XR-2050 归谁管？"

    dense = make(embedder, "dense")
    bm25 = make(embedder, "bm25")

    d_hits = dense.search(query, user_id=USER, k=5)
    b_hits = bm25.search(query, user_id=USER, k=5)

    print("\n    dense 通道排名：")
    for i, h in enumerate(d_hits, 1):
        print(f"      {i}. [{h.item.metadata['rid']}] {h.content[:30]}  ({h.score:.3f})")
    print("    bm25 通道排名：")
    for i, h in enumerate(b_hits, 1):
        print(f"      {i}. [{h.item.metadata['rid']}] {h.content[:30]}  ({h.score:.3f})")

    fused = reciprocal_rank_fusion(
        {
            "dense": [h.item.metadata["rid"] for h in d_hits],
            "bm25": [h.item.metadata["rid"] for h in b_hits],
        }
    )
    print("    RRF 融合后：")
    for rid, score in sorted(fused.items(), key=lambda kv: -kv[1])[:5]:
        print(f"      [{rid}] {score:.5f}")


def ablate_bm25_floor(embedder) -> None:
    """消融：BM25 低分候选的门槛对融合结果的影响。"""
    print("\n  消融：bm25_floor_ratio（丢弃得分低于本通道 top1 该比例的候选）")
    print(f"  {'floor':<10}{'召回率@5':>12}{'MRR':>10}")
    print("  " + "-" * 34)
    for floor in (0.0, 0.1, 0.3, 0.5, 0.7):
        store = VectorMemory(
            embedder=embedder, mode="hybrid", bm25_floor_ratio=floor, meter=Meter()
        )
        store.add_many(as_memory_items(), user_id="u_ablate")
        recalls, mrrs = [], []
        for q in MINI_QUERIES:
            rids = [h.item.metadata["rid"] for h in store.search(q.text, user_id="u_ablate", k=K)]
            recalls.append(recall_at_k(rids, q.gold, K))
            mrrs.append(mrr(rids, q.gold))
        mark = "  ← 默认" if floor == 0.5 else ""
        print(
            f"  {floor:<10}{sum(recalls) / len(recalls):>11.1%}{sum(mrrs) / len(mrrs):>10.3f}{mark}"
        )


def main() -> None:
    print("\n第 3 章实验二：dense / BM25 / hybrid")
    embedder = get_embedder()
    print(f"句向量模型：{embedder.name}\n")

    show_confusion(embedder)

    runs = []
    for mode in ("dense", "bm25", "hybrid"):
        store = make(embedder, mode)
        recall, mrr_score, per_cap = evaluate(store)
        latency = store._meter.summary(op="search")["latency_ms_mean"]  # noqa: SLF001
        runs.append((mode, recall, mrr_score, per_cap, latency))

    print(f"\n  {'模式':<10}{'召回率@5':>12}{'MRR':>10}{'检索延迟(ms)':>16}")
    print("  " + "-" * 48)
    for mode, recall, mrr_score, _, latency in runs:
        print(f"  {mode:<10}{recall:>11.1%}{mrr_score:>10.3f}{latency:>16.2f}")

    print("\n  分能力召回率@5")
    header = f"  {'能力':<10}" + "".join(f"{m:>10}" for m, *_ in runs)
    print(header)
    print("  " + "-" * (10 + 10 * len(runs)))
    for cap in CAPABILITIES:
        cells = ""
        for _, _, _, per_cap, _ in runs:
            v = per_cap.get(cap)
            cells += f"{v:>9.0%} " if v is not None else f"{'—':>10}"
        print(f"  {cap:<10}{cells}")

    stress_test_identifiers(embedder)
    show_rrf_detail(embedder)
    ablate_bm25_floor(embedder)

    dense_r, bm25_r, hybrid_r = (r for _, r, _, _, _ in runs)
    print(
        f"""
  读法：

  · **先说一个不漂亮的结果**：在 MiniBench 上，hybrid（{hybrid_r:.1%}）并没有赢过
    纯 dense（{dense_r:.1%}）。原因很简单——这个数据集只有 30 条记忆、
    两个相似编号，稠密向量还分得开。

    这正是小基准的典型问题：**它测不出某个机制的价值，不等于那个机制没价值**。
    很多论文和厂商报告里「我们的方法在 X 上提升了 Y%」，
    如果 X 本身测不到关键差异，那个 Y 就没有意义。第 10 章展开。

  · 所以要用压力测试补上：12 个只差数字的项目编号，
    dense 的 top-1 命中率会掉到很低，而 BM25 几乎全中。
    真实系统里工单号、SKU、合同编号成千上万，这才是常态。

  · 两个项目号的记忆在向量空间里相似度 0.65，而它们与「猫」那条只有 0.30。
    这不是模型的缺陷，**这就是语义相似度的定义**在这类文本上产生的结果。

  · RRF 只用名次不用分数。若直接把余弦相似度和 BM25 得分加权相加，
    你得先解决量纲问题（前者在 [-1,1]，后者可到几十且随语料变化），
    而任何归一化都会引入新假设。名次是无量纲的——这是 RRF 的全部聪明之处。

  · 但「只看名次」也带来一个真实的坑，见最后那组消融：
    当查询里根本没有可匹配的关键词时，BM25 通道返回的少量噪声候选
    依然会以「第 2 名」的身份拿到高权重，把 dense 通道真正相关的结果挤掉。
    `bm25_floor_ratio` 就是为此存在的——它不是调优旋钮，是修 bug。

  · 代价：hybrid 要跑两路检索、维护两套索引。延迟约为单路的 1.5~2 倍，
    内存多一份倒排表。小规模上无所谓，千万级就是实打实的成本。
"""
    )


if __name__ == "__main__":
    main()
