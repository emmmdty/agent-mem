# offline-ok
"""第 3 章实验一：语义检索到底解决了什么。

    pip install fastembed          # 或 pip install -e ".[vector]"
    python docs/chapter3/code/01_semantic_vs_literal.py

三个方案在同一份 MiniBench 上比召回：

    BufferMemory        第 1 章的字面重叠基线
    VectorMemory(fake)  哈希向量——形式上是向量检索，实质仍是字面匹配
    VectorMemory(real)  真实句向量

加入 fake 这一组是为了回答一个容易被跳过的问题：
**改善到底来自「换成了向量检索」这个架构，还是来自「向量真的懂语义」？**
只比 BufferMemory 和真实向量，这两个因素是混在一起的。

未安装向量模型时脚本会自动降级并提示，此时第三组的结果不具参考价值。
"""

from __future__ import annotations

import warnings

from minimem import BufferMemory
from minimem.eval import CAPABILITIES, MINI_QUERIES, as_memory_items, recall_at_k
from minimem.utils.embedding import FakeEmbedder, get_embedder
from minimem.utils.metering import Meter
from minimem.vector import VectorMemory

USER = "u_bench"
K = 5


def build(store, items) -> None:
    store.add_many(items, user_id=USER)


def evaluate(store, *, label: str) -> dict[str, object]:
    """跑一遍 MiniBench，返回总召回率与分能力召回率。"""
    per_capability: dict[str, list[float]] = {}
    overall: list[float] = []

    for q in MINI_QUERIES:
        hits = store.search(q.text, user_id=USER, k=K)
        retrieved = [h.item.metadata.get("rid", "") for h in hits]
        r = recall_at_k(retrieved, q.gold, K)
        overall.append(r)
        per_capability.setdefault(q.capability, []).append(r)

    return {
        "label": label,
        "recall": sum(overall) / len(overall),
        "by_capability": {c: sum(v) / len(v) for c, v in per_capability.items()},
    }


def main() -> None:
    print("\n第 3 章实验一：字面匹配 vs 哈希向量 vs 真实句向量")
    print(f"数据：MiniBench 30 条记忆 / {len(MINI_QUERIES)} 个查询，k={K}\n")

    items = as_memory_items()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        real_embedder = get_embedder()
        degraded = any(issubclass(w.category, RuntimeWarning) for w in caught)

    if degraded:
        print("  ⚠️  未能加载真实句向量模型，第三组已降级为 FakeEmbedder。")
        print("     安装后重跑可看到真实差距：pip install fastembed")
        print("     国内如下载失败，可先配置代理或 HF_ENDPOINT。\n")
    else:
        print(f"  使用的句向量模型：{real_embedder.name}（{real_embedder.dim} 维）\n")

    runs = []
    for label, store in [
        ("BufferMemory（字面）", BufferMemory(recency_weight=0.0, meter=Meter())),
        (
            "Vector + fake（哈希）",
            VectorMemory(embedder=FakeEmbedder(), mode="dense", meter=Meter()),
        ),
        ("Vector + 真实向量", VectorMemory(embedder=real_embedder, mode="dense", meter=Meter())),
    ]:
        build(store, items)
        runs.append(evaluate(store, label=label))

    # ---- 总表 ----
    print(f"  {'方案':<24}{'总召回率@5':>12}")
    print("  " + "-" * 38)
    for r in runs:
        print(f"  {r['label']:<24}{r['recall']:>11.1%}")

    # ---- 分能力 ----
    print(f"\n  分能力召回率@{K}")
    caps = list(CAPABILITIES)
    header = f"  {'能力':<10}" + "".join(f"{r['label'][:10]:>14}" for r in runs)
    print(header)
    print("  " + "-" * (10 + 14 * len(runs)))
    for cap in caps:
        cells = ""
        for r in runs:
            v = r["by_capability"].get(cap)
            cells += f"{v:>13.0%} " if v is not None else f"{'—':>14}"
        print(f"  {cap:<10}{cells}")

    # ---- k 敏感性：同一批方案，换个 k 就可能换个赢家 ----
    print("\n  换一个 k 会怎样（总召回率）")
    ks = [1, 3, 5, 10]
    print(f"  {'方案':<24}" + "".join(f"{f'k={k}':>10}" for k in ks))
    print("  " + "-" * (24 + 10 * len(ks)))
    for label, make in [
        ("BufferMemory（字面）", lambda: BufferMemory(recency_weight=0.0, meter=Meter())),
        (
            "Vector + fake（哈希）",
            lambda: VectorMemory(embedder=FakeEmbedder(), mode="dense", meter=Meter()),
        ),
        (
            "Vector + 真实向量",
            lambda: VectorMemory(embedder=real_embedder, mode="dense", meter=Meter()),
        ),
    ]:
        store = make()
        store.add_many(as_memory_items(), user_id=USER)
        cells = ""
        for k in ks:
            scores = [
                recall_at_k(
                    [
                        h.item.metadata.get("rid", "")
                        for h in store.search(q.text, user_id=USER, k=k)
                    ],
                    q.gold,
                    k,
                )
                for q in MINI_QUERIES
            ]
            cells += f"{sum(scores) / len(scores):>9.1%} "
        print(f"  {label:<24}{cells}")

    # ---- 逐题细看 ----
    print("\n  逐题细看：同义改写类（只取 top-1，最严格）")
    for q in MINI_QUERIES:
        if q.capability != "同义改写":
            continue
        print(f"\n    Q: {q.text}")
        for label, store in [
            ("字面", BufferMemory(recency_weight=0.0, meter=Meter())),
            ("向量", VectorMemory(embedder=real_embedder, mode="dense", meter=Meter())),
        ]:
            store.add_many(as_memory_items(), user_id=USER)
            hits = store.search(q.text, user_id=USER, k=1)
            top = hits[0].content if hits else "（无结果）"
            ok = hits and hits[0].item.metadata.get("rid") in q.gold
            print(f"      {label}：{top[:32]:<34}{'✅' if ok else '❌'}")

    print(
        """
  读法（请对照你自己跑出来的数字，下面写的是本书作者机器上的观察）：

  · **fake 向量并不比字面匹配更好，甚至更差。** 这一组的价值在于证伪：
    「换成向量检索」这个**架构动作本身**不带来任何东西，
    所有改善都来自向量真的编码了语义。
    这是个通用的验证手法——怀疑某个改进是不是真的时，
    把关键组件换成「形状相同但没有能力」的假件，看指标掉不掉。

  · **真实向量赢的地方，和直觉预期的不一样。**
    它在「知识更新」和「归纳」两类上大幅领先，因为这两类题需要
    **把散落在多条记忆里的相关内容都捞回来**，语义相似度天然擅长这件事。

    而在「同义改写」类上，两者可能打平——因为 k=5 已经很宽松，
    字面匹配靠一两个共享字也能把目标挤进前五。
    把 k 降到 1 再看那张逐题表，差距才显现出来。

  · **有的能力上语义检索反而更差**，比如多跳。
    多跳题要求两条 gold 都被召回，而语义相似度会把「语义相近但不是答案」的
    记忆排到前面，挤掉第二条 gold。这个问题第 4 章用图结构解决。

  · **k 敏感性那张表值得多看两眼**：k 从 1 变到 10，各方案的差距会显著变形。
    这意味着任何「A 比 B 好 X%」的说法，不指明 k 就没有意义。
    厂商基准里的对比数字经常缺这个前提——第 10 章会展开。

  · 最后一个陷阱：「拒答」类所有方案都是 100%，
    因为 gold 为空时召回率按约定记 1.0。**指标在这里什么也没测**。
    真正该测的是「系统会不会硬编一个答案」，那需要端到端评测，见第 10 章。

  一句必要的免责：MiniBench 只有 30 条记忆、24 个查询，
  单题权重高达 4%，**任何一两个百分点的差异都是噪声**。
  它能支撑的结论只有「量级」和「方向」，不能支撑排名。
"""
    )


if __name__ == "__main__":
    main()
