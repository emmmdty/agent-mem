# offline-ok
"""第 4 章实验一：多跳查询，向量检索差在哪。

    python docs/chapter4/code/01_multihop.py

第 3 章测出：语义向量在多跳查询上只有 50% 的召回率，**比字面匹配还差**。
这个实验解释原因，并给出图结构的解法。

多跳问题的形状是这样的：

    「我负责的那个项目带来了什么效果？」
      第一跳：「我负责的项目」 → XR-2049          （在 m06 里）
      第二跳：XR-2049 → 人工复核量下降四成        （在 m07 里）

向量检索做的是「找和整个问句最像的几条」。它没有「走一步」这个动作，
所以第二跳那条记忆（只提了 XR-2049 和复核量，没提「负责」「效果」）
在语义上离问句很远，排不进前几名。
"""

from __future__ import annotations

import warnings

from minimem import VectorMemory
from minimem.eval import CAPABILITIES, MINI_QUERIES, as_memory_items, recall_at_k
from minimem.graph import GraphMemory
from minimem.utils.embedding import get_embedder
from minimem.utils.metering import Meter

USER = "u_bench"
K = 5


def evaluate(store) -> tuple[float, dict[str, float]]:
    per: dict[str, list[float]] = {}
    for q in MINI_QUERIES:
        rids = [h.item.metadata.get("rid", "") for h in store.search(q.text, user_id=USER, k=K)]
        per.setdefault(q.capability, []).append(recall_at_k(rids, q.gold, K))
    allv = [v for vs in per.values() for v in vs]
    return sum(allv) / len(allv), {c: sum(v) / len(v) for c, v in per.items()}


def main() -> None:
    print("\n第 4 章实验一：多跳查询")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedder = get_embedder()
    print(f"句向量模型：{embedder.name}\n")

    stores = {
        "vector（第 3 章）": VectorMemory(embedder=embedder, mode="hybrid", meter=Meter()),
        "graph-ppr（纯图）": GraphMemory(mode="ppr", embedder=embedder, meter=Meter()),
        "graph-hybrid": GraphMemory(mode="hybrid", embedder=embedder, meter=Meter()),
    }
    results = {}
    for label, store in stores.items():
        store.add_many(as_memory_items(), user_id=USER)
        results[label] = evaluate(store)

    print(f"  {'方案':<20}{'总召回@5':>10}")
    print("  " + "-" * 32)
    for label, (total, _) in results.items():
        print(f"  {label:<20}{total:>9.1%}")

    print("\n  分能力召回率@5")
    print(f"  {'能力':<10}" + "".join(f"{lab[:12]:>16}" for lab in results))
    print("  " + "-" * (10 + 16 * len(results)))
    for cap in CAPABILITIES:
        row = f"  {cap:<10}"
        for _, per in results.values():
            v = per.get(cap)
            row += f"{v:>15.0%} " if v is not None else f"{'—':>16}"
        print(row)

    # ---- 逐题细看两道多跳题 ----
    print("\n  逐题细看：两道多跳题")
    for q in MINI_QUERIES:
        if q.capability != "多跳":
            continue
        print(f"\n    Q {q.qid}: {q.text}")
        print(f"    需要同时召回：{q.gold}")
        for label, store in stores.items():
            hits = store.search(q.text, user_id=USER, k=K)
            rids = [h.item.metadata.get("rid", "") for h in hits]
            got = [r for r in rids if r in q.gold]
            r = recall_at_k(rids, q.gold, K)
            detail = f"命中 {got}" if got else "（一条 gold 都没召回）"
            print(f"      {label:<20}{r:>5.0%}  {detail}")

    print(
        """
  读法：

  · **q10 是图检索该赢的地方，它也确实赢了。**
    「项目」这个实体在查询里，PPR 从它出发，沿着「项目—XR-2049—复核量」
    这条链把分数传过去，两条 gold 都进了前列。
    纯向量做不到，因为 m07 那句话（「XR-2049 上线后，人工复核量下降四成」）
    和问句在字面和语义上都不近。

  · **q11 是图检索失效的地方。** 「我的领导要去哪？」——
    「领导」不是一个能被抽出来的实体（规则词表里没有，它也不是专名），
    于是 PPR 连种子都没有，直接返回空。

    这是纯图检索的结构性弱点：**它要求查询里有可锚定的实体**。
    而日常提问里大量使用泛称、代词、描述性短语。
    这就是 GraphMemory 默认用 hybrid 而不是纯 ppr 的原因——
    向量通道负责兜住这类查询。

  · 看总召回那一栏：纯 ppr 只有 37.5%，远低于向量方案。
    **单独看它会得出「图记忆没用」的结论，那是错的**；
    单独看 hybrid 的总分提升（约 2 个百分点）又会觉得「提升太小不值得」，
    那也是错的。要看的是**分能力表**：图结构在多跳上带来了实质提升，
    而它在其他能力上没有拖后腿。

  · 代价没有出现在这张表里。实验三会算：建图要抽实体、PPR 每次要迭代、
    实体消解要人工维护词表。
"""
    )


if __name__ == "__main__":
    main()
