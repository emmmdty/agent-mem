# offline-ok
"""第 4 章实验二：PPR 怎么传播，以及一个我们已经踩过一次的坑。

    python docs/chapter4/code/02_ppr_and_the_trap.py

两部分：

1. 把 Personalized PageRank 的传播过程摊开——种子是谁、分数流到哪、衰减多快。
2. 复现第 3 章那个 RRF 陷阱。**同一个坑，换了个通道又出现了一次**，
   而且这次更隐蔽：PPR 会给图上**每一个连通节点**分配一点分数，哪怕小到 1e-8。
"""

from __future__ import annotations

import warnings

from minimem.eval import MINI_QUERIES, as_memory_items, recall_at_k
from minimem.graph import GraphMemory
from minimem.utils.embedding import get_embedder
from minimem.utils.metering import Meter

USER = "u_bench"
K = 5


def build(mode="hybrid", **kw):
    store = GraphMemory(mode=mode, meter=Meter(), **kw)
    store.add_many(as_memory_items(), user_id=USER)
    return store


def part1_propagation() -> None:
    print("\n  第一部分：PPR 的传播过程")
    print("  " + "-" * 66)

    store = build("ppr")
    print(f"\n  图的规模：{store.stats(user_id=USER)}")

    for query in ["我负责的那个项目带来了什么效果？", "我的领导要去哪？"]:
        print()
        for line in store.explain(query, user_id=USER).splitlines():
            print(f"    {line}")

    print(
        """
    读法：第一个查询里，「项目」是种子，分数一跳传到 XR-2049，
    再传到「反欺诈」。注意分数的衰减——一跳之后掉了约 40%，
    两跳之后掉到几乎为零。这个衰减由阻尼系数 damping 控制。

    第二个查询抽不出任何实体，PPR 连起点都没有。
    **这是纯图检索的结构性弱点**，不是实现缺陷。
"""
    )


def part2_the_trap() -> None:
    print("\n  第二部分：一个我们已经踩过一次的坑")
    print("  " + "-" * 66)

    store = build("ppr", ppr_floor_ratio=0.0)
    hits = store.search("我负责的那个项目带来了什么效果？", user_id=USER, k=5)
    print("\n  关掉门槛后，PPR 通道返回的 5 个候选：")
    for i, h in enumerate(hits, 1):
        rid = h.item.metadata["rid"]
        print(f"    {i}. [{rid}] {h.score:.8f}  {h.content[:30]}")

    print(
        """
    看第 3 名往后：分数比第 1 名小了约五个数量级，内容完全无关。
    它们之所以在，是因为 PPR 给**图上每个连通节点**都分了一点分数。

    单独看 PPR 通道，这些噪声排在后面，无伤大雅。
    但一旦进入 RRF 融合就不一样了——**RRF 只看名次**，
    「PPR 的第 3 名」和「向量的第 3 名」权重完全相同。
    于是一条分数近乎为零的垃圾，把向量通道里真正相关的结果挤了下去。

    这和第 3 章 BM25 通道的问题是同一个。修法也一样：给通道内的候选设相对门槛。
"""
    )

    print("  门槛消融（hybrid 模式）：")
    print(f"    {'floor':<10}{'总召回@5':>12}{'多跳':>10}")
    print("    " + "-" * 32)
    for floor in (0.0, 0.001, 0.01, 0.1):
        s = build("hybrid", ppr_floor_ratio=floor)
        per: dict[str, list[float]] = {}
        for q in MINI_QUERIES:
            rids = [h.item.metadata.get("rid", "") for h in s.search(q.text, user_id=USER, k=K)]
            per.setdefault(q.capability, []).append(recall_at_k(rids, q.gold, K))
        allv = [v for vs in per.values() for v in vs]
        mark = "  ← 默认" if floor == 0.01 else ""
        multi = sum(per["多跳"]) / len(per["多跳"])
        print(f"    {floor:<10}{sum(allv) / len(allv):>11.1%}{multi:>10.0%}{mark}")

    print(
        """
    多跳召回从 50% 回到 75%，总召回从 88.2% 到 90.3%。

    **这是本书第二次因为「通道里的噪声候选」损失召回率。**
    值得记住的规律是：

      任何要参与 RRF 融合的通道，都必须先把自己的低质量候选清理干净。
      因为融合看的是名次，而名次不区分「第 3 名很好」和「第 3 名很差」。

    注意 PPR 的门槛值（0.01）比 BM25 的（0.5）小两个数量级。
    因为 PPR 分数按跳数指数衰减，二跳邻居的分数天然就比种子低很多——
    **门槛要按通道的分数分布来定，不能照抄。**
"""
    )


def main() -> None:
    print("\n第 4 章实验二：PPR 传播与融合陷阱")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        get_embedder()
    part1_propagation()
    part2_the_trap()


if __name__ == "__main__":
    main()
