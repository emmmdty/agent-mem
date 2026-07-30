"""检索指标：召回率、命中率、MRR，以及它们各自会骗你的地方。

这些指标衡量的都是**检索**，不是**回答**。第 10 章会反复强调这个区别：
召回率 100% 的系统完全可能答错——因为召回的十条里混了三条互相矛盾的，
模型选错了那一条。所以本模块的每个函数都在文档里写了它测不到什么。
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["recall_at_k", "hit_rate", "mrr", "precision_at_k", "aggregate"]


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int | None = None) -> float:
    """召回率@k：gold 里有多少比例出现在前 k 个结果中。

    **测不到**：召回的顺序、召回的噪声量、模型能不能用好这些结果。
    gold 为空时返回 1.0（拒答题没有需要召回的东西）。
    """
    if not gold:
        return 1.0
    top = set(retrieved[:k] if k else retrieved)
    return len(top & set(gold)) / len(set(gold))


def hit_rate(retrieved: Sequence[str], gold: Sequence[str], k: int | None = None) -> float:
    """命中率@k：只要召回了任意一条 gold 就算 1。

    多跳问题上它会给出过于乐观的结论——召回了两条 gold 中的一条，
    命中率是 1.0，但那道题其实答不出来。这种时候要看召回率。
    """
    if not gold:
        return 1.0
    top = set(retrieved[:k] if k else retrieved)
    return 1.0 if top & set(gold) else 0.0


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int | None = None) -> float:
    """准确率@k：召回结果里有多少比例是 gold。

    这是本书里唯一直接衡量「噪声」的指标。检索方案常常靠增大 k 把召回率刷上去，
    代价就体现在这里——而噪声最终要用 token 付账，还会稀释模型的注意力。
    """
    top = list(retrieved[:k] if k else retrieved)
    if not top:
        return 0.0
    return len(set(top) & set(gold)) / len(top)


def mrr(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """平均倒数排名：第一条 gold 出现在第 r 位则得 1/r。

    它关心「好结果排得够不够靠前」。当上下文预算很紧、只能带 1~2 条时，
    MRR 比召回率更贴近实际体验。
    """
    if not gold:
        return 1.0
    gold_set = set(gold)
    for i, rid in enumerate(retrieved, start=1):
        if rid in gold_set:
            return 1.0 / i
    return 0.0


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    """把逐题指标求平均。空输入返回空字典而不是报错。"""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}
