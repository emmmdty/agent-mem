# offline-ok
"""第 5 章实验一：两条时间轴。

    python docs/chapter5/code/01_two_timelines.py

第 1 章的第三堵墙到这里正面处理。前四章的所有实现都会把矛盾的记忆
一起返回，让模型自己猜哪条是现在有效的：

    Q: 我在哪家公司上班？
       我在星辰银行工作。              ← 半年前
       我下周就要从星辰银行离职了。      ← 二十天前
       新工作定了，下个月入职长风科技。   ← 半个月前

三条都是真话。只有一条描述当前状态——而且答案取决于今天是几号。

本实验演示两条时间轴的区别：**说话时间**和**生效时间**。
为了让结果可复现，「今天」被固定为 MiniBench 的基准日 2026-07-30。
"""

from __future__ import annotations

import warnings
from datetime import timedelta

from minimem.eval import MINI_QUERIES, as_memory_items, recall_at_k
from minimem.eval.dataset import BASE_DATE
from minimem.graph import GraphMemory
from minimem.temporal import TemporalGraphMemory, parse_relative_time
from minimem.utils.embedding import get_embedder
from minimem.utils.metering import Meter

USER = "u_bench"
K = 5


def part1_two_timelines(store: TemporalGraphMemory) -> None:
    print("\n  第一部分：说话时间 ≠ 生效时间")
    print("  " + "-" * 66)

    samples = [
        "我在星辰银行工作。",
        "我下周就要从星辰银行离职了。",
        "新工作定了，我下个月入职长风科技。",
        "上周搬家了，现在住城东的沿江花园。",
    ]
    print(f"\n    假设这些话都说于 {BASE_DATE:%Y-%m-%d}：\n")
    for text in samples:
        vf = parse_relative_time(text, spoken_at=BASE_DATE)
        if vf is None:
            print(f"      「{text}」→ 说完即生效")
        else:
            lag = (vf - BASE_DATE).days
            print(f"      「{text}」→ 生效于 {vf:%Y-%m-%d}（{lag:+d} 天）")

    print(
        """
    「我下个月入职长风科技」说的是今天，指的是下个月。
    把这两个时间混为一谈，就会得出「他现在在长风科技」这个错误结论。
"""
    )


def part2_evolution(store: TemporalGraphMemory) -> None:
    print("\n  第二部分：事实的演化史")
    print("  " + "-" * 66)
    print()
    for predicate in ("雇主", "居住地"):
        for line in store.timeline(predicate, user_id=USER).splitlines():
            print(f"    {line}")
        print()

    print("    今天（2026-07-30）有效的事实：")
    for f in store.facts_at(user_id=USER):
        print(f"      {f.describe()}")

    print(
        """
    **注意「雇主」这一项：今天它没有有效值。**
    用户已从星辰银行离职（7-17 生效），而长风科技要 8-14 才入职。
    严格说，「我在哪家公司上班」这个问题在今天的正确答案是
    「已离职，下月入职长风科技」——而不是任何一家公司。

    前四章的任何一种实现都答不出这个，因为它们没有「事实何时生效」这个概念。
"""
    )


def part3_point_in_time(store: TemporalGraphMemory) -> None:
    print("\n  第三部分：时点查询")
    print("  " + "-" * 66)

    for label, when in [
        ("今天", BASE_DATE),
        ("三个月前", BASE_DATE - timedelta(days=90)),
        ("下个月", BASE_DATE + timedelta(days=30)),
    ]:
        print(f"\n    ── {label}（{when:%Y-%m-%d}）──")
        for predicate in ("雇主", "居住地"):
            facts = store.facts_at(when, user_id=USER, predicate=predicate)
            value = "、".join(f.object for f in facts) if facts else "（无有效值）"
            print(f"      {predicate}：{value}")

    print(
        """
    同一个问题，三个时点，三个不同的正确答案。
    这不是「模型答错了」能解释的——**没有时间轴，这个问题根本无解**。
"""
    )


def part4_context(store: TemporalGraphMemory) -> None:
    print("\n  第四部分：喂给模型的上下文长什么样")
    print("  " + "-" * 66)
    print()
    for line in store.answer_context("我在哪家公司上班？", user_id=USER).splitlines():
        print(f"    {line}")

    print(
        """
    对比第 4 章：那时给模型的是三条互相矛盾的裸文本。
    现在每条都带上了时间与状态，还附了一份当前有效事实的摘要。

    **这也是第 10 章讲投毒防御时说的「让上下文带上来源」**——
    模型有了判断依据，才可能判断对。
"""
    )


def part5_benchmark(embedder) -> None:
    print("\n  第五部分：在 MiniBench 上和第 4 章比一比")
    print("  " + "-" * 66)

    runs = []
    for label, store in [
        ("graph（第 4 章）", GraphMemory(embedder=embedder, meter=Meter())),
        (
            "temporal（本章）",
            TemporalGraphMemory(now=BASE_DATE, embedder=embedder, meter=Meter()),
        ),
    ]:
        store.add_many(as_memory_items(), user_id=USER)
        per: dict[str, list[float]] = {}
        for q in MINI_QUERIES:
            rids = [h.item.metadata.get("rid", "") for h in store.search(q.text, user_id=USER, k=K)]
            per.setdefault(q.capability, []).append(recall_at_k(rids, q.gold, K))
        allv = [v for vs in per.values() for v in vs]
        runs.append((label, sum(allv) / len(allv), per))

    caps = ["时序", "知识更新", "多跳", "直接检索"]
    print(f"\n    {'方案':<20}{'总召回':>10}" + "".join(f"{c:>10}" for c in caps))
    print("    " + "-" * (30 + 10 * len(caps)))
    for label, total, per in runs:
        row = f"    {label:<20}{total:>9.1%}"
        for c in caps:
            row += f"{sum(per[c]) / len(per[c]):>9.0%} "
        print(row)

    print(
        """
    **召回率几乎没有变化——这是意料之中的，也是重点。**

    时序处理改善的不是「能不能召回相关记忆」，而是「召回之后怎么排序、
    怎么告诉模型哪条有效」。召回率这个指标测不到它。

    这正是第 10 章那句话的又一个例证：
    **检索指标测不到端到端效果。** 要衡量时序能力，你需要的是
    「回答是否正确」，而那需要真实 LLM 参与评测。
"""
    )


def main() -> None:
    print("\n第 5 章实验一：两条时间轴")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedder = get_embedder()

    store = TemporalGraphMemory(now=BASE_DATE, embedder=embedder, meter=Meter())
    store.add_many(as_memory_items(), user_id=USER)

    part1_two_timelines(store)
    part2_evolution(store)
    part3_point_in_time(store)
    part4_context(store)
    part5_benchmark(embedder)


if __name__ == "__main__":
    main()
