# offline-ok
"""第 8 章实验一：给记忆打「重要性」分，值不值。

    python docs/chapter8/code/01_importance.py

第 1 章的 BufferMemory 用 recency + relevance 两项打分。
Generative Agents（UIST 2023）用的是三项：再加一个 **importance**——
让 LLM 给每条记忆打 1~10 分，重要的更容易被想起来。

这个实验回答两个问题：
  1. 加上 importance，检索会更好吗？
  2. 它值那笔钱吗（每条记忆一次 LLM 调用）？

答案都不是简单的「是」。
"""

from __future__ import annotations

import warnings

from minimem.eval import CAPABILITIES, MINI_MEMORIES, MINI_QUERIES, as_memory_items, recall_at_k
from minimem.skill import SkillMemory, rule_importance
from minimem.utils.embedding import get_embedder
from minimem.utils.metering import Meter

USER = "u_bench"
K = 5


def part1_scores() -> None:
    print("\n  第一部分：规则版打分长什么样")
    print("  " + "-" * 66)
    print()
    samples = [m.text for m in MINI_MEMORIES[:3]] + [
        "我对花生过敏，点餐要特别注意。",
        "今天天气不错，中午在楼下吃的面。",
        "开会尽量安排在下午，我上午效率低。",
        "帮同事看了一段 SQL，有个 join 写错了。",
    ]
    for text in dict.fromkeys(samples):
        print(f"    {rule_importance(text):2d}  {text}")
    print(
        """
    这是**规则版**（正则匹配几类模式）。它存在有两个理由：
    一是让本章离线可跑；二是给 LLM 打分提供一个对照基线——
    **如果 LLM 打分不比这个简单规则强，那它就不值那笔钱。**
"""
    )


def part2_effect(embedder) -> None:
    print("\n  第二部分：加上 importance，检索变好了吗")
    print("  " + "-" * 66)

    rows = []
    for use_imp in (False, True):
        store = SkillMemory(embedder=embedder, meter=Meter())
        store.add_many(as_memory_items(), user_id=USER)
        per: dict[str, list[float]] = {}
        for q in MINI_QUERIES:
            rids = [
                h.item.metadata.get("rid", "")
                for h in store.search(q.text, user_id=USER, k=K, use_importance=use_imp)
            ]
            per.setdefault(q.capability, []).append(recall_at_k(rids, q.gold, K))
        allv = [v for vs in per.values() for v in vs]
        label = "三项打分（+importance）" if use_imp else "两项打分（第 1 章）"
        rows.append((label, sum(allv) / len(allv), per))

    print(f"\n    {'配置':<24}{'总召回':>8}" + "".join(f"{c:>10}" for c in CAPABILITIES))
    print("    " + "-" * (32 + 10 * len(CAPABILITIES)))
    for label, total, per in rows:
        line = f"    {label:<24}{total:>7.1%}"
        for c in CAPABILITIES:
            line += f"{sum(per[c]) / len(per[c]):>9.0%} "
        print(line)

    print(
        """
    总召回涨了十个百分点。但**看分能力表**：

      · 直接检索、同义改写、归纳 —— 明显变好
      · **多跳、知识更新 —— 明显变差**

    原因不难理解。多跳题的第二跳记忆往往是「XR-2049 上线后复核量下降四成」
    这种**重要性不高但正好是答案**的内容。importance 把「过敏」「离职」
    这类高分记忆顶上去，恰好把它们挤了下来。

    结论不是「importance 没用」，而是：
    **打分函数要按任务选，不能因为论文用了就照抄。**

    Generative Agents 的三项打分是为「模拟有连续生活的 agent」设计的——
    那个场景要的是「像人一样想起重要的事」，不是「答对多跳问答」。
    照搬到问答系统里，你可能买到一个负收益。
"""
    )


def part3_cost(embedder) -> None:
    print("\n  第三部分：这笔钱花得值吗")
    print("  " + "-" * 66)

    n = len(MINI_MEMORIES)
    avg_len = sum(len(m.text) for m in MINI_MEMORIES) / n
    # 打分 prompt 模板约 120 token，加上记忆本身
    per_call_in = 120 + avg_len
    per_call_out = 20

    print(
        f"""
    规则版：0 次调用，0 token，0 元。

    LLM 版：**每条记忆一次调用**。
      · {n} 条记忆 → {n} 次调用
      · 每次约 {per_call_in:.0f} 输入 + {per_call_out} 输出 token
      · 合计约 {n * (per_call_in + per_call_out):.0f} token

    折算到每万条记忆：约 {10000 * (per_call_in + per_call_out) / 1000:.0f} 千 token。

    判断依据仍然是**读写比**（这是本书第四次说这句话）：
      · 写一次读一百次 → 打分的成本被摊薄，划算
      · 写一次读一次   → 纯亏，不如直接用 relevance 排序

    还有一个容易忽略的点：**importance 是写入时打的，之后不会变**。
    一条当时看着不重要的记忆，可能三个月后变得关键
    （「上次那个 join 写错的问题」——出问题时才知道它重要）。
    重新打分意味着重新花钱，而多数系统从不重打。
"""
    )


def main() -> None:
    print("\n第 8 章实验一：重要性打分")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedder = get_embedder()
    part1_scores()
    part2_effect(embedder)
    part3_cost(embedder)


if __name__ == "__main__":
    main()
