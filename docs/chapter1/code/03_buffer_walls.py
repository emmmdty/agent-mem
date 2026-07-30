# offline-ok
"""第 1 章实验三：BufferMemory 的三堵墙。

    python docs/chapter1/code/03_buffer_walls.py

BufferMemory 已经能用了——实验一里它答对了「我对什么过敏」。
但它会在三个地方撞墙，而这三堵墙恰好对应本书后面九章要解决的问题：

    墙一  语义鸿沟   换个说法就检索不到          → 第 3 章（向量检索）
    墙二  规模天花板  全表扫描，条数一多就慢       → 第 3 章（索引）/ 第 7 章（分层）
    墙三  知识演化   新旧矛盾的事实同时留在库里    → 第 5、6 章（时序 / 更新）

本脚本把三堵墙都测出来，而不是口头断言。
"""

from __future__ import annotations

import random
import time

from minimem import BufferMemory
from minimem.utils.metering import Meter

USER = "u1"


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


# ----------------------------------------------------------------------
# 墙一：语义鸿沟
# ----------------------------------------------------------------------


def wall_1_semantic_gap() -> None:
    rule("墙一：语义鸿沟——同一件事，换个说法就找不到了")

    store = BufferMemory(recency_weight=0.0, meter=Meter())
    store.add_many(
        [
            "我从事信贷风险评估工作，主要看制造业客户。",
            "上周去了趟青海湖，风景不错。",
            "我们家的猫叫毛毛，今年三岁。",
            "最近在读《证券分析》。",
        ],
        user_id=USER,
    )

    queries = [
        ("我做什么工作？", "我从事信贷风险评估工作，主要看制造业客户。"),
        ("他的职业是什么？", "我从事信贷风险评估工作，主要看制造业客户。"),
        ("我养宠物了吗？", "我们家的猫叫毛毛，今年三岁。"),
        ("我最近旅游去哪了？", "上周去了趟青海湖，风景不错。"),
    ]

    print(f"  {'查询':<20}{'命中':<36}{'是否正确'}")
    print("  " + "-" * 66)
    hits_correct = 0
    for query, expected in queries:
        hits = store.search(query, user_id=USER, k=1)
        top = hits[0].content if hits else "（无结果）"
        ok = top == expected
        hits_correct += ok
        print(f"  {query:<20}{top[:34]:<36}{'✅' if ok else '❌'}")

    print(f"\n  正确 {hits_correct}/{len(queries)}")
    print(
        """
  「我做什么工作？」能命中，是因为共享了「我、工作」两个字——运气好。
  「他的职业是什么？」一个字都不共享，于是彻底失败。

  这就是字面匹配的本质问题：它匹配的是**字符**，不是**意思**。
  中文里同一件事的表述方式极多（职业/工作/干什么的/从事什么），
  靠扩充同义词表补不完。第 3 章的句向量把「意思」变成了可计算的东西。
"""
    )


# ----------------------------------------------------------------------
# 墙二：规模天花板
# ----------------------------------------------------------------------


def wall_2_scale() -> None:
    rule("墙二：规模天花板——全表扫描的代价")

    random.seed(42)
    topics = ["项目进展", "客户拜访", "风控模型", "周末计划", "读书笔记", "系统告警"]
    sizes = [100, 1_000, 5_000, 20_000]

    print(f"  {'记忆条数':<12}{'单次检索均值(ms)':<20}{'相对 100 条':<14}{'每天 1 万次查询的耗时'}")
    print("  " + "-" * 74)

    baseline = None
    for n in sizes:
        meter = Meter()
        store = BufferMemory(meter=meter)
        store.add_many(
            [f"{random.choice(topics)}：第 {i} 条记录，涉及若干细节。" for i in range(n)],
            user_id=USER,
        )

        meter.reset()  # 只统计检索，不含写入
        for _ in range(20):
            store.search("风控模型的进展如何", user_id=USER, k=5)

        avg = meter.summary(op="search")["latency_ms_mean"]
        baseline = baseline or avg
        daily_hours = avg * 10_000 / 1000 / 3600
        ratio = f"{avg / baseline:.1f}×"
        print(f"  {n:<12,}{avg:<20.2f}{ratio:<14}{daily_hours:>14.2f} 小时")

    print(
        """
  检索耗时随记忆条数**线性增长**——因为 BufferMemory 每次都要把该用户的
  全部记忆扫一遍、算一遍分数。

  两万条记忆听起来很多？一个每天聊 50 轮的用户，一年就是一万八千条。
  而这还只是一个用户：真实系统要为每个用户维护一份。

  解法有两类，本书都会讲：
    · 建索引，让检索复杂度从 O(N) 降到近似 O(log N)——第 3 章的向量索引。
    · 分层，把绝大多数记忆挪到「冷层」，只在热层里做全扫描——第 7 章。
"""
    )


# ----------------------------------------------------------------------
# 墙三：知识演化
# ----------------------------------------------------------------------


def wall_3_knowledge_evolution() -> None:
    rule("墙三：知识演化——旧事实不会自己消失")

    store = BufferMemory(recency_weight=0.2, meter=Meter())

    timeline = [
        "我在星辰银行工作。",
        "今天开了一天的会。",
        "我下周就要从星辰银行离职了。",
        "新工作定了，我下个月入职长风科技。",
        "周末去看了场电影。",
    ]
    for line in timeline:
        store.add(line, user_id=USER)

    print("  用户按时间顺序说过：")
    for i, line in enumerate(timeline, 1):
        print(f"    {i}. {line}")

    print("\n  现在问：「我在哪工作？」")
    hits = store.search("我在哪工作", user_id=USER, k=3)
    for h in hits:
        print(f"    · {h.content}（分数 {h.score:.3f}）")

    print(
        """
  仔细看这个召回列表，有三处不对劲：

    1. 两条互相矛盾的事实同时出现（星辰银行 / 长风科技），谁新谁旧无从判断；
    2. 真正解释了矛盾的那条——「我下周就要从星辰银行离职了」——**没有被召回**，
       因为它和查询「我在哪工作」几乎没有字面重叠；
    3. 一条完全无关的「周末去看了场电影」凑数进了 top-3，
       因为 k=3 是死的，够不够、该不该都要凑满。

  而且「下个月入职长风科技」是一件**未来才生效**的事：如果今天还没到下个月，
  正确答案仍然是星辰银行，或者「离职待业」。

  BufferMemory 对此无能为力，因为它只做了一件事：把说过的话原样存下来。
  它没有：

    · 判断「离职」使前一条事实失效（→ 第 5 章：事实的有效期与 invalidation）
    · 区分「说这话的时间」和「事情生效的时间」（→ 第 5 章：双时间轴）
    · 把多条原始消息归纳成一条当前有效的事实（→ 第 6 章：extract-update）
    · 根据召回质量动态决定要不要多取几条（→ 第 3 章：阈值与重排）

  把矛盾信息一股脑塞给模型、让它自己判断——这是很多系统的实际做法，
  也是长对话里「模型说法前后不一」的主要来源之一。
"""
    )


def main() -> None:
    print("\n第 1 章实验三：BufferMemory 的三堵墙")
    print("（全部离线运行，约 2 秒完成）")
    t0 = time.perf_counter()

    wall_1_semantic_gap()
    wall_2_scale()
    wall_3_knowledge_evolution()

    rule("小结")
    print(
        f"""  三堵墙，三个方向：

    语义鸿沟  → 表征问题：怎么把「意思」变成可计算的东西      → 第 3、4 章
    规模天花板 → 索引与调度问题：怎么不扫全表                → 第 3、7 章
    知识演化  → 操作问题：怎么更新、失效、遗忘                → 第 5、6 章

  注意这三个方向恰好落在「表征 × 操作 × 载体」三轴上——
  这不是巧合，而是这套分类法为什么值得用的原因：
  **它能把你遇到的问题直接映射到该读哪一章**。

  （本次运行耗时 {time.perf_counter() - t0:.1f} 秒）
"""
    )


if __name__ == "__main__":
    main()
