# offline-ok
"""第 7 章实验一：core 层塞满时，谁被换出去。

    python docs/chapter7/code/01_paging.py

第 2 章的窗口方案按**年龄**丢弃，结果丢掉的恰好是开场交代的关键信息。
分层方案换一个依据：**热度**（访问次数 × 时间衰减）。

这个实验对比两种情况：
  1. 关键信息从没被访问过 —— 热度低，一样被换出去
  2. 关键信息被访问过几次 —— 热度上来了，留在 core

结论会有点扫兴：**热度换页并不能自动救回关键信息**，
它只是把「丢什么」的依据从年龄换成了使用频率。
"""

from __future__ import annotations

import warnings

from minimem.base import MemoryItem
from minimem.layered import LayeredMemory
from minimem.utils.embedding import get_embedder
from minimem.utils.metering import Meter

USER = "u_demo"
BUDGET = 160
KEY = "我对花生过敏，点餐要特别注意。"
KEYWORDS = ["花生", "过敏"]

FILLER = [f"今天第 {i} 件事：例行的项目沟通与文档整理。" for i in range(1, 16)]


def build(embedder) -> LayeredMemory:
    return LayeredMemory(core_budget=BUDGET, recall_capacity=6, embedder=embedder, meter=Meter())


def scenario(embedder, *, touch_key: int) -> tuple[LayeredMemory, str]:
    """写入关键信息 + 一堆填充，touch_key 次访问关键信息。"""
    store = build(embedder)
    key_id = store.add(MemoryItem(KEY, metadata={"layer": "core"}), user_id=USER)

    for _ in range(touch_key):
        store.search("过敏", user_id=USER, k=1)

    for text in FILLER:
        store.add(MemoryItem(text, metadata={"layer": "core"}), user_id=USER)
    return store, key_id


def main() -> None:
    print("\n第 7 章实验一：热度换页")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedder = get_embedder()

    print(f"\n  core 预算 {BUDGET} token，写入 1 条关键信息 + {len(FILLER)} 条填充\n")
    print(f"  {'场景':<28}{'关键信息所在层':<16}{'core 层条数':>12}{'换页次数':>10}")
    print("  " + "-" * 68)

    for touch, label in [(0, "关键信息从没被访问过"), (3, "关键信息被访问过 3 次")]:
        store, key_id = scenario(embedder, touch_key=touch)
        layer = store.layer_of(key_id, user_id=USER)
        stats = store.stats(user_id=USER)
        print(f"  {label:<28}{layer:<16}{stats['core']:>12}{stats['换页次数']:>10}")

    print("\n  换出决策的前几条（关键信息从没被访问过的场景）：")
    store, key_id = scenario(embedder, touch_key=0)
    for mid, src, dst, reason in store.evictions[:5]:
        mark = "  ← 关键信息" if mid == key_id else ""
        print(f"    {src} → {dst}：{reason}{mark}")

    print(
        """
  读法：

  · **热度换页不是银弹。** 如果关键信息从没被访问过，它的热度和填充内容
    没有区别，照样被换出 core。这和第 2 章「丢最旧的」犯的是同一类错误，
    只是换了个依据。

  · 被访问过几次之后，它的热度上来了，就能留在 core。这说明热度机制
    **在有反馈的场景里有效**——用户反复问同一件事，系统会自己学会把它常驻。

  · 但注意这个「学会」是滞后的：**它要先答错几次，才能知道什么重要**。
    第 8 章的 importance 打分是另一条路（写入时就判断），
    代价是每条记忆一次 LLM 调用。

    两条路可以合起来：写入时用 importance 给个初值，
    之后用访问热度动态调整。这是本章的 🟡 挑战。

  · 还要注意换页**没有丢数据**——被换出的记忆去了 recall 或 archival，
    仍然可以被检索到（archival 需要显式 include_archival=True）。
    这是分层相对第 2 章窗口方案的根本区别：**窗口是丢弃，分层是降级**。
"""
    )


if __name__ == "__main__":
    main()
