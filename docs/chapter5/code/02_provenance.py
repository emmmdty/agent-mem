# offline-ok
"""第 5 章实验二：来源追溯与级联删除。

    python docs/chapter5/code/02_provenance.py

第 10 章提了两个要求，这一章第一次真正实现它们：

1. **删除必须级联到派生数据**。删掉一条原始消息时，从它抽出的事实
   也必须处理掉——否则你「删掉」了消息，模型还是能从事实表里读到内容。
2. **上下文要带来源**。模型只有知道每条信息从哪来、什么时候说的，
   才可能判断该信谁。

还要区分两个容易混淆的操作：

    失效（invalidate）：事实不再有效，但记录还在，可以审计、可以回查历史
    删除（purge）：记录本身消失

**被遗忘权要求的是后者。** 把「不再检索」当成「已删除」，是合规上的事故。
"""

from __future__ import annotations

import warnings

from minimem.eval import as_memory_items
from minimem.eval.dataset import BASE_DATE
from minimem.temporal import TemporalGraphMemory
from minimem.utils.embedding import get_embedder
from minimem.utils.metering import Meter

USER = "u_bench"


def build(embedder) -> TemporalGraphMemory:
    store = TemporalGraphMemory(now=BASE_DATE, embedder=embedder, meter=Meter())
    store.add_many(as_memory_items(), user_id=USER)
    return store


def part1_provenance(store: TemporalGraphMemory) -> None:
    print("\n  第一部分：每条事实都知道自己从哪来")
    print("  " + "-" * 66)
    print()
    items = {i.id: i for i in store.all(user_id=USER)}
    for f in store.all_facts(user_id=USER)[:6]:
        src = items.get(f.source_episode)
        origin = src.content if src else "（原始记忆已删除）"
        print(f"    {f.predicate}={f.object}")
        print(f"      ← {origin}")
    print(
        """
    这条链路（事实 → source_episode → 原始记忆）有三个用途：

      · **回查**：模型说「你住在沿江花园」，你能立刻查到这是哪句话推出来的
      · **审计**：事实为什么失效？invalidated_by 指向让它失效的那条
      · **删除**：见第二部分
"""
    )


def part2_cascade_delete(store: TemporalGraphMemory) -> None:
    print("\n  第二部分：删除的两种做法")
    print("  " + "-" * 66)

    target = next(i for i in store.all(user_id=USER) if "过敏" in i.content)
    print(f"\n    要删除的记忆：{target.content}")

    before = [f for f in store.all_facts(user_id=USER) if f.source_episode == target.id]
    print(f"    它派生出的事实：{[f'{f.predicate}={f.object}' for f in before]}")

    print("\n    ── 做法一：只删记忆（很多系统实际做的） ──")
    print("       检索确实找不到那条消息了，但事实表里……")
    store.delete(target.id, user_id=USER)

    remaining = [f for f in store.all_facts(user_id=USER) if f.source_episode == target.id]
    print(f"       事实仍在：{[f'{f.predicate}={f.object}' for f in remaining]}")
    for f in remaining:
        state = "已失效" if f.valid_to else "仍有效"
        print(f"         {f.predicate}={f.object} → {state}")

    print(
        """
       本模块的 delete 至少把派生事实**标记为失效**了，
       所以它不会再出现在「当前有效事实」里。
       但**数据还在内存中**——这在合规上不算删除。
"""
    )

    print("    ── 做法二：物理清除（被遗忘权要求的） ──")
    n = store.purge_facts_of(target.id, user_id=USER)
    still = [f for f in store.all_facts(user_id=USER) if f.source_episode == target.id]
    print(f"       purge_facts_of 清除了 {n} 条派生事实，剩余 {len(still)} 条")

    print(
        """
       真实系统里这个「级联清单」还要长得多：
       向量缓存、图上的边与孤儿实体节点、倒排索引、
       从这条事实再派生出的摘要、备份、日志……

       **每加一种派生数据，删除路径就要加一处。**
       这就是为什么第 10 章说「删除必须是真删除」在记忆系统里格外难。
"""
    )


def part3_audit(embedder) -> None:
    print("\n  第三部分：为什么这条事实失效了")
    print("  " + "-" * 66)

    store = build(embedder)
    items = {i.id: i for i in store.all(user_id=USER)}
    facts = {f.fid: f for f in store.all_facts(user_id=USER)}

    print()
    for f in store.all_facts(user_id=USER):
        if f.predicate != "雇主":
            continue
        print(f"    {f.describe()}")
        if f.invalidated_by and f.invalidated_by in facts and not f.is_terminal:
            killer = facts[f.invalidated_by]
            src = items.get(killer.source_episode)
            print(f"      失效原因 ← {src.content if src else killer.fid}")
    print(
        """
    每一次状态变更都有据可查。这在两个场景下是硬需求：

      · **合规**：用户问「你为什么认为我在长风科技工作」，你要答得出来
      · **排错**：模型答错时，你能定位是抽取错了、冲突判定错了，还是检索错了

    对比一下没有这层的系统：你只能看到一堆原始消息，
    以及一个说不清从哪来的结论。
"""
    )


def main() -> None:
    print("\n第 5 章实验二：来源追溯与级联删除")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedder = get_embedder()

    store = build(embedder)
    part1_provenance(store)
    part2_cascade_delete(store)
    part3_audit(embedder)


if __name__ == "__main__":
    main()
