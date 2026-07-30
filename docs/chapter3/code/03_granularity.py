# offline-ok
"""第 3 章实验三：一条记忆应该多大？

    python docs/chapter3/code/03_granularity.py

RAG 教程里管这个叫 chunking，讨论的是「文档切多长」。
记忆场景的问题不同：对话本来就是一条条来的，你的选择是**要不要合并、要不要抽取**。

三种粒度：

    消息级   每条用户消息存一条            —— 最细，最忠实，噪声也最多
    会话级   把相邻若干条合并成一条         —— 上下文完整，但一条里混了多个主题
    事实级   从消息中抽出「属性 = 值」      —— 最省，但需要抽取，且会丢细节

这个实验只比前两种（第三种需要 LLM，留到第 6 章），
但会把事实级的**理论上界**算出来，让你知道值不值得为它引入 LLM 调用。
"""

from __future__ import annotations

from minimem.base import MemoryItem
from minimem.eval import MINI_MEMORIES, MINI_QUERIES, precision_at_k, recall_at_k
from minimem.utils.embedding import get_embedder
from minimem.utils.metering import Meter
from minimem.utils.tokens import count_tokens
from minimem.vector import VectorMemory

USER = "u_bench"
K = 5


def build_message_level() -> list[MemoryItem]:
    """消息级：一条消息一条记忆。"""
    return [
        MemoryItem(r.text, kind="message", metadata={"rid": r.rid, "rids": [r.rid]})
        for r in MINI_MEMORIES
    ]


def build_session_level(group_size: int = 5) -> list[MemoryItem]:
    """会话级：相邻 group_size 条合并成一条。

    注意合并后一条记忆对应多个 rid——评测时只要 gold 里任一 rid 在其中就算召回。
    这里有一个容易忽略的偏差：**合并会让召回率虚高**，因为一条大记忆盖住了更多 gold。
    所以必须同时看 precision 和返回的 token 量。
    """
    items: list[MemoryItem] = []
    for i in range(0, len(MINI_MEMORIES), group_size):
        chunk = MINI_MEMORIES[i : i + group_size]
        items.append(
            MemoryItem(
                "\n".join(c.text for c in chunk),
                kind="session",
                metadata={"rid": chunk[0].rid, "rids": [c.rid for c in chunk]},
            )
        )
    return items


def evaluate(items: list[MemoryItem], label: str, embedder) -> dict[str, float | str]:
    store = VectorMemory(embedder=embedder, mode="hybrid", meter=Meter())
    store.add_many(list(items), user_id=USER)

    recalls, precisions, ctx_tokens = [], [], []
    for q in MINI_QUERIES:
        hits = store.search(q.text, user_id=USER, k=K)
        # 一条记忆可能覆盖多个 rid，展开后再算指标
        retrieved: list[str] = []
        for h in hits:
            retrieved.extend(h.item.metadata.get("rids", []))
        recalls.append(recall_at_k(retrieved, q.gold))
        precisions.append(precision_at_k(retrieved, q.gold))
        ctx_tokens.append(sum(count_tokens(h.content) for h in hits))

    return {
        "label": label,
        "条数": len(items),
        "召回率": sum(recalls) / len(recalls),
        "准确率": sum(precisions) / len(precisions),
        "每次检索的上下文 token": sum(ctx_tokens) / len(ctx_tokens),
    }


def noise_analysis(embedder) -> None:
    """算一笔账：检索回来的 token 里，有多少是真正有用的？"""
    print("\n  噪声分析（消息级，k=5）")

    store = VectorMemory(embedder=embedder, mode="hybrid", meter=Meter())
    store.add_many(build_message_level(), user_id=USER)

    useful, total, n = 0, 0, 0
    for q in MINI_QUERIES:
        if not q.gold:
            continue
        hits = store.search(q.text, user_id=USER, k=K)
        total += sum(count_tokens(h.content) for h in hits)
        useful += sum(count_tokens(h.content) for h in hits if h.item.metadata["rid"] in q.gold)
        n += 1

    print(f"    平均每次检索返回：{total / n:.0f} token")
    print(f"    其中命中 gold 的：  {useful / n:.0f} token")
    print(f"    噪声占比：          {1 - useful / total:.0%}")
    print()
    print("    这就是事实级抽取真正的价值来源——它不只是压缩，更是**去噪**：")
    print("    把「我下周就要从星辰银行离职了，交接已经开始。」这样一条消息，")
    print("    压成「当前雇主：星辰银行（离职中）」，既短，又不带无关信息。")
    print("    代价是每条消息一次 LLM 调用。是否划算取决于**读写比**——")
    print("    写一次读一百次，抽取绝对划算；写一次读一次，就是纯亏。第 6 章算这笔账。")


def main() -> None:
    print("\n第 3 章实验三：记忆粒度")
    embedder = get_embedder()
    print(f"句向量模型：{embedder.name}\n")

    rows = [
        evaluate(build_message_level(), "消息级（30 条）", embedder),
        evaluate(build_session_level(3), "会话级 / 3 条一组", embedder),
        evaluate(build_session_level(5), "会话级 / 5 条一组", embedder),
        evaluate(build_session_level(10), "会话级 / 10 条一组", embedder),
    ]

    print(f"  {'粒度':<22}{'条数':>6}{'召回率':>10}{'准确率':>10}{'上下文 token':>16}")
    print("  " + "-" * 66)
    for r in rows:
        print(
            f"  {r['label']:<22}{r['条数']:>6}{r['召回率']:>10.1%}"
            f"{r['准确率']:>10.1%}{r['每次检索的上下文 token']:>16.0f}"
        )

    noise_analysis(embedder)

    print(
        """
  读法：

  · 合并粒度越粗，**召回率越高、准确率越低、上下文越贵**。
    这三者是同一个动作的三个侧面：一条大记忆当然更容易「盖住」答案，
    但它同时把大量无关内容一起塞进了 prompt。

  · 看最后一行「10 条一组」：全库只剩 3 条记忆，k=5 直接把整个库都返回了，
    召回率当然是 100%。**这个 100% 什么也没证明**——
    它等价于「不检索，全塞进去」，正是第 1 章我们想避免的方案。

  · 只看召回率会得出「粒度越粗越好」的荒谬结论。
    这是本章最需要记住的一件事：**检索指标必须成组看**。
    第 10 章会给出完整的指标组合，以及每个指标单独看时会怎么骗你。

  · 消息级 + top-5 通常是记忆场景的合理默认值。
    真正需要调整的情形是：单条消息很短（比如聊天记录一句一条），
    这时按「对话轮」或「话题段」合并会更好。

  · 事实级的上限收益可观，但它的代价是每条消息一次 LLM 调用。
    **判断依据是读写比**，不是「听起来更高级」。第 6 章会把这笔账算完整。
"""
    )


if __name__ == "__main__":
    main()
