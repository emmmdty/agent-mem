# offline-ok
"""第 2 章实验一：窗口塞满时，你丢掉的是什么。

    python docs/chapter2/code/01_what_gets_dropped.py

一段 40 轮的对话，关键信息（过敏史）在第 3 轮说出。
窗口预算只够装十几轮。三种淘汰策略，看谁还记得。

这个实验想说明的不是「哪种策略最好」，而是：
**「丢最旧的」是所有对话应用的默认行为，而它恰好丢掉的是最该留的东西。**
用户倾向于在对话开头交代背景——姓名、职业、禁忌、目标——
然后在几十轮之后期待你还记得。
"""

from __future__ import annotations

from minimem import BufferMemory, MemoryItem
from minimem.utils.metering import Meter
from minimem.utils.mock_llm import MockLLM
from minimem.utils.tokens import count_tokens
from minimem.window import WindowMemory

USER = "u1"
BUDGET = 260  # token，刻意调小以便在几十轮内看到效果

# 关键信息在最前面，后面是大量日常闲聊
CONVERSATION = [
    "你好，我叫小明。",
    "我在星辰银行工作，是一名信贷风控工程师。",
    "我对花生过敏，以后帮我点餐一定要注意。",
    *[f"今天第 {i} 件事：例行的项目沟通与文档整理，没有特别的进展。" for i in range(1, 38)],
]

PROBE = "我对什么过敏？"


def build(policy: str, budget: int = BUDGET) -> WindowMemory:
    store = WindowMemory(budget_tokens=budget, policy=policy, n_sink=3, meter=Meter())
    for i, text in enumerate(CONVERSATION):
        item = MemoryItem(text)
        # "pinned" 策略下，把过敏这条钉住。真实系统里由一个分类器或
        # LLM 判断「这条值不值得钉」——那本身就是一次额外的调用，见第 6 章。
        if policy == "pinned" and "过敏" in text:
            item = item.copy_with(metadata={"pinned": True})
        store.add(item, user_id=USER)
        _ = i
    return store


def answer_with(store) -> tuple[str, int]:
    """把窗口内容当上下文喂给 MockLLM，看它能不能答对。"""
    llm = MockLLM(context_limit=100_000)  # 上限设很大，让窗口策略成为唯一变量
    messages = [{"role": "user", "content": it.content} for it in store.visible(user_id=USER)]
    messages.append({"role": "user", "content": PROBE})
    resp = llm.chat(messages)
    return resp.content, resp.tokens_in


def main() -> None:
    print("\n第 2 章实验一：窗口塞满时，你丢掉的是什么")
    print(f"对话共 {len(CONVERSATION)} 轮，关键信息在第 3 轮，窗口预算 {BUDGET} token\n")

    # 基线：无窗口限制
    buf = BufferMemory(meter=Meter())
    buf.add_many(list(CONVERSATION), user_id=USER)
    full_tokens = sum(count_tokens(t) for t in CONVERSATION)

    print(f"  {'策略':<28}{'窗口内条数':>12}{'上下文 token':>14}{'能否答对':>12}")
    print("  " + "-" * 68)
    print(f"  {'全历史（无窗口，作基线）':<28}{len(CONVERSATION):>12}{full_tokens:>14}{'✅':>11}")

    for policy, label in [
        ("recent", "recent（丢最旧，默认行为）"),
        ("sink", "sink（保留开头 3 条）"),
        ("pinned", "pinned（钉住关键信息）"),
    ]:
        store = build(policy)
        answer, tokens = answer_with(store)
        ok = "花生" in answer
        usage = store.usage(user_id=USER)
        print(f"  {label:<28}{usage['窗口内条数']:>12}{tokens:>14}{'✅' if ok else '❌':>11}")

    # ---- 看看 recent 策略到底丢了什么 ----
    recent = build("recent")
    dropped = recent.evicted(user_id=USER)
    print(f"\n  recent 策略丢掉的前 3 条（共丢了 {len(dropped)} 条）：")
    for it in dropped[:3]:
        print(f"    · {it.content}")

    print(
        f"""
  读法：

  · **「丢最旧的」丢掉的恰好是最该留的。** 这不是巧合：
    用户倾向于在对话开头交代背景——姓名、职业、禁忌、目标——
    然后在几十轮之后期待你还记得。

  · sink 策略只是保留开头几条，代价几乎为零，却能救回大量这类信息。
    它的局限也很明显：如果关键信息出现在第 20 轮（比如中途说「我换工作了」），
    sink 一样救不了。

  · pinned 策略能救，但它把问题推给了另一个问题：
    **谁来判断哪条值得钉？** 规则（关键词表）覆盖不全，
    LLM 判断则意味着每条消息一次额外调用——第 6 章会算这笔账。

  · 注意上下文 token 那一列：三种窗口策略的成本几乎相同（都受同一个预算约束），
    但效果差别巨大。**这是本章最划算的一个优化**：
    不增加任何成本，只改变丢弃顺序。

  · 全历史基线当然全对，代价是 {full_tokens} token 且随轮数继续增长（第 1 章实验二）。
    窗口方案把成本压成了常数，代价是**彻底遗忘**——
    被挤出窗口的内容再也检索不到。

    第 3 章的检索方案是第三条路：不丢弃，但也不全带，用检索决定每轮带哪几条。
"""
    )


if __name__ == "__main__":
    main()
