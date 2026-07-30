# offline-ok
"""第 7 章实验二：摘要的摘要的摘要。

    python docs/chapter7/code/02_summary_decay.py

第 2 章末尾留了一道 🔴 挑战：

    「把一段对话反复摘要 5 次，每轮测一次关键信息覆盖率，画出退化曲线。」

这一章兑现它，因为分层调度会**真的这么干**：一条记忆从 recall 沉到 archival
时被摘要一次；如果它被捞回来又沉下去，再摘要一次。冷层数据在系统里
待得越久，被摘要的次数越多。

本脚本用规则式摘要（截首句 + 截断）。它退化得比 LLM 摘要快，
但**方向相同**——因为每次摘要都在丢信息，而丢掉的信息不会回来。
"""

from __future__ import annotations

from minimem.layered import summarize_by_rule
from minimem.utils.tokens import count_tokens

ORIGINAL = [
    "我在星辰银行工作，是一名信贷风控工程师，主要负责对公业务。",
    "我对花生过敏，点餐时一定要提前确认配料，这个很重要。",
    "我负责的项目代号是 XR-2049，是个反欺诈模型，上线后复核量降了四成。",
    "开会尽量安排在下午，我上午效率低，通常在处理邮件。",
    "我住在城东的沿江花园，通勤大概四十分钟。",
]

PROBES = [
    ("雇主", "星辰银行"),
    ("职业", "风控"),
    ("过敏原", "花生"),
    ("行动指引", "确认配料"),
    ("项目号", "XR-2049"),
    ("项目效果", "四成"),
    ("会议偏好", "下午"),
    ("住址", "沿江花园"),
]


def coverage(text: str) -> tuple[int, list[str]]:
    hit = [name for name, kw in PROBES if kw in text]
    missing = [name for name, kw in PROBES if kw not in text]
    return len(hit), missing


def main() -> None:
    print("\n第 7 章实验二：摘要退化曲线")
    print(f"\n  原始内容 {len(ORIGINAL)} 条，{len(PROBES)} 个关键信息点\n")

    current = list(ORIGINAL)
    rows = []

    text = "\n".join(current)
    n_hit, missing = coverage(text)
    rows.append((0, count_tokens(text), n_hit, missing))

    for round_no in range(1, 6):
        # 每一轮：把当前内容摘要成一条，再作为下一轮的输入
        summary = summarize_by_rule(current, max_chars=max(20, 90 - round_no * 12))
        current = [summary]
        n_hit, missing = coverage(summary)
        rows.append((round_no, count_tokens(summary), n_hit, missing))

    print(f"  {'摘要轮次':<10}{'token':>8}{'保留的关键点':>14}{'覆盖率':>10}   丢掉的")
    print("  " + "-" * 74)
    for round_no, tokens, n_hit, missing in rows:
        rate = n_hit / len(PROBES)
        lost = "、".join(missing[:4]) + ("…" if len(missing) > 4 else "")
        label = "原文" if round_no == 0 else f"第 {round_no} 次"
        print(f"  {label:<10}{tokens:>8}{n_hit:>10}/{len(PROBES)}{rate:>10.0%}   {lost}")

    print("\n  最后剩下的内容：")
    print(f"    「{current[0]}」")

    print(
        """
  读法：

  · **退化是单调的，而且很快。** 第一次摘要就丢掉了「行动指引」
    （「一定要提前确认配料」这类约束在摘要里最先消失，因为它们
    不是「属性=值」的形式）——这一点第 2 章已经预告过。

  · 更麻烦的是**退化不可见**。每一轮的输出看起来都像一个合理的摘要，
    没有任何迹象表明它丢了东西。只有拿原文对照才知道。

  · 看最后剩下的那句话里的项目号：**「XR-2049」被截成了「XR-2」**。
    这比「丢掉」更危险——丢掉至少是显性的缺失，而截断产生的是一个
    **看起来正确、实际错误**的值。下游拿它去查数据库会查不到，
    或者更糟，查到另一个项目。

    任何做截断的地方都要问一句：截出来的东西会不会被当成完整值使用。

  · 这对分层调度意味着什么：

      · **不要在下沉时无脑摘要**。至少保留原文指针（第 2 章的建议、
        第 5 章的 provenance），需要细节时能回查。
      · **限制摘要次数**。给每条记忆记一个 summarized_times，
        超过阈值就不再摘要，宁可占地方。
      · **摘要要从原文做，不要从上一次的摘要做**。
        这是本实验最重要的一条工程建议——它把「摘要的摘要」
        变回「原文的摘要」，退化就不会累积。

  · 规则式摘要退化得比 LLM 摘要快。但**方向是一样的**：
    每次摘要都在丢信息，而丢掉的信息不会回来。
    LLM 只是让你在丢得更慢的同时，多付一笔钱。
"""
    )


if __name__ == "__main__":
    main()
