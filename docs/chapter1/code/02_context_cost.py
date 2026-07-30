# offline-ok
"""第 1 章实验二：把「全历史注入」的账算清楚。

    python docs/chapter1/code/02_context_cost.py

实验一里 9 轮对话的差距还不刺眼。这个脚本把轮数拉到 500，比较四种策略的
**累计输入 token**——也就是你真正要付钱的那个数字。

关键结论：全历史注入的累计成本是 O(N²)。这不是因为它「效率低」，
而是因为**每一轮都要把前面所有轮重发一遍**：第 N 轮发 N 条，累计就是 1+2+…+N。
"""

from __future__ import annotations

from dataclasses import dataclass

from minimem.utils.tokens import TokenCost, count_tokens

# 一轮对话的典型规模。真实场景差异很大，这里取一个偏保守的估计。
USER_MSG = "帮我看看这份季度报告里的风险敞口，重点关注制造业客户的授信集中度。"
ASSISTANT_MSG = (
    "好的。从报告看，制造业客户占授信总额的 34%，高于行业均值。"
    "建议关注三家单一客户敞口超过一级资本 8% 的情况，并复核其抵质押覆盖率。"
)
SYSTEM_MSG = "你是一个金融风控助理。"

TURNS_TO_SHOW = [10, 50, 100, 500]
WINDOW_SIZE = 20  # 滑窗策略保留最近多少轮
TOP_K = 3  # 检索策略每轮带回几条


@dataclass
class Strategy:
    name: str
    note: str

    def tokens_at_turn(self, turn: int, per_turn: int, system: int) -> int:
        raise NotImplementedError


class NoMemory(Strategy):
    def tokens_at_turn(self, turn: int, per_turn: int, system: int) -> int:
        return system + per_turn // 2  # 只发当前这条用户消息


class FullHistory(Strategy):
    def tokens_at_turn(self, turn: int, per_turn: int, system: int) -> int:
        # 第 turn 轮要发：system + 前 turn-1 轮的完整问答 + 本轮用户消息
        return system + (turn - 1) * per_turn + per_turn // 2


class SlidingWindow(Strategy):
    def tokens_at_turn(self, turn: int, per_turn: int, system: int) -> int:
        kept = min(turn - 1, WINDOW_SIZE)
        return system + kept * per_turn + per_turn // 2


class Retrieval(Strategy):
    def tokens_at_turn(self, turn: int, per_turn: int, system: int) -> int:
        # 检索回 TOP_K 条历史消息（只是消息，不是完整问答对）
        recalled = min(turn - 1, TOP_K) * (per_turn // 2)
        return system + recalled + per_turn // 2


def main() -> None:
    per_turn = count_tokens(USER_MSG) + count_tokens(ASSISTANT_MSG)
    system = count_tokens(SYSTEM_MSG)
    cost = TokenCost()

    print("\n第 1 章实验二：四种策略的累计输入 token")
    print(f"（单轮问答约 {per_turn} token，system 约 {system} token）\n")

    strategies = [
        NoMemory("无记忆", "只发当前这句"),
        FullHistory("全历史注入", "每轮重发全部历史"),
        SlidingWindow("滑动窗口", f"只保留最近 {WINDOW_SIZE} 轮"),
        Retrieval("检索 top-k", f"每轮带回 {TOP_K} 条相关记忆"),
    ]

    header = f"  {'策略':<14}" + "".join(f"{f'{n} 轮':>12}" for n in TURNS_TO_SHOW)
    print(header)
    print("  " + "-" * (14 + 12 * len(TURNS_TO_SHOW)))

    totals: dict[str, list[int]] = {}
    for st in strategies:
        cumulative = []
        running = 0
        target = iter(TURNS_TO_SHOW)
        next_mark = next(target)
        for turn in range(1, max(TURNS_TO_SHOW) + 1):
            running += st.tokens_at_turn(turn, per_turn, system)
            if turn == next_mark:
                cumulative.append(running)
                next_mark = next(target, -1)
        totals[st.name] = cumulative
        cells = "".join(f"{v:>12,}" for v in cumulative)
        print(f"  {st.name:<14}{cells}")

    print(f"\n  {'策略':<14}{'500 轮累计成本（美元）':>24}{'相对最省':>12}")
    print("  " + "-" * 52)
    cheapest = min(t[-1] for t in totals.values())
    for st in strategies:
        total = totals[st.name][-1]
        # 只算输入侧：输出 token 与策略无关，四种方案一样多
        print(f"  {st.name:<14}{cost(total, 0):>24.4f}{total / cheapest:>11.1f}×")

    ratio = totals["全历史注入"][-1] / totals["检索 top-k"][-1]
    print(
        f"""
  读法：

  · 全历史注入在 500 轮时的累计输入 token 是检索方案的 {ratio:.0f} 倍。
    增长是**二次的**：轮数翻倍，成本翻四倍。

  · 滑动窗口把增长压回了线性，代价是**窗口外的信息彻底消失**——
    用户第 3 轮说的过敏史，到第 300 轮时已经不在窗口里了。
    第 2 章会讲怎么让窗口「聪明地」丢东西（保留 attention sink、保留摘要）。

  · 检索方案的单轮成本几乎是常数，这正是它值得存在的理由。
    但注意：这张表只算了**发给模型的 token**。检索本身的开销（嵌入、索引、
    向量搜索，第 6 章起还有 LLM 抽取）没有算进去。
    第 3 章起我们会把这部分补上——很多方案省下的 prompt 钱，
    转头就花在了写入侧。

  · 还有一个这张表算不出来的东西：**质量**。
    上下文长不等于模型用得好，检索少不等于漏掉了关键信息。
    第 10 章的评测 harness 就是为了把质量和成本放进同一张表。

  一句提醒：真实 API 的价格随模型、随时间变动，而且大多有 prompt 缓存机制
  （重复前缀可以便宜很多）。本脚本用的是 .env 里的占位价格，
  它给的是**量级**，不是账单。
"""
    )


if __name__ == "__main__":
    main()
