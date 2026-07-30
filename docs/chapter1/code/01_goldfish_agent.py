# offline-ok
"""第 1 章实验一：三种 Agent，同一段对话。

无需 API Key，无需联网，秒级完成。

    python docs/chapter1/code/01_goldfish_agent.py

对比三种取用历史的方式：

1. ``GoldfishAgent``   —— 只发当前这一句。金鱼记忆，七秒忘。
2. ``FullHistoryAgent``—— 把全部历史都发过去。答得对，但代价随轮数增长。
3. ``MemoryAgent``     —— 检索相关的几条再发。这是「记忆系统」的雏形。

结论会在最后打成一张表：**答对与否**和**花了多少 token**必须一起看。
"""

from __future__ import annotations

from dataclasses import dataclass

from minimem import BufferMemory, MemoryItem
from minimem.utils.metering import Meter
from minimem.utils.mock_llm import MockLLM

SYSTEM = "你是一个助理。只根据对话中出现过的信息回答，不要编造。"

# 一段普通的多轮对话：前面交代事实，中间闲聊，最后回头提问。
# 这是长期记忆最典型的失败场景——重要信息说得早，被问到时已经隔了很远。
CONVERSATION: list[tuple[str, bool]] = [
    ("你好，我叫小明。", False),
    ("我在星辰银行工作，是一名信贷风控工程师。", False),
    ("我对花生过敏，以后帮我点餐要注意。", False),
    ("我喜欢周末去爬山。", False),
    ("今天天气怎么样？", False),
    ("帮我把这段话翻译成英文。", False),
    ("推荐几部科幻电影吧。", False),
    ("对了，我叫什么来着？", True),  # ← 需要记忆才能答对
    ("我对什么过敏？", True),  # ← 需要记忆才能答对
]


class GoldfishAgent:
    """每一轮都从零开始：只把当前这句话发给模型。"""

    label = "无记忆（只发当前轮）"

    def __init__(self) -> None:
        self.llm = MockLLM()

    def reply(self, user_text: str) -> tuple[str, list[str]]:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text},
        ]
        return self.llm.chat(messages).content, []


class FullHistoryAgent:
    """把此前说过的每一句都重新发一遍。"""

    label = "全历史注入"

    def __init__(self) -> None:
        self.llm = MockLLM()
        self.history: list[dict[str, str]] = []

    def reply(self, user_text: str) -> tuple[str, list[str]]:
        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": SYSTEM}, *self.history]
        resp = self.llm.chat(messages)
        self.history.append({"role": "assistant", "content": resp.content})
        return resp.content, []


class MemoryAgent:
    """先检索最相关的几条历史，只把它们发过去。"""

    label = "记忆检索（k=3）"

    def __init__(self, k: int = 3) -> None:
        self.llm = MockLLM()
        self.k = k
        self.memory = BufferMemory(recency_weight=0.15, meter=Meter())
        self.user_id = "u_xiaoming"

    def reply(self, user_text: str) -> tuple[str, list[str]]:
        # 检索发生在写入之前：否则当前这句必然命中它自己，白占一个名额
        recalled = self.memory.search(user_text, user_id=self.user_id, k=self.k)
        self.memory.add(MemoryItem(user_text, kind="message"), user_id=self.user_id)

        context = "\n".join(f"- {r.content}" for r in recalled)
        messages = [{"role": "system", "content": SYSTEM}]
        if context:
            messages.append({"role": "user", "content": f"以下是相关的历史信息：\n{context}"})
        messages.append({"role": "user", "content": user_text})

        detail = [f"{r.content}（分数 {r.score:.3f}）" for r in recalled]
        return self.llm.chat(messages).content, detail


@dataclass
class RunResult:
    label: str
    correct: int
    total: int
    tokens: int


def run(agent) -> RunResult:
    print(f"\n{'=' * 70}")
    print(f"  {agent.label}")
    print("=" * 70)

    correct = 0
    total_probes = 0
    for text, is_probe in CONVERSATION:
        answer, recalled = agent.reply(text)
        print(f"  用户 > {text}")
        # 只在回头题上打印召回内容，否则刷屏
        if is_probe and recalled:
            print("         ↳ 召回：")
            for line in recalled:
                print(f"           · {line}")
        mark = ""
        if is_probe:
            total_probes += 1
            ok = "不知道" not in answer
            correct += ok
            mark = "  ✅" if ok else "  ❌"
        print(f"  助理 < {answer}{mark}")

    usage = agent.llm.usage()
    return RunResult(
        label=agent.label,
        correct=correct,
        total=total_probes,
        tokens=usage["输入 token"],
    )


def main() -> None:
    print("\n第 1 章实验一：同一段对话，三种取用历史的方式")
    print("（模型用 MockLLM：确定性、离线、只回答上下文里写着的东西，不会幻觉）")

    rows = [run(GoldfishAgent()), run(FullHistoryAgent()), run(MemoryAgent())]

    print(f"\n{'=' * 70}")
    print("  对比")
    print("=" * 70)
    print(f"  {'方案':<22}{'答对':<10}{'输入 token':<14}{'相对成本':<10}")
    print("  " + "-" * 62)
    base = max(1, rows[0].tokens)
    for r in rows:
        score = f"{r.correct}/{r.total}"
        print(f"  {r.label:<22}{score:<10}{r.tokens:<14}{r.tokens / base:.2f}×")

    print(
        """
  读法：

  · 无记忆方案 token 最省，两道回头题全错。省下的钱买不到能用的产品。

  · 全历史方案全对，代价是每一轮都把此前所有内容重发一遍。这段对话只有 9 轮，
    差距看着还能忍；实验二会把轮数拉到几百轮，让增长曲线自己说话。

  · 检索方案花了不到全历史一半的 token，但**只答对一题**——注意看「我叫什么来着」
    那一轮的召回列表：它召回的是最近的几句闲聊，而不是开场那句「我叫小明」。

    原因是 BufferMemory 用字面词重叠（Jaccard）打分：查询「对了，我叫什么来着？」
    和「你好，我叫小明。」只共享「我、叫」两个字，得分被稀释；再叠加一点新近性权重，
    近处的无关内容就排到了前面。

    这不是参数没调好，是方法本身的天花板：**字面重叠无法表达语义相近**。
    第 3 章会用句向量把这堵墙推倒，然后你会发现它又立起一堵新的。

  这三行就是全书的问题空间：在「答得对」和「花得少」之间，
  每一章都在尝试一种新的折中，并为此付出一种新的代价。
"""
    )


if __name__ == "__main__":
    main()
