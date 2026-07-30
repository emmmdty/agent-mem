"""``MockLLM``：一个只会做一件事的假模型——**只回答上下文里写着的东西**。

为什么教程需要它：

- **确定性**。同样的输入永远同样的输出，实验结果可复现，CI 里能跑。
- **零成本、可离线**。第 1~5 章因此不需要任何 API Key。
- **把「无状态」这件事变得可见**。真实 LLM 记不住上一轮时，往往会编一个听起来
  合理的答案（幻觉），反而模糊了「它到底知不知道」这个问题。MockLLM 不会编，
  它只会老实说「我不知道」。这让「记忆有没有起作用」变成一个二值的、可断言的事实。

它当然不是真模型：不会推理、不会改写、不会幻觉。凡是需要 LLM 判断力的章节
（第 6 章起的抽取、反思、归纳），必须换成真模型——那时候 MockLLM 会明确拒绝工作，
而不是给你一个看起来能跑的假象。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from minimem.utils.tokens import count_tokens

__all__ = ["MockLLM", "LLMResponse"]


@dataclass
class LLMResponse:
    """一次调用的返回，附带用量。用量字段与 OpenAI 的响应对齐，便于后续替换。"""

    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    matched_fact: str | None = None
    debug: dict[str, str] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


# 「事实」的抽取规则：从用户说过的话里认出「属性 = 值」。
# 刻意用正则而不是 LLM，是为了让第 1 章保持零依赖；
# 第 6 章会展示同一件事用 LLM 来做的效果差异与代价差异。
_FACT_PATTERNS: list[tuple[str, str]] = [
    (r"我叫([一-鿿 A-Za-z]{1,10})", "姓名"),
    (r"我(?:的名字)?是([一-鿿]{2,4})(?:[，。,.]|$)", "姓名"),
    (r"我在(.{2,20}?)(?:工作|上班)", "公司"),
    (r"我(?:现在)?(?:在|就职于)([一-鿿A-Za-z]{2,18}?(?:公司|集团|银行|科技|大学|研究院))", "公司"),
    (r"我(?:做|从事)(.{2,15}?)(?:方面的?工作|的工作|工作)", "职业"),
    (r"是(?:一名|一个|一位)(.{2,15}?)(?:[，。,.]|$)", "职业"),
    (r"我(?:喜欢|爱好是|平时喜欢)(.{1,20}?)(?:[，。,.]|$)", "爱好"),
    (r"我(?:住在|家在)(.{2,15}?)(?:[，。,.]|$)", "居住地"),
    (r"我(?:对|吃)(.{1,10}?)过敏", "过敏原"),
]

# 「问题」的识别规则：问的是哪个属性。
_QUESTION_PATTERNS: list[tuple[str, str]] = [
    (r"我叫什么|我的名字|你还记得我(?:是谁|叫什么)", "姓名"),
    (r"我在哪(?:里|儿)?(?:工作|上班)|我的公司|我(?:就职|供职)", "公司"),
    (r"我(?:是)?(?:做什么|干什么|什么职业|什么工作)|我的(?:职业|工作)", "职业"),
    (r"我喜欢什么|我的爱好", "爱好"),
    (r"我住在哪|我的住址|我家在哪", "居住地"),
    (r"我对什么过敏|我的过敏", "过敏原"),
]

_UNKNOWN = "抱歉，我不知道——你没有在我能看到的内容里提过这件事。"


class MockLLM:
    """一个确定性的假模型。

    行为定义得很窄，正是为了让实验的因果关系清楚：

    1. 扫描传入的 ``messages``（这就是它的**全部**可见范围，等同真实 LLM 的上下文窗口）；
    2. 用正则抽出其中出现过的「属性 = 值」；
    3. 若最后一条是提问且问的属性被抽到了，就回答它；否则回答「不知道」。

    换句话说：**它记住的只有你这次告诉它的**。这正是无状态模型的定义。

    Args:
        context_limit: 模拟的上下文窗口大小（token）。超出的部分会被从**最前面**
            截断——和真实推理服务在超长时的常见处理一致。第 2 章会用它演示
            「历史被悄悄截掉」这件事有多容易发生而不被察觉。
    """

    def __init__(self, *, context_limit: int = 4096) -> None:
        self.context_limit = context_limit
        self.call_count = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0

    # ------------------------------------------------------------------

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        """接收 OpenAI 风格的 messages，返回回复。"""
        self.call_count += 1

        visible, truncated = self._apply_context_limit(messages)
        tokens_in = sum(count_tokens(m["content"]) for m in visible)

        question = visible[-1]["content"] if visible else ""
        attr = self._detect_question(question)

        if attr is None:
            # 不是事实型提问就只作确认。注意它不会说「我记住了」——
            # 一个无状态模型没有资格这么说，说了就是在骗人。
            content = "好的。"
            matched = None
        else:
            facts = self._extract_facts(visible)
            value = facts.get(attr)
            if value:
                content = f"你的{attr}是{value}。"
                matched = f"{attr}={value}"
            else:
                content = _UNKNOWN
                matched = None

        tokens_out = count_tokens(content)
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out

        return LLMResponse(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            matched_fact=matched,
            debug={
                "问的属性": attr or "(未识别)",
                "可见消息数": str(len(visible)),
                "被截断的消息数": str(truncated),
            },
        )

    # ------------------------------------------------------------------

    def _apply_context_limit(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], int]:
        """从后往前保留消息，直到超出上下文预算。

        注意保留方向：**丢最旧的**。这是几乎所有对话应用的默认行为，
        也是「刚开始说的重要信息最先消失」这个经典 bug 的来源。
        """
        kept: list[dict[str, str]] = []
        used = 0
        for msg in reversed(messages):
            t = count_tokens(msg["content"])
            if used + t > self.context_limit and kept:
                break
            kept.append(msg)
            used += t
        kept.reverse()
        return kept, len(messages) - len(kept)

    @staticmethod
    def _extract_facts(messages: list[dict[str, str]]) -> dict[str, str]:
        """从可见消息里抽事实。后出现的覆盖先出现的——这是最朴素的「更新」策略。

        它的问题在第 5、6 章展开：如果用户说的是「我下个月要跳槽去 B 公司」，
        按出现顺序覆盖就会把一件**未来才生效**的事当成现在的事实。
        """
        facts: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") != "user":
                continue
            for pattern, attr in _FACT_PATTERNS:
                m = re.search(pattern, msg["content"])
                if m:
                    value = m.group(1).strip()
                    # 提问句里也含「我叫……」这样的字样（「我叫什么来着」），
                    # 不过滤掉的话，一句提问就会把真实事实覆盖成疑问词。
                    if value and not re.match(r"^(什么|谁|哪|啥|多少)", value):
                        facts[attr] = value
        return facts

    @staticmethod
    def _detect_question(text: str) -> str | None:
        for pattern, attr in _QUESTION_PATTERNS:
            if re.search(pattern, text):
                return attr
        return None

    # ------------------------------------------------------------------

    def usage(self) -> dict[str, int]:
        return {
            "调用次数": self.call_count,
            "输入 token": self.total_tokens_in,
            "输出 token": self.total_tokens_out,
        }

    def reset(self) -> None:
        self.call_count = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
