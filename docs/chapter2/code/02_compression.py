# offline-ok
"""第 2 章实验二：压缩——用信息损失换 token。

    python docs/chapter2/code/02_compression.py

窗口装不下时，除了「丢弃」还有第二条路：**压缩**。
把 20 轮对话压成一段摘要，token 少了一个数量级，信息保留多少？

本实验用一个**规则式**摘要器（正则抽取事实），而不是 LLM。
理由有两个：一是保持离线可跑；二是把「压缩」这个机制和「LLM 有多聪明」分开——
你会看到，即使是规则式压缩，压缩率也已经很可观，
而它丢失的东西（语气、细节、不确定性）恰恰是 LLM 摘要同样会丢的。

真正的 LLM 摘要在第 6、7 章出现，那时会连带算清它的代价。
"""

from __future__ import annotations

import re

from minimem.utils.tokens import count_tokens

# 20 轮对话：有价值的事实散落在大量寒暄里
DIALOG = [
    "你好，我叫小明。",
    "今天天气还不错。",
    "我在星辰银行工作，是一名信贷风控工程师。",
    "刚开完一个会，有点累。",
    "我对花生过敏，点餐要注意。",
    "中午在楼下随便吃了点。",
    "我负责的项目代号是 XR-2049，是个反欺诈模型。",
    "这个项目做了快一年了。",
    "我们用的特征平台叫 FeatBase，是自研的。",
    "文档写得不太全，回头要补。",
    "我住在城西的枫林小区。",
    "通勤大概四十分钟。",
    "我喜欢周末去爬山。",
    "上周去了灵山，人挺多。",
    "开会尽量安排在下午，我上午效率低。",
    "早上一般在处理邮件。",
    "文档麻烦发 Markdown，我不太爱看 Word。",
    "格式统一一点会省事。",
    "我的直属领导是李姐，她带我三年了。",
    "团队一共八个人。",
]

# 规则式事实抽取：只保留能匹配上「属性 = 值」的句子。
# 刻意做得简单，好让你看清它丢了什么。
FACT_RULES: list[tuple[str, str]] = [
    (r"我叫(.{1,6}?)[。，]", "姓名：{}"),
    (r"我在(.{2,12}?)工作", "雇主：{}"),
    (r"是一名(.{2,12}?)[。，]", "职业：{}"),
    (r"我对(.{1,6}?)过敏", "过敏原：{}"),
    (r"项目代号是\s*([A-Za-z0-9\-]+)", "负责项目：{}"),
    (r"特征平台叫\s*(\S+?)[，。]", "使用平台：{}"),
    (r"我住在(.{2,12}?)[。，]", "居住地：{}"),
    (r"我喜欢(.{2,10}?)[。，]", "爱好：{}"),
    (r"(会议|开会)尽量安排在(下午|上午)", "会议偏好：{1}"),
    (r"文档麻烦发\s*(\S+?)[，。]", "文档格式偏好：{}"),
    (r"直属领导是(.{1,5}?)[，。]", "上级：{}"),
]


def rule_summarize(lines: list[str]) -> list[str]:
    """从对话里抽出结构化事实，丢掉其余部分。"""
    facts: list[str] = []
    for line in lines:
        for pattern, template in FACT_RULES:
            m = re.search(pattern, line)
            if m:
                value = m.group(2) if "{1}" in template else m.group(1)
                facts.append(template.replace("{1}", "{}").format(value))
                break
    return facts


def head_truncate(lines: list[str], budget: int) -> list[str]:
    """截断：从最新往回保留，直到预算耗尽。"""
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        t = count_tokens(line)
        if used + t > budget:
            break
        kept.append(line)
        used += t
    return list(reversed(kept))


PROBES = [
    ("姓名", "小明"),
    ("雇主", "星辰"),
    ("过敏原", "花生"),
    ("项目代号", "XR-2049"),
    ("居住地", "枫林"),
    ("会议偏好", "下午"),
    ("上级", "李姐"),
]


def coverage(text: str) -> tuple[int, list[str]]:
    hit = [name for name, kw in PROBES if kw in text]
    return len(hit), [name for name, kw in PROBES if kw not in text]


def main() -> None:
    print("\n第 2 章实验二：压缩")
    print(f"原始对话 {len(DIALOG)} 轮\n")

    full = "\n".join(DIALOG)
    full_tokens = count_tokens(full)

    facts = rule_summarize(DIALOG)
    summary = "\n".join(facts)
    summary_tokens = count_tokens(summary)

    truncated = "\n".join(head_truncate(DIALOG, summary_tokens))
    trunc_tokens = count_tokens(truncated)

    print(f"  {'方案':<22}{'token':>8}{'压缩率':>10}{'关键信息覆盖':>14}")
    print("  " + "-" * 56)
    for label, text, tokens in [
        ("原文全带", full, full_tokens),
        ("截断（保留最近）", truncated, trunc_tokens),
        ("规则式摘要", summary, summary_tokens),
    ]:
        n_hit, missing = coverage(text)
        ratio = f"{tokens / full_tokens:.0%}"
        print(f"  {label:<22}{tokens:>8}{ratio:>10}{f'{n_hit}/{len(PROBES)}':>14}")

    print("\n  规则式摘要的产物：")
    for f in facts:
        print(f"    · {f}")

    _, missing_trunc = coverage(truncated)
    print(f"\n  相同 token 预算下，截断方案漏掉了：{'、'.join(missing_trunc) or '（无）'}")

    print(
        """
  读法：

  · **在相同 token 预算下，摘要的信息密度远高于截断。**
    截断保留的是「最近说的」，摘要保留的是「重要的」——
    而这两者在对话里往往不重合。

  · 但请注意摘要丢掉了什么：

      「我对花生过敏，点餐要注意。」  →  「过敏原：花生」

    丢掉的是「点餐要注意」这个**行动指引**。用户说这句话时，
    真正想表达的是一个持续生效的约束，而不只是一个属性值。
    规则式摘要抽不出它，LLM 摘要也经常抽不出——
    因为它不在「属性 = 值」这个模式里。

  · 更危险的是**不确定性被抹掉**。原文「这个项目做了快一年了」
    是个模糊表述，压成「项目周期：一年」就变成了确定断言。
    压缩会系统性地把「大概、可能、我记得」变成事实——
    这是摘要式记忆最隐蔽的失真来源。

  · 还有一个成本问题这里看不出来：规则式摘要是免费的，
    LLM 摘要不是。每压缩一次要一次调用，而且**压缩后的内容还会被再压缩**
    （第 7 章的分层调度里，冷层数据会被反复摘要）。
    信息在多次摘要中会持续退化——这个现象第 7 章会专门测。

  · 实践建议：**压缩要保留原文指针**。摘要里存一条「来源：第 5 轮消息」，
    需要细节时能回查。第 5 章的 provenance 与第 7 章的分层都依赖这个设计。
"""
    )


if __name__ == "__main__":
    main()
