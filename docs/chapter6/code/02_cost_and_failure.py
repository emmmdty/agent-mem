# offline-ok
"""第 6 章实验二：自组织的账单，以及它出错时会怎样。

    python docs/chapter6/code/02_cost_and_failure.py

第 3~5 章的写入侧成本一直是 0（除了嵌入）。这一章不是了。

两部分：
  1. 把账算清楚：写入贵了多少，检索省了多少，读写比多少时打平
  2. 让 LLM 出错，看系统怎么办——**抽取错误会固化进记忆**，
     这比一次回答错误严重得多
"""

from __future__ import annotations

import json
import warnings

from minimem import VectorMemory
from minimem.agentic import AgenticMemory
from minimem.eval import MINI_QUERIES, as_memory_items, recall_at_k
from minimem.utils.embedding import get_embedder
from minimem.utils.llm import ScriptedLLM
from minimem.utils.metering import Meter
from minimem.utils.tokens import TokenCost, count_tokens

USER = "u_bench"
K = 5


def make_llm(meter: Meter, *, fail_rate: float = 0.0, seed: int = 0) -> ScriptedLLM:
    """注册预设逻辑；fail_rate > 0 时按比例返回非法 JSON。"""
    llm = ScriptedLLM(meter=meter, store_label="agentic")
    counter = {"n": 0}

    def maybe_fail() -> bool:
        counter["n"] += 1
        if fail_rate <= 0:
            return False
        # 确定性的「每 N 次失败一次」，保证实验可复现
        return counter["n"] % max(1, int(1 / fail_rate)) == 0

    def note(prompt: str) -> str:
        if maybe_fail():
            return "抱歉，我需要更多上下文才能处理这条消息。"
        text = prompt.split("消息：", 1)[-1].strip()
        # 模拟「去掉寒暄」这个收益：截断到第一个句号
        summary = text.split("。")[0].lstrip("你好，").lstrip("对了，")
        return json.dumps(
            {"summary": summary, "keywords": [summary[:4]], "tags": ["临时"]},
            ensure_ascii=False,
        )

    def decide(prompt: str) -> str:
        return json.dumps({"action": "ADD", "target": None, "reason": "新信息"}, ensure_ascii=False)

    llm.register("加工成一条结构化记忆笔记", note)
    llm.register("判断新信息与已有记忆的关系", decide)
    return llm


def evaluate(store) -> tuple[float, float]:
    recalls, ctx = [], []
    for q in MINI_QUERIES:
        hits = store.search(q.text, user_id=USER, k=K)
        rids = [h.item.metadata.get("rid", "") for h in hits]
        recalls.append(recall_at_k(rids, q.gold, K))
        # 自组织方案注入的是 summary，不是原文——这是它在检索侧的收益
        texts = [h.debug.get("summary") or h.content for h in hits]
        ctx.append(sum(count_tokens(t) for t in texts))
    return sum(recalls) / len(recalls), sum(ctx) / len(ctx)


def part1_cost(embedder) -> None:
    print("\n  第一部分：账单")
    print("  " + "-" * 66)

    rows = []

    m1 = Meter()
    v = VectorMemory(embedder=embedder, mode="hybrid", meter=m1)
    v.add_many(as_memory_items(), user_id=USER)
    r1, c1 = evaluate(v)
    rows.append(("vector（第 3 章）", r1, c1, m1))

    m2 = Meter()
    a = AgenticMemory(llm=make_llm(m2), embedder=embedder, meter=m2)
    a.add_many(as_memory_items(), user_id=USER)
    r2, c2 = evaluate(a)
    rows.append(("agentic（本章）", r2, c2, m2))

    print(
        f"\n  {'方案':<20}{'召回@5':>9}{'检索 ctx':>10}{'LLM 调用':>10}{'写入 token':>12}{'成本$':>10}"
    )
    print("  " + "-" * 74)
    cost = TokenCost()
    for label, recall, ctx, meter in rows:
        s = meter.summary()
        print(
            f"  {label:<20}{recall:>8.1%}{ctx:>10.0f}{s['llm_calls']:>10}"
            f"{s['tokens_total']:>12}{cost(s['tokens_total'], 0):>10.5f}"
        )

    write_tokens = rows[1][3].summary()["tokens_total"]
    saved_per_query = max(rows[0][2] - rows[1][2], 0.001)
    breakeven = write_tokens / saved_per_query

    print(
        f"""
  读法：

  · 写入侧从 0 次 LLM 调用变成 {rows[1][3].summary()["llm_calls"]} 次
    （30 条记忆 × 2 次：note 生成 + update 决策）。

  · 检索侧注入的是 summary 而不是原文，每次省下约
    {rows[0][2] - rows[1][2]:.0f} token。

  · **打平点**：写入的 {write_tokens} token，需要约 {breakeven:.0f} 次检索才能省回来。

    也就是说，如果这批记忆总共被检索不到 {breakeven:.0f} 次，
    自组织就是纯亏。**这就是「读写比」的具体算法**——
    本书说了五次，这里给出它的公式：

        打平所需检索次数 = 写入侧额外 token ÷ 每次检索省下的 token

  · 注意召回率的变化。summary 去掉了寒暄，理论上信噪比更高；
    但它也**丢掉了原文里的细节**，而有些细节正好是答案。
    这两个效应哪个占上风，取决于你的语料和问题类型——
    所以必须测，不能推。
"""
    )


def part2_failure(embedder) -> None:
    print("\n  第二部分：LLM 出错时会怎样")
    print("  " + "-" * 66)

    print(f"\n  {'失败率':<10}{'加工失败数':>12}{'召回@5':>10}{'记忆条数':>10}")
    print("  " + "-" * 44)
    for rate in (0.0, 0.2, 0.5):
        meter = Meter()
        store = AgenticMemory(llm=make_llm(meter, fail_rate=rate), embedder=embedder, meter=meter)
        store.add_many(as_memory_items(), user_id=USER)
        recall, _ = evaluate(store)
        stats = store.stats(user_id=USER)
        print(f"  {rate:<10.0%}{stats['加工失败']:>12}{recall:>10.1%}{stats['记忆数']:>10}")

    print(
        """
  读法：

  · **记忆条数没有减少。** 这是刻意的设计：加工失败时退回原文，
    而不是丢掉这条消息。丢消息是记忆系统能犯的最严重的错误之一——
    用户说过的话凭空消失，而且没有任何报错。

  · 但退回原文意味着**悄悄退化**：这条记忆没有 summary、没有标签、
    没有链接，混在一堆加工好的记忆里，谁也看不出来。
    所以 `Note.degraded` 这个字段是必须的——
    **它把「LLM 失败了多少次」变成可观测的量**。

    一个实践建议：把 degraded 比例接进监控。它突然升高通常意味着
    prompt 被改坏了、模型版本变了、或者上游数据格式变了。

  · 还有一类更危险的失败，这个实验测不出来：**抽取错了但格式正确**。
    LLM 把「我不对花生过敏」总结成「过敏原：花生」，JSON 完全合法，
    系统欣然接受，然后这条错误事实会被**反复检索、反复使用**。

    这就是第 6 章最需要记住的一句话：

      **抽取错误会固化进记忆，比一次回答错误严重得多。**

    一次回答错误只影响一次对话；一条错误记忆影响之后的每一次对话，
    而且它看起来和正确记忆一模一样。

  · 三个可行的防御（都不便宜）：
      1. **校验**：抽出的事实要能在原文里找到依据（字符串包含或蕴含判断）
      2. **保留原文指针**：第 5 章的 provenance，出问题时能回查
      3. **置信度标注**：让 LLM 同时输出置信度，低置信的只存不用
"""
    )


def main() -> None:
    print("\n第 6 章实验二：成本与失败")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedder = get_embedder()
    part1_cost(embedder)
    part2_failure(embedder)


if __name__ == "__main__":
    main()
