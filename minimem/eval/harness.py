"""``EvalHarness``：把效果和代价放进同一张表。

这是全书结构上的收口。第 1 章定的 ``MemoryStore`` 接口、``Meter`` 计量器，
到这里终于兑现——任意两种实现都能在同一份数据上跑出可比的
「召回 / 准确 / 排序 / 上下文 token / 延迟 / LLM 调用 / 成本」七列。

**它测的是什么，不测什么**，请务必看清楚：

============  ====================================  ==========================
指标           测的是                                 测不到的
============  ====================================  ==========================
recall        该召回的召回了多少                       召回的顺序、召回的噪声
precision     召回的里面有多少是该召回的                漏掉了什么
mrr           第一条正确结果排多靠前                   后面的结果有没有用
answerable    召回的文本里有没有出现答案关键词           模型能不能**用好**这些文本
abstain       该拒答时有没有返回空                     模型会不会硬编答案
ctx_tokens    每次检索往 prompt 里塞了多少 token       这些 token 值不值
latency       检索耗时                                写入侧的摊销成本
cost          LLM 调用花的钱                          自建模型的机器成本
============  ====================================  ==========================

最重要的一条：``answerable`` **不是端到端准确率**。它只回答「检索层有没有
把答案送到模型面前」，不回答「模型有没有答对」。两者的差距在第 10 章正文里
有专门一节——那个差距正是很多系统「召回率很高但用户体验很差」的来源。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from minimem.base import MemoryStore
from minimem.eval.dataset import MINI_MEMORIES, MINI_QUERIES, MemoryRecord, Query, as_memory_items
from minimem.eval.metrics import mrr, precision_at_k, recall_at_k
from minimem.utils.metering import Meter
from minimem.utils.tokens import count_tokens

__all__ = ["EvalResult", "EvalHarness"]


@dataclass
class EvalResult:
    """一次评测的完整结果。效果与代价放在同一个对象里，刻意不允许只取一半。"""

    label: str
    n_queries: int
    recall: float
    precision: float
    mrr: float
    answerable: float
    abstain_rate: float
    ctx_tokens: float
    search_ms_mean: float
    search_ms_p95: float
    write_s: float
    llm_calls: int
    cost_usd: float
    by_capability: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "方案": self.label,
            "召回": self.recall,
            "准确": self.precision,
            "MRR": self.mrr,
            "可答": self.answerable,
            "ctx_token": self.ctx_tokens,
            "延迟ms": self.search_ms_mean,
            "p95ms": self.search_ms_p95,
            "成本$": self.cost_usd,
        }


class EvalHarness:
    """在一份数据上评测任意多个 ``MemoryStore`` 实现。

    Args:
        memories: 待写入的记忆。默认用 MiniBench。
        queries: 查询集。默认用 MiniBench。
        k: 每次检索取几条。**这个值必须在结果里一起报告**——
            第 3 章的 k 敏感性实验说明了不指明 k 的对比没有意义。

    Example:
        >>> harness = EvalHarness()                                 # doctest: +SKIP
        >>> r1 = harness.run(lambda m: BufferMemory(meter=m), "buffer")   # doctest: +SKIP
        >>> print(harness.table([r1]))                              # doctest: +SKIP
    """

    def __init__(
        self,
        memories: list[MemoryRecord] | None = None,
        queries: list[Query] | None = None,
        *,
        k: int = 5,
        user_id: str = "u_eval",
    ) -> None:
        self.memories = memories if memories is not None else MINI_MEMORIES
        self.queries = queries if queries is not None else MINI_QUERIES
        self.k = k
        self.user_id = user_id

    # ------------------------------------------------------------------

    def run(self, factory: Callable[[Meter], MemoryStore], label: str) -> EvalResult:
        """构建一个 store、写入全部记忆、跑完全部查询。

        ``factory`` 接收一个 ``Meter`` 而不是直接接收 store 实例，
        是为了保证每次评测用的是**干净的计量器**——否则上一次评测的
        延迟和 token 会混进来，这类污染在对比实验里非常隐蔽。
        """
        meter = Meter()
        store = factory(meter)

        items = as_memory_items(self.memories)
        t0 = time.perf_counter()
        store.add_many(list(items), user_id=self.user_id)
        write_s = time.perf_counter() - t0

        # 写入的计量不算进检索统计，但 LLM 调用与成本要累计
        write_records = list(meter.records)
        meter.reset()

        recalls, precisions, mrrs, answerables, ctx_tokens = [], [], [], [], []
        abstained, abstain_total = 0, 0
        per_cap: dict[str, list[float]] = {}

        for q in self.queries:
            hits = store.search(q.text, user_id=self.user_id, k=self.k)
            rids = [h.item.metadata.get("rid", "") for h in hits]
            texts = [h.content for h in hits]

            recalls.append(recall_at_k(rids, q.gold, self.k))
            precisions.append(precision_at_k(rids, q.gold, self.k))
            mrrs.append(mrr(rids, q.gold))
            ctx_tokens.append(sum(count_tokens(t) for t in texts))
            per_cap.setdefault(q.capability, []).append(recalls[-1])

            if q.should_abstain:
                abstain_total += 1
                abstained += not hits
                # 拒答题没有正确答案可言，不计入 answerable
            else:
                joined = "".join(texts)
                answerables.append(float(any(kw in joined for kw in q.answer_keywords)))

        search_summary = meter.summary(op="search")
        all_records = write_records + meter.records

        return EvalResult(
            label=label,
            n_queries=len(self.queries),
            recall=_mean(recalls),
            precision=_mean(precisions),
            mrr=_mean(mrrs),
            answerable=_mean(answerables),
            abstain_rate=abstained / abstain_total if abstain_total else 0.0,
            ctx_tokens=_mean(ctx_tokens),
            search_ms_mean=search_summary["latency_ms_mean"],
            search_ms_p95=search_summary["latency_ms_p95"],
            write_s=write_s,
            llm_calls=sum(r.llm_calls for r in all_records),
            cost_usd=sum(r.cost_usd for r in all_records),
            by_capability={c: _mean(v) for c, v in per_cap.items()},
        )

    def compare(self, factories: dict[str, Callable[[Meter], MemoryStore]]) -> list[EvalResult]:
        return [self.run(f, label) for label, f in factories.items()]

    # ------------------------------------------------------------------

    def table(self, results: list[EvalResult]) -> str:
        """效果与代价并列的主表。**刻意不提供只显示效果的选项。**"""
        lines = [
            f"  评测集：{len(self.memories)} 条记忆 / {len(self.queries)} 个查询，k={self.k}",
            "",
            f"  {'方案':<20}{'召回':>8}{'准确':>8}{'MRR':>8}{'可答':>8}"
            f"{'ctx tok':>10}{'延迟ms':>9}{'p95':>8}{'写入s':>8}{'成本$':>10}",
            "  " + "-" * 97,
        ]
        for r in results:
            lines.append(
                f"  {r.label:<20}{r.recall:>7.1%}{r.precision:>8.1%}{r.mrr:>8.3f}"
                f"{r.answerable:>8.1%}{r.ctx_tokens:>10.0f}{r.search_ms_mean:>9.2f}"
                f"{r.search_ms_p95:>8.2f}{r.write_s:>8.2f}{r.cost_usd:>10.4f}"
            )
        return "\n".join(lines)

    def capability_table(self, results: list[EvalResult]) -> str:
        """分能力召回率。总分掩盖的差异都在这张表里。"""
        caps: list[str] = []
        for r in results:
            for c in r.by_capability:
                if c not in caps:
                    caps.append(c)

        lines = [
            f"  {'能力':<10}" + "".join(f"{r.label[:12]:>14}" for r in results),
            "  " + "-" * (10 + 14 * len(results)),
        ]
        for cap in caps:
            row = f"  {cap:<10}"
            for r in results:
                v = r.by_capability.get(cap)
                row += f"{v:>13.0%} " if v is not None else f"{'—':>14}"
            lines.append(row)
        return "\n".join(lines)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
