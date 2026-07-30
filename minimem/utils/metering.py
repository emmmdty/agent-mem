"""计量：让每一次记忆操作的代价都可见。

本书的一条编写原则是「代价必须一等呈现」。要做到这点，代价就不能靠事后估算，
而要在每次操作发生时记录下来。``Meter`` 就是这个记录器：

- ``MemoryStore`` 的公开方法自动把 add / search / update / delete 记进来；
- 需要调 LLM 的实现（第 6 章起）额外把 token 数与估算成本记进来；
- 第 10 章的 ``EvalHarness`` 直接消费这些记录，产出「准确率 / 延迟 / token / 成本」四列对比表。

刻意不引入 OpenTelemetry 之类的重型依赖：教程需要的是**能看懂的一百行**，
生产环境请换成你团队已有的可观测体系。
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

__all__ = ["OpRecord", "Meter", "get_default_meter", "reset_default_meter"]


@dataclass
class OpRecord:
    """一次操作的计量记录。

    ``tokens_in`` / ``tokens_out`` / ``cost_usd`` 只有在操作真的调用了 LLM 时才非零。
    朴素的向量检索不花 LLM token，但花时间；反过来，一次「反思」可能只有一次调用，
    却烧掉几千 token。两类代价都要看，只看一个会得出错误结论。
    """

    op: str
    store: str
    user_id: str = ""
    duration_ms: float = 0.0
    n_items: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class Meter:
    """收集 ``OpRecord`` 并给出汇总。

    用法::

        meter = Meter()
        with meter.measure(op="search", store="vector", user_id="u1") as rec:
            results = do_search()
            rec.n_items = len(results)

        print(meter.summary())
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.records: list[OpRecord] = []

    @contextmanager
    def measure(self, *, op: str, store: str, user_id: str = "") -> Generator[OpRecord, None, None]:
        """上下文管理器：进入时开始计时，退出时把记录存下来。

        即使被包裹的代码抛异常，记录也会被保存（并标记 ``failed``），
        否则失败路径的耗时会在统计里凭空消失。
        """
        rec = OpRecord(op=op, store=store, user_id=user_id)
        t0 = time.perf_counter()
        try:
            yield rec
        except Exception as exc:
            rec.extra["failed"] = type(exc).__name__
            raise
        finally:
            rec.duration_ms = (time.perf_counter() - t0) * 1000
            if self.enabled:
                self.records.append(rec)

    def add_llm_usage(
        self,
        *,
        op: str,
        store: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float = 0.0,
        user_id: str = "",
    ) -> None:
        """记录一次独立的 LLM 调用（不在 measure 上下文内时使用）。"""
        if not self.enabled:
            return
        self.records.append(
            OpRecord(
                op=op,
                store=store,
                user_id=user_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                llm_calls=1,
                cost_usd=cost_usd,
            )
        )

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------

    def filter(self, *, op: str | None = None, store: str | None = None) -> list[OpRecord]:
        return [
            r
            for r in self.records
            if (op is None or r.op == op) and (store is None or r.store == store)
        ]

    def summary(self, *, op: str | None = None, store: str | None = None) -> dict[str, Any]:
        """汇总统计。

        延迟同时给出均值和 p95：记忆系统的延迟分布通常是长尾的（缓存未命中、
        触发了一次巩固、图检索碰到高度节点），只报均值会掩盖最糟糕的用户体验。
        """
        rows = self.filter(op=op, store=store)
        if not rows:
            return {
                "count": 0,
                "latency_ms_mean": 0.0,
                "latency_ms_p95": 0.0,
                "tokens_total": 0,
                "llm_calls": 0,
                "cost_usd": 0.0,
            }

        latencies = sorted(r.duration_ms for r in rows)
        p95_idx = min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))
        return {
            "count": len(rows),
            "latency_ms_mean": round(statistics.fmean(latencies), 3),
            "latency_ms_p95": round(latencies[p95_idx], 3),
            "tokens_total": sum(r.tokens for r in rows),
            "tokens_per_op": round(sum(r.tokens for r in rows) / len(rows), 1),
            "llm_calls": sum(r.llm_calls for r in rows),
            "cost_usd": round(sum(r.cost_usd for r in rows), 6),
        }

    def report(self) -> str:
        """按 (store, op) 分组的可读报表。"""
        keys = sorted({(r.store, r.op) for r in self.records})
        if not keys:
            return "（无计量记录）"

        header = f"{'store':<16}{'op':<10}{'次数':>6}{'均值ms':>10}{'p95ms':>10}{'token':>10}{'成本$':>10}"
        lines = [header, "-" * len(header)]
        for store, op in keys:
            s = self.summary(op=op, store=store)
            lines.append(
                f"{store:<16}{op:<10}{s['count']:>6}{s['latency_ms_mean']:>10.2f}"
                f"{s['latency_ms_p95']:>10.2f}{s['tokens_total']:>10}{s['cost_usd']:>10.4f}"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        self.records.clear()


_DEFAULT_METER = Meter()


def get_default_meter() -> Meter:
    """进程级默认计量器。教学脚本里直接用它即可；评测时请显式注入独立实例。"""
    return _DEFAULT_METER


def reset_default_meter() -> None:
    _DEFAULT_METER.reset()
