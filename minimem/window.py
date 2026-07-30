"""``WindowMemory``：上下文放不下时，到底该丢什么。

第 1 章的 ``BufferMemory`` 假装上下文无限大。真实系统里它是有限的，
于是「丢什么」成了一个必须回答的问题——而**默认答案（丢最旧的）是最糟的一个**。

本模块实现三种策略，可以在同一个接口下横向比较：

===============  ==========================================================
策略              做法
===============  ==========================================================
``"recent"``     只保留最近的，丢最旧——几乎所有对话应用的默认行为
``"sink"``       始终保留开头若干条 + 最近的，中间丢弃
``"pinned"``     在 sink 的基础上，额外保留被标记为「钉住」的记忆
===============  ==========================================================

``"sink"`` 这个名字来自 StreamingLLM 的 attention sink 现象
（📄 Xiao et al., arXiv:2309.17453）：Transformer 会把大量注意力分配给
序列最开始的几个 token，即使它们语义上无关紧要；一旦这些 token 被驱逐出
KV cache，模型输出质量会急剧下降。保留最初几个 token 就能稳住流式解码。

**必须说清楚的一点**：这里借的只是「保留开头」这个形状，
两者解决的**不是同一个问题**。StreamingLLM 处理的是 KV cache 层面的
数值稳定性，它并不扩展模型的有效上下文长度；我们这里处理的是应用层
「历史太长塞不下」的信息取舍。把两者混为一谈是这个领域常见的以讹传讹。
"""

from __future__ import annotations

from typing import Any, Literal

from minimem.base import (
    CrossUserAccessError,
    MemoryItem,
    MemoryNotFoundError,
    MemoryStore,
    SearchResult,
)
from minimem.buffer import _tokenize
from minimem.utils.tokens import count_tokens

__all__ = ["WindowMemory", "EvictionPolicy"]

EvictionPolicy = Literal["recent", "sink", "pinned"]


class WindowMemory(MemoryStore):
    """带 token 预算的滑动窗口。

    与 ``BufferMemory`` 的关键区别：**窗口外的记忆检索不到**。
    这不是缺陷，正是本章要演示的性质——窗口方案用「彻底遗忘」换来了
    恒定的上下文成本，而遗忘的是什么，完全由淘汰策略决定。

    Args:
        budget_tokens: 窗口的 token 预算。
        policy: 淘汰策略，见模块文档。
        n_sink: ``"sink"`` / ``"pinned"`` 策略下，开头保留几条。
        drop_from_store: 被挤出窗口的记忆是否**真的删除**。
            默认 False——只是不再可见，数据还在。这个区分很重要：
            第 10 章的删除权要求的是真删除，而「移出窗口」不是删除。

    Note:
        钉住一条记忆：写入时设置 ``metadata={"pinned": True}``，
        或事后 ``store.pin(memory_id, user_id=...)``。
    """

    name = "window"

    def __init__(
        self,
        *,
        budget_tokens: int = 2000,
        policy: EvictionPolicy = "sink",
        n_sink: int = 2,
        drop_from_store: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if budget_tokens <= 0:
            raise ValueError("budget_tokens 必须为正")
        if n_sink < 0:
            raise ValueError("n_sink 不能为负")
        self.budget_tokens = budget_tokens
        self.policy: EvictionPolicy = policy
        self.n_sink = n_sink
        self.drop_from_store = drop_from_store

        self._store: dict[str, list[MemoryItem]] = {}
        self._owner: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 窗口计算
    # ------------------------------------------------------------------

    def visible(self, *, user_id: str) -> list[MemoryItem]:
        """返回当前窗口内可见的记忆，保持原始时间顺序。

        这是本类的核心。三种策略的差别全在这个方法里。
        """
        items = self._store.get(user_id, [])
        if not items:
            return []

        budget = self.budget_tokens
        costs = [count_tokens(it.content) for it in items]

        keep: set[int] = set()
        used = 0

        # 第一步：先把必须保留的占位（sink 与 pinned）。
        # 顺序很重要——如果先填最近的，预算可能在轮到 sink 之前就用光了。
        must: list[int] = []
        if self.policy in ("sink", "pinned"):
            must.extend(range(min(self.n_sink, len(items))))
        if self.policy == "pinned":
            must.extend(i for i, it in enumerate(items) if it.metadata.get("pinned"))

        for i in sorted(set(must)):
            if used + costs[i] <= budget:
                keep.add(i)
                used += costs[i]

        # 第二步：从最新往回填，直到预算耗尽
        for i in range(len(items) - 1, -1, -1):
            if i in keep:
                continue
            if used + costs[i] > budget:
                break
            keep.add(i)
            used += costs[i]

        return [items[i] for i in sorted(keep)]

    def evicted(self, *, user_id: str) -> list[MemoryItem]:
        """返回被挤出窗口的记忆。调试与教学用——让「丢了什么」看得见。"""
        visible_ids = {it.id for it in self.visible(user_id=user_id)}
        return [it for it in self._store.get(user_id, []) if it.id not in visible_ids]

    def pin(self, memory_id: str, *, user_id: str) -> None:
        """把一条记忆钉住，使其在 ``"pinned"`` 策略下不被淘汰。"""
        self.update(memory_id, {"metadata": {"pinned": True}}, user_id=user_id)

    def usage(self, *, user_id: str) -> dict[str, int]:
        items = self._store.get(user_id, [])
        vis = self.visible(user_id=user_id)
        return {
            "总条数": len(items),
            "窗口内条数": len(vis),
            "窗口内 token": sum(count_tokens(it.content) for it in vis),
            "预算": self.budget_tokens,
        }

    # ------------------------------------------------------------------
    # 钩子实现
    # ------------------------------------------------------------------

    def _add(self, item: MemoryItem, *, user_id: str) -> str:
        bucket = self._store.setdefault(user_id, [])
        bucket.append(item)
        self._owner[item.id] = user_id

        if self.drop_from_store:
            # 真删除：被挤出窗口的记忆从存储里消失，不可恢复
            visible_ids = {it.id for it in self.visible(user_id=user_id)}
            for gone in [it for it in bucket if it.id not in visible_ids]:
                bucket.remove(gone)
                self._owner.pop(gone.id, None)
        return item.id

    def _search(self, query: str, *, user_id: str, k: int, **kwargs: Any) -> list[SearchResult]:
        """只在窗口内检索。窗口外的记忆**检索不到**——这正是要演示的性质。"""
        candidates = self.visible(user_id=user_id)
        if not candidates:
            return []

        q_tokens = set(_tokenize(query))
        results: list[SearchResult] = []
        for pos, item in enumerate(candidates):
            i_tokens = set(_tokenize(item.content))
            overlap = (
                len(q_tokens & i_tokens) / len(q_tokens | i_tokens)
                if q_tokens and i_tokens
                else 0.0
            )
            recency = pos / (len(candidates) - 1) if len(candidates) > 1 else 1.0
            score = 0.85 * overlap + 0.15 * recency
            if score > 0:
                results.append(
                    SearchResult(
                        item=item,
                        score=score,
                        source="window",
                        debug={"overlap": round(overlap, 4), "in_window": True},
                    )
                )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def _update(self, memory_id: str, patch: dict[str, Any], *, user_id: str) -> None:
        bucket = self._locate(memory_id, user_id)
        for i, item in enumerate(bucket):
            if item.id == memory_id:
                bucket[i] = self.apply_patch(item, patch)
                return
        raise MemoryNotFoundError(memory_id)

    def _delete(self, memory_id: str, *, user_id: str) -> None:
        bucket = self._locate(memory_id, user_id)
        for i, item in enumerate(bucket):
            if item.id == memory_id:
                del bucket[i]
                self._owner.pop(memory_id, None)
                return
        raise MemoryNotFoundError(memory_id)

    def _all(self, *, user_id: str) -> list[MemoryItem]:
        return list(self._store.get(user_id, []))

    # ------------------------------------------------------------------

    def _locate(self, memory_id: str, user_id: str) -> list[MemoryItem]:
        owner = self._owner.get(memory_id)
        if owner is None:
            raise MemoryNotFoundError(memory_id)
        if owner != user_id:
            raise CrossUserAccessError(f"记忆 {memory_id} 不属于用户 {user_id}")
        return self._store.setdefault(user_id, [])
