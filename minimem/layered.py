"""``LayeredMemory``：记忆越来越多，而上下文就那么大。

前六章各有各的办法：第 2 章丢掉，第 3 章挑几条，第 6 章合并。
本章是第四种——**分层**：把记忆放在不同的层里，按热度在层之间搬。

::

    core      常驻上下文，有硬性 token 上限，每次都注入
      ↑↓  换页
    recall    近期可检索，不常驻
      ↑↓  归档 / 回捞
    archival  冷存，需要显式捞取，且可能只保留摘要

这个结构来自 MemGPT（📄 arXiv:2310.08560）的 core / recall / archival 三层，
以及 MemoryOS 的热度换页思路。

**关于「memory OS」这个说法，需要先泼一盆冷水。**

「操作系统」的隐喻主要是**调度与分页的类比**，而不是说记忆真的成了
一等资源。真正落地的机制其实只有两样：

1. **self-editing memory blocks**——让 agent 用工具调用改自己的 core 区
   （MemGPT 的 ``core_memory_replace`` / ``archival_memory_insert``）；
2. **异步 ingestion**——写入不阻塞对话。

这两样都是实实在在的工程设计，值得学。但「first-class resource」
「memory OS」这类表述属于叙事，不属于机制。本模块实现前者，不复述后者。

顺带一提：**MemoryOS 和 MemOS 是两个不同的项目**，前者是分层换页
（STM/MTM/LPM），后者是 MemCube 抽象与 MemTensor 生态。名字像，别混。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from minimem.base import (
    CrossUserAccessError,
    MemoryItem,
    MemoryNotFoundError,
    MemoryStore,
    SearchResult,
)
from minimem.buffer import _tokenize
from minimem.utils.embedding import Embedder, get_embedder
from minimem.utils.tokens import count_tokens

__all__ = ["Layer", "LayeredMemory", "summarize_by_rule"]

Layer = Literal["core", "recall", "archival"]

_DECAY_HALF_LIFE_S = 7 * 24 * 3600.0  # 热度的半衰期：一周


def summarize_by_rule(texts: list[str], max_chars: int = 60) -> str:
    """规则式摘要：取每条的首句拼起来再截断。

    刻意做得很差。它的作用是让「反复摘要会退化」这件事**可测量**——
    真实的 LLM 摘要退化得慢一些，但方向相同，因为每次摘要都在丢信息，
    而丢掉的信息不会回来。
    """
    heads = [t.split("。")[0].split("，")[0] for t in texts if t.strip()]
    joined = "；".join(heads)
    return joined[:max_chars]


@dataclass
class _Entry:
    item: MemoryItem
    layer: Layer = "recall"
    hits: int = 0
    last_access: datetime | None = None
    summarized_times: int = 0

    def heat(self, now: datetime) -> float:
        """热度 = 访问次数 × 时间衰减。

        两个因子缺一不可：只看次数，老热点永远占着 core；
        只看新近度，就退化成第 2 章的滑动窗口了。
        """
        last = self.last_access or self.item.created_at
        age = max((now - last).total_seconds(), 0.0)
        decay = 0.5 ** (age / _DECAY_HALF_LIFE_S)
        return (1.0 + self.hits) * decay


class LayeredMemory(MemoryStore):
    """三层记忆 + 热度换页 + self-editing core 区。

    Args:
        core_budget: core 层的 token 上限。**这是本章最核心的约束**——
            没有硬上限就不存在换页问题，分层也就没有意义。
        recall_capacity: recall 层最多放多少条，超出的下沉到 archival。
        archival_summarize: 下沉到 archival 时是否压成摘要。
            开启能省空间，代价是**信息不可逆地损失**，而且反复下沉会反复摘要。
        summarizer: 摘要函数。默认规则版；传 LLM 版会更好但每次要花钱。
        embedder: 句向量模型。

    Note:
        ``search`` 默认只搜 core + recall。archival 需要显式 ``include_archival=True``——
        这不是偷懒，是 MemGPT 式设计的核心：**冷层要显式捞取**，
        否则分层就退化成了「一个带排序的普通库」。
    """

    name = "layered"

    def __init__(
        self,
        *,
        core_budget: int = 300,
        recall_capacity: int = 20,
        archival_summarize: bool = True,
        summarizer: Callable[[list[str]], str] | None = None,
        embedder: Embedder | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if core_budget <= 0:
            raise ValueError("core_budget 必须为正")
        self.core_budget = core_budget
        self.recall_capacity = recall_capacity
        self.archival_summarize = archival_summarize
        self.summarizer = summarizer or (lambda texts: summarize_by_rule(texts))
        self._embedder = embedder

        self._entries: dict[str, dict[str, _Entry]] = {}
        self._owner: dict[str, str] = {}
        self._vectors: dict[str, Any] = {}
        self.evictions: list[tuple[str, Layer, Layer, str]] = []

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # 分层与换页
    # ------------------------------------------------------------------

    def _add(self, item: MemoryItem, *, user_id: str) -> str:
        entries = self._entries.setdefault(user_id, {})
        layer: Layer = item.metadata.get("layer", "core")
        entries[item.id] = _Entry(item=item, layer=layer, last_access=item.created_at)
        self._owner[item.id] = user_id
        self._vectors[item.id] = self.embedder.encode_one(item.content)
        self._rebalance(user_id=user_id)
        return item.id

    def _rebalance(self, *, user_id: str) -> None:
        """把超出预算的记忆往下一层推。

        顺序很重要：先处理 core→recall，再处理 recall→archival。
        反过来的话，刚从 core 换出的记忆可能立刻被推进 archival。
        """
        now = self._now()
        entries = self._entries.get(user_id, {})

        core = [e for e in entries.values() if e.layer == "core"]
        used = sum(count_tokens(e.item.content) for e in core)
        if used > self.core_budget:
            # 热度最低的先走
            for entry in sorted(core, key=lambda e: e.heat(now)):
                if used <= self.core_budget:
                    break
                used -= count_tokens(entry.item.content)
                entry.layer = "recall"
                self.evictions.append(
                    (entry.item.id, "core", "recall", f"热度 {entry.heat(now):.3f} 最低")
                )

        recall = [e for e in entries.values() if e.layer == "recall"]
        if len(recall) > self.recall_capacity:
            overflow = sorted(recall, key=lambda e: e.heat(now))[
                : len(recall) - self.recall_capacity
            ]
            for entry in overflow:
                self._archive(entry)

    def _archive(self, entry: _Entry) -> None:
        entry.layer = "archival"
        self.evictions.append((entry.item.id, "recall", "archival", "recall 层已满"))
        if not self.archival_summarize:
            return

        # 下沉即摘要。**注意这是不可逆的**——而且如果一条记忆多次进出
        # archival，它会被反复摘要，信息持续退化。第 7 章的实验会画出这条曲线。
        summary = self.summarizer([entry.item.content])
        if summary and summary != entry.item.content:
            entry.item = entry.item.copy_with(content=summary)
            entry.summarized_times += 1
            self._vectors[entry.item.id] = self.embedder.encode_one(summary)

    # ------------------------------------------------------------------
    # core 区的自编辑（MemGPT 式）
    # ------------------------------------------------------------------

    def core_context(self, *, user_id: str) -> str:
        """core 层的内容，每轮都注入 prompt。"""
        entries = [e for e in self._entries.get(user_id, {}).values() if e.layer == "core"]
        entries.sort(key=lambda e: e.item.created_at)
        return "\n".join(f"- {e.item.content}" for e in entries)

    def core_memory_append(self, text: str, *, user_id: str) -> str:
        """往 core 区加一条。超预算会立刻触发换页。"""
        return self.add(MemoryItem(text, metadata={"layer": "core"}), user_id=user_id)

    def core_memory_replace(self, memory_id: str, new_text: str, *, user_id: str) -> None:
        """就地改写 core 区的一条。这是 MemGPT 最实在的那个机制。"""
        self.update(memory_id, {"content": new_text}, user_id=user_id)

    def promote(self, memory_id: str, *, user_id: str) -> None:
        """把一条记忆提到 core 层（回捞）。"""
        entry = self._locate(memory_id, user_id)
        old = entry.layer
        entry.layer = "core"
        entry.hits += 1
        entry.last_access = self._now()
        if old != "core":
            self.evictions.append((memory_id, old, "core", "显式回捞"))
        self._rebalance(user_id=user_id)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def _search(self, query: str, *, user_id: str, k: int, **kwargs: Any) -> list[SearchResult]:
        import numpy as np

        entries = self._entries.get(user_id, {})
        if not entries:
            return []

        include_archival: bool = kwargs.get("include_archival", False)
        pool = [
            e
            for e in entries.values()
            if e.layer in ("core", "recall") or (include_archival and e.layer == "archival")
        ]
        if not pool:
            return []

        qv = self.embedder.encode_one(query)
        matrix = np.vstack([self._vectors[e.item.id] for e in pool])
        sims = matrix @ qv

        # core 层加成：它是常驻的，本来就该优先被看到
        boost = {"core": 1.15, "recall": 1.0, "archival": 0.85}
        scored = [(pool[i], float(sims[i]) * boost[pool[i].layer]) for i in range(len(pool))]
        scored.sort(key=lambda kv: kv[1], reverse=True)

        now = self._now()
        results = []
        for entry, score in scored[:k]:
            entry.hits += 1
            entry.last_access = now
            results.append(
                SearchResult(
                    item=entry.item,
                    score=score,
                    source="layered",
                    debug={
                        "layer": entry.layer,
                        "hits": entry.hits,
                        "heat": round(entry.heat(now), 4),
                        "summarized": entry.summarized_times,
                    },
                )
            )
        return results

    # ------------------------------------------------------------------

    def layer_of(self, memory_id: str, *, user_id: str) -> Layer:
        return self._locate(memory_id, user_id).layer

    def stats(self, *, user_id: str) -> dict[str, Any]:
        entries = self._entries.get(user_id, {}).values()
        by_layer: dict[str, int] = {"core": 0, "recall": 0, "archival": 0}
        for e in entries:
            by_layer[e.layer] += 1
        core_tokens = sum(count_tokens(e.item.content) for e in entries if e.layer == "core")
        return {
            **by_layer,
            "core token": core_tokens,
            "core 预算": self.core_budget,
            "换页次数": len(self.evictions),
            "被摘要过的": sum(1 for e in entries if e.summarized_times),
        }

    def recall_rate(self, keywords: list[str], *, user_id: str) -> float:
        """还能在 core+recall 里找到多少关键信息。教学用。"""
        visible = " ".join(
            e.item.content
            for e in self._entries.get(user_id, {}).values()
            if e.layer in ("core", "recall")
        )
        tokens = set(_tokenize(visible))
        hit = sum(1 for kw in keywords if set(_tokenize(kw)) <= tokens)
        return hit / len(keywords) if keywords else 0.0

    # ------------------------------------------------------------------

    def _update(self, memory_id: str, patch: dict[str, Any], *, user_id: str) -> None:
        entry = self._locate(memory_id, user_id)
        entry.item = self.apply_patch(entry.item, patch)
        if "content" in patch:
            self._vectors[memory_id] = self.embedder.encode_one(entry.item.content)
        self._rebalance(user_id=user_id)

    def _delete(self, memory_id: str, *, user_id: str) -> None:
        self._locate(memory_id, user_id)
        self._entries[user_id].pop(memory_id, None)
        self._owner.pop(memory_id, None)
        self._vectors.pop(memory_id, None)

    def _all(self, *, user_id: str) -> list[MemoryItem]:
        entries = sorted(self._entries.get(user_id, {}).values(), key=lambda e: e.item.created_at)
        return [e.item for e in entries]

    def _locate(self, memory_id: str, user_id: str) -> _Entry:
        owner = self._owner.get(memory_id)
        if owner is None:
            raise MemoryNotFoundError(memory_id)
        if owner != user_id:
            raise CrossUserAccessError(f"记忆 {memory_id} 不属于用户 {user_id}")
        entry = self._entries.get(user_id, {}).get(memory_id)
        if entry is None:
            raise MemoryNotFoundError(memory_id)
        return entry
