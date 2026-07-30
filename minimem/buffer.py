"""``BufferMemory``：最朴素的记忆——把所有东西按顺序存下来。

这是全书的基线（baseline）。它几乎没有任何"智能"：检索就是关键词匹配加时间排序。
但它有两个不可替代的价值：

1. **它常常够用**。几十轮的短对话里，把历史全塞进上下文的效果就是最好的，
   而且没有任何检索误差。第 10 章的评测里，很多"聪明"方案在小数据集上打不过它。
2. **它是代价的下界**。零 LLM 调用、零嵌入计算。后面每一章新增的机制，
   都要回答"比 BufferMemory 多花的这些钱，换来了什么"。
"""

from __future__ import annotations

import re
from typing import Any

from minimem.base import (
    CrossUserAccessError,
    MemoryItem,
    MemoryNotFoundError,
    MemoryStore,
    SearchResult,
)

__all__ = ["BufferMemory"]

# 一个刻意粗糙的分词：中文按字切，英文数字按词切。
# 第 3 章会说明这种切法在中文上为什么会导致 BM25 召回虚高，以及正经做法。
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BufferMemory(MemoryStore):
    """按写入顺序保存全部记忆，检索用词重叠打分。

    Args:
        max_items: 每个用户最多保留多少条。超出后丢弃最旧的——这是全书出现的
            第一种「遗忘」，也是最粗暴的一种：它不看重要性，只看年龄。
            第 7 章会用热度换页、第 5 章会用事实失效来替代它。
        recency_weight: 时间新近性在最终分数里的权重（0~1）。设为 0 则纯粹按
            词重叠排序。Generative Agents 的检索分数是 recency / relevance /
            importance 三项加权，这里先只做前两项，importance 留到第 8 章。

    Note:
        实现刻意保持 O(N) 全扫描。当记忆条数到几千以上时它会明显变慢——
        这个性能墙正是第 3 章引入向量索引的动机，我们会在第 3 章把它测出来。
    """

    name = "buffer"

    def __init__(
        self,
        *,
        max_items: int | None = None,
        recency_weight: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if max_items is not None and max_items <= 0:
            raise ValueError("max_items 必须为正整数或 None")
        if not 0.0 <= recency_weight <= 1.0:
            raise ValueError("recency_weight 必须在 [0, 1] 区间")
        self.max_items = max_items
        self.recency_weight = recency_weight
        # user_id -> 有序的记忆列表
        self._store: dict[str, list[MemoryItem]] = {}
        # 记忆 id -> user_id，用于快速定位与越权检测
        self._owner: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 钩子实现
    # ------------------------------------------------------------------

    def _add(self, item: MemoryItem, *, user_id: str) -> str:
        bucket = self._store.setdefault(user_id, [])
        bucket.append(item)
        self._owner[item.id] = user_id

        if self.max_items is not None and len(bucket) > self.max_items:
            dropped = bucket.pop(0)
            self._owner.pop(dropped.id, None)
        return item.id

    def _search(self, query: str, *, user_id: str, k: int, **kwargs: Any) -> list[SearchResult]:
        bucket = self._store.get(user_id, [])
        if not bucket:
            return []

        q_tokens = set(_tokenize(query))
        scored: list[SearchResult] = []
        n = len(bucket)

        for pos, item in enumerate(bucket):
            i_tokens = set(_tokenize(item.content))
            if not q_tokens or not i_tokens:
                overlap = 0.0
            else:
                # Jaccard 相似度：交集 / 并集。选它是因为一行能讲清楚，
                # 而不是因为它好——第 3 章会展示它在同义表述上如何完全失效。
                overlap = len(q_tokens & i_tokens) / len(q_tokens | i_tokens)

            # 新近性：最新一条为 1，最旧一条为 0
            recency = pos / (n - 1) if n > 1 else 1.0
            score = (1 - self.recency_weight) * overlap + self.recency_weight * recency

            if score > 0:
                scored.append(
                    SearchResult(
                        item=item,
                        score=score,
                        source="buffer",
                        debug={"overlap": round(overlap, 4), "recency": round(recency, 4)},
                    )
                )

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def _update(self, memory_id: str, patch: dict[str, Any], *, user_id: str) -> None:
        bucket = self._locate_bucket(memory_id, user_id)
        for i, item in enumerate(bucket):
            if item.id == memory_id:
                bucket[i] = self.apply_patch(item, patch)
                return
        raise MemoryNotFoundError(memory_id)

    def _delete(self, memory_id: str, *, user_id: str) -> None:
        bucket = self._locate_bucket(memory_id, user_id)
        for i, item in enumerate(bucket):
            if item.id == memory_id:
                del bucket[i]
                self._owner.pop(memory_id, None)
                return
        raise MemoryNotFoundError(memory_id)

    def _all(self, *, user_id: str) -> list[MemoryItem]:
        return list(self._store.get(user_id, []))

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _locate_bucket(self, memory_id: str, user_id: str) -> list[MemoryItem]:
        owner = self._owner.get(memory_id)
        if owner is None:
            raise MemoryNotFoundError(memory_id)
        if owner != user_id:
            # 注意这里没有返回「未找到」：把越权和不存在区分开，才能在日志里
            # 看见有人在扫别人的记忆。生产环境要不要对外暴露这个区别，是另一个
            # 权衡（暴露了等于确认了 ID 存在）——第 10 章会讨论。
            raise CrossUserAccessError(f"记忆 {memory_id} 不属于用户 {user_id}")
        return self._store.setdefault(user_id, [])

    def history(self, *, user_id: str, limit: int | None = None) -> list[MemoryItem]:
        """按时间顺序返回最近的记忆。

        这是「把历史全塞进上下文」那种用法的入口，与 ``search`` 的语义不同：
        它不做任何相关性筛选。第 1 章的实验会对比这两种取用方式。
        """
        bucket = self._store.get(user_id, [])
        return list(bucket) if limit is None else list(bucket[-limit:])
