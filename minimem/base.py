"""MiniMem 的核心抽象：一条记忆是什么，一个记忆库能做什么。

全书所有记忆实现（缓冲、滑窗、向量、图、时序、自组织、分层、技能）都实现同一个
``MemoryStore`` 接口。这不是为了美观——**不能被比较的方案，等于没有结论**。
统一接口是第 10 章能把八种实现放进同一个评测 harness 的前提。

设计上的两条硬约束：

1. **多用户隔离**：所有读写方法都带 ``user_id``，跨用户访问必须失败而不是静默返回空。
   这既是工程刚需，也是第 10 章隐私一节的前提。
2. **可观测**：公开方法负责校验与计量，真正的逻辑放在子类的 ``_add`` / ``_search`` 等
   钩子里（模板方法模式）。子类作者不用操心计时和统计，harness 也不用给每种实现
   写一遍埋点。
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from minimem.utils.metering import Meter, get_default_meter

__all__ = [
    "MemoryItem",
    "SearchResult",
    "MemoryStore",
    "MemoryNotFoundError",
    "CrossUserAccessError",
]


def _now() -> datetime:
    """统一使用带时区的 UTC 时间。

    第 5 章会解释为什么记忆系统里的朴素时间戳（naive datetime）是个陷阱：
    一旦要回答"三个月前他在哪家公司"这类问题，时区歧义会直接变成答案错误。
    """
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


class MemoryNotFoundError(KeyError):
    """访问不存在的记忆 ID。"""


class CrossUserAccessError(PermissionError):
    """试图访问其他用户的记忆。

    刻意设计成抛异常而不是返回空结果：静默失败会让越权访问的 bug 一直藏到线上。
    """


@dataclass
class MemoryItem:
    """一条记忆。

    刻意保持字段极少。后续章节需要的额外信息（向量、实体、有效期、热度、
    技能签名……）一律放进 ``metadata``，而不是往这里加字段——否则到第 8 章
    这个类会膨胀成一个谁也看不懂的大杂烩。

    Attributes:
        content: 记忆的文本内容。
        id: 唯一标识，默认随机生成。
        created_at: 写入时间（UTC）。注意这是**记录时间**，不是事实**发生时间**，
            两者的区别是第 5 章双时间轴的核心。
        kind: 记忆类型标签，如 ``"message"`` / ``"fact"`` / ``"note"`` / ``"skill"``。
            仅作过滤用，不承载语义。
        metadata: 各章实现自用的附加信息。
        user_id: 归属用户。由 ``MemoryStore.add`` 在写入时填充，调用方不必设置。
    """

    content: str
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_now)
    kind: str = "message"
    metadata: dict[str, Any] = field(default_factory=dict)
    user_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError(f"content 必须是 str，收到 {type(self.content).__name__}")

    def copy_with(self, **changes: Any) -> MemoryItem:
        """返回一个应用了改动的副本（不修改原对象）。"""
        data = {
            "content": self.content,
            "id": self.id,
            "created_at": self.created_at,
            "kind": self.kind,
            "metadata": dict(self.metadata),
            "user_id": self.user_id,
        }
        data.update(changes)
        return MemoryItem(**data)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "kind": self.kind,
            "metadata": self.metadata,
            "user_id": self.user_id,
        }


@dataclass
class SearchResult:
    """一次检索命中。

    Attributes:
        item: 命中的记忆。
        score: 相关性分数。**不同实现的分数不可直接比较**——余弦相似度、BM25
            得分、PageRank 值的量纲完全不同。跨实现比较请用第 10 章的端到端指标。
        source: 产生这条结果的检索通道，如 ``"dense"`` / ``"bm25"`` / ``"ppr"``。
            混合检索时用它来分析各通道的贡献。
        debug: 可选的中间信息，如各通道原始排名，供教学与排错使用。
    """

    item: MemoryItem
    score: float
    source: str = "default"
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return self.item.content


class MemoryStore(ABC):
    """记忆库的统一接口。

    子类只需实现五个下划线开头的钩子方法，参数校验、用户隔离和计量由基类负责。

    典型用法::

        store = BufferMemory()
        store.add(MemoryItem("我叫小明，是做金融风控的"), user_id="u1")
        hits = store.search("他是做什么的", user_id="u1", k=3)
    """

    #: 供 harness 与日志使用的可读名称，子类可覆盖
    name: str = "memory"

    def __init__(self, *, meter: Meter | None = None) -> None:
        self._meter = meter if meter is not None else get_default_meter()

    # ------------------------------------------------------------------
    # 公开 API：校验 + 计量 + 委托
    # ------------------------------------------------------------------

    def add(self, item: MemoryItem | str, *, user_id: str) -> str:
        """写入一条记忆，返回其 ID。

        传字符串是常见场景的快捷方式，内部会包装成 ``MemoryItem``。
        """
        self._check_user_id(user_id)
        if isinstance(item, str):
            item = MemoryItem(item)
        item = item.copy_with(user_id=user_id)

        with self._meter.measure(op="add", store=self.name, user_id=user_id) as rec:
            memory_id = self._add(item, user_id=user_id)
            rec.n_items = 1
        return memory_id

    def add_many(self, items: list[MemoryItem | str], *, user_id: str) -> list[str]:
        """批量写入。默认逐条调用 ``add``，子类可覆盖以利用批量嵌入等优化。"""
        return [self.add(it, user_id=user_id) for it in items]

    def search(self, query: str, *, user_id: str, k: int = 5, **kwargs: Any) -> list[SearchResult]:
        """检索最相关的 k 条记忆，按分数降序。"""
        self._check_user_id(user_id)
        if k <= 0:
            raise ValueError(f"k 必须为正整数，收到 {k}")

        with self._meter.measure(op="search", store=self.name, user_id=user_id) as rec:
            results = self._search(query, user_id=user_id, k=k, **kwargs)
            rec.n_items = len(results)
        return results

    def update(self, memory_id: str, patch: dict[str, Any], *, user_id: str) -> None:
        """局部更新一条记忆。

        ``patch`` 支持 ``content`` / ``kind`` / ``metadata`` 三个键；``metadata``
        采用**浅合并**而非整体替换——第 6 章的 extract-update 流水线依赖这个语义。
        """
        self._check_user_id(user_id)
        unknown = set(patch) - {"content", "kind", "metadata"}
        if unknown:
            raise ValueError(f"不支持的 patch 字段：{sorted(unknown)}")

        with self._meter.measure(op="update", store=self.name, user_id=user_id) as rec:
            self._update(memory_id, patch, user_id=user_id)
            rec.n_items = 1

    def delete(self, memory_id: str, *, user_id: str) -> None:
        """删除一条记忆。

        「删除」在记忆系统里比看上去复杂：第 5 章会讨论「事实失效」与「物理删除」
        的区别，第 10 章会讨论被遗忘权对应的删除必须是真删除。
        """
        self._check_user_id(user_id)
        with self._meter.measure(op="delete", store=self.name, user_id=user_id) as rec:
            self._delete(memory_id, user_id=user_id)
            rec.n_items = 1

    def all(self, *, user_id: str) -> list[MemoryItem]:
        """返回该用户的全部记忆，按写入时间升序。主要用于调试与评测。"""
        self._check_user_id(user_id)
        return self._all(user_id=user_id)

    def clear(self, *, user_id: str) -> None:
        """清空该用户的全部记忆。"""
        for item in self.all(user_id=user_id):
            self.delete(item.id, user_id=user_id)

    # ------------------------------------------------------------------
    # 便利方法
    # ------------------------------------------------------------------

    def as_context(self, query: str, *, user_id: str, k: int = 5, max_chars: int = 2000) -> str:
        """把检索结果拼成可直接塞进 prompt 的一段文本。

        ``max_chars`` 是最朴素的预算控制。第 2 章会讲为什么按字符截断是个坏主意，
        以及按 token 预算裁剪要怎么做。
        """
        hits = self.search(query, user_id=user_id, k=k)
        lines: list[str] = []
        used = 0
        for i, hit in enumerate(hits, 1):
            line = f"[{i}] {hit.content}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def __len__(self) -> int:
        raise TypeError("记忆库的大小与用户有关，请用 len(store.all(user_id=...))")

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"

    # ------------------------------------------------------------------
    # 子类需要实现的钩子
    # ------------------------------------------------------------------

    @abstractmethod
    def _add(self, item: MemoryItem, *, user_id: str) -> str: ...

    @abstractmethod
    def _search(self, query: str, *, user_id: str, k: int, **kwargs: Any) -> list[SearchResult]: ...

    @abstractmethod
    def _update(self, memory_id: str, patch: dict[str, Any], *, user_id: str) -> None: ...

    @abstractmethod
    def _delete(self, memory_id: str, *, user_id: str) -> None: ...

    @abstractmethod
    def _all(self, *, user_id: str) -> list[MemoryItem]: ...

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _check_user_id(user_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id 必须是非空字符串；记忆库不允许匿名读写")

    @staticmethod
    def apply_patch(item: MemoryItem, patch: dict[str, Any]) -> MemoryItem:
        """把 patch 应用到 item 上，metadata 走浅合并。供子类复用。"""
        changes: dict[str, Any] = {}
        if "content" in patch:
            changes["content"] = patch["content"]
        if "kind" in patch:
            changes["kind"] = patch["kind"]
        if "metadata" in patch:
            merged = dict(item.metadata)
            merged.update(patch["metadata"])
            changes["metadata"] = merged
        return item.copy_with(**changes)
