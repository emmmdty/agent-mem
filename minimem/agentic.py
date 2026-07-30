"""``AgenticMemory``：让记忆自己组织自己。

前五章一路积累了一串「留给 LLM 的活」：

- 事实抽取靠正则，抓不到没有标志的专名（第 4 章）
- 冲突判定靠「一个属性一个值」，分不清单值多值（第 5 章）
- 什么记忆值得钉住、值得抽取，无从判断（第 2、4 章）

这些都需要**判断力**，而规则给不了判断力。本章接上 LLM，并把账算清楚。

两条代表性路线：

============  ==========================================  ==========================
路线           核心动作                                     代表工作
============  ==========================================  ==========================
自组织         写入时生成结构化 note，与旧 note 建立链接、     A-MEM
              并可能触发旧 note 的演化                       (arXiv:2502.12110)
extract-update 抽出候选事实，与已有记忆比对，                Mem0
              决定 ADD / UPDATE / DELETE / NOOP            (arXiv:2504.19413)
============  ==========================================  ==========================

本模块两条都实现，因为它们解决的是不同的问题：前者让记忆之间**长出结构**，
后者让记忆**保持一致**。

**这是全书第一次成本列不为 0 的章节。** 每条记忆的写入从「一次编码」
变成了「一到三次 LLM 调用」，而检索侧几乎没变。所有关于「值不值」的判断，
最后都会归结到一个数：**读写比**。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from minimem.base import (
    CrossUserAccessError,
    MemoryItem,
    MemoryNotFoundError,
    MemoryStore,
    SearchResult,
)
from minimem.utils.embedding import Embedder, get_embedder
from minimem.utils.llm import LLMClient

__all__ = ["Note", "Decision", "AgenticMemory", "NOTE_PROMPT", "UPDATE_PROMPT"]

Action = Literal["ADD", "UPDATE", "NOOP", "DELETE"]

NOTE_PROMPT = """把下面这条消息加工成一条结构化记忆笔记。

要求：
1. summary 用一句话概括，去掉寒暄和口头语
2. keywords 是 2~5 个便于检索的词
3. tags 从这些里选（可多选）：身份、偏好、健康、工作、生活、人际、学习、临时
4. 只输出 JSON：{{"summary": "...", "keywords": ["..."], "tags": ["..."]}}

消息：{text}"""

UPDATE_PROMPT = """判断新信息与已有记忆的关系。

已有记忆：
{existing}

新信息：{new}

从下面选一个动作：
  ADD    —— 新信息与已有记忆无冲突，应作为新记忆加入
  UPDATE —— 新信息修正/推翻了某条已有记忆，给出它的编号
  NOOP   —— 新信息只是重复，不必存
  DELETE —— 新信息表明某条已有记忆应被删除，给出它的编号

只输出 JSON：{{"action": "...", "target": <编号或 null>, "reason": "<二十字以内>"}}"""


@dataclass
class Note:
    """一条加工过的记忆笔记。

    Attributes:
        summary: 一句话概括，检索时用它而不是原文——**这是自组织的收益来源**：
            去掉寒暄后，同样的 token 预算能装下更多有效信息（第 3 章的噪声分析）。
        keywords: 便于检索的词。
        tags: 粗分类，用于过滤。
        links: 与之相关的其他记忆 id。
        source_episode: 原始消息（provenance，沿用第 5 章的设计）。
        degraded: 加工失败时为 True——此时 summary 就是原文。
            **这个字段很重要**：它让「LLM 失败了多少次」变成可观测的，
            而不是悄悄地退化。
    """

    summary: str
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    source_episode: str = ""
    degraded: bool = False


@dataclass
class Decision:
    """一次 extract-update 决策，保留下来供审计与教学展示。"""

    action: Action
    target: str | None
    reason: str
    new_text: str
    degraded: bool = False


class AgenticMemory(MemoryStore):
    """自组织记忆：note 生成 + 动态链接 + extract-update。

    Args:
        llm: 必需。没有它，本模块退化成一个更贵的 VectorMemory——
            所以构造时不给 LLM 会直接报错，而不是悄悄降级。
            （教学脚本请传 ``ScriptedLLM``。）
        embedder: 句向量模型，用于链接候选与检索。
        link_top_k: 新 note 与最相似的几条建立链接。
        link_threshold: 相似度低于此值不建链接。这是个真实的权衡：
            **设太高（试过 0.55）一条链接都建不起来**，自组织退化成普通向量库；
            **设太低会让图变成全连接**，那样链接同样没有信息量。
            默认 0.40 是在 MiniBench 上试出来的，换语料要重调。
        enable_update: 是否跑 extract-update 决策。关掉它可以省一半调用，
            代价是矛盾记忆会并存（回到第 4 章的状态）。

    Note:
        写入一条记忆的调用次数：
        1 次 note 生成 + （若开启）1 次 update 决策 = **最多 2 次**。
        这是本书第一次写入侧的开销超过检索侧。
    """

    name = "agentic"

    def __init__(
        self,
        *,
        llm: LLMClient,
        embedder: Embedder | None = None,
        link_top_k: int = 3,
        link_threshold: float = 0.40,
        enable_update: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if llm is None:  # pragma: no cover - 防御性
            raise ValueError(
                "AgenticMemory 必须提供 llm。没有 LLM 时它退化成一个更贵的 VectorMemory，"
                "与其悄悄降级，不如直接报错。"
            )
        self.llm = llm
        # 让 LLM 的用量记进**同一个**计量器。
        # 不做这件事的话，harness 里的成本列会显示 0——
        # store 的 meter 只记检索延迟，LLM 的 token 全落到了另一个计量器上。
        # 「代价必须一等呈现」这条原则，靠的就是这种地方不出岔子。
        if getattr(llm, "_meter", None) is not self._meter:
            llm._meter = self._meter  # noqa: SLF001 —— 刻意共享计量器
        self._embedder = embedder
        self.link_top_k = link_top_k
        self.link_threshold = link_threshold
        self.enable_update = enable_update

        self._items: dict[str, list[MemoryItem]] = {}
        self._notes: dict[str, Note] = {}
        self._owner: dict[str, str] = {}
        self._vectors: dict[str, Any] = {}
        self.decisions: list[Decision] = []
        self.degraded_count = 0

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    # ------------------------------------------------------------------
    # note 生成
    # ------------------------------------------------------------------

    def make_note(self, text: str, *, source_episode: str = "") -> Note:
        """把原始消息加工成笔记。失败时退回原文并标记 degraded。"""
        result = self.llm.complete(NOTE_PROMPT.format(text=text), op="note")
        data = result.json(default=None)

        if not isinstance(data, dict) or not data.get("summary"):
            # 加工失败**不能丢掉这条记忆**——退回原文，但标记出来。
            # 悄悄降级是最糟的处理：你会以为系统在自组织，其实它在裸奔。
            self.degraded_count += 1
            return Note(summary=text, source_episode=source_episode, degraded=True)

        return Note(
            summary=str(data["summary"]).strip(),
            keywords=[str(k) for k in data.get("keywords", [])][:5],
            tags=[str(t) for t in data.get("tags", [])][:4],
            source_episode=source_episode,
        )

    # ------------------------------------------------------------------
    # extract-update
    # ------------------------------------------------------------------

    def decide(self, new_text: str, *, user_id: str, candidates: list[MemoryItem]) -> Decision:
        """判断新信息与已有记忆的关系。"""
        if not candidates:
            return Decision("ADD", None, "没有相关的已有记忆", new_text)

        listing = "\n".join(
            f"[{i}] {self._notes[c.id].summary if c.id in self._notes else c.content}"
            for i, c in enumerate(candidates)
        )
        result = self.llm.complete(
            UPDATE_PROMPT.format(existing=listing, new=new_text), op="update_decision"
        )
        data = result.json(default=None)

        if not isinstance(data, dict) or data.get("action") not in (
            "ADD",
            "UPDATE",
            "NOOP",
            "DELETE",
        ):
            # 决策失败时**默认 ADD**——宁可多存一条，不可错删或错更。
            # 这个默认值的选择很关键：默认 UPDATE 会让一次解析失败
            # 覆盖掉一条正确的记忆。
            self.degraded_count += 1
            return Decision("ADD", None, "决策解析失败，保守处理", new_text, degraded=True)

        target_idx = data.get("target")
        target_id = None
        if isinstance(target_idx, int) and 0 <= target_idx < len(candidates):
            target_id = candidates[target_idx].id

        action: Action = data["action"]
        if action in ("UPDATE", "DELETE") and target_id is None:
            # 说要改但没说改哪条——同样保守处理
            return Decision("ADD", None, "目标编号无效，保守处理", new_text, degraded=True)

        return Decision(action, target_id, str(data.get("reason", ""))[:40], new_text)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def _add(self, item: MemoryItem, *, user_id: str) -> str:
        note = self.make_note(item.content, source_episode=item.id)
        vec = self.embedder.encode_one(note.summary)

        if self.enable_update:
            candidates = self._nearest(vec, user_id=user_id, k=self.link_top_k)
            decision = self.decide(note.summary, user_id=user_id, candidates=candidates)
            self.decisions.append(decision)

            if decision.action == "NOOP":
                # 不存，但仍返回一个 id 以满足接口。调用方通过 decisions 得知发生了什么。
                return item.id
            if decision.action == "UPDATE" and decision.target:
                self._apply_update(decision.target, note, vec, user_id=user_id)
                return decision.target
            if decision.action == "DELETE" and decision.target:
                self.delete(decision.target, user_id=user_id)

        self._store_new(item, note, vec, user_id=user_id)
        return item.id

    def _store_new(self, item: MemoryItem, note: Note, vec: Any, *, user_id: str) -> None:
        note.links = [c.id for c in self._nearest(vec, user_id=user_id, k=self.link_top_k)]
        self._items.setdefault(user_id, []).append(item)
        self._notes[item.id] = note
        self._owner[item.id] = user_id
        self._vectors[item.id] = vec

        # 双向链接：新 note 连出去，旧 note 也要连回来，
        # 否则从旧记忆出发永远找不到新记忆——这是 A-MEM 式「演化」的最小形式
        for other in note.links:
            if other in self._notes and item.id not in self._notes[other].links:
                self._notes[other].links.append(item.id)

    def _apply_update(self, target: str, note: Note, vec: Any, *, user_id: str) -> None:
        old = self._notes.get(target)
        merged_links = list(dict.fromkeys((old.links if old else []) + note.links))
        note.links = merged_links
        self._notes[target] = note
        self._vectors[target] = vec
        for item in self._items.get(user_id, []):
            if item.id == target:
                item.metadata["updated"] = item.metadata.get("updated", 0) + 1

    def _nearest(self, vec: Any, *, user_id: str, k: int) -> list[MemoryItem]:
        import numpy as np

        items = self._items.get(user_id, [])
        if not items:
            return []
        matrix = np.vstack([self._vectors[i.id] for i in items])
        sims = matrix @ vec
        order = np.argsort(-sims)[: k * 2]
        return [items[i] for i in order if sims[i] >= self.link_threshold][:k]

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def _search(self, query: str, *, user_id: str, k: int, **kwargs: Any) -> list[SearchResult]:
        import numpy as np

        items = self._items.get(user_id, [])
        if not items:
            return []

        follow_links: bool = kwargs.get("follow_links", True)
        qv = self.embedder.encode_one(query)
        matrix = np.vstack([self._vectors[i.id] for i in items])
        sims = matrix @ qv

        scored = {items[i].id: float(sims[i]) for i in range(len(items))}

        if follow_links:
            # 沿链接扩散一跳：被高分记忆链接到的，也给一点分。
            # 这是 A-MEM 式链接的实际用途——它让「相关但措辞不同」的记忆
            # 有机会被带出来。系数 0.3 是个经验值，调大会引入噪声。
            top_ids = sorted(scored, key=lambda i: -scored[i])[:k]
            for mid in top_ids:
                for linked in self._notes.get(mid, Note("")).links:
                    if linked in scored:
                        scored[linked] = max(scored[linked], scored[mid] * 0.3)

        by_id = {i.id: i for i in items}
        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            SearchResult(
                item=by_id[mid],
                score=score,
                source="agentic",
                debug={
                    "summary": self._notes[mid].summary if mid in self._notes else "",
                    "tags": self._notes[mid].tags if mid in self._notes else [],
                    "links": len(self._notes[mid].links) if mid in self._notes else 0,
                    "degraded": self._notes[mid].degraded if mid in self._notes else False,
                },
            )
            for mid, score in ranked
        ]

    # ------------------------------------------------------------------

    def note_of(self, memory_id: str) -> Note | None:
        return self._notes.get(memory_id)

    def stats(self, *, user_id: str) -> dict[str, Any]:
        items = self._items.get(user_id, [])
        notes = [self._notes[i.id] for i in items if i.id in self._notes]
        actions: dict[str, int] = {}
        for d in self.decisions:
            actions[d.action] = actions.get(d.action, 0) + 1
        return {
            "记忆数": len(items),
            "链接总数": sum(len(n.links) for n in notes),
            "加工失败": sum(1 for n in notes if n.degraded),
            "决策分布": actions,
            "LLM 调用": self.llm.call_count,
        }

    # ------------------------------------------------------------------

    def _update(self, memory_id: str, patch: dict[str, Any], *, user_id: str) -> None:
        bucket = self._locate(memory_id, user_id)
        for i, item in enumerate(bucket):
            if item.id == memory_id:
                new_item = self.apply_patch(item, patch)
                bucket[i] = new_item
                if "content" in patch:
                    note = self.make_note(new_item.content, source_episode=memory_id)
                    self._notes[memory_id] = note
                    self._vectors[memory_id] = self.embedder.encode_one(note.summary)
                return
        raise MemoryNotFoundError(memory_id)

    def _delete(self, memory_id: str, *, user_id: str) -> None:
        bucket = self._locate(memory_id, user_id)
        for i, item in enumerate(bucket):
            if item.id == memory_id:
                del bucket[i]
                self._notes.pop(memory_id, None)
                self._owner.pop(memory_id, None)
                self._vectors.pop(memory_id, None)
                # 清理指向它的链接，否则会留下悬空引用
                for note in self._notes.values():
                    if memory_id in note.links:
                        note.links.remove(memory_id)
                return
        raise MemoryNotFoundError(memory_id)

    def _all(self, *, user_id: str) -> list[MemoryItem]:
        return list(self._items.get(user_id, []))

    def _locate(self, memory_id: str, user_id: str) -> list[MemoryItem]:
        owner = self._owner.get(memory_id)
        if owner is None:
            raise MemoryNotFoundError(memory_id)
        if owner != user_id:
            raise CrossUserAccessError(f"记忆 {memory_id} 不属于用户 {user_id}")
        return self._items.setdefault(user_id, [])


def format_decisions(decisions: list[Decision], limit: int = 10) -> str:
    """把决策序列打成可读表格。教学脚本用。"""
    lines = [f"  {'动作':<10}{'原因':<24}{'内容'}"]
    lines.append("  " + "-" * 70)
    for d in decisions[:limit]:
        mark = "⚠️ " if d.degraded else "  "
        lines.append(f"  {mark}{d.action:<8}{d.reason[:22]:<24}{d.new_text[:26]}")
    return "\n".join(lines)


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
