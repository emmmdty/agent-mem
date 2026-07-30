"""``TemporalGraphMemory``：给记忆装上时间轴。

第 1 章的第三堵墙到这里才正面处理。前四章的所有实现都有同一个盲点：
它们把「说过的话」原样存下来，然后在检索时把矛盾的几条一起返回，
让模型自己猜哪条是现在有效的。

```
Q: 我在哪家公司上班？
A: 我在星辰银行工作。          ← 半年前说的
   新工作定了，下个月入职长风科技。 ← 半个月前说的
   我下周就要从星辰银行离职了。    ← 二十天前说的
```

三条都是真话，但**只有一条描述当前状态**——而且答案取决于「今天是几号」。

本模块引入三样东西：

1. **双时间轴（bi-temporal）**。区分两个完全不同的时间：

   - ``recorded_at``：这句话**什么时候被说的**
   - ``valid_from`` / ``valid_to``：这个事实**什么时候生效/失效**

   「我下个月入职长风科技」的 recorded_at 是今天，valid_from 是下个月。
   把两者混为一谈，是时序问答出错的头号原因。
   这个设计取自 Zep/Graphiti（📄 arXiv:2501.13956）。

2. **失效而非删除**。新事实与旧事实冲突时，给旧事实盖上 ``valid_to``，
   而不是删掉它。因为「他三个月前在哪工作」仍然是一个合法的问题。

3. **provenance**。每条事实记录 ``source_episode``（它来自哪条原始记忆）。
   这既支持回查，也让第 10 章要求的**级联删除**成为可能——
   删掉一条原始消息时，从它派生出的事实必须一起失效。

刻意保留的局限（正文会展开）：本模块的事实抽取和冲突判定都是规则式的。
真实系统需要 LLM 来做这两件事，而那会引入第 6 章要处理的一整类新问题。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from minimem.base import MemoryItem, SearchResult
from minimem.graph import GraphMemory

__all__ = ["Fact", "TemporalGraphMemory", "parse_relative_time", "extract_facts"]

SUBJECT = "用户"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ----------------------------------------------------------------------
# 时间表达式
# ----------------------------------------------------------------------

#: 中文相对时间表达 → 相对说话时刻的天数偏移。
#: 刻意做得很小，因为**重点不是解析多全，而是「说话时间 ≠ 生效时间」这件事**。
_REL_TIME: list[tuple[str, int]] = [
    (r"下个?月", 30),
    (r"下周|下星期", 7),
    (r"明天", 1),
    (r"后天", 2),
    (r"上个?月", -30),
    (r"上周|上星期", -7),
    (r"昨天", -1),
    (r"前天", -2),
    (r"去年", -365),
    (r"今天|现在|目前", 0),
]

#: 「N 个月前」「三周后」这类带数量的表达
_CN_NUM = {
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_UNIT_DAYS = {"天": 1, "周": 7, "星期": 7, "个月": 30, "月": 30, "年": 365}


def parse_relative_time(text: str, *, spoken_at: datetime) -> datetime | None:
    """从文本里解析出「事实何时生效」，返回 None 表示说话即生效。

    Examples:
        >>> from datetime import datetime, timezone
        >>> t = datetime(2026, 7, 30, tzinfo=timezone.utc)
        >>> parse_relative_time("我下个月入职长风科技", spoken_at=t).month
        8
        >>> parse_relative_time("我在星辰银行工作", spoken_at=t) is None
        True
    """
    # 「三个月前」「两周后」
    m = re.search(
        rf"([{''.join(_CN_NUM)}\d]+)\s*({'|'.join(_UNIT_DAYS)})\s*(前|后|以前|以后)", text
    )
    if m:
        raw, unit, direction = m.groups()
        n = int(raw) if raw.isdigit() else _CN_NUM.get(raw, 1)
        days = n * _UNIT_DAYS[unit]
        return spoken_at + timedelta(days=days if "后" in direction else -days)

    for pattern, offset in _REL_TIME:
        if re.search(pattern, text):
            return spoken_at + timedelta(days=offset)
    return None


# ----------------------------------------------------------------------
# 事实
# ----------------------------------------------------------------------


@dataclass
class Fact:
    """一条带双时间轴的事实。

    Attributes:
        predicate: 属性名，如 ``"雇主"`` / ``"居住地"``。
        object: 属性值。
        recorded_at: **说这句话的时间**。
        valid_from: **事实开始生效的时间**。默认等于 recorded_at，
            但「我下个月入职」会让它晚于 recorded_at。
        valid_to: 失效时间。``None`` 表示仍然有效。
        source_episode: 来自哪条原始记忆（provenance）。
        invalidated_by: 被哪条事实推翻的。用于审计「为什么它失效了」。
        subject: 主语，本模块默认都是「用户」。
    """

    predicate: str
    object: str
    recorded_at: datetime
    valid_from: datetime
    source_episode: str
    valid_to: datetime | None = None
    invalidated_by: str | None = None
    subject: str = SUBJECT
    fid: str = ""

    def is_valid_at(self, when: datetime) -> bool:
        """在给定时刻这条事实是否有效。

        注意用的是 ``valid_from`` / ``valid_to`` 而不是 ``recorded_at``——
        一条今天说的、下个月才生效的事实，在今天是**无效**的。
        """
        if when < self.valid_from:
            return False
        return self.valid_to is None or when < self.valid_to

    @property
    def is_terminal(self) -> bool:
        """终止型事件（离职、搬走）：它只给旧事实划句号，自己不描述新状态。"""
        return self.invalidated_by == self.fid

    def describe(self) -> str:
        if self.is_terminal:
            state = f"终止事件，于 {self.valid_from:%Y-%m-%d} 结束此项"
        elif self.valid_to is None:
            state = "有效"
        else:
            state = f"已失效于 {self.valid_to:%Y-%m-%d}"
        lag = (self.valid_from - self.recorded_at).days
        lag_note = f" · 说话与生效相差 {lag:+d} 天" if lag else ""
        return (
            f"{self.predicate}={self.object}  "
            f"[生效 {self.valid_from:%Y-%m-%d} · 记录 {self.recorded_at:%Y-%m-%d}"
            f"{lag_note} · {state}]"
        )


#: 事实抽取规则：(正则, 属性名, 是否为「终止型」事件)
#: 终止型事件（离职、搬走）不产生新事实，而是让既有事实失效。
_FACT_RULES: list[tuple[str, str, bool]] = [
    (r"从(.{2,12}?)(?:离职|辞职)", "雇主", True),
    (r"入职([一-鿿A-Za-z]{2,12}?)(?:[，。,.]|$|，|做)", "雇主", False),
    (r"我在([一-鿿A-Za-z]{2,12}?)(?:工作|上班)", "雇主", False),
    (r"(?:现在)?住(?:在)?(?:城[东南西北]的)?([一-鿿]{2,10}?)(?:[，。,.]|$)", "居住地", False),
    (r"搬到(?:城[东南西北]的)?([一-鿿]{2,10}?)(?:[，。,.]|$)", "居住地", False),
    (r"我对(.{1,8}?)过敏", "过敏原", False),
    (r"直属领导是(.{1,6}?)(?:[，。,.]|$)", "上级", False),
    (r"我叫(.{1,6}?)(?:[，。,.]|$)", "姓名", False),
]


def extract_facts(item: MemoryItem) -> list[Fact]:
    """从一条记忆里抽事实，并解析它的生效时间。

    这是**规则式**的，覆盖面很窄。它存在的意义是让你看清双时间轴的机制，
    而不是提供一个能用的抽取器。真实系统需要 LLM——第 6 章会接上，
    并把 LLM 抽取引入的新问题（抽错、抽漏、把假设当事实）一并处理。
    """
    text = item.content
    spoken_at = item.created_at
    valid_from = parse_relative_time(text, spoken_at=spoken_at) or spoken_at

    facts: list[Fact] = []
    for pattern, predicate, terminal in _FACT_RULES:
        m = re.search(pattern, text)
        if not m:
            continue
        value = m.group(1).strip()
        if not value:
            continue
        facts.append(
            Fact(
                predicate=predicate,
                object=value,
                recorded_at=spoken_at,
                valid_from=valid_from,
                source_episode=item.id,
                # 终止型事件用一个哨兵值表示「结束，但没有新值」
                invalidated_by="__terminal__" if terminal else None,
            )
        )
    return facts


# ----------------------------------------------------------------------


class TemporalGraphMemory(GraphMemory):
    """在图记忆之上加一层双时间轴的事实表。

    Args:
        now: 「当前时刻」的来源。测试与教学时可以注入一个固定时间，
            让实验可复现——**时间相关的代码不注入时钟就没法测**。
        demote_invalid: 检索时，把在该时点无效的事实所对应的记忆降权到多少倍。
            这个值是一个**真实的权衡**，不是随便取的：

            - 设为 0.0 等于过滤掉历史，「我三个月前住哪」就答不出来了；
            - 设得太低（试过 0.35）会把历史记忆挤出 top-k，效果和过滤差不多；
            - 设得太高则起不到「当前有效的排前面」的作用。

            默认 0.6：足以让当前有效的排到前面，又不至于让历史记忆消失。
            调它之前先想清楚你的用户问「现在」多还是问「当时」多。

    Example:
        >>> store = TemporalGraphMemory()                       # doctest: +SKIP
        >>> store.search("我现在住哪", user_id="u1")              # doctest: +SKIP
        >>> store.search("我三个月前住哪", user_id="u1",
        ...              as_of=three_months_ago)                # doctest: +SKIP
    """

    name = "temporal"

    def __init__(
        self,
        *,
        now: datetime | None = None,
        demote_invalid: float = 0.6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._now = now
        self.demote_invalid = demote_invalid
        #: user_id -> 事实列表（按写入顺序，失效的也留着）
        self._facts: dict[str, list[Fact]] = {}
        self._fact_seq = 0

    def now(self) -> datetime:
        return self._now if self._now is not None else _now()

    # ------------------------------------------------------------------
    # 写入：抽事实 + 冲突消解
    # ------------------------------------------------------------------

    def _add(self, item: MemoryItem, *, user_id: str) -> str:
        memory_id = super()._add(item, user_id=user_id)
        facts = self._facts.setdefault(user_id, [])

        for fact in extract_facts(item):
            self._fact_seq += 1
            fact.fid = f"f{self._fact_seq:04d}"
            terminal = fact.invalidated_by == "__terminal__"
            fact.invalidated_by = None

            self._invalidate_conflicts(facts, fact, terminal=terminal)

            if terminal:
                # 「我从星辰银行离职」不产生新事实——它只是给旧事实划上句号。
                # 把它也记下来是为了可审计：谁让那条事实失效的。
                fact.valid_to = fact.valid_from
                fact.invalidated_by = fact.fid
            facts.append(fact)

        return memory_id

    def _invalidate_conflicts(self, facts: list[Fact], new: Fact, *, terminal: bool) -> None:
        """让与新事实冲突的旧事实失效。

        规则很简单：同一个 ``predicate`` 上，旧的有效事实在新事实生效时终止。

        **这条规则的局限必须说清楚**：它假设一个属性同时只能有一个值。
        对「雇主」「居住地」成立，对「爱好」「技能」就不成立——
        一个人可以同时喜欢爬山和读书。真实系统需要区分
        「单值属性」和「多值属性」，而这个判断本身往往需要 LLM。
        """
        for old in facts:
            if old.predicate != new.predicate or old.valid_to is not None:
                continue
            if not terminal and old.object == new.object:
                # 重复陈述同一件事，不算冲突
                continue
            old.valid_to = new.valid_from
            old.invalidated_by = new.fid

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def facts_at(
        self, when: datetime | None = None, *, user_id: str, predicate: str | None = None
    ) -> list[Fact]:
        """返回在给定时刻有效的事实。这是时点查询的核心。"""
        when = when or self.now()
        return [
            f
            for f in self._facts.get(user_id, [])
            if f.is_valid_at(when) and (predicate is None or f.predicate == predicate)
        ]

    def all_facts(self, *, user_id: str) -> list[Fact]:
        """包括已失效的。用于审计与教学展示。"""
        return list(self._facts.get(user_id, []))

    def timeline(self, predicate: str, *, user_id: str) -> str:
        """把某个属性的演化历史打出来。"""
        rows = [f for f in self._facts.get(user_id, []) if f.predicate == predicate]
        if not rows:
            return f"（没有关于「{predicate}」的事实）"
        lines = [f"「{predicate}」的演化："]
        for f in sorted(rows, key=lambda x: x.valid_from):
            lines.append(f"  {f.describe()}  ← 来自 {f.source_episode[:8]}")
        return "\n".join(lines)

    def _search(self, query: str, *, user_id: str, k: int, **kwargs: Any) -> list[SearchResult]:
        """检索时按时点调整排序。

        做法刻意保守：**不过滤，只降权**。因为「我三个月前住哪」这类问题
        需要的恰恰是当时有效、现在失效的事实。直接过滤会让它无解。
        """
        as_of: datetime | None = kwargs.pop("as_of", None)
        when = as_of or self.now()

        results = super()._search(query, user_id=user_id, k=k * 3, **kwargs)
        if not results:
            return []

        valid_sources = {f.source_episode for f in self.facts_at(when, user_id=user_id)}
        known_sources = {f.source_episode for f in self._facts.get(user_id, [])}

        adjusted: list[SearchResult] = []
        for r in results:
            score = r.score
            state = "无关事实"
            if r.item.id in valid_sources:
                state = "当时有效"
            elif r.item.id in known_sources:
                score *= self.demote_invalid
                state = "当时无效"
            adjusted.append(
                SearchResult(
                    item=r.item,
                    score=score,
                    source=r.source,
                    debug={**r.debug, "时点": when.strftime("%Y-%m-%d"), "事实状态": state},
                )
            )

        adjusted.sort(key=lambda x: x.score, reverse=True)
        return adjusted[:k]

    def answer_context(self, query: str, *, user_id: str, as_of: datetime | None = None) -> str:
        """给模型的上下文：带时间标注的记忆 + 当前有效事实摘要。

        这是本章最实用的一个产出。把 provenance 和时间一起注入，
        模型才有依据判断该信哪条——第 10 章讲投毒防御时提到的
        「让上下文带上来源」，在这里落地。
        """
        when = as_of or self.now()
        hits = self.search(query, user_id=user_id, k=5, as_of=when)
        lines = [f"（以下信息的判断时点：{when:%Y-%m-%d}）", "", "相关记忆："]
        for h in hits:
            stamp = h.item.created_at.strftime("%Y-%m-%d")
            state = h.debug.get("事实状态", "")
            lines.append(f"  [{stamp} · {state}] {h.content}")

        facts = self.facts_at(when, user_id=user_id)
        if facts:
            lines += ["", "当前有效的事实："]
            lines += [f"  {f.predicate}：{f.object}（自 {f.valid_from:%Y-%m-%d}）" for f in facts]
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def _delete(self, memory_id: str, *, user_id: str) -> None:
        """删除记忆时，级联失效由它派生的事实。

        第 10 章的删除权要求删除必须级联到派生数据。这里是本书第一次
        真正实现它——``source_episode`` 就是为此存在的。
        """
        super()._delete(memory_id, user_id=user_id)
        for fact in self._facts.get(user_id, []):
            if fact.source_episode == memory_id and fact.valid_to is None:
                fact.valid_to = self.now()
                fact.invalidated_by = "__deleted__"

    def purge_facts_of(self, memory_id: str, *, user_id: str) -> int:
        """物理删除某条记忆派生的全部事实，返回删除条数。

        与 ``delete`` 的区别：``delete`` 只让事实失效（保留可审计的痕迹），
        这个方法是真删除。**被遗忘权要求的是后者**——
        「不再检索」和「不再存在」是两回事。
        """
        facts = self._facts.get(user_id, [])
        before = len(facts)
        self._facts[user_id] = [f for f in facts if f.source_episode != memory_id]
        return before - len(self._facts[user_id])
