"""``SkillMemory``：从「记住发生了什么」到「记住该怎么做」。

前七章的记忆都是**关于世界的事实**——他叫小明、他对花生过敏、他换了工作。
本章处理另一类：**关于怎么做事的经验**。

这是表征轴上的一次跨越。用第 1 章的话说，前者是陈述性的（declarative），
后者是程序性的（procedural）。但请注意本书的用词纪律：我们不说
「实现了程序记忆」，而说「存了带触发条件和步骤的可复用流程」——
后者才是你要维护的那张表。

三个机制：

1. **importance 打分**。第 1 章的 ``BufferMemory`` 用 recency + relevance
   两项打分，这里补上第三项。它来自 Generative Agents
   （📄 Park et al., UIST 2023, DOI 10.1145/3586183.3606763）。

2. **reflection（反思）**。累积重要性越过阈值时，让 LLM 从近期记忆里
   归纳出更高层的洞察，作为新记忆写回。
   📄 论文报告去掉 reflection 后，agent 在 48 模拟小时内退化为
   重复的、无上下文的响应。

3. **技能库**。把成功的执行轨迹归纳成带 trigger 和参数槽的可复用流程。
   Voyager 存的是**可执行代码**（需要沙箱），AWM（Agent Workflow Memory）
   存的是**结构化 workflow**。本模块走后者——更通用，也不需要执行环境。

**必须提前说清的一件事**：所有已发表系统的 reflection 触发条件都是**启发式**的
（定时、计数、阈值）。没有哪个系统真的实现了「自发反思」。
这和第 1 章批评「巩固并不自动」是同一个问题，只是换了个位置出现。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from minimem.base import (
    CrossUserAccessError,
    MemoryItem,
    MemoryNotFoundError,
    MemoryStore,
    SearchResult,
)
from minimem.buffer import _tokenize
from minimem.utils.embedding import Embedder, get_embedder
from minimem.utils.llm import LLMClient

__all__ = ["Skill", "SkillMemory", "IMPORTANCE_PROMPT", "REFLECT_PROMPT", "DISTILL_PROMPT"]


IMPORTANCE_PROMPT = """给下面这条记忆打一个重要性分数（1~10）。

评分参考：
  1~3  日常琐事，过几天就无所谓（今天天气不错、中午吃了面）
  4~6  有用的背景信息（他的职业、他在读什么书）
  7~10 长期影响行为的信息（过敏史、核心偏好、身份、重大变更）

只输出 JSON：{{"score": <数字>, "reason": "<十个字以内>"}}

记忆：{text}"""

REFLECT_PROMPT = """下面是一个用户最近的一些记忆。请归纳出 1~3 条更高层的洞察。

要求：
1. 洞察必须是**从多条记忆里综合出来的**，不能是单条的复述
2. 不要推测记忆里没有的信息
3. 只输出 JSON：{{"insights": ["...", "..."]}}

记忆：
{memories}"""

DISTILL_PROMPT = """下面是一次成功完成任务的操作轨迹。请把它归纳成一个可复用的流程。

要求：
1. trigger 要写清「什么情况下该用这个流程」
2. steps 要去掉本次特有的具体值，用 {{参数名}} 占位
3. 只输出 JSON：
   {{"name": "...", "trigger": "...", "steps": ["...", "..."], "params": ["..."]}}

轨迹：
{trace}"""


# ----------------------------------------------------------------------


@dataclass
class Skill:
    """一条可复用的流程。

    Attributes:
        name: 技能名。
        trigger: 什么情况下该用它。**检索时匹配的是这个字段**，
            而不是步骤内容——这是 AWM 的关键设计：
            触发条件描述的是「问题」，步骤描述的是「解法」，
            而用户提问时给出的是问题。
        steps: 步骤，用 ``{参数名}`` 占位本次特有的值。
        params: 参数名列表。
        uses: 被复用了几次。
        successes: 其中成功几次。**失败也要记**——
            Reflexion 的核心正是「把失败的教训写进记忆，下次避开」。
        source_episodes: 从哪些原始记忆归纳出来的（provenance，见第 5 章）。
    """

    name: str
    trigger: str
    steps: list[str]
    params: list[str] = field(default_factory=list)
    uses: int = 0
    successes: int = 0
    source_episodes: list[str] = field(default_factory=list)
    sid: str = ""

    @property
    def success_rate(self) -> float:
        """成功率。没用过时返回 0.0——**不是 1.0**。

        把「没试过」当成「一定成功」，是技能库最容易犯的错：
        一个刚归纳出来、从未验证过的流程会被优先推荐。
        """
        return self.successes / self.uses if self.uses else 0.0

    def render(self, **values: str) -> str:
        text = "\n".join(f"  {i}. {s}" for i, s in enumerate(self.steps, 1))
        for key, val in values.items():
            text = text.replace(f"{{{key}}}", val)
        return f"【{self.name}】适用于：{self.trigger}\n{text}"

    def to_memory_text(self) -> str:
        return f"技能「{self.name}」：{self.trigger} → " + " → ".join(self.steps)


# ----------------------------------------------------------------------

#: 规则版 importance 打分：命中这些模式的记忆被认为长期重要。
#: 它的存在是为了让本章**离线可跑**，也为了给 LLM 打分提供一个对照基线——
#: 如果 LLM 打分不比这个简单规则强，那它就不值那笔钱。
_IMPORTANT_PATTERNS: list[tuple[str, int]] = [
    (r"过敏|禁忌|忌口|不能吃", 9),
    (r"我叫|我是一名|我的名字", 8),
    (r"离职|入职|跳槽|换工作|搬家", 8),
    (r"喜欢|不喜欢|偏好|习惯|尽量|麻烦", 6),
    (r"负责|项目|工作|团队|领导", 5),
    (r"在读|在学|课程|教材", 4),
]
_DEFAULT_IMPORTANCE = 2


def rule_importance(text: str) -> int:
    for pattern, score in _IMPORTANT_PATTERNS:
        if re.search(pattern, text):
            return score
    return _DEFAULT_IMPORTANCE


class SkillMemory(MemoryStore):
    """记忆流 + 三项打分 + 反思 + 技能库。

    Args:
        llm: 用于 importance 打分、反思、技能归纳。为 ``None`` 时全部走规则版
            （离线可跑，但反思和技能归纳会退化成很粗糙的模板）。
        embedder: 句向量模型，用于 relevance。
        weights: recency / relevance / importance 三项的权重。
            Generative Agents 用的是等权，本模块默认略偏 relevance——
            因为在问答场景里「答不对题」比「记不住重点」更致命。
        reflect_threshold: 累积重要性超过它就触发一次反思。
            **这个阈值是纯启发式的**，没有理论依据，
            调它等于调「多久反思一次」，直接影响成本。
        max_reflect_window: 反思时看最近多少条记忆。

    Note:
        importance 打分在写入时进行。如果用 LLM，**每条记忆一次调用**——
        这是继第 6 章之后又一处写入侧的固定开销。
    """

    name = "skill"

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        weights: tuple[float, float, float] = (0.2, 0.5, 0.3),
        reflect_threshold: int = 30,
        max_reflect_window: int = 12,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.llm = llm
        self._embedder = embedder
        self.w_recency, self.w_relevance, self.w_importance = weights
        self.reflect_threshold = reflect_threshold
        self.max_reflect_window = max_reflect_window

        self._items: dict[str, list[MemoryItem]] = {}
        self._owner: dict[str, str] = {}
        self._vectors: dict[str, Any] = {}
        self._skills: dict[str, list[Skill]] = {}
        self._accumulated: dict[str, int] = {}
        self._reflect_count = 0
        self._skill_seq = 0

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    # ------------------------------------------------------------------
    # importance
    # ------------------------------------------------------------------

    def score_importance(self, text: str) -> int:
        """给一条记忆打重要性分。有 LLM 就用 LLM，否则走规则。"""
        if self.llm is None:
            return rule_importance(text)

        result = self.llm.complete(IMPORTANCE_PROMPT.format(text=text), op="importance")
        data = result.json(default=None)
        if not isinstance(data, dict) or "score" not in data:
            # 打分失败退回规则版。**不要默认给高分**——
            # 那会让所有解析失败的记忆挤占 core 层。
            return rule_importance(text)
        try:
            return max(1, min(10, int(data["score"])))
        except (TypeError, ValueError):
            return rule_importance(text)

    # ------------------------------------------------------------------
    # 写入与反思
    # ------------------------------------------------------------------

    def _add(self, item: MemoryItem, *, user_id: str) -> str:
        if "importance" not in item.metadata:
            item.metadata["importance"] = self.score_importance(item.content)

        self._items.setdefault(user_id, []).append(item)
        self._owner[item.id] = user_id
        self._vectors[item.id] = self.embedder.encode_one(item.content)

        # 反思产物本身不该再触发反思，否则会自激振荡
        if item.kind not in ("reflection", "skill"):
            self._accumulated[user_id] = (
                self._accumulated.get(user_id, 0) + item.metadata["importance"]
            )
        return item.id

    def should_reflect(self, *, user_id: str) -> bool:
        return self._accumulated.get(user_id, 0) >= self.reflect_threshold

    def maybe_reflect(self, *, user_id: str, force: bool = False) -> list[MemoryItem]:
        """如果累积重要性够了就反思一次，返回新生成的洞察。

        **触发条件是启发式的**，这一点必须反复强调：它不是「agent 觉得
        该反思了」，而是「计数器到了」。调 ``reflect_threshold`` 等于调
        「多久花一次钱」。
        """
        if not force and not self.should_reflect(user_id=user_id):
            return []

        recent = [i for i in self._items.get(user_id, []) if i.kind not in ("reflection", "skill")][
            -self.max_reflect_window :
        ]
        if len(recent) < 3:
            return []

        self._accumulated[user_id] = 0
        self._reflect_count += 1
        insights = self._generate_insights(recent)

        out: list[MemoryItem] = []
        for text in insights:
            item = MemoryItem(
                text,
                kind="reflection",
                metadata={
                    "importance": 7,
                    "derived_from": [i.id for i in recent],
                    "reflection_round": self._reflect_count,
                },
            )
            self.add(item, user_id=user_id)
            out.append(item)
        return out

    def _generate_insights(self, recent: list[MemoryItem]) -> list[str]:
        if self.llm is None:
            # 规则版：把高分记忆罗列成一条「偏好摘要」。
            # 这**不是**反思——反思要求综合出单条记忆里没有的东西。
            # 保留它只是为了让离线流程能跑通，正文会明确指出这个差别。
            top = sorted(recent, key=lambda i: -i.metadata.get("importance", 0))[:3]
            if not top:
                return []
            return ["近期要点：" + "；".join(i.content.rstrip("。") for i in top)]

        joined = "\n".join(f"- {i.content}" for i in recent)
        result = self.llm.complete(REFLECT_PROMPT.format(memories=joined), op="reflect")
        data = result.json(default=None)
        if not isinstance(data, dict):
            return []
        return [str(x) for x in data.get("insights", []) if str(x).strip()][:3]

    # ------------------------------------------------------------------
    # 技能
    # ------------------------------------------------------------------

    def distill_skill(
        self, trace: list[str], *, user_id: str, source_episodes: list[str] | None = None
    ) -> Skill | None:
        """从一次成功的轨迹归纳出可复用流程。"""
        if self.llm is None:
            return None

        result = self.llm.complete(
            DISTILL_PROMPT.format(trace="\n".join(f"{i}. {s}" for i, s in enumerate(trace, 1))),
            op="distill",
        )
        data = result.json(default=None)
        if not isinstance(data, dict) or not data.get("steps"):
            return None

        self._skill_seq += 1
        skill = Skill(
            name=str(data.get("name") or f"技能{self._skill_seq}"),
            trigger=str(data.get("trigger") or ""),
            steps=[str(s) for s in data["steps"]],
            params=[str(p) for p in data.get("params", [])],
            source_episodes=source_episodes or [],
            sid=f"s{self._skill_seq:03d}",
        )
        self._skills.setdefault(user_id, []).append(skill)
        self.add(
            MemoryItem(
                skill.to_memory_text(),
                kind="skill",
                metadata={"importance": 7, "sid": skill.sid},
            ),
            user_id=user_id,
        )
        return skill

    def find_skill(self, task: str, *, user_id: str) -> Skill | None:
        """按 trigger 匹配技能。

        匹配的是 trigger 而不是 steps——因为用户描述的是**问题**，
        而 trigger 描述的正是「什么问题适用」。拿 steps 去匹配问题
        是技能库检索最常见的设计错误。
        """
        skills = self._skills.get(user_id, [])
        if not skills:
            return None

        q = set(_tokenize(task))
        best, best_score = None, 0.0
        for s in skills:
            t = set(_tokenize(s.trigger))
            if not t:
                continue
            overlap = len(q & t) / len(q | t)
            # 用成功率做加权，但**没用过的技能不加分也不减分**
            score = overlap * (1.0 + s.success_rate)
            if score > best_score:
                best, best_score = s, score
        return best if best_score > 0.08 else None

    def record_use(self, sid: str, *, user_id: str, success: bool) -> None:
        for s in self._skills.get(user_id, []):
            if s.sid == sid:
                s.uses += 1
                s.successes += success
                return

    def skills(self, *, user_id: str) -> list[Skill]:
        return list(self._skills.get(user_id, []))

    # ------------------------------------------------------------------
    # 检索：三项打分
    # ------------------------------------------------------------------

    def _search(self, query: str, *, user_id: str, k: int, **kwargs: Any) -> list[SearchResult]:
        items = self._items.get(user_id, [])
        if not items:
            return []

        use_importance: bool = kwargs.get("use_importance", True)
        qv = self.embedder.encode_one(query)
        now = max(i.created_at for i in items)
        oldest = min(i.created_at for i in items)
        span = max((now - oldest).total_seconds(), 1.0)

        results: list[SearchResult] = []
        for item in items:
            relevance = float(self._vectors[item.id] @ qv)
            recency = 1.0 - (now - item.created_at).total_seconds() / span
            importance = item.metadata.get("importance", _DEFAULT_IMPORTANCE) / 10.0

            score = self.w_recency * recency + self.w_relevance * relevance
            if use_importance:
                score += self.w_importance * importance

            results.append(
                SearchResult(
                    item=item,
                    score=score,
                    source="skill",
                    debug={
                        "recency": round(recency, 3),
                        "relevance": round(relevance, 3),
                        "importance": item.metadata.get("importance"),
                        "kind": item.kind,
                    },
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    # ------------------------------------------------------------------

    def _update(self, memory_id: str, patch: dict[str, Any], *, user_id: str) -> None:
        bucket = self._locate(memory_id, user_id)
        for i, item in enumerate(bucket):
            if item.id == memory_id:
                new_item = self.apply_patch(item, patch)
                bucket[i] = new_item
                if "content" in patch:
                    self._vectors[memory_id] = self.embedder.encode_one(new_item.content)
                return
        raise MemoryNotFoundError(memory_id)

    def _delete(self, memory_id: str, *, user_id: str) -> None:
        bucket = self._locate(memory_id, user_id)
        for i, item in enumerate(bucket):
            if item.id == memory_id:
                del bucket[i]
                self._owner.pop(memory_id, None)
                self._vectors.pop(memory_id, None)
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

    # ------------------------------------------------------------------

    def stats(self, *, user_id: str) -> dict[str, Any]:
        items = self._items.get(user_id, [])
        return {
            "记忆总数": len(items),
            "反思产物": sum(1 for i in items if i.kind == "reflection"),
            "技能": len(self._skills.get(user_id, [])),
            "累积重要性": self._accumulated.get(user_id, 0),
            "反思次数": self._reflect_count,
        }


def parse_json_list(text: str) -> list[str]:
    """从 LLM 回复里抠出字符串列表，失败返回空。教学脚本用。"""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return [str(x) for x in v]
    return []
