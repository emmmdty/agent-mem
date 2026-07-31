"""进度与间隔复习。

两件事：

1. **记住你学到哪了**，这样 ``learn next`` 知道该给你什么。
   进度可见本身就是动力——完成过的章节会显示为 ✅，而不是让你回忆。
2. **安排复习**。用 Leitner 盒子（间隔重复的最简形式）：
   答对升一格，答错回第一格，每格对应一个复习间隔。

   ==========  ==========  ================================
   盒子         间隔         含义
   ==========  ==========  ================================
   1           1 天        刚学会或刚答错，明天再问
   2           3 天
   3           7 天
   4           16 天
   5           35 天       基本记牢了
   ==========  ==========  ================================

   为什么用 Leitner 而不是 SM-2？因为它足够好，而且**你能看懂它**——
   一个你理解其规则的复习系统，比一个黑箱算法更容易坚持用下去。

进度文件默认放在仓库根的 ``.learn/progress.json``（已在 .gitignore 中），
所以它是**你个人的**，不会被提交，也不会和别人的冲突。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

__all__ = ["CardState", "Progress", "BOX_INTERVALS"]

BOX_INTERVALS: dict[int, int] = {1: 1, 2: 3, 3: 7, 4: 16, 5: 35}
MAX_BOX = 5

DEFAULT_DIR = Path(os.getenv("AGENT_MEM_LEARN_DIR", ".learn"))


@dataclass
class CardState:
    """一张卡片的复习状态。"""

    box: int = 1
    due: str = ""
    seen: int = 0
    correct: int = 0

    def schedule(self, *, got_it: bool, today: date | None = None) -> None:
        today = today or date.today()
        self.seen += 1
        if got_it:
            self.correct += 1
            self.box = min(MAX_BOX, self.box + 1)
        else:
            # 答错直接回第一格。不做「降一格」的温柔处理——
            # 记不住就是记不住，明天再来一次比三天后再来更有效。
            self.box = 1
        self.due = (today + timedelta(days=BOX_INTERVALS[self.box])).isoformat()

    def is_due(self, today: date | None = None) -> bool:
        if not self.due:
            return True
        return date.fromisoformat(self.due) <= (today or date.today())


@dataclass
class Progress:
    """学习进度。

    Attributes:
        completed: 已完成的章节 slug（``"ch01"``）。
        current: 当前学到的 slug。
        step_index: 当前章内进行到第几步，支持中断续学。
        cards: 卡片状态，key 为卡片正面的哈希。
        predictions: 预测题的答题记录——**保留错题**，
            因为「我当时猜错了」这个记忆点比答案本身更有价值。
        started: 首次使用日期。
        streak_days: 连续学习天数。
        last_active: 上次活动日期。
    """

    completed: list[str] = field(default_factory=list)
    current: str = ""
    step_index: int = 0
    cards: dict[str, CardState] = field(default_factory=dict)
    predictions: dict[str, bool] = field(default_factory=dict)
    started: str = ""
    streak_days: int = 0
    last_active: str = ""

    def __post_init__(self) -> None:
        # 实例属性而非 dataclass 字段：asdict() 不会把它写进 JSON
        self._base: Path | None = None

    # ------------------------------------------------------------------

    @classmethod
    def path(cls, base: Path | None = None) -> Path:
        return (base or DEFAULT_DIR) / "progress.json"

    @classmethod
    def load(cls, base: Path | None = None) -> Progress:
        """从 ``base`` 目录加载进度，并**记住这个目录**。

        记住它是为了让后续无参数的 ``save()`` 写回同一个地方。
        早期版本不记，于是测试里 ``Progress.load(tmp_path)`` 之后
        runner 内部的 ``progress.save()`` 会写到仓库根的 ``.learn/``——
        测试污染了工作区，而且因为 ``.learn/`` 在 .gitignore 里，
        这个副作用一直没被发现。
        """
        p = cls.path(base)
        if not p.exists():
            obj = cls(started=date.today().isoformat())
        else:
            raw = json.loads(p.read_text(encoding="utf-8"))
            cards = {k: CardState(**v) for k, v in raw.pop("cards", {}).items()}
            obj = cls(**raw, cards=cards)
        obj._base = base
        return obj

    def save(self, base: Path | None = None) -> None:
        p = self.path(base if base is not None else self._base)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["cards"] = {k: asdict(v) for k, v in self.cards.items()}
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------

    def touch(self, today: date | None = None) -> None:
        """记录一次活动，维护连续天数。

        连续天数是个双刃剑：它能提供动力，但断掉时也会让人放弃。
        所以这里**不惩罚断更**——断了就重新从 1 开始，
        不显示「你毁掉了 30 天的记录」这种话。
        """
        today = today or date.today()
        iso = today.isoformat()
        if self.last_active == iso:
            return
        if self.last_active == (today - timedelta(days=1)).isoformat():
            self.streak_days += 1
        else:
            self.streak_days = 1
        self.last_active = iso
        if not self.started:
            self.started = iso

    def mark_done(self, slug: str) -> None:
        if slug not in self.completed:
            self.completed.append(slug)
        self.step_index = 0

    def is_done(self, slug: str) -> bool:
        return slug in self.completed

    # ------------------------------------------------------------------

    def card_state(self, front: str) -> CardState:
        return self.cards.setdefault(_key(front), CardState())

    def due_cards(self, all_cards: list, today: date | None = None) -> list:
        """挑出到期的卡片。只从**已学过的章节**里挑——
        复习没学过的内容只会造成挫败。
        """
        out = []
        for card in all_cards:
            if f"ch{card.chapter:02d}" not in self.completed:
                continue
            if self.card_state(card.front).is_due(today):
                out.append(card)
        return out

    def record_prediction(self, qid: str, correct: bool) -> None:
        self.predictions[qid] = correct

    @property
    def prediction_accuracy(self) -> float:
        if not self.predictions:
            return 0.0
        return sum(self.predictions.values()) / len(self.predictions)


def _key(text: str) -> str:
    import hashlib

    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
