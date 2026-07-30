"""交互式课程的积木：一节课由哪些「步骤」组成。

每种步骤对应一条有实证支持的学习原理。之所以把原理直接写进代码注释和
课程输出里，是因为**知道自己为什么在做某个动作，本身就能提升学习效果**
（元认知策略）。读者不该只是被动地被引导，而应该知道这套流程在利用什么。

============  ================================  ==========================
步骤           做什么                             背后的原理
============  ================================  ==========================
``Motto``     一句话点题                          组块化（chunking）
``Predict``   先猜答案，再揭晓                     生成效应 + 预期违背
``Run``       跑一个脚本，看真实输出                主动实践
``Recall``    合上书回答问题                       提取练习（testing effect）
``Explain``   用自己的话讲一遍                     费曼技巧 / 精细加工
``Apply``     动手改代码                          迁移（transfer）
``Cliff``     留一个未解决的问题                    蔡格尼克效应
============  ================================  ==========================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

__all__ = [
    "Step",
    "Motto",
    "Predict",
    "Run",
    "Recall",
    "Explain",
    "Apply",
    "Cliff",
    "Lesson",
    "Card",
    "PRINCIPLES",
]

PRINCIPLES: dict[str, str] = {
    "motto": "组块化：把一章压成一句能记住的话，减轻工作记忆负担",
    "predict": "生成效应：先猜再看，哪怕猜错，记忆也比直接读强得多",
    "run": "主动实践：亲手跑出来的数字，比读到的数字牢固",
    "recall": "提取练习：从脑子里「取」比再「看」一遍有效数倍",
    "explain": "费曼技巧：讲不清楚的地方，就是你没懂的地方",
    "apply": "迁移：能改动它，才算真的会用它",
    "cliff": "蔡格尼克效应：未完成的问题会一直挂在心里，把你带到下一章",
}


@dataclass
class Step:
    """所有步骤的基类。

    ``kind`` 是 ``ClassVar`` 而不是 dataclass 字段——这一点很关键：
    如果它是字段，就会占用子类的第一个位置参数，
    于是 ``Motto("上下文总会满")`` 会把那句话赋给 kind 而不是 text。
    （这个 bug 真的发生过，修复后留下这段注释。）
    """

    kind: ClassVar[str] = "step"

    @property
    def principle(self) -> str:
        return PRINCIPLES.get(self.kind, "")


@dataclass
class Motto(Step):
    """一句话点题。放在每节开头，也是复习时的锚点。"""

    text: str = ""
    kind: ClassVar[str] = "motto"


@dataclass
class Predict(Step):
    """先猜后验。

    这是本课程最重要的一种步骤。**答错比答对更有价值**——
    预期与现实的落差会制造一个「惊讶」，而惊讶是记忆的强力粘合剂。
    所以选项里一定要有一个「听起来非常合理但实际是错的」。

    Attributes:
        question: 问题。
        options: 选项，形如 ``{"A": "……", "B": "……"}``。
        answer: 正确选项的 key。
        reveal: 揭晓时的解释——重点解释**为什么直觉会错**。
        trap: 可选，点明哪个选项是「合理但错误」的陷阱及其来由。
    """

    question: str = ""
    options: dict[str, str] = field(default_factory=dict)
    answer: str = ""
    reveal: str = ""
    trap: str = ""
    kind: ClassVar[str] = "predict"


@dataclass
class Run(Step):
    """跑一个脚本。runner 会问你要不要直接执行。

    Attributes:
        script: 相对仓库根的路径。
        why: 为什么要跑它——**先说目的，再让人动手**，否则就成了照着敲命令。
        look_for: 提示读者该盯着输出里的哪一处。
        minutes: 预计耗时，用于降低启动阻力。
    """

    script: str = ""
    why: str = ""
    look_for: str = ""
    minutes: int = 1
    kind: ClassVar[str] = "run"


@dataclass
class Recall(Step):
    """自测。答案默认折叠，逼你先自己想。"""

    question: str = ""
    answer: str = ""
    hint: str = ""
    kind: ClassVar[str] = "recall"


@dataclass
class Explain(Step):
    """费曼式：用自己的话讲给别人听。

    没有标准答案，只给一个「合格的解释应该包含什么」的清单，
    让你自己对照。自评比给标准答案更有效——因为对照清单的过程
    本身就是一次结构化的复述。
    """

    task: str = ""
    checklist: list[str] = field(default_factory=list)
    kind: ClassVar[str] = "explain"


@dataclass
class Apply(Step):
    """动手挑战。三个难度，读者自选。

    Attributes:
        level: ``"🟢"`` 改参数看现象 / ``"🟡"`` 补全实现 / ``"🔴"`` 开放设计。
        task: 任务描述。
        starting_point: 从哪个文件、哪一行开始改。
        success_looks_like: 做对了会看到什么——**必须可观察**，
            否则读者不知道自己完成没有。
        verify: 可选的自动验证命令。
    """

    level: str = "🟢"
    task: str = ""
    starting_point: str = ""
    success_looks_like: str = ""
    verify: str = ""
    kind: ClassVar[str] = "apply"


@dataclass
class Cliff(Step):
    """章末悬念：留一个本章解决不了的问题。"""

    text: str = ""
    next_chapter: str = ""
    kind: ClassVar[str] = "cliff"


@dataclass
class Card:
    """复习卡片。用于间隔重复，也可导出到 Anki。

    卡片刻意只问「为什么」和「什么时候」，不问「是什么」——
    定义类的知识查得到，判断类的知识查不到。
    """

    front: str
    back: str
    chapter: int
    tag: str = ""


@dataclass
class Lesson:
    """一节课。

    Attributes:
        chapter: 章号。
        title: 标题。
        minutes: 预计时长。**必须诚实**——低估时长会让读者中途放弃，
            而「说好 15 分钟结果 15 分钟真的做完了」会显著提升继续的意愿。
        difficulty: 1~3。
        prerequisites: 前置章节号。
        steps: 步骤序列。
        cards: 本章的复习卡片。
    """

    chapter: int
    title: str
    minutes: int
    difficulty: int = 1
    prerequisites: list[int] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    cards: list[Card] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return f"ch{self.chapter:02d}"

    @property
    def stars(self) -> str:
        return "●" * self.difficulty + "○" * (3 - self.difficulty)
