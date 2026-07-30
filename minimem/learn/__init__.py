"""交互式课程：让这本书能被「学进去」，而不只是被读完。

    python -m minimem.learn

设计依据的几条学习科学结论（每一条都对应代码里的一种步骤类型）：

- **提取练习优于重读**。所以每节有检查点，答案默认藏起来。
- **先猜再看（生成效应）**。所以关键结论前都有一道预测题，
  而且选项里一定有一个「听起来非常合理但实际是错的」——猜错比猜对更有价值。
- **间隔重复**。所以有 Leitner 盒子的复习队列，答错的明天再问。
- **合意困难**。所以练习分三级，让你能选一个刚好够不着的。
- **元认知**。所以本模块把这些原理直接写出来告诉你，
  而不是偷偷用——知道自己为什么在做某个动作，本身就能提升效果。

完整说明见 ``docs/学习方法.md``。
"""

from minimem.learn.models import (
    Apply,
    Card,
    Cliff,
    Explain,
    Lesson,
    Motto,
    Predict,
    Recall,
    Run,
    Step,
)
from minimem.learn.progress import CardState, Progress

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
    "Progress",
    "CardState",
]
