"""各章的交互式课程定义。

新增一章时，在这里加一行 import 并追加到 ``LESSONS``。
课程内容与正文 markdown 是**互补**而非重复的关系：
markdown 负责完整论述，课程负责「先猜、动手、自测、复述」这条主动学习链路。
"""

from __future__ import annotations

from minimem.learn.lessons import ch01, ch02, ch03, ch04, ch05, ch06, ch08, ch10
from minimem.learn.models import Card, Lesson

__all__ = ["LESSONS", "ALL_CARDS", "get_lesson"]

LESSONS: list[Lesson] = [
    ch01.LESSON,
    ch02.LESSON,
    ch03.LESSON,
    ch04.LESSON,
    ch05.LESSON,
    ch06.LESSON,
    ch08.LESSON,
    ch10.LESSON,
]

ALL_CARDS: list[Card] = [card for lesson in LESSONS for card in lesson.cards]


def get_lesson(chapter: int) -> Lesson | None:
    for lesson in LESSONS:
        if lesson.chapter == chapter:
            return lesson
    return None
