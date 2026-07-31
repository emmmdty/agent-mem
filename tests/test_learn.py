"""交互式课程的测试。

课程内容也需要测试——一道选项写错的预测题，比一段写错的代码更难被发现。
这里检查的都是**结构性错误**：答案 key 不在选项里、脚本路径不存在、
卡片正反面为空、章节缺少关键环节。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minimem.learn.lessons import ALL_CARDS, LESSONS, get_lesson
from minimem.learn.models import PRINCIPLES, Apply, Card, Motto, Predict, Recall, Run
from minimem.learn.progress import BOX_INTERVALS, CardState, Progress

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestLessonIntegrity:
    def test_章节号唯一且有序(self):
        chapters = [lesson.chapter for lesson in LESSONS]
        assert chapters == sorted(chapters)
        assert len(chapters) == len(set(chapters))

    @pytest.mark.parametrize("lesson", LESSONS, ids=lambda x: x.slug)
    def test_每章都有motto和悬念(self, lesson):
        kinds = [s.kind for s in lesson.steps]
        assert kinds[0] == "motto", "每章第一步应该是一句话点题"
        assert "cliff" in kinds, "每章末尾应该留一个悬念，把读者带到下一章"

    @pytest.mark.parametrize("lesson", LESSONS, ids=lambda x: x.slug)
    def test_motto不为空(self, lesson):
        """曾经因为 kind 是 dataclass 字段，motto 文本被吃掉过。"""
        for step in lesson.steps:
            if isinstance(step, Motto):
                assert step.text.strip(), f"{lesson.slug} 的 motto 是空的"

    @pytest.mark.parametrize("lesson", LESSONS, ids=lambda x: x.slug)
    def test_每章至少一道预测题(self, lesson):
        assert any(isinstance(s, Predict) for s in lesson.steps)

    @pytest.mark.parametrize("lesson", LESSONS, ids=lambda x: x.slug)
    def test_每章至少一个检查点(self, lesson):
        assert any(isinstance(s, Recall) for s in lesson.steps)

    @pytest.mark.parametrize("lesson", LESSONS, ids=lambda x: x.slug)
    def test_动手挑战覆盖三个难度(self, lesson):
        levels = {s.level for s in lesson.steps if isinstance(s, Apply)}
        assert levels == {"🟢", "🟡", "🔴"}, f"{lesson.slug} 的挑战难度不全：{levels}"

    @pytest.mark.parametrize("lesson", LESSONS, ids=lambda x: x.slug)
    def test_时长与难度合理(self, lesson):
        assert 10 <= lesson.minutes <= 120
        assert 1 <= lesson.difficulty <= 3


class TestPredictQuality:
    def test_答案必须在选项里(self):
        for lesson in LESSONS:
            for step in lesson.steps:
                if isinstance(step, Predict):
                    assert step.answer in step.options, (
                        f"{lesson.slug} 的预测题「{step.question[:20]}…」"
                        f"答案 {step.answer} 不在选项 {list(step.options)} 中"
                    )

    def test_至少两个选项(self):
        for lesson in LESSONS:
            for step in lesson.steps:
                if isinstance(step, Predict):
                    assert len(step.options) >= 2

    def test_必须有揭晓说明(self):
        """预测题的价值全在 reveal——它要解释「为什么直觉会错」。"""
        for lesson in LESSONS:
            for step in lesson.steps:
                if isinstance(step, Predict):
                    assert len(step.reveal.strip()) > 20, f"{lesson.slug} 有预测题缺少揭晓说明"


class TestRunSteps:
    def test_脚本路径都存在(self):
        for lesson in LESSONS:
            for step in lesson.steps:
                if isinstance(step, Run):
                    assert (REPO_ROOT / step.script).exists(), f"找不到 {step.script}"

    def test_都说明了为什么要跑(self):
        for lesson in LESSONS:
            for step in lesson.steps:
                if isinstance(step, Run):
                    assert step.why.strip(), f"{step.script} 没说明为什么要跑"


class TestApplySteps:
    def test_都有可观察的成功标准(self):
        """没有可观察的标准，读者不知道自己做完没有，很容易半途而废。"""
        for lesson in LESSONS:
            for step in lesson.steps:
                if isinstance(step, Apply):
                    assert step.success_looks_like.strip(), f"{lesson.slug} 有挑战缺少成功标准"


class TestCards:
    def test_卡片正反面都不为空(self):
        for card in ALL_CARDS:
            assert card.front.strip()
            assert card.back.strip()

    def test_卡片章节号有对应课程(self):
        chapters = {lesson.chapter for lesson in LESSONS}
        for card in ALL_CARDS:
            assert card.chapter in chapters

    def test_每章至少三张卡片(self):
        for lesson in LESSONS:
            assert len(lesson.cards) >= 3, f"{lesson.slug} 的复习卡片太少"

    def test_卡片正面唯一(self):
        fronts = [c.front for c in ALL_CARDS]
        assert len(fronts) == len(set(fronts))


class TestPrinciples:
    def test_每种步骤都标注了原理(self):
        """元认知：读者应该知道每个环节在利用什么原理。"""
        for lesson in LESSONS:
            for step in lesson.steps:
                assert step.kind in PRINCIPLES, f"{step.kind} 缺少原理说明"


class TestProgress:
    def test_首次加载给出默认值(self, tmp_path):
        p = Progress.load(tmp_path)
        assert p.completed == []
        assert p.started

    def test_存取往返(self, tmp_path):
        p = Progress.load(tmp_path)
        p.mark_done("ch01")
        p.save(tmp_path)
        assert Progress.load(tmp_path).is_done("ch01")

    def test_完成不重复记录(self, tmp_path):
        p = Progress.load(tmp_path)
        p.mark_done("ch01")
        p.mark_done("ch01")
        assert p.completed == ["ch01"]

    def test_连续天数(self, tmp_path):
        from datetime import date, timedelta

        p = Progress.load(tmp_path)
        today = date(2026, 7, 30)
        p.touch(today - timedelta(days=1))
        p.touch(today)
        assert p.streak_days == 2

    def test_断更后重新从一开始且不惩罚(self, tmp_path):
        from datetime import date, timedelta

        p = Progress.load(tmp_path)
        p.touch(date(2026, 7, 30) - timedelta(days=10))
        p.touch(date(2026, 7, 30))
        assert p.streak_days == 1


class TestSpacedRepetition:
    def test_答对升格答错回第一格(self):
        st = CardState()
        st.schedule(got_it=True)
        assert st.box == 2
        st.schedule(got_it=True)
        assert st.box == 3
        st.schedule(got_it=False)
        assert st.box == 1, "答错应直接回第一格，不做降一格的温柔处理"

    def test_盒子上限(self):
        st = CardState()
        for _ in range(10):
            st.schedule(got_it=True)
        assert st.box == max(BOX_INTERVALS)

    def test_到期判断(self):
        from datetime import date, timedelta

        today = date(2026, 7, 30)
        st = CardState()
        st.schedule(got_it=True, today=today)
        assert not st.is_due(today)
        assert st.is_due(today + timedelta(days=BOX_INTERVALS[2]))

    def test_新卡片立即到期(self):
        assert CardState().is_due()

    def test_只复习学过的章节(self, tmp_path):
        p = Progress.load(tmp_path)
        cards = [
            Card(front="问题一", back="答案", chapter=1),
            Card(front="问题二", back="答案", chapter=3),
        ]
        assert p.due_cards(cards) == []
        p.mark_done("ch01")
        assert [c.front for c in p.due_cards(cards)] == ["问题一"]

    def test_预测题正确率(self, tmp_path):
        p = Progress.load(tmp_path)
        p.record_prediction("q1", True)
        p.record_prediction("q2", False)
        assert p.prediction_accuracy == 0.5


class TestLookup:
    def test_按章号取课程(self):
        assert get_lesson(1) is not None
        assert get_lesson(999) is None


class TestRunnerSmoke:
    def test_非交互模式下能跑完一章(self, monkeypatch, tmp_path, capsys):
        """CI 里没有 tty，runner 必须自动跳过输入而不是卡住。"""
        from minimem.learn import runner

        monkeypatch.setattr(runner, "_interactive", lambda: False)
        progress = Progress.load(tmp_path)
        lesson = get_lesson(1)
        assert lesson is not None
        runner.run_lesson(lesson, progress, REPO_ROOT)
        out = capsys.readouterr().out
        assert "第 1 章" in out
        assert progress.is_done("ch01")

    def test_不污染工作区(self, monkeypatch, tmp_path):
        """runner 内部的 save() 必须写回 load() 时的目录，而不是仓库根。

        这个副作用曾经存在很久没被发现——因为 .learn/ 在 .gitignore 里。
        """
        from minimem.learn import runner

        monkeypatch.setattr(runner, "_interactive", lambda: False)
        monkeypatch.chdir(tmp_path)  # 换个 CWD，默认路径会落在这里
        store_dir = tmp_path / "elsewhere"

        progress = Progress.load(store_dir)
        lesson = get_lesson(1)
        assert lesson is not None
        runner.run_lesson(lesson, progress, REPO_ROOT)

        assert (store_dir / "progress.json").exists(), "应写回 load 时指定的目录"
        assert not (tmp_path / ".learn").exists(), "不得写到默认目录"

    def test_总览能渲染(self, tmp_path, capsys):
        from minimem.learn import runner

        runner.cmd_overview(LESSONS, Progress.load(tmp_path))
        assert "进度" in capsys.readouterr().out
