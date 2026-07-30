"""交互式课程运行器。

    python -m minimem.learn              # 总览与下一步
    python -m minimem.learn next         # 继续学
    python -m minimem.learn ch03         # 跳到某一章
    python -m minimem.learn review       # 今日复习
    python -m minimem.learn progress     # 进度
    python -m minimem.learn cards ch03   # 看某章的复习卡片
    python -m minimem.learn export-anki  # 导出 Anki CSV
    python -m minimem.learn reset

设计上的两个克制：

1. **不强制**。课程是正文的辅助，不是替代。你完全可以只读 markdown。
2. **不打断**。每一步都能按回车跳过，进度会保存，随时可以走。
   学习工具最常见的失败模式是「用起来比学本身还累」。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from minimem.learn.models import Apply, Card, Cliff, Explain, Lesson, Motto, Predict, Recall, Run
from minimem.learn.progress import Progress

__all__ = ["main", "run_lesson"]

W = 72


def _hr(ch: str = "─") -> str:
    return ch * W


def _wrap(text: str, indent: str = "  ") -> str:
    """按中文宽度折行。终端里读长段落很累，控制在 W 以内。"""
    import textwrap

    out = []
    for para in text.strip().split("\n"):
        if not para.strip():
            out.append("")
            continue
        out.extend(
            textwrap.wrap(
                para.strip(),
                width=W - len(indent),
                initial_indent=indent,
                subsequent_indent=indent,
            )
            or [indent]
        )
    return "\n".join(out)


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _ask(prompt: str, default: str = "") -> str:
    if not _interactive():
        return default
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  进度已保存，随时用 `python -m minimem.learn next` 回来。\n")
        raise SystemExit(0) from None


# ----------------------------------------------------------------------
# 单个步骤的呈现
# ----------------------------------------------------------------------


def _show_motto(step: Motto) -> None:
    print(f"\n{_hr('═')}")
    print(f"  💡 {step.text}")
    print(f"{_hr('═')}")


def _show_predict(step: Predict, progress: Progress) -> None:
    print(f"\n  🤔 先猜一猜  ({step.principle})")
    print(_hr())
    print(_wrap(step.question))
    print()
    for key, text in step.options.items():
        print(_wrap(f"{key}. {text}", indent="    "))
    print()

    guess = _ask("  你的选择（直接回车看答案）：", default="").upper()
    correct = guess == step.answer.upper()

    if guess:
        progress.record_prediction(step.question[:40], correct)
        print(f"\n  {'✅ 猜对了' if correct else '❌ 猜错了'}——正确答案是 {step.answer}")
        if not correct:
            print(_wrap("猜错在这里是好事：预期和现实的落差会让接下来的解释记得更牢。"))
    else:
        print(f"\n  正确答案是 {step.answer}")

    print()
    print(_wrap(step.reveal))
    if step.trap:
        print()
        print(_wrap(f"⚠️  {step.trap}"))


def _show_run(step: Run, repo_root: Path) -> None:
    print(f"\n  ▶️  跑一下  （约 {step.minutes} 分钟）")
    print(_hr())
    print(_wrap(step.why))
    print()
    print(f"    python {step.script}")
    if step.look_for:
        print()
        print(_wrap(f"👀 重点看：{step.look_for}"))

    ans = _ask("\n  现在就跑吗？[Y/n] ", default="n").lower()
    if ans in ("", "y", "yes"):
        script = repo_root / step.script
        if not script.exists():
            print(f"  ⚠️  找不到 {script}")
            return
        print()
        subprocess.run([sys.executable, str(script)], cwd=repo_root, check=False)


def _show_recall(step: Recall, progress: Progress) -> None:
    print(f"\n  ✅ 检查点  ({step.principle})")
    print(_hr())
    print(_wrap(step.question))
    if step.hint:
        print(_wrap(f"（提示：{step.hint}）"))

    _ask("\n  想好了按回车看答案…", default="")
    print()
    print(_wrap(step.answer))

    got = _ask("\n  答上来了吗？[y/N] ", default="y").lower() in ("y", "yes", "")
    state = progress.card_state(step.question)
    state.schedule(got_it=got)
    if not got:
        print(_wrap("  记下了，明天的 `learn review` 会再问你一次。"))


def _show_explain(step: Explain) -> None:
    print(f"\n  🗣️  讲给别人听  ({step.principle})")
    print(_hr())
    print(_wrap(step.task))
    print()
    print(_wrap("一个合格的解释应该包含："))
    for item in step.checklist:
        print(_wrap(f"□ {item}", indent="    "))
    print()
    print(_wrap("没有标准答案。对照上面的清单自评——讲不清楚的那条，就是没懂的那条。"))
    _ask("\n  讲完了按回车继续…", default="")


def _show_apply(step: Apply) -> None:
    print(f"\n  🔧 动手挑战 {step.level}  ({step.principle})")
    print(_hr())
    print(_wrap(step.task))
    if step.starting_point:
        print(_wrap(f"从这里开始：{step.starting_point}"))
    print(_wrap(f"做对了会看到：{step.success_looks_like}"))
    if step.verify:
        print(_wrap(f"验证：{step.verify}"))


def _show_cliff(step: Cliff) -> None:
    print(f"\n{_hr('═')}")
    print(_wrap(f"🚪 {step.text}"))
    if step.next_chapter:
        print(_wrap(f"→ {step.next_chapter}"))
    print(_hr("═"))


# ----------------------------------------------------------------------


def run_lesson(lesson: Lesson, progress: Progress, repo_root: Path, *, start: int = 0) -> None:
    print(f"\n\n{_hr('━')}")
    print(f"  第 {lesson.chapter} 章 · {lesson.title}")
    print(f"  预计 {lesson.minutes} 分钟   难度 {lesson.stars}")
    print(_hr("━"))

    steps = lesson.steps
    for i in range(start, len(steps)):
        step = steps[i]
        progress.current = lesson.slug
        progress.step_index = i
        progress.save()

        match step:
            case Motto():
                _show_motto(step)
            case Predict():
                _show_predict(step, progress)
            case Run():
                _show_run(step, repo_root)
            case Recall():
                _show_recall(step, progress)
            case Explain():
                _show_explain(step)
            case Apply():
                _show_apply(step)
            case Cliff():
                _show_cliff(step)

        if not _interactive():
            continue
        if i < len(steps) - 1 and step.kind not in ("motto", "cliff"):
            cmd = _ask("\n  [回车] 继续  [q] 先歇会儿 > ", default="")
            if cmd.lower() == "q":
                print(_wrap("\n进度已存。回来时执行：python -m minimem.learn next\n"))
                progress.save()
                return

    progress.mark_done(lesson.slug)
    progress.touch()
    progress.save()
    print(f"\n  🎉 第 {lesson.chapter} 章完成。")
    print(_wrap(f"复习卡片已加入队列（{len(lesson.cards)} 张），明天用 `learn review` 巩固。"))


# ----------------------------------------------------------------------


def cmd_overview(lessons: list[Lesson], progress: Progress) -> None:
    print(f"\n{_hr('━')}")
    print("  agent-mem · 交互式课程")
    print(_hr("━"))

    done = len(progress.completed)
    total = len(lessons)
    bar_len = 30
    filled = int(bar_len * done / max(total, 1))
    print(f"\n  进度  [{'█' * filled}{'░' * (bar_len - filled)}]  {done}/{total} 章")
    if progress.streak_days:
        print(f"  连续学习 {progress.streak_days} 天")
    if progress.predictions:
        print(f"  预测题正确率 {progress.prediction_accuracy:.0%}（猜错不扣分，反而记得更牢）")

    print(f"\n  {'章节':<34}{'时长':>6}{'难度':>8}{'状态':>8}")
    print("  " + _hr("─"))
    for lesson in lessons:
        mark = "✅" if progress.is_done(lesson.slug) else "  "
        title = f"第 {lesson.chapter} 章 {lesson.title}"
        print(f"  {title:<34}{lesson.minutes:>4}分{lesson.stars:>8}{mark:>8}")

    nxt = _next_lesson(lessons, progress)
    print()
    if nxt:
        print(_wrap(f"下一步：第 {nxt.chapter} 章 {nxt.title}"))
        print(_wrap("开始：python -m minimem.learn next", indent="    "))
    else:
        print(_wrap("全部完成。用 `python -m minimem.learn review` 保持记忆。"))
    print()


def _next_lesson(lessons: list[Lesson], progress: Progress) -> Lesson | None:
    for lesson in lessons:
        if not progress.is_done(lesson.slug):
            return lesson
    return None


def cmd_review(all_cards: list[Card], progress: Progress) -> None:
    due = progress.due_cards(all_cards)
    print(f"\n{_hr('━')}")
    print(f"  今日复习：{len(due)} 张卡片")
    print(_hr("━"))

    if not due:
        print(_wrap("\n没有到期的卡片。间隔重复的重点是**不要多复习**——"))
        print(_wrap("在快忘掉的时候复习，效果最好；提前复习是浪费时间。\n"))
        return

    right = 0
    for i, card in enumerate(due, 1):
        print(f"\n  [{i}/{len(due)}] 第 {card.chapter} 章")
        print(_wrap(card.front))
        _ask("\n  想好了按回车…", default="")
        print()
        print(_wrap(card.back))
        got = _ask("\n  答对了吗？[y/N] ", default="y").lower() in ("y", "yes", "")
        right += got
        progress.card_state(card.front).schedule(got_it=got)

    progress.touch()
    progress.save()
    print(f"\n  本次 {right}/{len(due)} 张答对。答错的明天会再出现。\n")


def cmd_cards(lessons: list[Lesson], chapter: str | None) -> None:
    for lesson in lessons:
        if chapter and lesson.slug != chapter:
            continue
        if not lesson.cards:
            continue
        print(f"\n  第 {lesson.chapter} 章 · {lesson.title}")
        print("  " + _hr("─"))
        for card in lesson.cards:
            print(f"\n  Q: {card.front}")
            print(_wrap(f"A: {card.back}", indent="     "))
    print()


def cmd_export_anki(lessons: list[Lesson], out: Path) -> None:
    import csv

    rows = [
        (c.front, c.back, f"agent-mem::ch{c.chapter:02d}")
        for lesson in lessons
        for c in lesson.cards
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\n  已导出 {len(rows)} 张卡片到 {out}")
    print(_wrap("Anki 导入时选「字段用逗号分隔」，三列分别是正面、背面、牌组标签。\n"))


def main(argv: list[str] | None = None) -> int:
    from minimem.learn.lessons import ALL_CARDS, LESSONS

    argv = argv if argv is not None else sys.argv[1:]
    repo_root = Path(__file__).resolve().parents[2]
    progress = Progress.load()

    cmd = argv[0] if argv else ""

    if not cmd:
        cmd_overview(LESSONS, progress)
        return 0

    if cmd == "next":
        lesson = _next_lesson(LESSONS, progress)
        if lesson is None:
            print(_wrap("\n所有章节都完成了。用 `learn review` 保持记忆。\n"))
            return 0
        start = progress.step_index if progress.current == lesson.slug else 0
        run_lesson(lesson, progress, repo_root, start=start)
        return 0

    if cmd == "progress":
        cmd_overview(LESSONS, progress)
        return 0

    if cmd == "review":
        cmd_review(ALL_CARDS, progress)
        return 0

    if cmd == "cards":
        cmd_cards(LESSONS, argv[1] if len(argv) > 1 else None)
        return 0

    if cmd == "export-anki":
        cmd_export_anki(LESSONS, Path(argv[1] if len(argv) > 1 else "agent-mem-cards.csv"))
        return 0

    if cmd == "reset":
        if _ask("  确定要清空进度吗？[y/N] ", default="n").lower() in ("y", "yes"):
            Progress().save()
            print("  已重置。\n")
        return 0

    for lesson in LESSONS:
        if cmd in (lesson.slug, f"ch{lesson.chapter}", str(lesson.chapter)):
            run_lesson(lesson, progress, repo_root)
            return 0

    print(f"\n  不认识的命令：{cmd}")
    print(_wrap("可用：next / progress / review / cards / export-anki / reset / chNN\n"))
    return 1
