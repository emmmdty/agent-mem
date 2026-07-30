# offline-ok
"""第 8 章实验二：反思与技能库。

    python docs/chapter8/code/02_reflection_and_skills.py

两个机制：

  **反思**：累积重要性越过阈值时，让 LLM 从近期记忆归纳出更高层的洞察。
  **技能库**：把成功的执行轨迹归纳成带触发条件和参数槽的可复用流程。

本脚本用 ScriptedLLM 模拟这两次调用（离线可跑）。**它模拟的是格式和用量，
不是 LLM 的判断力**——真实模型归纳出的洞察质量差别很大，而这一章的
结论恰恰依赖那个判断力。正文会明确指出哪些结论换真模型会变。
"""

from __future__ import annotations

import json
import warnings

from minimem.eval import as_memory_items
from minimem.skill import SkillMemory
from minimem.utils.embedding import get_embedder
from minimem.utils.llm import ScriptedLLM
from minimem.utils.metering import Meter

USER = "u_bench"


def make_llm(meter: Meter) -> ScriptedLLM:
    """注册三种调用的预设逻辑。"""
    llm = ScriptedLLM(meter=meter, store_label="skill")

    # importance：按关键词给分，模拟一个「和规则版差不多」的模型
    def importance(prompt: str) -> str:
        high = ("过敏", "离职", "入职", "搬家", "我叫")
        mid = ("喜欢", "尽量", "麻烦", "习惯", "负责")
        score = 9 if any(w in prompt for w in high) else 5 if any(w in prompt for w in mid) else 2
        return json.dumps({"score": score, "reason": "模拟打分"}, ensure_ascii=False)

    # reflection：从记忆里综合出偏好类洞察
    def reflect(prompt: str) -> str:
        insights = []
        if "下午" in prompt and "邮件" in prompt:
            insights.append("沟通偏好：会议排下午、用消息而非邮件、文档要 Markdown")
        if "离职" in prompt or "入职" in prompt:
            insights.append("正处于职业变动期：已从原东家离职，下月入职新公司")
        if "毛毛" in prompt or "猫" in prompt:
            insights.append("养了一只叫毛毛的猫，最近在关注它的健康")
        return json.dumps({"insights": insights[:3]}, ensure_ascii=False)

    # 技能归纳
    def distill(prompt: str) -> str:
        return json.dumps(
            {
                "name": "排查慢 SQL",
                "trigger": "同事反馈某个查询很慢，需要定位原因",
                "steps": [
                    "拿到 {查询语句} 与执行环境",
                    "看执行计划，确认是否走了索引",
                    "检查 join 条件是否写错或缺失",
                    "对 {关键字段} 补索引后复测",
                    "把结论同步到 {文档位置}",
                ],
                "params": ["查询语句", "关键字段", "文档位置"],
            },
            ensure_ascii=False,
        )

    llm.register("重要性分数", importance)
    llm.register("归纳出 1~3 条更高层的洞察", reflect)
    llm.register("归纳成一个可复用的流程", distill)
    return llm


def part1_reflection(embedder) -> None:
    print("\n  第一部分：反思")
    print("  " + "-" * 66)

    meter = Meter()
    store = SkillMemory(llm=make_llm(meter), embedder=embedder, reflect_threshold=40, meter=meter)
    store.add_many(as_memory_items(), user_id=USER)

    print(f"\n    写入 30 条记忆后：{store.stats(user_id=USER)}")
    print(f"    累积重要性已达阈值？{store.should_reflect(user_id=USER)}")

    insights = store.maybe_reflect(user_id=USER)
    print(f"\n    反思产出 {len(insights)} 条洞察：")
    for item in insights:
        print(f"      · {item.content}")

    s = meter.summary()
    print(
        f"""
    这一轮的账：{s["llm_calls"]} 次 LLM 调用，{s["tokens_total"]} token。
    其中绝大部分是 importance 打分（每条记忆一次），反思本身只有 1 次。

    **但反思那一次是重操作**：它要把最近十几条记忆全塞进 prompt。
    真实场景里一次反思几千 token 很常见，而 importance 打分每次只有一两百。

    触发条件是 `reflect_threshold`——一个**纯启发式**的阈值。
    调它等于调「多久花一次这个钱」。所有已发表系统都是这么做的：
    定时、计数、或者阈值。**没有哪个系统真的实现了「自发反思」。**
    这和第 1 章批评「巩固并不自动」是同一个问题换了个位置。
"""
    )


def part2_skills(embedder) -> None:
    print("\n  第二部分：技能库")
    print("  " + "-" * 66)

    meter = Meter()
    store = SkillMemory(llm=make_llm(meter), embedder=embedder, meter=meter)

    trace = [
        "同事说订单查询接口很慢",
        "拉了那条 SQL，看执行计划",
        "发现 join 条件漏了一个字段，走了全表扫描",
        "补上索引后从 3.2s 降到 40ms",
        "把结论写进了团队文档",
    ]
    print("\n    第一次遇到这类问题，走了 5 步：")
    for i, step in enumerate(trace, 1):
        print(f"      {i}. {step}")

    before = meter.summary()["tokens_total"]
    skill = store.distill_skill(trace, user_id=USER)
    distill_cost = meter.summary()["tokens_total"] - before

    if skill is None:
        print("\n    ⚠️  归纳失败（LLM 返回的格式不对）")
        return

    print(f"\n    归纳出的可复用流程（花了 {distill_cost} token）：")
    for line in skill.render().splitlines():
        print(f"      {line}")

    print("\n    ── 第二次遇到类似问题 ──")
    task = "线上有个查询特别慢，帮我看看什么原因"
    found = store.find_skill(task, user_id=USER)
    print(f"    任务：「{task}」")
    if found:
        print(
            f"    命中技能：{found.name}（成功率 {found.success_rate:.0%}，用过 {found.uses} 次）"
        )
        print("    → 不必从头摸索，直接照着 5 步走")
        store.record_use(found.sid, user_id=USER, success=True)
        print(f"    执行后回填：用过 {found.uses} 次，成功率 {found.success_rate:.0%}")
    else:
        print("    没有命中任何技能")

    print(
        """
    三个设计细节值得注意：

    · **匹配的是 trigger，不是 steps。** 用户描述的是「问题」，
      而 trigger 描述的正是「什么问题适用」。拿 steps 去匹配问题，
      是技能库检索最常见的设计错误。

    · **没用过的技能成功率是 0%，不是 100%。** 把「没试过」当成
      「一定成功」，会让刚归纳出来、从未验证的流程被优先推荐。

    · **失败也要记。** Reflexion 的核心正是把失败的教训写进记忆、下次避开。
      只记成功的技能库，会一直重复同一个错误。
"""
    )


def part3_caveat() -> None:
    print("\n  第三部分：关于这些数字的免责说明")
    print("  " + "-" * 66)
    print(
        """
    本脚本的 LLM 是**模拟的**。它按预设返回格式正确的洞察和流程，
    不会归纳错、不会漏、不会返回非法 JSON。

    换成真实模型后，**下面这些会变**：

      · 反思质量。模拟器输出的「洞察」是我手写的；真实模型可能
        综合得更好，也可能只是把几条记忆复述一遍——**后者很常见**，
        而复述型洞察会污染记忆库（占地方、被检索、还显得像新知识）。
      · 技能归纳的泛化程度。哪些值该抽成参数、哪些该保留，
        是个需要判断力的问题，模拟器给不了。
      · 失败率。真实调用会超时、会返回非法 JSON、会拒答。
        第 6 章讨论过这类失败的代价——它们会**固化进记忆**。

    **不变的是成本结构**：importance 每条一次、反思每轮一次且很贵、
    技能归纳每次一次。这部分结论可以直接带走。
"""
    )


def main() -> None:
    print("\n第 8 章实验二：反思与技能库")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedder = get_embedder()
    part1_reflection(embedder)
    part2_skills(embedder)
    part3_caveat()


if __name__ == "__main__":
    main()
