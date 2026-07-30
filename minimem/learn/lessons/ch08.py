"""第 8 章的交互式课程。"""

from __future__ import annotations

from minimem.learn.models import Apply, Card, Cliff, Explain, Lesson, Motto, Predict, Recall, Run

LESSON = Lesson(
    chapter=8,
    title="经验与技能记忆",
    minutes=35,
    difficulty=2,
    prerequisites=[1, 3],
    steps=[
        Motto("前七章记的是「发生了什么」，这一章记的是「该怎么做」。"),
        Predict(
            question=(
                "Generative Agents 在 recency + relevance 之外加了第三项打分：importance"
                "（让 LLM 给每条记忆打 1~10 分）。在 MiniBench 上加入它，分能力表会怎么变？"
            ),
            options={
                "A": "全面提升",
                "B": "大部分提升，但有几类明显变差",
                "C": "基本没变化",
            },
            answer="B",
            reveal=(
                "总召回从 54.9% 涨到 65.3%，直接检索 50%→100%、同义改写 40%→60%。\n"
                "但**多跳从 25% 掉到 0%**，知识更新从 58% 掉到 33%。"
            ),
            trap=(
                "原因：多跳题的第二跳记忆往往是「XR-2049 上线后复核量下降四成」"
                "这种**重要性不高但正好是答案**的内容。importance 把「过敏」「离职」"
                "这类高分记忆顶上去，恰好挤掉了它们。\n"
                "**结论不是「importance 没用」，而是打分函数要按任务选。**"
                "Generative Agents 的三项打分是为「模拟有连续生活的 agent」设计的，"
                "照搬到问答系统里，你可能买到一个负收益。"
            ),
        ),
        Run(
            script="docs/chapter8/code/01_importance.py",
            why="看 importance 在哪几类能力上是正收益、哪几类是负收益。",
            look_for="分能力表里「多跳」那一列——从 25% 掉到 0%。",
            minutes=1,
        ),
        Recall(
            question="为什么加入 importance 之后多跳召回率反而掉到 0？",
            answer=(
                "多跳题需要召回「第二跳」的中间记忆，而这类记忆通常重要性不高"
                "（既不是身份也不是禁忌）。importance 把高分记忆顶到前列，恰好挤掉了它们。\n"
                "说明打分函数有**任务偏好**，不能因为论文用了就照搬。"
            ),
        ),
        Predict(
            question=("一轮反思实验里，importance 打分调用了 30 次、反思调用了 1 次。哪部分更贵？"),
            options={
                "A": "打分 —— 30 次对 1 次，差 30 倍",
                "B": "反思 —— 它单次就可能比 30 次打分加起来还贵",
                "C": "差不多",
            },
            answer="B",
            reveal=(
                "反思是**低频高价**操作：它要把最近十几条记忆全塞进 prompt，"
                "单次几千 token；而 importance 打分单次只有一两百。"
            ),
            trap=(
                "**按调用次数估算成本会严重低估。** 你看到「只调用了 1 次」就放心了，"
                "而那 1 次可能比前面 30 次加起来还贵。这类误判在按次计费的场景里非常常见。"
            ),
        ),
        Run(
            script="docs/chapter8/code/02_reflection_and_skills.py",
            why="看反思产出什么、花多少钱，以及技能库怎么被复用。",
            look_for="反思那条洞察——它**不在任何单条记忆里**，是从三条偏好综合出来的。",
            minutes=1,
        ),
        Recall(
            question="「自发反思」存在吗？",
            answer=(
                "不存在。所有已发表系统的 reflection 触发条件都是启发式的——"
                "定时、计数、或阈值。\n"
                "这和第 1 章批评「情景记忆到语义记忆的巩固并不自动」"
                "是同一个问题换了个位置出现。调 reflect_threshold 等于调「多久花一次钱」。"
            ),
        ),
        Predict(
            question="技能库检索时，该拿用户的任务描述去匹配技能的哪个字段？",
            options={
                "A": "steps（步骤）—— 那是技能的主体内容",
                "B": "trigger（触发条件）",
            },
            answer="B",
            reveal=(
                "用户描述的是**问题**（「查询特别慢」），而 trigger 描述的正是"
                "「什么问题适用」；steps 描述的是**解法**。"
                "拿解法去匹配问题，方向反了。这是 AWM 的关键设计。"
            ),
            trap=(
                "另外两个常见错误：**把「没用过」当成「一定成功」**"
                "（未验证的流程会压过久经考验的）；**只记成功不记失败**"
                "（Reflexion 的核心恰恰是把失败教训写进记忆、下次避开）。"
            ),
        ),
        Explain(
            task="向同事解释：我们要不要给客服机器人加上「重要性打分 + 定期反思」。",
            checklist=[
                "先说 importance 的任务偏好，举多跳掉到 0 这个实测反例",
                "算清成本结构：打分每条一次、反思低频高价",
                "指出触发条件是启发式的，调阈值就是调花钱频率",
                "给出判断依据：读写比，以及「先上规则版看是正收益还是负收益」",
            ],
        ),
        Apply(
            level="🟢",
            task="把 weights 从 (0.2, 0.5, 0.3) 改成等权 (1/3, 1/3, 1/3)，重跑实验一。",
            starting_point="docs/chapter8/code/01_importance.py 里构造 SkillMemory 的地方",
            success_looks_like="recency 权重变大后「时序」类可能变好而「同义改写」变差——权重就是任务偏好",
        ),
        Apply(
            level="🟡",
            task=(
                "实现动态 importance：一条记忆每被成功召回一次，重要性 +0.5（上限 10）。"
                "让「当时不重要但后来常被用到」的记忆自己浮上来，"
                "验证它对多跳的伤害是否减轻。"
            ),
            starting_point="minimem/skill.py 的 _search，在返回结果时回写",
            success_looks_like="多次检索后多跳类召回率回升",
            verify="python -m pytest tests/test_skill.py -q",
        ),
        Apply(
            level="🔴",
            task=(
                "实现失败技能库：记录失败轨迹与原因，检索时把「上次这么做失败了」"
                "一并注入上下文。设计实验证明它减少了重复犯错——"
                "注意你得先构造一个「会重复犯错」的场景。"
            ),
            starting_point="minimem/skill.py 的 Skill 与 find_skill",
            success_looks_like="同一类任务的第二次尝试避开了第一次的坑",
        ),
        Cliff(
            text=(
                "到这里，全书的记忆都存在**上下文里**——不管是原文、向量、图、事实还是技能，"
                "最终都要拼进 prompt 才起作用。还有第三种可能：写进模型权重。"
            ),
            next_chapter="第 9 章：参数化记忆，以及它为什么还不能作为主路线。",
        ),
    ],
    cards=[
        Card(
            front="为什么加入 importance 打分后，多跳类召回率反而下降？",
            back=(
                "多跳题需要「第二跳」的中间记忆，而这类记忆重要性通常不高。"
                "importance 把高分记忆顶到前列，恰好挤掉了它们。"
                "**打分函数有任务偏好**——Generative Agents 的三项打分是为"
                "「模拟有连续生活的 agent」设计的，照搬到问答上可能是负收益。"
            ),
            chapter=8,
        ),
        Card(
            front="「自发反思」存在吗？",
            back=(
                "不存在。所有已发表系统都是定时/计数/阈值触发。"
                "这和第 1 章批评「巩固并不自动」是同一个问题换了位置。"
                "调 reflect_threshold 等于调「多久花一次这个钱」。"
            ),
            chapter=8,
        ),
        Card(
            front="为什么「反思调用次数少所以便宜」是错的？",
            back=(
                "反思是**低频高价**：要把最近十几条记忆全塞进 prompt，单次几千 token；"
                "importance 打分单次只有一两百。按调用次数估算会严重低估——"
                "1 次反思可能比 30 次打分加起来还贵。"
            ),
            chapter=8,
        ),
        Card(
            front="技能库检索该匹配 trigger 还是 steps？",
            back=(
                "**trigger**。用户描述的是「问题」，trigger 描述「什么问题适用」，"
                "steps 描述的是解法。拿解法匹配问题，方向反了——"
                "这是技能库最常见的设计错误。"
            ),
            chapter=8,
        ),
        Card(
            front="没用过的技能，成功率该记 0 还是 1？为什么？",
            back=(
                "**0**。记 1 的话，刚归纳出来、从未验证过的流程会被优先推荐，"
                "而久经考验的老技能排在后面。另外还要记失败——"
                "只记成功的技能库会一直重复同一个错误。"
            ),
            chapter=8,
        ),
    ],
)
