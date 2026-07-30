"""第 9 章的交互式课程。"""

from __future__ import annotations

from minimem.learn.models import Apply, Card, Cliff, Explain, Lesson, Motto, Predict, Recall, Run

LESSON = Lesson(
    chapter=9,
    title="参数化记忆与持续学习",
    minutes=30,
    difficulty=3,
    prerequisites=[1],
    steps=[
        Motto("把记忆写进权重是可行的，但它目前不适合作为「用户说什么你记什么」这条主路径。"),
        Predict(
            question=(
                "用模型编辑把「法国的首都」从巴黎改成里昂。改完之后，"
                "问「巴黎位于哪个国家」，模型会怎么答？"
            ),
            options={
                "A": "会说不知道 —— 相关知识被一起改了",
                "B": "仍然说法国，且仍然认为巴黎是首都 —— 旧推论没被动过",
            },
            answer="B",
            reveal=(
                "这就是**涟漪效应**：改了一条事实，但由它推出的结论没跟着改。"
                "模型会同时相信「法国的首都是里昂」和「巴黎是法国的首都」。"
            ),
            trap=(
                "要处理它，你得知道「哪些结论依赖这条事实」——"
                "而在真实模型里**没人知道**。知识在权重里是纠缠的，没有依赖图可查。\n"
                "对比上下文类记忆：第 5 章的 provenance 让这件事变得可查。"
            ),
        ),
        Run(
            script="docs/chapter9/code/02_why_editing_is_hard.py",
            why="用玩具知识库把三个失败模式演一遍，不需要 GPU。",
            look_for="失败模式二：三次编辑之后，知识状态自相矛盾且无法逐条撤销。",
            minutes=1,
        ),
        Recall(
            question="模型编辑的三个评估维度是什么？看到「编辑成功率 99%」该问什么？",
            answer=(
                "**Reliability**（目标问题改对没）、**Generalization**（换个问法还对吗）、"
                "**Specificity/Locality**（不该改的没被改吧）。\n"
                "该问：**在哪个维度上？** Reliability 几乎总是很高——"
                "你改的地方当然改成功了。难点全在后两个。"
            ),
        ),
        Predict(
            question="参数化记忆相比上下文类记忆，失去了哪些能力？",
            options={
                "A": "只是检索方式不同，能力一样",
                "B": "失去可审计、可回滚、可解释",
                "C": "上面三样，外加**可删除**",
            },
            answer="C",
            reveal=(
                "最后一样最要命：**GDPR 的删除权在参数化记忆上目前没有可靠实现方式**。"
                "你没法保证一条信息从权重里被彻底移除。"
            ),
            trap=(
                "machine unlearning（机器遗忘）是活跃但**远未成熟**的方向。"
                "在它成熟之前，任何涉及个人信息的记忆都不该只放在权重里。"
            ),
        ),
        Predict(
            question=(
                "在 RTX 4090 上真跑一次：把 Qwen2.5-0.5B 的「法国的首都」编辑成「里昂」，"
                "10 步收敛、目标问题改对了。此时用**英文**问 "
                "「The capital of France is」，输出会怎样？"
            ),
            options={
                "A": "也变成 Lyon —— 知识是语言无关的",
                "B": "变成一个中英混杂的奇怪答案",
                "C": "一个字都没变，还是 Paris",
            },
            answer="C",
            reveal=(
                "**一个字都没变**：编辑前后都是「Paris. It was founded in」。\n"
                "编辑发生在中文的表示上，英文路径完全没被触及。\n"
                "整体 Generalization 只有 25%（4 个改写问法只有 1 个跟着改）。"
            ),
            trap=(
                "同一次实验的 Specificity 是 50%，但**仔细看那两条「变了」的**："
                "「德国的首都是」从「哪个城市？」变成了「柏林」——**它变得更对了**。\n"
                "这不是知识被污染，是**回答格式发生了溢出**。\n"
                "自动化指标只能测「输出是否改变」，分不清「被污染」和「被带偏」——"
                "这正是文献里编辑成功率难以直接比较的原因之一。"
            ),
        ),
        Run(
            script="docs/chapter9/code/03_minimal_edit.py",
            why="（需要 GPU）自己跑一次真实的权重编辑，一百行实现，不依赖 EasyEdit。",
            look_for="三个维度的汇总——只报第一行的话，这会是一次「100% 成功」的编辑。",
            minutes=5,
        ),
        Recall(
            question="什么场景适合参数化记忆？",
            answer=(
                "**稳定的领域知识**（医学术语、法条、产品规格）、"
                "**风格与语气适配**（本来就不是「一条事实」，检索也表达不了）、"
                "**高频且不变的事实**（省去反复检索）。\n"
                "实用划分：**变化频率高 / 需要审计 / 需要删除 → 上下文类记忆；"
                "稳定 / 通用 / 不需溯源 → 可考虑参数化。**"
            ),
        ),
        Explain(
            task="向同事解释：为什么我们不把用户偏好微调进模型里。",
            checklist=[
                "说出三个失败模式，各举一个具体例子",
                "对比可审计/可回滚/可解释/可删除四样能力",
                "承认参数化确实适合某些场景，并说出是哪些",
                "给出判断依据：变化频率、审计需求、删除需求",
            ],
        ),
        Apply(
            level="🟢",
            task=(
                "在玩具模型里加一条新规则（如「首都的市长是 X」），"
                "看编辑「法国的首都」之后这条推论会怎样。"
            ),
            starting_point="docs/chapter9/code/02_why_editing_is_hard.py 的 build()",
            success_looks_like="涟漪效应扩散到了更多层推论",
        ),
        Apply(
            level="🟡",
            task=(
                "（需要 GPU）跑 03_minimal_edit.py，然后换一层（AGENT_MEM_EDIT_LAYER）"
                "或换个模型重跑，看三个维度怎么变。"
            ),
            starting_point="docs/chapter9/code/requirements.txt，建议单独建环境；0.5B 约需 4GB 显存",
            success_looks_like=(
                "换层/换模型后三个维度明显不同——"
                "**说明模型编辑对超参数相当敏感，而这些超参数没有普适取法**"
            ),
        ),
        Apply(
            level="🔴",
            task=(
                "设计一个混合记忆方案：稳定领域知识用 LoRA，用户个人信息用第 2~8 章的方案。"
                "设计路由策略决定新信息该进哪边，并说明这个路由本身的判断成本。"
            ),
            starting_point="从第 8 章的 importance 打分出发考虑路由信号",
            success_looks_like=(
                "一张「信息类型 × 存储位置 × 判断依据」的决策表——这才是参数化记忆在实践中真正的用法"
            ),
        ),
        Cliff(
            text=(
                "全书三种表征到这里讲完了。剩下的问题是："
                "**你怎么知道自己做的这套东西是好的、安全的？**"
            ),
            next_chapter="第 10 章：评测、安全与工程落地。（按 1→3→10 路径读的话，你已经读过了。）",
        ),
    ],
    cards=[
        Card(
            front="模型编辑的三个失败模式是什么？",
            back=(
                "**涟漪效应**（改了事实但推论没跟着改，因为没人知道哪些结论依赖它）、"
                "**多编辑冲突**（累积且不可逐条撤销，多次编辑后状态无法审计）、"
                "**跨域泛化失败**（换个问法、换种语言就拿到旧答案）。"
            ),
            chapter=9,
        ),
        Card(
            front="看到「编辑成功率 99%」，第一个该问什么？",
            back=(
                "**在哪个维度上？** 三个维度：Reliability（目标问题）、"
                "Generalization（换问法）、Specificity（不该改的没被改）。"
                "Reliability 几乎总是很高，难点在后两个。"
            ),
            chapter=9,
        ),
        Card(
            front="参数化记忆失去了哪四样能力？",
            back=(
                "可审计、可回滚、可解释、**可删除**。"
                "最后一样最要命——GDPR 删除权在权重里目前没有可靠实现方式，"
                "machine unlearning 远未成熟。"
            ),
            chapter=9,
        ),
        Card(
            front="怎么判断一条信息该进权重还是进上下文类记忆？",
            back=(
                "**变化频率高 / 需要审计 / 需要删除 → 上下文类记忆；"
                "稳定 / 通用 / 不需溯源 → 可考虑参数化。**"
                "多数产品的记忆需求落在前者。"
            ),
            chapter=9,
        ),
    ],
)
