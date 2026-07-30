"""第 6 章的交互式课程。"""

from __future__ import annotations

from minimem.learn.models import Apply, Card, Cliff, Explain, Lesson, Motto, Predict, Recall, Run

LESSON = Lesson(
    chapter=6,
    title="agentic memory 与自组织",
    minutes=35,
    difficulty=2,
    prerequisites=[3, 5],
    steps=[
        Motto("让记忆自己组织自己，第一次要为写入付真金白银。值不值，看读写比。"),
        Predict(
            question=(
                "extract-update 会把矛盾记忆合并成一条最新的"
                "（「枫林小区」被「沿江花园」就地覆盖）。"
                "那么「我三个月前住哪？」这个问题，现在还答得出来吗？"
            ),
            options={
                "A": "能 —— 记忆更干净了，检索更准",
                "B": "不能 —— UPDATE 是就地覆盖，旧值没了",
            },
            answer="B",
            reveal=(
                "UPDATE 是**就地覆盖**，旧值不再存在。"
                "而第 5 章的做法（失效而非删除）**是能答的**——"
                "它给旧事实盖 valid_to，记录留着。"
            ),
            trap=(
                "两种设计解决的是不同问题：\n"
                "  extract-update → 让**当前状态**唯一干净（代价：历史丢失）\n"
                "  bi-temporal    → 让**历史**可查（代价：检索要处理多版本）\n"
                "真实系统往往两个都要，代价是两套机制都要维护。"
            ),
        ),
        Run(
            script="docs/chapter6/code/01_extract_update.py",
            why="看 ADD/UPDATE/NOOP 决策怎么把 6 条输入压成 3 条记忆。",
            look_for="NOOP 那一行——重复陈述被识别出来，根本没写进库。",
            minutes=1,
        ),
        Recall(
            question="extract-update 决策失败时，默认动作该选哪个？为什么？",
            answer=(
                "**ADD**。因为它是唯一「错了也能补救」的选项——多存一条只是浪费空间。\n"
                "默认 UPDATE 会让一次 JSON 解析失败**覆盖掉一条正确的记忆**；"
                "默认 NOOP 会让用户说过的话凭空消失且无报错。\n"
                "一般原则：**写入路径上，失败的默认行为要选可逆的那个。**"
            ),
        ),
        Predict(
            question=(
                "自组织让写入侧多花了 11159 token（30 条记忆 × 2 次 LLM 调用），"
                "检索侧每次省下约 5 token。这批记忆要被检索多少次才回本？"
            ),
            options={"A": "约 60 次", "B": "约 500 次", "C": "约 2300 次"},
            answer="C",
            reveal=(
                "11159 ÷ 5 ≈ **2289 次**。\n"
                "30 条记忆被检索两千多次是什么场景？高频个人助理可能达到，"
                "用完即走的客服会话绝对达不到。"
            ),
            trap=(
                "「判断依据是读写比」这句话本书说了五次，这里是它的**具体公式**：\n"
                "  打平所需检索次数 = 写入侧额外 token ÷ 每次检索省下的 token\n"
                "从现在起，任何「让记忆更聪明」的提议，你都可以要求先算这个数。"
            ),
        ),
        Predict(
            question="把 note 加工的失败率拉到 50%（一半记忆退回原文），召回率会掉多少？",
            options={"A": "掉一半左右", "B": "掉几个百分点", "C": "一点不掉"},
            answer="C",
            reveal=(
                "88.2% → 88.2% → 88.2%，**完全不变**。\n"
                "这说明：在这份语料上，note 加工带来的检索收益**微乎其微**。"
            ),
            trap=(
                "把它和打平点放在一起看：写入多花 11159 token、检索每次省 5 token、"
                "加工失败一半效果不变——三个数字指向同一个结论："
                "**这一步在这个场景下不划算**。\n"
                "这不代表 A-MEM 类方法没价值，而是**你必须在自己的语料上验证**。"
                "验证方法就是这个实验：**把组件弄坏，看指标掉不掉**"
                "（第 3 章 FakeEmbedder 手法的第三次应用）。"
            ),
        ),
        Run(
            script="docs/chapter6/code/02_cost_and_failure.py",
            why="算清账单，并看 LLM 出错时系统怎么办。",
            look_for="打平点那个数字，以及失败率拉到 50% 时召回率的变化。",
            minutes=1,
        ),
        Recall(
            question="为什么「抽取错误」比「回答错误」严重得多？",
            answer=(
                "一次回答错误只影响一次对话；一条错误记忆影响之后的**每一次**对话，"
                "而且它看起来和正确记忆一模一样。\n"
                "最危险的是「抽错了但格式正确」——「我不对花生过敏」被总结成"
                "「过敏原：花生」，JSON 完全合法，系统欣然接受。"
            ),
        ),
        Explain(
            task="向同事解释：我们该不该给产品加上 A-MEM 式的自组织记忆。",
            checklist=[
                "先算打平点，给出具体数字而不是「感觉有用」",
                "说明 extract-update 与双时间轴的取舍（当前状态 vs 历史可查）",
                "指出失败模式：退回原文会悄悄退化，必须有 degraded 监控",
                "强调抽取错误会固化，比回答错误严重",
            ],
        ),
        Apply(
            level="🟢",
            task="把 enable_update 关掉，重跑实验二，看写入成本降多少、记忆条数变多少。",
            starting_point="docs/chapter6/code/02_cost_and_failure.py",
            success_looks_like="LLM 调用减半，但矛盾记忆重新并存（回到第 4 章的状态）",
        ),
        Apply(
            level="🟡",
            task=(
                "实现抽取校验：note 的 summary 必须能在原文里找到依据"
                "（关键词覆盖率 ≥ 50%），否则标记为低置信。"
                "构造一个「LLM 抽反了」的用例验证它能拦住。"
            ),
            starting_point="minimem/agentic.py 的 make_note",
            success_looks_like="「我不对花生过敏 → 过敏原：花生」这类反转被标记出来",
            verify="python -m pytest tests/test_agentic.py -q",
        ),
        Apply(
            level="🔴",
            task=(
                "把 extract-update 和第 5 章的双时间轴合起来：UPDATE 时不覆盖，"
                "而是给旧记忆盖 valid_to，同时维护一份「当前视图」。"
                "验证它既能答「我现在住哪」也能答「我三个月前住哪」。"
            ),
            starting_point="minimem/agentic.py 与 minimem/temporal.py",
            success_looks_like="两个问题都答对，代价是存储与写入复杂度都上升",
        ),
        Cliff(
            text=(
                "自组织解决了「记忆怎么变干净」，但没解决一个更基础的问题："
                "**记忆越来越多，而上下文就那么大**。"
            ),
            next_chapter="第 7 章：分层调度，以及那个 OS 隐喻到底兑现了多少。",
        ),
    ],
    cards=[
        Card(
            front="extract-update 和双时间轴分别解决什么问题？",
            back=(
                "extract-update 让**当前状态**唯一干净（代价：历史丢失）；"
                "双时间轴让**历史**可查（代价：检索要处理多版本）。"
                "真实系统往往两个都要：前者维护当前视图，后者保留完整历史。"
            ),
            chapter=6,
        ),
        Card(
            front="写入路径上，操作失败时的默认动作该怎么选？",
            back=(
                "选**可逆**的那个。extract-update 里就是 ADD——"
                "多存一条只是浪费空间，可以事后清理。"
                "默认 UPDATE 会让一次解析失败覆盖正确记忆；"
                "默认 NOOP 会让消息凭空消失且无报错。"
            ),
            chapter=6,
        ),
        Card(
            front="怎么算「让记忆更聪明」的方案值不值？",
            back=(
                "**打平所需检索次数 = 写入侧额外 token ÷ 每次检索省下的 token。**"
                "本章配置算出来约 2289 次——这批记忆要被检索两千多次才回本。"
                "「值得」是可计算的问题，不是态度问题。"
            ),
            chapter=6,
        ),
        Card(
            front="怎么验证某个记忆组件到底有没有用？",
            back=(
                "**把它弄坏，看指标掉不掉。** 本章把 note 加工的失败率拉到 50%，"
                "召回率一点没掉——说明这一步在该语料上收益微乎其微。"
                "这是第 3 章 FakeEmbedder 手法的推广。"
            ),
            chapter=6,
            tag="方法论",
        ),
        Card(
            front="为什么「抽取错误」比「回答错误」严重得多？",
            back=(
                "一次回答错误只影响一次对话；一条错误记忆影响之后的**每一次**对话，"
                "且看起来和正确记忆一模一样。最危险的是「抽错但格式正确」——"
                "JSON 合法，系统欣然接受，错误就此固化。"
            ),
            chapter=6,
        ),
    ],
)
