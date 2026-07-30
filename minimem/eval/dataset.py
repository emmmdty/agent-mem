"""MiniBench：一个刻意做得很小的中文记忆评测集。

**先说清楚它不是什么**：它不是 LongMemEval，不是 LoCoMo，**它不能用来证明任何
方案是 SOTA**。它只有 30 条记忆、24 个查询，一个人在两小时内编的，覆盖面窄，
统计上没有任何显著性可言。

那它有什么用？三件事：

1. **让实验秒级可复现**。全书每一章的新实现都能立刻跑一遍，看看有没有把
   前面章节能答对的题答错——这是回归测试，不是排名。
2. **让不同能力可分离**。24 个查询按能力分成 8 类，你能看到「加了图记忆之后，
   多跳类涨了、时序类没动」这种有解释力的结果，而不是一个笼统的总分。
3. **让评测本身成为教学内容**。第 10 章会用它演示：为什么样本量小的基准
   容易得出反转的结论，为什么 LLM 评判会放行「主题相邻但答错」的回答。

真实项目请用你自己业务的数据构造评测集。**任何公开基准都无法替代它**，
这一点第 10 章会详细展开。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

__all__ = [
    "MemoryRecord",
    "Query",
    "MINI_MEMORIES",
    "MINI_QUERIES",
    "CAPABILITIES",
    "load_mini_bench",
    "as_memory_items",
]

# 所有时间以「基准日」为锚点，方便测试时时间旅行。
BASE_DATE = datetime(2026, 7, 30, tzinfo=timezone.utc)

CAPABILITIES: dict[str, str] = {
    "直接检索": "答案就在某一条记忆的字面里",
    "同义改写": "查询用了和记忆完全不同的说法",
    "专有名词": "需要精确匹配编号、代号、型号",
    "多跳": "需要串联两条以上记忆才能回答",
    "时序": "答案取决于「什么时候」",
    "知识更新": "存在被后来信息推翻的旧事实",
    "归纳": "答案不在任何单条记忆里，要从多条中抽象",
    "拒答": "记忆里根本没有相关信息，应当承认不知道",
}


@dataclass
class MemoryRecord:
    """一条待写入的记忆。

    Attributes:
        rid: 稳定标识，供查询的 ``gold`` 字段引用。
        text: 内容。
        days_ago: 相对基准日多少天前发生。用于第 5 章的时序实验。
    """

    rid: str
    text: str
    days_ago: int = 0

    def timestamp(self, base: datetime | None = None) -> datetime:
        return (base or BASE_DATE) - timedelta(days=self.days_ago)


@dataclass
class Query:
    """一个查询。

    Attributes:
        qid: 标识。
        text: 查询文本。
        capability: 考察哪种能力，取值见 ``CAPABILITIES``。
        gold: 应当被召回的记忆 rid 集合。用于算召回率。
        answer_keywords: 正确回答中应当出现的关键词（任一命中即算对）。
            空列表表示这题应当拒答。
        note: 出题意图，供教学使用。
    """

    qid: str
    text: str
    capability: str
    gold: list[str] = field(default_factory=list)
    answer_keywords: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def should_abstain(self) -> bool:
        return not self.answer_keywords


# ----------------------------------------------------------------------
# 记忆：一个金融风控工程师小明，跨越约半年的零散对话
# ----------------------------------------------------------------------

MINI_MEMORIES: list[MemoryRecord] = [
    # --- 身份与工作 ---
    MemoryRecord("m01", "你好，我叫小明。", 180),
    MemoryRecord("m02", "我在星辰银行工作，是一名信贷风控工程师。", 180),
    MemoryRecord("m03", "我们团队主要负责对公业务的授信审批模型。", 175),
    MemoryRecord("m04", "我下周就要从星辰银行离职了，交接已经开始。", 20),
    MemoryRecord("m05", "新工作定了，我下个月入职长风科技，做风控算法。", 15),
    # --- 项目细节（含专有名词） ---
    MemoryRecord("m06", "我负责的项目代号是 XR-2049，是个反欺诈模型。", 120),
    MemoryRecord("m07", "XR-2049 上线后，人工复核量下降了大概四成。", 90),
    MemoryRecord("m08", "另一个项目 XR-2050 是做小微企业评分的，不归我管。", 88),
    MemoryRecord("m09", "我们用的特征平台叫 FeatBase，是自研的。", 100),
    MemoryRecord("m10", "FeatBase 的实时特征延迟大概在 80 毫秒左右。", 95),
    # --- 生活 ---
    MemoryRecord("m11", "我对花生过敏，点餐要特别注意。", 170),
    MemoryRecord("m12", "我喜欢周末去爬山，最近去了趟灵山。", 60),
    MemoryRecord("m13", "我们家的猫叫毛毛，今年三岁了。", 150),
    MemoryRecord("m14", "毛毛最近有点掉毛，带去医院看了一次。", 30),
    MemoryRecord("m15", "我住在城西的枫林小区。", 160),
    MemoryRecord("m16", "上周搬家了，现在住城东的沿江花园。", 10),
    # --- 偏好（用于归纳） ---
    MemoryRecord("m17", "开会尽量安排在下午，我上午效率低。", 140),
    MemoryRecord("m18", "文档麻烦发 Markdown，我不太爱看 Word。", 130),
    MemoryRecord("m19", "邮件我基本不看，有事直接发消息。", 125),
    MemoryRecord("m20", "周报我习惯周五下午写完。", 110),
    # --- 学习与阅读 ---
    MemoryRecord("m21", "最近在读《证券分析》，进度大概三分之一。", 45),
    MemoryRecord("m22", "我想系统学一下因果推断，在找教材。", 40),
    MemoryRecord("m23", "报了个在线课程，讲双重差分和断点回归的。", 35),
    # --- 人际 ---
    MemoryRecord("m24", "我的直属领导是李姐，她带我三年了。", 165),
    MemoryRecord("m25", "小赵是我们组新来的实习生，负责数据标注。", 70),
    MemoryRecord("m26", "李姐下个季度要调去总行了。", 25),
    # --- 干扰项：与查询主题相近但不是答案 ---
    MemoryRecord("m27", "今天天气不错，中午在楼下吃的面。", 5),
    MemoryRecord("m28", "开了一下午的会，讨论明年的预算。", 3),
    MemoryRecord("m29", "帮同事看了一段 SQL，有个 join 写错了。", 2),
    MemoryRecord("m30", "周末想去看场电影，还没定片子。", 1),
]

# ----------------------------------------------------------------------
# 查询：按能力分类
# ----------------------------------------------------------------------

MINI_QUERIES: list[Query] = [
    # --- 直接检索 ---
    Query("q01", "我叫什么名字？", "直接检索", ["m01"], ["小明"]),
    Query("q02", "我对什么过敏？", "直接检索", ["m11"], ["花生"]),
    Query("q03", "我家的猫叫什么？", "直接检索", ["m13"], ["毛毛"]),
    # --- 同义改写：查询与记忆没有字面重叠 ---
    Query(
        "q04",
        "他是干哪一行的？",
        "同义改写",
        ["m02"],
        ["风控", "信贷"],
        note="「干哪一行」与「信贷风控工程师」零字面重叠，第 1 章的 BufferMemory 必错",
    ),
    Query("q05", "我平时业余时间干什么？", "同义改写", ["m12"], ["爬山"]),
    Query("q06", "我最近在看什么书？", "同义改写", ["m21"], ["证券分析"]),
    # --- 专有名词：稠密向量的弱项 ---
    Query(
        "q07",
        "XR-2049 是什么项目？",
        "专有名词",
        ["m06"],
        ["反欺诈"],
        note="XR-2049 和 XR-2050 在向量空间里几乎重合，需要 BM25 精确匹配",
    ),
    Query("q08", "FeatBase 的延迟是多少？", "专有名词", ["m10"], ["80", "毫秒"]),
    Query("q09", "XR-2050 归谁管？", "专有名词", ["m08"], ["不归我管", "不是我"]),
    # --- 多跳：需要串联 ---
    Query(
        "q10",
        "我负责的那个项目带来了什么效果？",
        "多跳",
        ["m06", "m07"],
        ["四成", "复核"],
        note="要先从「我负责的项目」定位到 XR-2049，再找它的效果",
    ),
    Query("q11", "我的领导要去哪？", "多跳", ["m24", "m26"], ["总行"]),
    # --- 时序 ---
    Query(
        "q12",
        "我现在住在哪？",
        "时序",
        ["m15", "m16"],
        ["沿江花园", "城东"],
        note="两条住址记忆都会被召回，必须靠时间判断哪条有效",
    ),
    Query("q13", "我三个月前住哪？", "时序", ["m15"], ["枫林", "城西"]),
    # --- 知识更新 ---
    Query(
        "q14",
        "我在哪家公司上班？",
        "知识更新",
        ["m02", "m04", "m05"],
        ["长风", "星辰", "离职"],
        note="答案取决于「今天是哪天」：已离职、尚未入职，严格说两者都不完全对。"
        "第 5 章会讨论这类问题的正确答法",
    ),
    Query("q15", "我还养猫吗？", "知识更新", ["m13", "m14"], ["毛毛", "养"]),
    # --- 归纳：答案不在任何单条记忆里 ---
    Query(
        "q16",
        "跟我沟通有什么要注意的？",
        "归纳",
        ["m17", "m18", "m19"],
        ["下午", "Markdown", "消息", "邮件"],
        note="三条偏好散落在不同时间，需要归纳成一条「沟通偏好」",
    ),
    Query("q17", "我的工作习惯是怎样的？", "归纳", ["m17", "m20"], ["下午", "周五"]),
    # --- 拒答：记忆里没有 ---
    Query(
        "q18",
        "我的血型是什么？",
        "拒答",
        [],
        [],
        note="记忆里完全没有。会编答案的系统在这里暴露；只召回不判断的系统也会暴露",
    ),
    Query("q19", "我孩子上几年级了？", "拒答", [], [], note="预设了不存在的前提"),
    Query("q20", "我车牌号是多少？", "拒答", [], []),
    # --- 混合难度 ---
    Query("q21", "实习生负责什么？", "直接检索", ["m25"], ["数据标注", "标注"]),
    Query("q22", "我在学什么方法？", "同义改写", ["m22", "m23"], ["因果推断", "双重差分"]),
    Query("q23", "我们组做什么业务？", "同义改写", ["m03"], ["对公", "授信"]),
    Query(
        "q24",
        "我上个月爬的哪座山？",
        "时序",
        ["m12"],
        ["灵山"],
        note="时间限定其实不匹配（那是两个月前），考察系统会不会强行给答案",
    ),
]


def load_mini_bench() -> tuple[list[MemoryRecord], list[Query]]:
    """返回 (记忆, 查询) 两份列表的副本。"""
    return list(MINI_MEMORIES), list(MINI_QUERIES)


def as_memory_items(records: list[MemoryRecord] | None = None, *, base: datetime | None = None):
    """把 ``MemoryRecord`` 转成可直接写入 ``MemoryStore`` 的 ``MemoryItem``。

    ``rid`` 会存进 ``metadata["rid"]``——评测时靠它判断召回是否命中 gold。
    """
    from minimem.base import MemoryItem

    records = records if records is not None else MINI_MEMORIES
    return [
        MemoryItem(
            content=r.text,
            created_at=r.timestamp(base),
            kind="message",
            metadata={"rid": r.rid, "days_ago": r.days_ago},
        )
        for r in records
    ]
