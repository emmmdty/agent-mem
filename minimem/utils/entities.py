"""实体抽取：把一句话里的「东西」找出来。

第 4 章建图需要节点，节点来自实体。这里提供两条路：

- ``RuleExtractor``：正则 + 模式，零依赖、零成本、确定性。
- ``LLMExtractor``：用 LLM 抽三元组，召回好得多，但每条记忆一次调用。

本模块刻意先给规则版，并且**故意让你看到它的召回有多差**——
第 4 章的实验会量化这个差距，而那正是第 6 章引入 LLM 抽取的实证动机。
如果一上来就用 LLM，你就不会知道自己为那点召回率付了多少钱。

关于中文实体抽取的现实：没有分词器和 NER 模型的情况下，规则能抓住的主要是
**带标志的实体**——英文标识符、书名号、带后缀的机构名。像「毛毛」（猫名）、
「灵山」（地名）这类光秃秃的专有名词，规则基本无能为力。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

__all__ = ["Entity", "Triple", "EntityExtractor", "RuleExtractor", "LLMExtractor"]


@dataclass(frozen=True)
class Entity:
    """一个实体。

    Attributes:
        name: 规范化后的名字，作为图节点的 id。
        kind: 类型标签（``org`` / ``id`` / ``person`` / ``work`` / ``term``）。
            只用于展示与过滤，不承载语义。
    """

    name: str
    kind: str = "term"


@dataclass(frozen=True)
class Triple:
    """一条 (主语, 谓语, 宾语) 三元组。规则版抽不出谓语时用 ``"相关"``。"""

    subject: str
    predicate: str
    object: str


class EntityExtractor(ABC):
    name: str = "extractor"

    @abstractmethod
    def extract(self, text: str) -> list[Entity]: ...

    def extract_triples(self, text: str) -> list[Triple]:
        """默认实现：同一句话里出现的实体两两相关。

        这是一个**很弱**的近似——它把「共现」当成「有关系」，
        既不知道关系类型，也不知道方向。第 6 章用 LLM 抽取时会好很多。
        """
        ents = self.extract(text)
        return [Triple(a.name, "相关", b.name) for i, a in enumerate(ents) for b in ents[i + 1 :]]


# ----------------------------------------------------------------------
# 规则抽取
# ----------------------------------------------------------------------

#: 机构、项目、产品一类带后缀的名字
# 「系统」「部门」这类后缀太泛（「我想系统学一下」也会中招），刻意不放进来。
_SUFFIXES = "银行|公司|集团|科技|大学|学院|研究院|小区|花园|平台|模型|团队|项目"

#: 常见姓氏，用于约束称呼类实体的左边界。
_SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦许何吕施张孔曹严华金魏陶姜谢邹苏潘葛范彭鲁韦马苗方俞任袁柳史唐费岑薛雷贺倪汤殷罗毕郝安常乐于时傅齐康伍余元顾孟平黄萧尹"

_PATTERNS: list[tuple[str, str]] = [
    # 英文/数字标识符：XR-2049、FeatBase、GPT-4o
    (r"\b([A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*)\b", "id"),
    # 书名号
    (r"《([^》]{1,20})》", "work"),
    # 带后缀的机构名：星辰银行、长风科技、枫林小区
    # 前缀限 2~4 字并在事后剥离噪声字，否则会把「我在星辰银行」整个吃进去——
    # 那样同一个实体在不同句子里会变成不同的节点，图就连不起来了。
    (rf"([一-鿿]{{2,4}}(?:{_SUFFIXES}))", "org"),
    # 「小X」「老X」「X姐/哥/总」这类称呼
    (r"((?:小|老)[一-鿿]{1,2})(?:[，。、！？]|$|说|是|的)", "person"),
    # 称呼类必须以姓氏打头，否则「调去总行」会被抽成「调去总」
    (rf"([{_SURNAMES}](?:姐|哥|总|工|老师))", "person"),
]

#: 抽出来但没有信息量的词，直接丢掉。
#: 这份表是手工维护的——**这本身就是规则方法的成本**，换个领域要重写一遍。
_STOP_ENTITIES = {
    "的",
    "了",
    "是",
    "我",
    "你",
    "他",
    "在",
    "有",
    "和",
    "这个",
    "那个",
    "什么",
    "怎么",
    "为什么",
    "a",
    "an",
    "the",
    "is",
    "are",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
}

#: 会被误粘到实体名前面的虚词与动词。逐字从左剥离。
#: 这份表和下面的领域词表一样，是**规则方法的隐藏成本**——换个领域就要重写。
_PREFIX_NOISE = set("我你他她们在从于对把和与的了是就要还也才那这个月周天日年后前入职离开来回上下")

#: 领域关键词表：规则抓不到的裸专有名词，靠这张表兜底。
#: 真实项目里这张表通常来自业务词典或已有的实体库。
_DOMAIN_TERMS = {
    "过敏",
    "花生",
    "风控",
    "信贷",
    "授信",
    "反欺诈",
    "评分",
    "特征",
    "爬山",
    "离职",
    "入职",
    "搬家",
    "实习生",
    "标注",
    "周报",
    "会议",
    "邮件",
    "文档",
    "因果推断",
    "双重差分",
    "断点回归",
}


class RuleExtractor(EntityExtractor):
    """基于正则与词表的实体抽取。

    Args:
        min_len: 中文实体的最短长度。设为 1 会引入大量噪声。
        extra_terms: 追加的领域词表。**换一个业务领域就要换一份**——
            这份人工成本是规则方法的隐藏代价，第 4 章会把它和 LLM 抽取的
            金钱成本放在一起比较。
    """

    name = "rule"

    def __init__(self, *, min_len: int = 2, extra_terms: set[str] | None = None) -> None:
        self.min_len = min_len
        self.terms = _DOMAIN_TERMS | (extra_terms or set())

    def extract(self, text: str) -> list[Entity]:
        found: dict[str, str] = {}

        for pattern, kind in _PATTERNS:
            for m in re.finditer(pattern, text):
                name = self._strip_noise(m.group(1).strip(), kind)
                if self._keep(name):
                    found.setdefault(name.lower() if name.isascii() else name, kind)

        for term in self.terms:
            if term in text:
                found.setdefault(term, "term")

        return [Entity(name=n, kind=k) for n, k in found.items()]

    @staticmethod
    def _strip_noise(name: str, kind: str) -> str:
        """从左侧逐字剥掉粘上来的虚词。

        「我在星辰银行」→「星辰银行」，「个月入职长风科技」→「长风科技」。
        只对机构类做，因为其他类型的模式已经有明确边界。
        """
        if kind not in ("org", "person"):
            return name
        # 「西的枫林小区」→「枫林小区」：定语和中心语之间的「的」是个好切点
        if "的" in name:
            tail = name.rsplit("的", 1)[1]
            if len(tail) >= 2:
                name = tail
        while len(name) > 2 and name[0] in _PREFIX_NOISE:
            name = name[1:]
        return name

    def _keep(self, name: str) -> bool:
        if not name or name.lower() in _STOP_ENTITIES:
            return False
        if name.isascii():
            # 单个字母或纯数字没有信息量
            return len(name) >= 2 and not name.isdigit()
        return len(name) >= self.min_len


# ----------------------------------------------------------------------
# LLM 抽取
# ----------------------------------------------------------------------

_TRIPLE_PROMPT = """从下面这句话里抽取知识三元组。

要求：
1. 只抽句子里明确说了的，不要推断
2. 主语和宾语用具体的名字，不要用代词
3. 输出 JSON：{{"triples": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}

句子：{text}"""


class LLMExtractor(EntityExtractor):
    """用 LLM 抽三元组。

    召回和关系类型都比规则版好得多，代价是**每条记忆一次调用**。
    第 4 章会把这笔账算清楚：抽取 30 条记忆要多少 token、多少钱，
    换来召回率提升多少。

    Args:
        llm: 任意 ``LLMClient``。离线时传 ``ScriptedLLM`` 并注册
            ``"抽取知识三元组"`` 这个标记。
        fallback: LLM 失败（超时、返回非法 JSON）时的兜底抽取器。
            **不设兜底会让一次网络抖动变成一条记忆的永久丢失。**
    """

    name = "llm"

    def __init__(self, llm, *, fallback: EntityExtractor | None = None) -> None:
        self.llm = llm
        self.fallback = fallback if fallback is not None else RuleExtractor()

    def extract_triples(self, text: str) -> list[Triple]:
        result = self.llm.complete(_TRIPLE_PROMPT.format(text=text), op="extract_triples")
        data = result.json(default=None)

        if not isinstance(data, dict) or "triples" not in data:
            # 解析失败就退回规则版，而不是丢掉这条记忆
            return self.fallback.extract_triples(text)

        triples: list[Triple] = []
        for item in data.get("triples", []):
            if not isinstance(item, dict):
                continue
            s, p, o = item.get("subject"), item.get("predicate"), item.get("object")
            if s and o:
                triples.append(Triple(str(s).strip(), str(p or "相关").strip(), str(o).strip()))
        return triples or self.fallback.extract_triples(text)

    def extract(self, text: str) -> list[Entity]:
        """从三元组反推实体；抽不出三元组时退回规则抽取。

        最后那个退路不是可有可无的。**查询**通常只含一个实体
        （「我负责的那个项目带来了什么效果？」里只有「项目」），
        而一个实体构不成三元组——如果这里直接返回空，
        图检索就会连种子都没有，静默退化成纯向量检索。

        这个 bug 真实发生过：第 4 章实验三里 LLM 抽取版的多跳召回率
        莫名比规则版低，查了半天才发现问题不在建图，在**查询侧抽不出种子**。
        """
        triples = self.extract_triples(text)
        names = {t.subject for t in triples} | {t.object for t in triples}
        if not names:
            return self.fallback.extract(text)
        return [Entity(name=n) for n in sorted(names)]
