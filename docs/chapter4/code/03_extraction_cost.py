# offline-ok
"""第 4 章实验三：建图要花多少钱。

    python docs/chapter4/code/03_extraction_cost.py

前两个实验只谈了效果。这个实验算账。

图记忆的成本分三块：

1. **抽取**：每条记忆抽一次实体。规则版免费但召回差，LLM 版召回好但每条一次调用。
2. **检索**：PPR 每次要在整张图上做幂迭代。
3. **人工**：规则版需要维护词表和模式——**这块最容易被忽略，因为它不出现在账单上**。

本脚本用 ScriptedLLM 模拟 LLM 抽取（离线可跑），token 数如实记录，
金额按 .env 里的占位价格估算。**它模拟的是格式和用量，不是 LLM 的判断力**——
真实模型的抽取质量会更好也更不稳定。
"""

from __future__ import annotations

import json
import time
import warnings

from minimem.eval import MINI_MEMORIES, MINI_QUERIES, as_memory_items, recall_at_k
from minimem.graph import GraphMemory
from minimem.utils.embedding import get_embedder
from minimem.utils.entities import LLMExtractor, RuleExtractor
from minimem.utils.llm import ScriptedLLM
from minimem.utils.metering import Meter
from minimem.utils.tokens import TokenCost

USER = "u_bench"
K = 5

# 手工标注的「理想抽取结果」，用来模拟一个抽得比规则版好的 LLM。
# 重点是补上规则版抓不到的裸专有名词：毛毛、灵山、沿江花园……
GOLD_TRIPLES: dict[str, list[tuple[str, str, str]]] = {
    "m01": [("用户", "叫", "小明")],
    "m02": [("小明", "就职于", "星辰银行"), ("小明", "职业", "信贷风控工程师")],
    "m03": [("团队", "负责", "对公授信审批模型")],
    "m04": [("小明", "即将离职", "星辰银行")],
    "m05": [("小明", "将入职", "长风科技"), ("小明", "新职位", "风控算法")],
    # 注意 ("XR-2049", "是", "项目") 这条：它把泛称「项目」也放进了图。
    # 少了它，查询「我负责的那个项目」就找不到锚点——见本脚本末尾关于
    # 「写入侧与查询侧的实体空间必须对齐」的讨论。
    "m06": [
        ("小明", "负责", "XR-2049"),
        ("XR-2049", "类型", "反欺诈模型"),
        ("XR-2049", "是", "项目"),
    ],
    "m07": [("XR-2049", "效果", "人工复核量下降四成")],
    "m08": [
        ("XR-2050", "类型", "小微企业评分"),
        ("XR-2050", "负责人", "非小明"),
        ("XR-2050", "是", "项目"),
    ],
    "m09": [("团队", "使用", "FeatBase"), ("FeatBase", "性质", "自研")],
    "m10": [("FeatBase", "实时特征延迟", "80 毫秒")],
    "m11": [("小明", "过敏原", "花生")],
    "m12": [("小明", "爱好", "爬山"), ("小明", "去过", "灵山")],
    "m13": [("小明", "养", "毛毛"), ("毛毛", "物种", "猫"), ("毛毛", "年龄", "三岁")],
    "m14": [("毛毛", "症状", "掉毛"), ("毛毛", "去过", "医院")],
    "m15": [("小明", "曾住", "枫林小区")],
    "m16": [("小明", "现住", "沿江花园")],
    "m24": [("小明", "上级", "李姐")],
    "m25": [("小赵", "角色", "实习生"), ("小赵", "负责", "数据标注")],
    "m26": [("李姐", "将调往", "总行")],
}


def make_scripted_llm(meter: Meter) -> ScriptedLLM:
    """注册一个按记忆内容返回预设三元组的模拟器。"""
    llm = ScriptedLLM(meter=meter, store_label="graph")

    def handler(prompt: str) -> str:
        for rec in MINI_MEMORIES:
            if rec.text in prompt:
                triples = GOLD_TRIPLES.get(rec.rid, [])
                return json.dumps(
                    {
                        "triples": [
                            {"subject": s, "predicate": p, "object": o} for s, p, o in triples
                        ]
                    },
                    ensure_ascii=False,
                )
        return '{"triples": []}'

    llm.register("抽取知识三元组", handler)
    return llm


def evaluate(store) -> tuple[float, float]:
    per: dict[str, list[float]] = {}
    for q in MINI_QUERIES:
        rids = [h.item.metadata.get("rid", "") for h in store.search(q.text, user_id=USER, k=K)]
        per.setdefault(q.capability, []).append(recall_at_k(rids, q.gold, K))
    allv = [v for vs in per.values() for v in vs]
    return sum(allv) / len(allv), sum(per["多跳"]) / len(per["多跳"])


def main() -> None:
    print("\n第 4 章实验三：建图的代价")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedder = get_embedder()

    rows = []

    # ---- 规则抽取 ----
    meter = Meter()
    store = GraphMemory(extractor=RuleExtractor(), embedder=embedder, meter=meter)
    t0 = time.perf_counter()
    store.add_many(as_memory_items(), user_id=USER)
    write_s = time.perf_counter() - t0
    total, multi = evaluate(store)
    rows.append(("规则抽取", store.stats(user_id=USER), write_s, meter, total, multi))

    # ---- LLM 抽取（模拟） ----
    meter2 = Meter()
    llm = make_scripted_llm(meter2)
    store2 = GraphMemory(extractor=LLMExtractor(llm), embedder=embedder, meter=meter2)
    t0 = time.perf_counter()
    store2.add_many(as_memory_items(), user_id=USER)
    write2_s = time.perf_counter() - t0
    total2, multi2 = evaluate(store2)
    rows.append(("LLM 抽取（模拟）", store2.stats(user_id=USER), write2_s, meter2, total2, multi2))

    print(f"\n  {'抽取方式':<20}{'实体节点':>10}{'边':>8}{'没抽到实体':>12}")
    print("  " + "-" * 52)
    for label, stats, *_ in rows:
        print(
            f"  {label:<20}{stats['实体节点']:>10}{stats['边']:>8}{stats['没抽到实体的记忆']:>12}"
        )

    print(
        f"\n  {'抽取方式':<20}{'总召回@5':>10}{'多跳':>8}{'写入耗时(s)':>14}{'LLM 调用':>10}{'token':>9}"
    )
    print("  " + "-" * 72)
    cost = TokenCost()
    for label, _, write_s, meter, total, multi in rows:
        s = meter.summary()
        print(
            f"  {label:<20}{total:>9.1%}{multi:>8.0%}{write_s:>14.2f}"
            f"{s['llm_calls']:>10}{s['tokens_total']:>9}"
        )

    llm_tokens = rows[1][3].summary()["tokens_total"]
    print(
        f"\n  按占位价格估算，抽取 {len(MINI_MEMORIES)} 条记忆的 LLM 成本：${cost(llm_tokens, 0):.6f}"
    )
    print(f"  折算到每万条记忆：约 ${cost(llm_tokens, 0) * 10000 / len(MINI_MEMORIES):.2f}")

    print(
        """
  读法：

  · **先看一个反直觉的结果**：LLM 抽取建了一张更密的图
    （实体更多、边更多、没抽到实体的记忆更少），
    但它的多跳召回率**比规则版还低**。

    原因值得记住。查询「我负责的那个项目」的种子是「项目」这个泛称，
    而 LLM 抽取把 XR-2049 和 XR-2050 **都**连到了「项目」节点上。
    于是 PPR 从「项目」出发时，分数被两个项目均分；
    m08（XR-2050，一跳就到记忆）反而压过了 m07（XR-2049 的效果，要走两跳）。

    这是图检索的一个经典问题：**枢纽节点会稀释分数**。
    连接越多的实体，从它出发的传播就越发散。
    HippoRAG 用 node specificity（类似 IDF，给高频节点降权）缓解这个问题，
    本章的挑战题里留了这个实现。

  · 更一般的结论：**「更好的抽取」不等于「更好的检索」**。
    图的效果取决于抽取质量、图结构、查询锚定三者的配合，
    单独优化其中一个，另外两个可能反过来拖后腿。
    这和第 3 章「换成向量检索这个架构动作本身不带来任何东西」是同一类教训。

  · 成本是实打实的：每条记忆一次调用。上面的金额看着小，
    因为只有 30 条记忆。**折算到每万条**那一行才是你要看的数字，
    而真实系统里一个活跃用户一年就能产生上万条。

  · 还有一块成本不在这张表里：**规则版需要人工维护词表和模式**。
    本章的 RuleExtractor 里有一份领域词表、一份姓氏表、一份前缀噪声表，
    换一个业务领域全都要重写。这份人力成本不出现在账单上，
    但它是真实的，而且不会随规模摊薄。

  · 一个实践建议：**混合用**。规则抽取兜底（免费、快、确定），
    对「重要」的记忆再走一次 LLM 抽取。怎么判断重要？
    第 8 章会讲 importance 打分——但那本身又是一次 LLM 调用。
    这类「为了省钱而引入的判断，本身也要花钱」的循环，
    在记忆系统里非常常见，判断依据仍然是读写比。

  · 写这个实验时踩到过两个坑，都留在代码里了：

      1. `_add` 里既调 extract_triples 又调 extract，用 LLM 抽取器时
         等于每条记忆付两次钱——30 条记忆发了 84 次请求。
         **在成本表出来之前，这个 bug 完全看不出来。**
      2. `LLMExtractor.extract()` 从三元组反推实体，而**查询**通常只含
         一个实体、构不成三元组，于是它静默返回空，
         图检索连种子都没有就退化成了纯向量检索。

    第二个坑尤其值得注意：它不报错、不崩溃，只是让你的图记忆
    悄悄变回了向量检索。**没有分能力表，这种退化不会被发现。**

  · 最后提醒：这里的 LLM 是**模拟的**。它按预设返回完美的三元组，
    不会抽错、不会漏、不会返回非法 JSON。真实模型这三件事都会发生，
    而且错误会**固化进图里**并影响未来所有检索。第 6 章专门处理这个问题。
"""
    )


if __name__ == "__main__":
    main()
