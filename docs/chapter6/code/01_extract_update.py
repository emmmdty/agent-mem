# offline-ok
"""第 6 章实验一：让记忆自己决定该不该存。

    python docs/chapter6/code/01_extract_update.py

前五章的写入都是无脑的：给什么存什么。于是「我在星辰银行工作」和
「我下个月入职长风科技」会并排躺在库里，等着检索时一起被捞出来。

本章让 LLM 在写入时先判断一次：这条新信息和已有记忆是什么关系？

    ADD    —— 无冲突，作为新记忆加入
    UPDATE —— 修正了某条已有记忆
    NOOP   —— 只是重复，不必存
    DELETE —— 表明某条已有记忆该被删掉

本脚本用 ScriptedLLM 模拟这些判断（离线可跑）。**它模拟的是流程和用量，
不是 LLM 的判断力**——真实模型在这里的表现差异很大，正文有专门说明。
"""

from __future__ import annotations

import json
import warnings

from minimem.agentic import AgenticMemory, format_decisions
from minimem.base import MemoryItem
from minimem.utils.embedding import get_embedder
from minimem.utils.llm import ScriptedLLM
from minimem.utils.metering import Meter

USER = "u_demo"

# 一段带冲突的时间线：换工作 + 搬家 + 一次重复陈述
TIMELINE = [
    "我在星辰银行工作，是一名信贷风控工程师。",
    "我住在城西的枫林小区。",
    "我对花生过敏，点餐要特别注意。",
    "对了，我是在星辰银行做风控的。",  # 重复，应 NOOP
    "我下周就要从星辰银行离职了。",  # 修正雇主，应 UPDATE
    "上周搬家了，现在住城东的沿江花园。",  # 修正住址，应 UPDATE
]


def make_llm(meter: Meter) -> ScriptedLLM:
    llm = ScriptedLLM(meter=meter, store_label="agentic")

    def note(prompt: str) -> str:
        text = prompt.split("消息：", 1)[-1].strip()
        table = {
            "星辰银行": (["工作", "星辰银行", "风控"], ["工作", "身份"]),
            "枫林小区": (["住址", "枫林小区"], ["生活"]),
            "花生": (["过敏", "花生"], ["健康"]),
            "离职": (["离职", "星辰银行"], ["工作"]),
            "沿江花园": (["搬家", "沿江花园"], ["生活"]),
        }
        keywords, tags = ["记忆"], ["临时"]
        for key, (kw, tg) in table.items():
            if key in text:
                keywords, tags = kw, tg
                break
        return json.dumps(
            {"summary": text.rstrip("。"), "keywords": keywords, "tags": tags},
            ensure_ascii=False,
        )

    def decide(prompt: str) -> str:
        new = prompt.split("新信息：", 1)[-1].split("\n", 1)[0]
        existing = prompt.split("已有记忆：", 1)[-1]

        def index_of(keyword: str) -> int | None:
            for line in existing.splitlines():
                if line.startswith("[") and keyword in line:
                    return int(line[1 : line.index("]")])
            return None

        if "是在星辰银行做风控" in new:
            return json.dumps(
                {"action": "NOOP", "target": None, "reason": "与已有记忆重复"}, ensure_ascii=False
            )
        if "离职" in new:
            i = index_of("星辰银行")
            if i is not None:
                return json.dumps(
                    {"action": "UPDATE", "target": i, "reason": "雇主状态变更"}, ensure_ascii=False
                )
        if "沿江花园" in new:
            i = index_of("枫林小区")
            if i is not None:
                return json.dumps(
                    {"action": "UPDATE", "target": i, "reason": "住址已搬迁"}, ensure_ascii=False
                )
        return json.dumps({"action": "ADD", "target": None, "reason": "新信息"}, ensure_ascii=False)

    llm.register("加工成一条结构化记忆笔记", note)
    llm.register("判断新信息与已有记忆的关系", decide)
    return llm


def main() -> None:
    print("\n第 6 章实验一：extract-update 流水线")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedder = get_embedder()

    meter = Meter()
    store = AgenticMemory(llm=make_llm(meter), embedder=embedder, meter=meter)

    print("\n  按时间顺序写入 6 条消息：\n")
    for text in TIMELINE:
        store.add(MemoryItem(text), user_id=USER)

    print(format_decisions(store.decisions))

    print("\n  写入后库里实际存了几条：")
    for item in store.all(user_id=USER):
        note = store.note_of(item.id)
        updated = item.metadata.get("updated", 0)
        mark = f"（被更新过 {updated} 次）" if updated else ""
        print(f"    · {note.summary if note else item.content}{mark}")
        if note and note.tags:
            print(f"        标签 {note.tags}　链接 {len(note.links)} 条")

    print(f"\n  {store.stats(user_id=USER)}")

    print(
        """
  读法：

  · 6 条输入，库里只留下 3 条。一条被判为 NOOP（重复陈述），
    两条触发了 UPDATE（雇主与住址被就地修正，不新增记录）。

    对比第 4 章：那时这 6 条会原样存下 6 条，检索「我在哪工作」时
    「星辰银行」和「离职」会一起返回，让模型自己猜。

  · **但请注意这个「干净」是有代价的**：UPDATE 是**就地覆盖**，
    旧值没了。这意味着「我三个月前住哪」这个问题现在答不出来——
    而第 5 章的做法（失效而非删除）是能答的。

    两种设计解决的是不同的问题：
      extract-update  → 让**当前状态**保持唯一和干净
      bi-temporal     → 让**历史**可查

    真实系统往往两个都要：用 extract-update 维护一份「当前视图」，
    同时用双时间轴保留完整历史。代价是两套机制都要维护。

  · 决策失败时默认 **ADD**，不是 UPDATE。这个默认值的选择很关键——
    默认 UPDATE 会让一次 JSON 解析失败**覆盖掉一条正确的记忆**。
    宁可多存一条，不可错删或错更。
"""
    )


if __name__ == "__main__":
    main()
