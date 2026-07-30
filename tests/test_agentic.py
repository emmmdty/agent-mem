"""AgenticMemory 的测试。

三条断言钉住本章的核心论点：
1. 决策失败时默认 ADD（可逆），不是 UPDATE 或 NOOP；
2. 加工失败退回原文而非丢消息，且必须标记 degraded；
3. 链接是双向的，否则旧记忆永远找不到新记忆。
"""

from __future__ import annotations

import json

import pytest

from minimem.agentic import AgenticMemory
from minimem.base import MemoryItem
from minimem.utils.embedding import FakeEmbedder
from minimem.utils.llm import ScriptedLLM
from minimem.utils.metering import Meter

USER = "u1"


def _note(summary_from_text: bool = True):
    def fn(prompt: str) -> str:
        text = prompt.split("消息：", 1)[-1].strip()
        return json.dumps(
            {"summary": text if summary_from_text else "摘要", "keywords": ["k"], "tags": ["临时"]},
            ensure_ascii=False,
        )

    return fn


def make_store(decide_fn=None, note_fn=None, **kwargs) -> AgenticMemory:
    llm = ScriptedLLM()
    llm.register("加工成一条结构化记忆笔记", note_fn or _note())
    llm.register(
        "判断新信息与已有记忆的关系",
        decide_fn or (lambda p: json.dumps({"action": "ADD", "target": None, "reason": "新"})),
    )
    return AgenticMemory(llm=llm, embedder=FakeEmbedder(), meter=Meter(), **kwargs)


class TestNote:
    def test_生成摘要与标签(self):
        s = make_store()
        mid = s.add("我在星辰银行工作。", user_id=USER)
        note = s.note_of(mid)
        assert note is not None
        assert note.summary
        assert note.tags

    def test_加工失败退回原文而非丢消息(self):
        s = make_store(note_fn=lambda p: "模型今天不想输出 JSON")
        s.add("我对花生过敏。", user_id=USER)
        assert len(s.all(user_id=USER)) == 1, "加工失败不能让消息凭空消失"

    def test_加工失败必须可观测(self):
        s = make_store(note_fn=lambda p: "不是 JSON")
        mid = s.add("我对花生过敏。", user_id=USER)
        note = s.note_of(mid)
        assert note is not None and note.degraded, "悄悄降级是最糟的处理"
        assert s.stats(user_id=USER)["加工失败"] == 1

    def test_退回原文时摘要就是原文(self):
        s = make_store(note_fn=lambda p: "不是 JSON")
        mid = s.add("我对花生过敏。", user_id=USER)
        note = s.note_of(mid)
        assert note is not None and note.summary == "我对花生过敏。"


class TestDecisions:
    def test_无候选时直接ADD(self):
        s = make_store()
        s.add("第一条记忆", user_id=USER)
        assert s.decisions[0].action == "ADD"

    def test_NOOP不写入(self):
        s = make_store(
            decide_fn=lambda p: json.dumps({"action": "NOOP", "target": None, "reason": "重复"})
        )
        s.add("我在星辰银行工作。", user_id=USER)
        s.add("我在星辰银行工作。", user_id=USER)
        assert len(s.all(user_id=USER)) == 1

    def test_UPDATE就地覆盖不新增(self):
        # 第一条写入时候选为空，decide 不会被调用（直接 ADD）；
        # 第二条才会真正走到这个 handler
        s = make_store(
            decide_fn=lambda p: json.dumps({"action": "UPDATE", "target": 0, "reason": "修正"}),
            link_threshold=0.0,
        )
        s.add("我在星辰银行工作。", user_id=USER)
        s.add("我从星辰银行离职了。", user_id=USER)
        assert len(s.all(user_id=USER)) == 1
        assert s.all(user_id=USER)[0].metadata["updated"] == 1

    def test_决策失败默认ADD而非覆盖(self):
        """默认 UPDATE 会让一次解析失败覆盖掉正确记忆。"""
        s = make_store(decide_fn=lambda p: "模型返回了一段散文", link_threshold=0.0)
        s.add("第一条", user_id=USER)
        s.add("第二条", user_id=USER)
        assert len(s.all(user_id=USER)) == 2
        assert s.decisions[-1].action == "ADD"
        assert s.decisions[-1].degraded

    def test_UPDATE缺目标时降级为ADD(self):
        s = make_store(
            decide_fn=lambda p: json.dumps({"action": "UPDATE", "target": None, "reason": "改"}),
            link_threshold=0.0,
        )
        s.add("第一条", user_id=USER)
        s.add("第二条", user_id=USER)
        assert s.decisions[-1].action == "ADD"
        assert len(s.all(user_id=USER)) == 2

    def test_可关闭update省一半调用(self):
        s = make_store(enable_update=False)
        s.add_many(["第一条", "第二条", "第三条"], user_id=USER)
        assert s.llm.call_count == 3, "关掉决策后每条只调一次"
        assert s.decisions == []


class TestLinks:
    def test_链接是双向的(self):
        """否则从旧记忆出发永远找不到新记忆。"""
        s = make_store(link_threshold=0.0)
        first = s.add("信贷风控相关的内容甲", user_id=USER)
        second = s.add("信贷风控相关的内容乙", user_id=USER)
        note_first = s.note_of(first)
        note_second = s.note_of(second)
        assert note_second is not None and first in note_second.links
        assert note_first is not None and second in note_first.links

    def test_阈值过高时建不起链接(self):
        """这是个真实的坑：自组织会悄悄退化成普通向量库。"""
        s = make_store(link_threshold=0.99)
        s.add_many(["内容甲", "内容乙", "内容丙"], user_id=USER)
        assert s.stats(user_id=USER)["链接总数"] == 0

    def test_删除会清理悬空链接(self):
        s = make_store(link_threshold=0.0)
        a = s.add("内容甲", user_id=USER)
        b = s.add("内容乙", user_id=USER)
        s.delete(a, user_id=USER)
        note_b = s.note_of(b)
        assert note_b is not None and a not in note_b.links


class TestSearch:
    def test_检索返回摘要(self):
        s = make_store()
        s.add("我在星辰银行工作。", user_id=USER)
        hit = s.search("星辰银行", user_id=USER, k=1)[0]
        assert "summary" in hit.debug

    def test_沿链接扩散可关闭(self):
        s = make_store(link_threshold=0.0)
        s.add_many(["内容甲", "内容乙"], user_id=USER)
        with_links = s.search("内容", user_id=USER, k=2, follow_links=True)
        without = s.search("内容", user_id=USER, k=2, follow_links=False)
        assert len(with_links) == len(without) == 2

    def test_用户隔离(self):
        s = make_store()
        s.add("用户1的内容", user_id="u1")
        s.add("用户2的内容", user_id="u2")
        assert all(h.item.user_id == "u2" for h in s.search("内容", user_id="u2", k=5))


class TestCost:
    def test_每条最多两次调用(self):
        s = make_store()
        s.add_many(["第一条", "第二条"], user_id=USER)
        assert s.llm.call_count <= 4

    def test_成本被记进计量器(self):
        s = make_store()
        s.add("我在星辰银行工作。", user_id=USER)
        assert s._meter.summary()["llm_calls"] >= 1


class TestConstruction:
    def test_必须提供llm(self):
        with pytest.raises((ValueError, TypeError)):
            AgenticMemory(llm=None, embedder=FakeEmbedder())  # type: ignore[arg-type]

    def test_更新内容会重新加工(self):
        s = make_store()
        mid = s.add("原始内容", user_id=USER)
        s.update(mid, {"content": "全新的内容"}, user_id=USER)
        note = s.note_of(mid)
        assert note is not None and "全新" in note.summary

    def test_接受MemoryItem(self):
        s = make_store()
        s.add(MemoryItem("一条记忆", kind="message"), user_id=USER)
        assert len(s.all(user_id=USER)) == 1
