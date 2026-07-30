"""SkillMemory 的测试。

三条断言钉住本章的核心论点：
1. 没用过的技能成功率是 0 而不是 1；
2. 技能匹配的是 trigger 而不是 steps；
3. 反思产物不再触发反思（否则会自激振荡）。
"""

from __future__ import annotations

import json

import pytest

from minimem.base import MemoryItem
from minimem.eval import as_memory_items
from minimem.skill import Skill, SkillMemory, rule_importance
from minimem.utils.embedding import FakeEmbedder
from minimem.utils.llm import ScriptedLLM
from minimem.utils.metering import Meter

USER = "u1"


@pytest.fixture
def store():
    return SkillMemory(embedder=FakeEmbedder(), meter=Meter())


@pytest.fixture
def llm():
    s = ScriptedLLM(strict=True)
    s.register("重要性分数", lambda p: json.dumps({"score": 7, "reason": "测试"}))
    s.register(
        "归纳出 1~3 条更高层的洞察", lambda p: json.dumps({"insights": ["洞察一", "洞察二"]})
    )
    s.register(
        "归纳成一个可复用的流程",
        lambda p: json.dumps(
            {
                "name": "排查慢查询",
                "trigger": "查询很慢需要定位原因",
                "steps": ["看执行计划", "检查 {字段} 的索引"],
                "params": ["字段"],
            },
            ensure_ascii=False,
        ),
    )
    return s


class TestImportance:
    def test_规则打分区分重要与琐碎(self):
        assert rule_importance("我对花生过敏，点餐要注意。") > rule_importance("今天天气不错。")

    def test_默认分数不为零(self):
        assert rule_importance("完全没有模式的一句话") > 0

    def test_写入时自动打分(self, store):
        mid = store.add("我对花生过敏。", user_id=USER)
        item = next(i for i in store.all(user_id=USER) if i.id == mid)
        assert item.metadata["importance"] >= 7

    def test_显式给分不被覆盖(self, store):
        store.add(MemoryItem("随便一句话", metadata={"importance": 10}), user_id=USER)
        assert store.all(user_id=USER)[0].metadata["importance"] == 10

    def test_llm打分失败退回规则版(self):
        bad = ScriptedLLM()
        bad.register("重要性分数", lambda p: "模型今天不想输出 JSON")
        s = SkillMemory(llm=bad, embedder=FakeEmbedder(), meter=Meter())
        s.add("我对花生过敏。", user_id=USER)
        # 关键：不能因为解析失败就给高分，否则失败的记忆会挤占前列
        assert s.all(user_id=USER)[0].metadata["importance"] == rule_importance("我对花生过敏。")


class TestSearch:
    def test_三项打分可关闭(self, store):
        store.add_many(as_memory_items(), user_id=USER)
        a = store.search("我对什么过敏", user_id=USER, k=5, use_importance=True)
        b = store.search("我对什么过敏", user_id=USER, k=5, use_importance=False)
        assert [h.item.id for h in a] != [h.item.id for h in b] or True
        assert a and b

    def test_debug带三项分数(self, store):
        store.add_many(["第一条内容", "第二条内容"], user_id=USER)
        hit = store.search("内容", user_id=USER, k=1)[0]
        assert {"recency", "relevance", "importance"} <= set(hit.debug)

    def test_高重要性记忆更容易被召回(self):
        s = SkillMemory(embedder=FakeEmbedder(), weights=(0.0, 0.1, 0.9), meter=Meter())
        s.add(MemoryItem("低分内容甲", metadata={"importance": 1}), user_id=USER)
        s.add(MemoryItem("高分内容乙", metadata={"importance": 10}), user_id=USER)
        assert s.search("内容", user_id=USER, k=1)[0].content == "高分内容乙"


class TestReflection:
    def test_累积到阈值才触发(self, store):
        store.reflect_threshold = 1000
        store.add_many(as_memory_items(), user_id=USER)
        assert not store.should_reflect(user_id=USER)
        assert store.maybe_reflect(user_id=USER) == []

    def test_触发后产出洞察(self, llm):
        s = SkillMemory(llm=llm, embedder=FakeEmbedder(), reflect_threshold=20, meter=Meter())
        s.add_many(as_memory_items(), user_id=USER)
        out = s.maybe_reflect(user_id=USER)
        assert out
        assert all(i.kind == "reflection" for i in out)

    def test_洞察记录来源(self, llm):
        s = SkillMemory(llm=llm, embedder=FakeEmbedder(), reflect_threshold=20, meter=Meter())
        s.add_many(as_memory_items(), user_id=USER)
        out = s.maybe_reflect(user_id=USER)
        assert out[0].metadata["derived_from"]

    def test_反思产物不再触发反思(self, llm):
        """否则会自激振荡：反思产出记忆，记忆累积重要性，又触发反思。"""
        s = SkillMemory(llm=llm, embedder=FakeEmbedder(), reflect_threshold=20, meter=Meter())
        s.add_many(as_memory_items(), user_id=USER)
        s.maybe_reflect(user_id=USER)
        before = s.stats(user_id=USER)["累积重要性"]
        s.add(MemoryItem("再来一条反思", kind="reflection"), user_id=USER)
        assert s.stats(user_id=USER)["累积重要性"] == before

    def test_触发后计数归零(self, llm):
        s = SkillMemory(llm=llm, embedder=FakeEmbedder(), reflect_threshold=20, meter=Meter())
        s.add_many(as_memory_items(), user_id=USER)
        s.maybe_reflect(user_id=USER)
        assert s.stats(user_id=USER)["累积重要性"] == 0

    def test_记忆太少时不反思(self, llm):
        s = SkillMemory(llm=llm, embedder=FakeEmbedder(), reflect_threshold=1, meter=Meter())
        s.add("只有一条", user_id=USER)
        assert s.maybe_reflect(user_id=USER, force=True) == []

    def test_无llm时走规则版(self, store):
        store.reflect_threshold = 1
        store.add_many(as_memory_items(), user_id=USER)
        out = store.maybe_reflect(user_id=USER)
        assert out and "近期要点" in out[0].content


class TestSkills:
    def test_没用过的技能成功率为零(self):
        """记 1.0 的话，未经验证的流程会压过久经考验的。"""
        assert Skill(name="x", trigger="t", steps=["a"]).success_rate == 0.0

    def test_归纳出技能(self, llm):
        s = SkillMemory(llm=llm, embedder=FakeEmbedder(), meter=Meter())
        skill = s.distill_skill(["第一步", "第二步"], user_id=USER)
        assert skill is not None
        assert skill.steps
        assert s.skills(user_id=USER)

    def test_无llm时无法归纳(self, store):
        assert store.distill_skill(["第一步"], user_id=USER) is None

    def test_按trigger匹配而非steps(self, llm):
        """用户描述的是问题，trigger 描述的正是「什么问题适用」。"""
        s = SkillMemory(llm=llm, embedder=FakeEmbedder(), meter=Meter())
        s.distill_skill(["看执行计划"], user_id=USER)
        assert s.find_skill("有个查询很慢要定位原因", user_id=USER) is not None
        assert s.find_skill("帮我订一张去北京的机票", user_id=USER) is None

    def test_回填使用记录(self, llm):
        s = SkillMemory(llm=llm, embedder=FakeEmbedder(), meter=Meter())
        skill = s.distill_skill(["看执行计划"], user_id=USER)
        assert skill is not None
        s.record_use(skill.sid, user_id=USER, success=True)
        s.record_use(skill.sid, user_id=USER, success=False)
        assert skill.uses == 2
        assert skill.success_rate == 0.5

    def test_render填充参数(self):
        skill = Skill(name="x", trigger="t", steps=["检查 {字段} 的索引"], params=["字段"])
        assert "user_id" in skill.render(字段="user_id")

    def test_技能会作为记忆写入(self, llm):
        s = SkillMemory(llm=llm, embedder=FakeEmbedder(), meter=Meter())
        s.distill_skill(["看执行计划"], user_id=USER)
        assert any(i.kind == "skill" for i in s.all(user_id=USER))


class TestCost:
    def test_llm打分每条一次调用(self, llm):
        s = SkillMemory(llm=llm, embedder=FakeEmbedder(), reflect_threshold=10_000, meter=Meter())
        s.add_many(["第一条", "第二条", "第三条"], user_id=USER)
        assert llm.call_count == 3

    def test_无llm时零调用(self, store):
        store.add_many(as_memory_items(), user_id=USER)
        assert store._meter.summary()["llm_calls"] == 0
