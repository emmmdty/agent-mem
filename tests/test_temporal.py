"""TemporalGraphMemory 的测试。

时间相关的代码不注入时钟就没法测，所以这里全部用固定的 ``BASE_DATE``。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from minimem.base import MemoryItem
from minimem.eval import as_memory_items
from minimem.eval.dataset import BASE_DATE
from minimem.temporal import Fact, TemporalGraphMemory, extract_facts, parse_relative_time
from minimem.utils.embedding import FakeEmbedder
from minimem.utils.metering import Meter

USER = "u1"
T0 = datetime(2026, 7, 30, tzinfo=timezone.utc)


@pytest.fixture
def store():
    s = TemporalGraphMemory(now=BASE_DATE, embedder=FakeEmbedder(), meter=Meter())
    s.add_many(as_memory_items(), user_id=USER)
    return s


class TestRelativeTime:
    @pytest.mark.parametrize(
        ("text", "days"),
        [
            ("我下个月入职长风科技", 30),
            ("我下周就要离职了", 7),
            ("明天开会", 1),
            ("上周搬家了", -7),
            ("昨天去了医院", -1),
        ],
    )
    def test_相对表达(self, text, days):
        assert parse_relative_time(text, spoken_at=T0) == T0 + timedelta(days=days)

    def test_带数量的表达(self):
        assert parse_relative_time("三个月前我还在北京", spoken_at=T0) == T0 - timedelta(days=90)
        assert parse_relative_time("两周后交付", spoken_at=T0) == T0 + timedelta(days=14)

    def test_没有时间表达返回None(self):
        assert parse_relative_time("我在星辰银行工作", spoken_at=T0) is None

    def test_生效时间可以早于说话时间(self):
        """「上周搬家了」是在追述已经发生的事，这个方向也必须支持。"""
        vf = parse_relative_time("上周搬家了，现在住沿江花园", spoken_at=T0)
        assert vf is not None and vf < T0


class TestFactExtraction:
    def test_抽出雇主(self):
        item = MemoryItem("我在星辰银行工作。", created_at=T0)
        facts = extract_facts(item)
        assert any(f.predicate == "雇主" and f.object == "星辰银行" for f in facts)

    def test_未来生效的事实(self):
        item = MemoryItem("新工作定了，我下个月入职长风科技。", created_at=T0)
        fact = next(f for f in extract_facts(item) if f.predicate == "雇主")
        assert fact.valid_from > fact.recorded_at
        assert not fact.is_valid_at(T0), "下个月才生效的事实，今天应当无效"
        assert fact.is_valid_at(T0 + timedelta(days=31))

    def test_记录来源(self):
        item = MemoryItem("我对花生过敏。", created_at=T0)
        assert all(f.source_episode == item.id for f in extract_facts(item))


class TestFactValidity:
    def test_未生效时无效(self):
        f = Fact("雇主", "长风科技", T0, T0 + timedelta(days=30), "m1")
        assert not f.is_valid_at(T0)

    def test_已失效时无效(self):
        f = Fact("雇主", "星辰银行", T0, T0, "m1", valid_to=T0 + timedelta(days=10))
        assert f.is_valid_at(T0)
        assert not f.is_valid_at(T0 + timedelta(days=11))

    def test_valid_to为None表示仍有效(self):
        assert Fact("姓名", "小明", T0, T0, "m1").is_valid_at(T0 + timedelta(days=3650))


class TestEvolution:
    def test_新事实让旧事实失效(self, store):
        rows = [f for f in store.all_facts(user_id=USER) if f.predicate == "居住地"]
        assert len(rows) == 2
        old = next(f for f in rows if f.object == "枫林小区")
        assert old.valid_to is not None, "搬家后旧住址应失效"
        assert old.invalidated_by

    def test_当前有效事实不含已失效的(self, store):
        values = {f.object for f in store.facts_at(user_id=USER, predicate="居住地")}
        assert values == {"沿江花园"}

    def test_离职后雇主没有有效值(self, store):
        """已离职、尚未入职——「当前无有效值」是正确答案，不是 bug。"""
        assert store.facts_at(user_id=USER, predicate="雇主") == []

    def test_时点查询给出不同答案(self, store):
        past = BASE_DATE - timedelta(days=90)
        assert [f.object for f in store.facts_at(past, user_id=USER, predicate="居住地")] == [
            "枫林小区"
        ]
        assert [f.object for f in store.facts_at(user_id=USER, predicate="居住地")] == ["沿江花园"]

    def test_未来时点雇主生效(self, store):
        future = BASE_DATE + timedelta(days=30)
        assert [f.object for f in store.facts_at(future, user_id=USER, predicate="雇主")] == [
            "长风科技"
        ]

    def test_重复陈述不算冲突(self):
        s = TemporalGraphMemory(now=T0, embedder=FakeEmbedder(), meter=Meter())
        s.add(MemoryItem("我在星辰银行工作。", created_at=T0), user_id=USER)
        s.add(MemoryItem("对了，我在星辰银行工作。", created_at=T0), user_id=USER)
        assert len(s.facts_at(user_id=USER, predicate="雇主")) >= 1
        assert all(f.object == "星辰银行" for f in s.facts_at(user_id=USER, predicate="雇主"))

    def test_timeline可读(self, store):
        out = store.timeline("雇主", user_id=USER)
        assert "星辰银行" in out and "长风科技" in out
        assert "（没有关于" in store.timeline("不存在的属性", user_id=USER)


class TestProvenance:
    def test_每条事实都能追到来源(self, store):
        ids = {i.id for i in store.all(user_id=USER)}
        assert all(f.source_episode in ids for f in store.all_facts(user_id=USER))

    def test_删除记忆会级联失效派生事实(self, store):
        target = next(i for i in store.all(user_id=USER) if "过敏" in i.content)
        assert store.facts_at(user_id=USER, predicate="过敏原")
        store.delete(target.id, user_id=USER)
        assert store.facts_at(user_id=USER, predicate="过敏原") == []

    def test_失效不等于删除(self, store):
        """delete 只标记失效，数据还在——这在合规上不算删除。"""
        target = next(i for i in store.all(user_id=USER) if "过敏" in i.content)
        store.delete(target.id, user_id=USER)
        still = [f for f in store.all_facts(user_id=USER) if f.source_episode == target.id]
        assert still, "delete 之后事实记录应仍保留（可审计）"

    def test_purge才是真删除(self, store):
        target = next(i for i in store.all(user_id=USER) if "过敏" in i.content)
        n = store.purge_facts_of(target.id, user_id=USER)
        assert n >= 1
        assert not [f for f in store.all_facts(user_id=USER) if f.source_episode == target.id]


class TestSearch:
    def test_检索仍然可用(self, store):
        assert store.search("我住在哪", user_id=USER, k=5)

    def test_debug带时点与状态(self, store):
        hits = store.search("我住在哪", user_id=USER, k=3)
        assert "时点" in hits[0].debug
        assert hits[0].debug["事实状态"] in ("当时有效", "当时无效", "无关事实")

    def test_同一条记忆在不同时点得分不同(self, store):
        """当时有效的排前面，当时无效的降权——这是本章对检索的全部影响。"""
        past = BASE_DATE - timedelta(days=90)

        def score_of(keyword: str, as_of):
            hits = store.search("我住在哪里", user_id=USER, k=10, as_of=as_of)
            return next((h.score for h in hits if keyword in h.content), None)

        old_now = score_of("枫林", BASE_DATE)
        old_past = score_of("枫林", past)
        assert old_now is not None and old_past is not None
        assert old_past > old_now, "旧住址在它有效的那个时点应当得分更高"

    def test_失效记忆被降权而非过滤(self, store):
        """「三个月前住哪」需要旧记忆仍可召回，所以只降权不过滤。

        降权幅度是个真实权衡：试过 0.35，结果历史记忆被挤出 top-k，
        效果和直接过滤没区别。
        """
        hits = store.search("我住在哪里", user_id=USER, k=10)
        assert any("枫林" in h.content for h in hits), "历史记忆必须仍可召回"

    def test_answer_context带时间与来源(self, store):
        ctx = store.answer_context("我在哪家公司上班？", user_id=USER)
        assert "判断时点" in ctx
        assert "当前有效的事实" in ctx
