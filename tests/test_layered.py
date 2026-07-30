"""LayeredMemory 的测试。

三条断言钉住本章的核心论点：
1. 换出不等于丢失（分层是降级，不是丢弃）；
2. 冷层必须显式捞取，否则分层没有意义；
3. 反复摘要会持续退化。
"""

from __future__ import annotations

import pytest

from minimem.base import MemoryItem
from minimem.layered import LayeredMemory, summarize_by_rule
from minimem.utils.embedding import FakeEmbedder
from minimem.utils.metering import Meter
from minimem.utils.tokens import count_tokens

USER = "u1"
FILLER = [f"第 {i} 条例行记录，内容并不重要。" for i in range(20)]


@pytest.fixture
def store():
    return LayeredMemory(core_budget=120, recall_capacity=5, embedder=FakeEmbedder(), meter=Meter())


class TestConstruction:
    def test_预算必须为正(self):
        with pytest.raises(ValueError):
            LayeredMemory(core_budget=0)


class TestPaging:
    def test_预算充足时不换页(self):
        s = LayeredMemory(core_budget=100_000, embedder=FakeEmbedder(), meter=Meter())
        s.add_many(list(FILLER[:5]), user_id=USER)
        assert s.stats(user_id=USER)["换页次数"] == 0

    def test_超预算触发换页(self, store):
        store.add_many(list(FILLER), user_id=USER)
        assert store.stats(user_id=USER)["换页次数"] > 0

    def test_core层不超预算(self, store):
        store.add_many(list(FILLER), user_id=USER)
        stats = store.stats(user_id=USER)
        assert stats["core token"] <= stats["core 预算"]

    def test_换出不等于丢失(self, store):
        """分层是降级，不是丢弃——这是与第 2 章窗口方案的根本区别。"""
        store.add_many(list(FILLER), user_id=USER)
        assert len(store.all(user_id=USER)) == len(FILLER)

    def test_热度低的先被换出(self, store):
        hot = store.add(MemoryItem("需要保留的热点内容", metadata={"layer": "core"}), user_id=USER)
        for _ in range(5):
            store.search("热点", user_id=USER, k=1)
        store.add_many(list(FILLER), user_id=USER)
        assert store.layer_of(hot, user_id=USER) == "core"

    def test_从未访问的记忆会被换出(self):
        """热度换页不是银弹——这条断言记录的是局限，不是能力。"""
        s = LayeredMemory(
            core_budget=100, recall_capacity=3, embedder=FakeEmbedder(), meter=Meter()
        )
        key = s.add(MemoryItem("从未被访问的关键信息", metadata={"layer": "core"}), user_id=USER)
        s.add_many(list(FILLER), user_id=USER)
        assert s.layer_of(key, user_id=USER) != "core"

    def test_换页记录可审计(self, store):
        store.add_many(list(FILLER), user_id=USER)
        assert store.evictions
        mid, src, dst, reason = store.evictions[0]
        assert src in ("core", "recall") and dst in ("recall", "archival")
        assert reason


class TestLayers:
    def test_冷层默认不参与检索(self, store):
        store.add_many(list(FILLER), user_id=USER)
        hits = store.search("例行记录", user_id=USER, k=20)
        assert all(h.debug["layer"] != "archival" for h in hits)

    def test_显式捞取冷层(self, store):
        store.add_many(list(FILLER), user_id=USER)
        n_without = len(store.search("例行记录", user_id=USER, k=20))
        n_with = len(store.search("例行记录", user_id=USER, k=20, include_archival=True))
        assert n_with > n_without

    def test_debug带层信息(self, store):
        store.add("一条内容", user_id=USER)
        hit = store.search("内容", user_id=USER, k=1)[0]
        assert hit.debug["layer"] in ("core", "recall", "archival")
        assert "heat" in hit.debug


class TestSelfEditing:
    def test_append进core(self, store):
        mid = store.core_memory_append("用户希望被称呼为小明", user_id=USER)
        assert store.layer_of(mid, user_id=USER) == "core"
        assert "小明" in store.core_context(user_id=USER)

    def test_replace就地改写(self, store):
        mid = store.core_memory_append("旧的称呼", user_id=USER)
        store.core_memory_replace(mid, "新的称呼", user_id=USER)
        assert "新的称呼" in store.core_context(user_id=USER)
        assert "旧的称呼" not in store.core_context(user_id=USER)

    def test_promote回捞(self, store):
        store.add_many(list(FILLER), user_id=USER)
        cold = next(
            i.id for i in store.all(user_id=USER) if store.layer_of(i.id, user_id=USER) != "core"
        )
        store.promote(cold, user_id=USER)
        assert store.layer_of(cold, user_id=USER) == "core"


class TestSummaryDecay:
    def test_摘要会变短(self):
        original = "我在星辰银行工作，是一名信贷风控工程师，主要负责对公业务。"
        assert count_tokens(summarize_by_rule([original])) < count_tokens(original)

    def test_反复摘要持续退化(self):
        texts = [
            "我在星辰银行工作，是一名信贷风控工程师。",
            "我对花生过敏，点餐时一定要提前确认配料。",
            "我负责的项目代号是 XR-2049，上线后复核量降了四成。",
        ]
        lengths = []
        current = texts
        for i in range(4):
            summary = summarize_by_rule(current, max_chars=max(15, 70 - i * 15))
            lengths.append(count_tokens(summary))
            current = [summary]
        assert lengths == sorted(lengths, reverse=True), "反复摘要应单调变短"

    def test_下沉时摘要可关闭(self):
        s = LayeredMemory(
            core_budget=60,
            recall_capacity=2,
            archival_summarize=False,
            embedder=FakeEmbedder(),
            meter=Meter(),
        )
        s.add_many(list(FILLER), user_id=USER)
        assert s.stats(user_id=USER)["被摘要过的"] == 0

    def test_下沉时默认摘要(self):
        s = LayeredMemory(core_budget=60, recall_capacity=2, embedder=FakeEmbedder(), meter=Meter())
        s.add_many(list(FILLER), user_id=USER)
        assert s.stats(user_id=USER)["被摘要过的"] > 0


class TestMutation:
    def test_删除(self, store):
        mid = store.add("要删除的内容", user_id=USER)
        store.delete(mid, user_id=USER)
        assert not any(i.id == mid for i in store.all(user_id=USER))

    def test_用户隔离(self, store):
        store.add("用户1的内容", user_id="u1")
        store.add("用户2的内容", user_id="u2")
        assert all(h.item.user_id == "u2" for h in store.search("内容", user_id="u2", k=5))

    def test_recall_rate工具(self, store):
        store.add(MemoryItem("我对花生过敏", metadata={"layer": "core"}), user_id=USER)
        assert store.recall_rate(["花生"], user_id=USER) == 1.0
        assert store.recall_rate(["海鲜"], user_id=USER) == 0.0
