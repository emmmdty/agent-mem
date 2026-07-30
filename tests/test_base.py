"""MemoryStore 接口契约的测试。

这些测试对**所有**记忆实现都应当成立。后续章节新增实现时，只需把类加进
``ALL_STORES``，契约就自动被检验——这是保证「八种实现可横向比较」的最低成本手段。
"""

from __future__ import annotations

import functools

import pytest

from minimem import (
    BufferMemory,
    CrossUserAccessError,
    MemoryItem,
    MemoryNotFoundError,
)
from minimem.graph import GraphMemory
from minimem.temporal import TemporalGraphMemory
from minimem.utils.embedding import FakeEmbedder
from minimem.utils.metering import Meter
from minimem.vector import VectorMemory
from minimem.window import WindowMemory

# 随着章节推进，这个列表会变长。每加一个实现，上面全部契约测试自动生效——
# 这是「八种实现可横向比较」这个承诺的最低成本保障。
# 向量实现统一用 FakeEmbedder：契约测试验证的是接口行为，不是模型质量。
ALL_STORES = [
    pytest.param(BufferMemory, id="buffer"),
    pytest.param(functools.partial(VectorMemory, embedder=FakeEmbedder()), id="vector"),
    pytest.param(functools.partial(WindowMemory, budget_tokens=100_000), id="window"),
    pytest.param(
        functools.partial(GraphMemory, embedder=FakeEmbedder(), mode="hybrid"), id="graph"
    ),
    pytest.param(functools.partial(TemporalGraphMemory, embedder=FakeEmbedder()), id="temporal"),
]


@pytest.fixture(params=ALL_STORES)
def store(request):
    return request.param(meter=Meter())


class TestMemoryItem:
    def test_默认字段(self):
        item = MemoryItem("我叫小明")
        assert item.content == "我叫小明"
        assert len(item.id) == 32
        assert item.kind == "message"
        assert item.metadata == {}
        assert item.created_at.tzinfo is not None, "时间戳必须带时区"

    def test_内容必须是字符串(self):
        with pytest.raises(TypeError):
            MemoryItem(123)  # type: ignore[arg-type]

    def test_copy_with不修改原对象(self):
        item = MemoryItem("原文", metadata={"a": 1})
        new = item.copy_with(content="新文")
        assert item.content == "原文"
        assert new.content == "新文"
        assert new.id == item.id
        new.metadata["b"] = 2
        assert "b" not in item.metadata, "metadata 必须深拷贝一层"


class TestStoreContract:
    def test_写入后可读出(self, store):
        mid = store.add("我在做金融风控", user_id="u1")
        items = store.all(user_id="u1")
        assert len(items) == 1
        assert items[0].id == mid
        assert items[0].user_id == "u1"

    def test_接受字符串快捷写法(self, store):
        store.add("纯字符串", user_id="u1")
        assert store.all(user_id="u1")[0].content == "纯字符串"

    def test_用户之间互相不可见(self, store):
        store.add("用户1的秘密", user_id="u1")
        store.add("用户2的秘密", user_id="u2")

        assert len(store.all(user_id="u1")) == 1
        hits = store.search("秘密", user_id="u2", k=10)
        assert all(h.item.user_id == "u2" for h in hits)
        assert "用户1" not in "".join(h.content for h in hits)

    def test_越权访问抛异常而非静默失败(self, store):
        mid = store.add("用户1的秘密", user_id="u1")
        store.add("占位", user_id="u2")
        with pytest.raises(CrossUserAccessError):
            store.delete(mid, user_id="u2")

    def test_空user_id被拒绝(self, store):
        for bad in ["", "   "]:
            with pytest.raises(ValueError):
                store.add("x", user_id=bad)

    def test_k必须为正(self, store):
        store.add("x", user_id="u1")
        with pytest.raises(ValueError):
            store.search("x", user_id="u1", k=0)

    def test_更新走浅合并(self, store):
        mid = store.add(MemoryItem("原文", metadata={"a": 1}), user_id="u1")
        store.update(mid, {"metadata": {"b": 2}}, user_id="u1")
        item = store.all(user_id="u1")[0]
        assert item.metadata == {"a": 1, "b": 2}, "metadata 应合并而非整体替换"
        assert item.content == "原文"

    def test_更新未知字段被拒绝(self, store):
        mid = store.add("x", user_id="u1")
        with pytest.raises(ValueError):
            store.update(mid, {"created_at": "2020-01-01"}, user_id="u1")

    def test_删除不存在的记忆(self, store):
        store.add("x", user_id="u1")
        with pytest.raises(MemoryNotFoundError):
            store.delete("不存在的id", user_id="u1")

    def test_清空(self, store):
        store.add_many(["a", "b", "c"], user_id="u1")
        store.clear(user_id="u1")
        assert store.all(user_id="u1") == []

    def test_空库检索返回空列表(self, store):
        assert store.search("任意查询", user_id="u1", k=5) == []

    def test_检索结果按分数降序(self, store):
        store.add_many(["苹果和香蕉", "苹果", "完全无关的内容"], user_id="u1")
        hits = store.search("苹果", user_id="u1", k=10)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_检索数量不超过k(self, store):
        store.add_many([f"记忆{i}" for i in range(20)], user_id="u1")
        assert len(store.search("记忆", user_id="u1", k=3)) <= 3

    def test_as_context受字符预算约束(self, store):
        store.add_many(["苹果" * 50, "苹果" * 50, "苹果" * 50], user_id="u1")
        ctx = store.as_context("苹果", user_id="u1", k=3, max_chars=120)
        assert len(ctx) <= 120

    def test_len被显式禁止(self, store):
        with pytest.raises(TypeError, match="用户"):
            len(store)


class TestMetering:
    def test_操作被记录(self):
        meter = Meter()
        store = BufferMemory(meter=meter)
        store.add("x", user_id="u1")
        store.search("x", user_id="u1", k=1)

        ops = [r.op for r in meter.records]
        assert ops == ["add", "search"]
        assert all(r.duration_ms >= 0 for r in meter.records)
        assert meter.summary(op="add")["count"] == 1

    def test_失败的操作也被记录(self):
        meter = Meter()
        store = BufferMemory(meter=meter)
        store.add("x", user_id="u1")
        with pytest.raises(MemoryNotFoundError):
            store.delete("不存在", user_id="u1")

        failed = [r for r in meter.records if "failed" in r.extra]
        assert len(failed) == 1, "失败路径的耗时不应从统计里消失"
