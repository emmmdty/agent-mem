"""BufferMemory 特有行为的测试。"""

from __future__ import annotations

import pytest

from minimem import BufferMemory
from minimem.utils.metering import Meter


@pytest.fixture
def store():
    return BufferMemory(meter=Meter())


def test_按写入顺序保存(store):
    store.add_many(["第一句", "第二句", "第三句"], user_id="u1")
    assert [i.content for i in store.all(user_id="u1")] == ["第一句", "第二句", "第三句"]


def test_max_items触发最旧丢弃(store):
    s = BufferMemory(max_items=3, meter=Meter())
    s.add_many([f"消息{i}" for i in range(5)], user_id="u1")
    contents = [i.content for i in s.all(user_id="u1")]
    assert contents == ["消息2", "消息3", "消息4"], "超出容量应丢最旧的"


def test_max_items必须为正():
    with pytest.raises(ValueError):
        BufferMemory(max_items=0)


def test_recency_weight取值范围():
    with pytest.raises(ValueError):
        BufferMemory(recency_weight=1.5)


def test_词重叠检索命中(store):
    store.add("小明在做金融风控", user_id="u1")
    store.add("今天天气不错", user_id="u1")
    hits = store.search("金融风控", user_id="u1", k=1)
    assert hits[0].content == "小明在做金融风控"


def test_同义表述检索失败_这正是第三章的动机(store):
    """BufferMemory 靠字面重叠，遇到同义改写就完全失效。

    这个测试断言的是**缺陷**而不是能力：它是第 3 章引入语义向量的实证动机。
    如果哪天它失败了，说明有人给 BufferMemory 加了语义能力——那就该改章节了。
    """
    store.add("我从事信贷风险评估工作", user_id="u1")
    store.add("我喜欢周末去爬山", user_id="u1")

    hits = store.search("他的职业是什么", user_id="u1", k=1)
    # 「职业」与「信贷风险评估」零字面重叠，命中的很可能是无关项
    assert hits == [] or hits[0].content != "我从事信贷风险评估工作"


def test_recency_weight为零时纯按重叠排序():
    s = BufferMemory(recency_weight=0.0, meter=Meter())
    s.add("苹果 香蕉 橙子", user_id="u1")
    s.add("苹果", user_id="u1")
    hits = s.search("苹果", user_id="u1", k=2)
    # 「苹果」与查询完全一致，Jaccard = 1；三词那条只有 1/3
    assert hits[0].content == "苹果"


def test_recency_weight提升新记忆排名():
    s = BufferMemory(recency_weight=0.9, meter=Meter())
    s.add("苹果", user_id="u1")
    s.add("苹果 香蕉 橙子", user_id="u1")
    hits = s.search("苹果", user_id="u1", k=2)
    assert hits[0].content == "苹果 香蕉 橙子", "高 recency 权重下新记忆应排前"


def test_history不做相关性筛选(store):
    store.add_many(["a", "b", "c", "d"], user_id="u1")
    assert [i.content for i in store.history(user_id="u1", limit=2)] == ["c", "d"]
    assert len(store.history(user_id="u1")) == 4


def test_中英混合分词(store):
    store.add("使用 RAG 做检索增强", user_id="u1")
    hits = store.search("RAG", user_id="u1", k=1)
    assert len(hits) == 1
