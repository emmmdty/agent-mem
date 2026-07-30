"""VectorMemory 与检索工具的测试。

刻意全部用 ``FakeEmbedder``：测试要验证的是**检索管线的行为**（融合、门槛、
惰性索引、用户隔离），不是嵌入模型的质量。用真实模型会让测试变慢、
需要联网，而且断言会随模型版本漂移。
"""

from __future__ import annotations

import numpy as np
import pytest

from minimem.base import MemoryItem
from minimem.utils.bm25 import BM25, tokenize
from minimem.utils.embedding import FakeEmbedder, cosine_similarity, get_embedder
from minimem.utils.metering import Meter
from minimem.vector import VectorMemory, reciprocal_rank_fusion


@pytest.fixture
def store():
    return VectorMemory(embedder=FakeEmbedder(), meter=Meter())


class TestTokenize:
    def test_英文按词切且保留标识符(self):
        assert "xr-2049" in tokenize("项目 XR-2049 上线了")

    def test_中文切出单字与bigram(self):
        tokens = tokenize("信贷风控")
        assert "信" in tokens
        assert "信贷" in tokens
        assert "贷风" in tokens

    def test_空串(self):
        assert tokenize("") == []


class TestBM25:
    def test_命中查询词得分为正(self):
        bm25 = BM25().fit(["我在星辰银行工作", "今天天气不错", "我们家的猫叫毛毛"])
        scores = bm25.score("星辰银行")
        assert scores[0] > 0
        assert scores[0] > scores[1]

    def test_完全不命中得零分(self):
        bm25 = BM25().fit(["abc", "def"])
        assert bm25.score("zzz").sum() == 0

    def test_空索引不报错(self):
        assert len(BM25().fit([]).score("任意")) == 0

    def test_罕见词权重高于常见词(self):
        docs = [f"这是第 {i} 条普通记录" for i in range(20)]
        docs.append("这是第 99 条普通记录，提到了 XR-2049")
        bm25 = BM25().fit(docs)
        # 罕见的 XR-2049 应该把最后一条顶到最高
        assert int(np.argmax(bm25.score("XR-2049"))) == len(docs) - 1


class TestEmbedder:
    def test_fake向量已归一化(self):
        v = FakeEmbedder().encode(["测试文本", "另一段"])
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=1e-5)

    def test_fake向量确定性(self):
        a = FakeEmbedder().encode_one("同样的输入")
        b = FakeEmbedder().encode_one("同样的输入")
        np.testing.assert_array_equal(a, b)

    def test_fake保留字面重叠(self):
        e = FakeEmbedder()
        v = e.encode(["信贷风控工程师", "信贷风控专家", "今天天气真好"])
        assert float(v[0] @ v[1]) > float(v[0] @ v[2])

    def test_空输入(self):
        assert FakeEmbedder().encode([]).shape[0] == 0

    def test_get_embedder的fake快捷方式(self):
        assert isinstance(get_embedder("fake"), FakeEmbedder)

    def test_余弦相似度(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        sims = cosine_similarity(a, b)[0]
        np.testing.assert_allclose(sims, [1.0, 0.0, -1.0], atol=1e-6)

    def test_零向量不产生nan(self):
        sims = cosine_similarity(np.zeros((1, 2)), np.zeros((1, 2)))
        assert not np.isnan(sims).any()


class TestRRF:
    def test_两个通道都排第一的文档得分最高(self):
        fused = reciprocal_rank_fusion({"a": ["x", "y"], "b": ["x", "z"]})
        assert max(fused, key=lambda kv: fused[kv]) == "x"

    def test_权重生效(self):
        fused = reciprocal_rank_fusion({"a": ["x"], "b": ["y"]}, weights={"a": 10.0, "b": 1.0})
        assert fused["x"] > fused["y"]

    def test_空输入(self):
        assert reciprocal_rank_fusion({}) == {}


class TestVectorMemory:
    def test_基本检索(self, store):
        store.add_many(["我在星辰银行工作", "今天天气不错"], user_id="u1")
        hits = store.search("星辰银行", user_id="u1", k=1)
        assert hits[0].content == "我在星辰银行工作"

    def test_三种模式都能跑(self, store):
        store.add_many(["项目 XR-2049 是反欺诈模型", "我喜欢爬山"], user_id="u1")
        for mode in ("dense", "bm25", "hybrid"):
            hits = store.search("XR-2049", user_id="u1", k=2, mode=mode)
            assert hits, f"{mode} 模式返回空"

    def test_debug字段带各通道排名(self, store):
        store.add_many(["项目 XR-2049 是反欺诈模型", "我喜欢爬山"], user_id="u1")
        hits = store.search("XR-2049", user_id="u1", k=1, mode="hybrid")
        assert "dense_rank" in hits[0].debug
        assert "bm25_score" in hits[0].debug

    def test_更新内容会重算向量(self, store):
        mid = store.add("原始内容与检索目标无关", user_id="u1")
        store.add("干扰项", user_id="u1")
        store.update(mid, {"content": "星辰银行信贷风控"}, user_id="u1")
        hits = store.search("星辰银行信贷风控", user_id="u1", k=1)
        assert hits[0].item.id == mid, "更新内容后向量必须重算，否则检索仍按旧语义命中"

    def test_删除后不再被检索到(self, store):
        mid = store.add("需要被删除的内容", user_id="u1")
        store.add("保留的内容", user_id="u1")
        store.delete(mid, user_id="u1")
        assert all(h.item.id != mid for h in store.search("需要被删除", user_id="u1", k=5))
        assert mid not in store._vectors, "删除后向量缓存也要清理"

    def test_用户隔离(self, store):
        store.add("用户1的秘密内容", user_id="u1")
        store.add("用户2的内容", user_id="u2")
        hits = store.search("秘密", user_id="u2", k=5)
        assert all(h.item.user_id == "u2" for h in hits)

    def test_批量写入与逐条写入结果一致(self):
        texts = ["第一条内容", "第二条内容", "第三条内容"]
        a = VectorMemory(embedder=FakeEmbedder(), meter=Meter())
        a.add_many(list(texts), user_id="u1")
        b = VectorMemory(embedder=FakeEmbedder(), meter=Meter())
        for t in texts:
            b.add(t, user_id="u1")

        ra = [h.content for h in a.search("第二条", user_id="u1", k=3)]
        rb = [h.content for h in b.search("第二条", user_id="u1", k=3)]
        assert ra == rb

    def test_bm25门槛过滤低分候选(self):
        items = [MemoryItem("项目 XR-2049 是反欺诈模型")] + [
            MemoryItem(f"这是第 {i} 条无关记录") for i in range(10)
        ]
        strict = VectorMemory(
            embedder=FakeEmbedder(), mode="bm25", bm25_floor_ratio=0.9, meter=Meter()
        )
        loose = VectorMemory(
            embedder=FakeEmbedder(), mode="bm25", bm25_floor_ratio=0.0, meter=Meter()
        )
        for s in (strict, loose):
            s.add_many([i.copy_with() for i in items], user_id="u1")

        n_strict = len(strict.search("XR-2049 是什么", user_id="u1", k=10))
        n_loose = len(loose.search("XR-2049 是什么", user_id="u1", k=10))
        assert n_strict <= n_loose

    def test_min_score过滤(self):
        s = VectorMemory(embedder=FakeEmbedder(), mode="dense", min_score=2.0, meter=Meter())
        s.add_many(["内容甲", "内容乙"], user_id="u1")
        # 余弦相似度不可能达到 2.0，全部应被过滤
        assert s.search("完全不相关的查询", user_id="u1", k=5) == []

    def test_惰性索引_dense模式不建bm25(self, store):
        store.add_many(["一些内容", "另一些内容"], user_id="u1")
        store.search("内容", user_id="u1", k=1, mode="dense")
        idx = store._index["u1"]
        assert idx.matrix is not None
        assert idx.bm25 is None, "纯 dense 模式不应付 BM25 的索引成本"

    def test_top_k_indices取前k且降序(self):
        scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        top = VectorMemory._top_k_indices(scores, 3)
        assert list(top) == [1, 4, 2]

    def test_top_k_indices的k大于长度(self):
        scores = np.array([0.1, 0.9])
        assert len(VectorMemory._top_k_indices(scores, 10)) == 2
