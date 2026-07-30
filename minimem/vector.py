"""``VectorMemory``：用句向量把「意思」变成可计算的东西，再和 BM25 融合。

相比第 1 章的 ``BufferMemory``，这里换掉了两样东西：

1. **表征**：从「字符集合」换成「稠密向量」。同义改写不再是死穴。
2. **检索**：从单路 Jaccard 换成 dense + BM25 双路 + RRF 融合。

没换的是接口。这是刻意的——第 10 章要把它和其它七种实现放进同一个 harness。

三个值得注意的实现选择：

- **暴力检索而非 ANN 索引**。numpy 的一次矩阵乘法比 Python 循环快一个数量级以上，
  在几万条的量级上已经够快，且没有召回损失。真正需要 ANN（HNSW/IVF）的是
  百万级以上，那时的取舍见第 3 章正文与附录 B。
- **惰性重建，且两个通道各自独立**。写入只打脏标记；dense 与 BM25 的索引
  分别按需构建，纯向量检索不会白白付 BM25 的分词统计成本。
  若「写一条重建一次」，批量写入会变成 O(N²)。
- **归一化后用内积**。``Embedder`` 保证输出是单位向量，于是余弦相似度就是内积。
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from minimem.base import (
    CrossUserAccessError,
    MemoryItem,
    MemoryNotFoundError,
    MemoryStore,
    SearchResult,
)
from minimem.utils.bm25 import BM25
from minimem.utils.embedding import Embedder, get_embedder

__all__ = ["VectorMemory", "reciprocal_rank_fusion"]

SearchMode = Literal["dense", "bm25", "hybrid"]


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]], *, k: int = 60, weights: dict[str, float] | None = None
) -> dict[str, float]:
    """RRF（Reciprocal Rank Fusion）：只看名次，不看分数。

    .. math::

        \\text{RRF}(d) = \\sum_{r \\in \\text{rankers}} \\frac{w_r}{k + \\text{rank}_r(d)}

    为什么不直接把余弦相似度和 BM25 得分加权求和？因为**两者量纲完全不同**：
    余弦相似度在 [-1, 1]，BM25 得分可以是 0 到几十，且分布随语料变化。
    做分数归一化需要估计分布，估计得不好会引入新的偏差；
    而名次是无量纲的，天然可比。

    常数 ``k=60`` 来自 RRF 的原始论文（Cormack et al., SIGIR 2009），
    作用是压低头部名次的绝对优势，让第 1 名和第 2 名的差距不至于碾压其他通道。

    Args:
        rankings: ``{通道名: [按相关性降序排列的 id]}``
        k: 平滑常数
        weights: 各通道权重，缺省为 1.0
    """
    weights = weights or {}
    fused: dict[str, float] = {}
    for channel, ids in rankings.items():
        w = weights.get(channel, 1.0)
        for rank, doc_id in enumerate(ids, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    return fused


class _UserIndex:
    """单个用户的记忆索引。

    惰性重建：写入只打脏标记，检索时才真正重算。而且**两个通道的脏标记是分开的**——
    纯 dense 模式下永远不会去构建 BM25 索引。

    这不是过度设计：BM25 的 ``fit`` 要对全部文档分词并统计词频，
    五万条量级上是百毫秒级的操作。如果无条件构建，
    纯向量检索会白白背上这个成本，而且它藏在第一次检索的延迟里，很难发现。
    """

    def __init__(self) -> None:
        self.items: list[MemoryItem] = []
        self.matrix: np.ndarray | None = None
        self.bm25: BM25 | None = None
        self.dense_dirty = True
        self.bm25_dirty = True

    def mark_dirty(self) -> None:
        self.dense_dirty = True
        self.bm25_dirty = True


class VectorMemory(MemoryStore):
    """向量 + BM25 混合检索的记忆库。

    Args:
        embedder: 句向量模型。缺省按 ``AGENT_MEM_EMBED_MODEL`` 自动选择，
            依赖缺失时降级为 ``FakeEmbedder`` 并打印警告。
        mode: ``"dense"`` 只用向量、``"bm25"`` 只用关键词、``"hybrid"`` 两路融合。
            默认 hybrid。第 3 章的实验会分别测三种模式，你会看到没有一种通吃。
        rrf_k: RRF 的平滑常数。
        weights: 各通道在融合时的权重，如 ``{"dense": 1.0, "bm25": 0.5}``。
        candidate_k: 每个通道各取多少候选进入融合。取太小会让融合失去意义
            （两个通道的 top-k 完全重合），一般取最终 k 的 3~5 倍。
        min_score: 融合后分数低于此值的结果直接丢弃。默认 0.0 表示不过滤。
            第 1 章墙三里「无关结果凑数占满 k 个名额」的问题，就靠它解决。
        bm25_floor_ratio: BM25 候选的相对门槛——得分低于本通道 top1 的这个比例时丢弃。
            **这个参数不是可有可无的调优项，而是修一个真实的 bug**：
            RRF 只看名次，于是当查询里根本没有可匹配的关键词时，
            BM25 返回的少量噪声候选依然会拿到「第 2 名」这个高权重，
            把 dense 通道真正相关的第 2、3 名挤出去。第 3 章实验二会把这个现象测出来。
            默认 0.5 是在 MiniBench 上扫出来的经验值，换语料需要重新扫。

    Example:
        >>> store = VectorMemory()                        # doctest: +SKIP
        >>> store.add("我从事信贷风险评估工作", user_id="u1")   # doctest: +SKIP
        >>> store.search("他的职业是什么", user_id="u1")       # doctest: +SKIP
    """

    name = "vector"

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        mode: SearchMode = "hybrid",
        rrf_k: int = 60,
        weights: dict[str, float] | None = None,
        candidate_k: int | None = None,
        min_score: float = 0.0,
        bm25_floor_ratio: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.embedder = embedder if embedder is not None else get_embedder()
        self.mode: SearchMode = mode
        self.rrf_k = rrf_k
        self.weights = weights or {"dense": 1.0, "bm25": 1.0}
        self.candidate_k = candidate_k
        self.min_score = min_score
        self.bm25_floor_ratio = bm25_floor_ratio

        self._index: dict[str, _UserIndex] = {}
        self._owner: dict[str, str] = {}
        # 向量缓存：id -> vector。删除记忆时同步清理，避免内存泄漏
        self._vectors: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 钩子实现
    # ------------------------------------------------------------------

    def _add(self, item: MemoryItem, *, user_id: str) -> str:
        idx = self._index.setdefault(user_id, _UserIndex())
        idx.items.append(item)
        idx.mark_dirty()
        self._owner[item.id] = user_id
        self._vectors[item.id] = self.embedder.encode_one(item.content)
        return item.id

    def add_many(self, items: list[MemoryItem | str], *, user_id: str) -> list[str]:
        """批量写入：一次性编码，比逐条调用快得多。

        句向量模型的单次调用有固定开销（张量分配、图执行），
        批量编码 100 条通常只比编码 1 条慢几倍，而不是 100 倍。
        """
        self._check_user_id(user_id)
        objs = [MemoryItem(it) if isinstance(it, str) else it for it in items]
        objs = [o.copy_with(user_id=user_id) for o in objs]

        with self._meter.measure(op="add", store=self.name, user_id=user_id) as rec:
            vectors = self.embedder.encode([o.content for o in objs])
            idx = self._index.setdefault(user_id, _UserIndex())
            for obj, vec in zip(objs, vectors, strict=True):
                idx.items.append(obj)
                self._owner[obj.id] = user_id
                self._vectors[obj.id] = vec
            idx.mark_dirty()
            rec.n_items = len(objs)
        return [o.id for o in objs]

    def _search(self, query: str, *, user_id: str, k: int, **kwargs: Any) -> list[SearchResult]:
        idx = self._index.get(user_id)
        if idx is None or not idx.items:
            return []

        mode: SearchMode = kwargs.get("mode", self.mode)
        cand_k = self.candidate_k or max(k * 4, 20)
        cand_k = min(cand_k, len(idx.items))
        id_to_item = {it.id: it for it in idx.items}

        channels: dict[str, list[str]] = {}
        raw_scores: dict[str, dict[str, float]] = {}

        if mode in ("dense", "hybrid"):
            self._ensure_dense(idx)
            ids, scores = self._dense_rank(idx, query, cand_k)
            channels["dense"] = ids
            raw_scores["dense"] = dict(zip(ids, scores, strict=True))

        if mode in ("bm25", "hybrid"):
            self._ensure_bm25(idx)
            ids, scores = self._bm25_rank(idx, query, cand_k)
            channels["bm25"] = ids
            raw_scores["bm25"] = dict(zip(ids, scores, strict=True))

        # 单通道时直接用原始分数排序；融合只在多通道时才有意义
        if len(channels) == 1:
            channel = next(iter(channels))
            fused = {i: raw_scores[channel][i] for i in channels[channel]}
        else:
            fused = reciprocal_rank_fusion(channels, k=self.rrf_k, weights=self.weights)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        results: list[SearchResult] = []
        for doc_id, score in ranked[:k]:
            if score < self.min_score:
                continue
            debug: dict[str, Any] = {
                f"{ch}_score": round(raw_scores[ch].get(doc_id, 0.0), 4) for ch in raw_scores
            }
            for ch, ids in channels.items():
                debug[f"{ch}_rank"] = ids.index(doc_id) + 1 if doc_id in ids else None
            results.append(
                SearchResult(
                    item=id_to_item[doc_id],
                    score=float(score),
                    source=mode,
                    debug=debug,
                )
            )
        return results

    def _update(self, memory_id: str, patch: dict[str, Any], *, user_id: str) -> None:
        idx = self._locate(memory_id, user_id)
        for i, item in enumerate(idx.items):
            if item.id == memory_id:
                new_item = self.apply_patch(item, patch)
                idx.items[i] = new_item
                if "content" in patch:
                    # 内容变了，向量必须重算——忘了这一步是个安静的 bug：
                    # 检索仍然按旧内容的语义命中，但返回的是新内容
                    self._vectors[memory_id] = self.embedder.encode_one(new_item.content)
                    idx.mark_dirty()
                return
        raise MemoryNotFoundError(memory_id)

    def _delete(self, memory_id: str, *, user_id: str) -> None:
        idx = self._locate(memory_id, user_id)
        for i, item in enumerate(idx.items):
            if item.id == memory_id:
                del idx.items[i]
                idx.mark_dirty()
                self._owner.pop(memory_id, None)
                self._vectors.pop(memory_id, None)
                return
        raise MemoryNotFoundError(memory_id)

    def _all(self, *, user_id: str) -> list[MemoryItem]:
        idx = self._index.get(user_id)
        return list(idx.items) if idx else []

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _locate(self, memory_id: str, user_id: str) -> _UserIndex:
        owner = self._owner.get(memory_id)
        if owner is None:
            raise MemoryNotFoundError(memory_id)
        if owner != user_id:
            raise CrossUserAccessError(f"记忆 {memory_id} 不属于用户 {user_id}")
        return self._index[user_id]

    def _ensure_dense(self, idx: _UserIndex) -> None:
        if not idx.dense_dirty:
            return
        idx.matrix = np.vstack([self._vectors[it.id] for it in idx.items]) if idx.items else None
        idx.dense_dirty = False

    def _ensure_bm25(self, idx: _UserIndex) -> None:
        if not idx.bm25_dirty:
            return
        idx.bm25 = BM25().fit([it.content for it in idx.items]) if idx.items else None
        idx.bm25_dirty = False

    @staticmethod
    def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
        """取分数最高的 k 个下标（降序）。

        用 ``argpartition``（O(N)）先划分出前 k 个，再只对这 k 个排序，
        而不是对全部 N 个做 ``argsort``（O(N log N)）。
        N=5 万、k=20 时这一步能省下大部分时间——检索的瓶颈往往不在算相似度，
        而在**排序**。
        """
        n = len(scores)
        if k >= n:
            return np.argsort(-scores)
        part = np.argpartition(-scores, k)[:k]
        return part[np.argsort(-scores[part])]

    def _dense_rank(
        self, idx: _UserIndex, query: str, cand_k: int
    ) -> tuple[list[str], list[float]]:
        assert idx.matrix is not None
        qv = self.embedder.encode_one(query)
        sims = idx.matrix @ qv  # 向量已归一化，内积即余弦相似度
        top = self._top_k_indices(sims, cand_k)
        return [idx.items[i].id for i in top], [float(sims[i]) for i in top]

    def _bm25_rank(self, idx: _UserIndex, query: str, cand_k: int) -> tuple[list[str], list[float]]:
        assert idx.bm25 is not None
        scores = idx.bm25.score(query)
        top = self._top_k_indices(scores, cand_k)
        if len(top) == 0 or scores[top[0]] <= 0:
            return [], []

        # 两道门槛，缺一不可：
        # 1. 得分为 0 表示一个查询词都没命中；
        # 2. 得分远低于本通道 top1 的候选，说明它只是蹭上了一两个常见字。
        #    RRF 只看名次，这种候选会以「第 2 名」的身份拿到高权重，
        #    把另一通道真正相关的结果挤下去——实测能让混合检索反而不如单路。
        floor = scores[top[0]] * self.bm25_floor_ratio
        keep = [i for i in top if scores[i] > 0 and scores[i] >= floor]
        return [idx.items[i].id for i in keep], [float(scores[i]) for i in keep]
