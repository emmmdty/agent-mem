"""``GraphMemory``：当记忆之间有关系。

第 3 章的向量检索在多跳查询上只有 50% 的召回率。原因很直接：
它把每条记忆当成孤立的点，而多跳问题要求你**顺着关系走**。

「我负责的那个项目带来了什么效果？」需要两步：
先从「我负责的项目」找到 XR-2049，再从 XR-2049 找到它的效果。
向量检索只能一次性地找「和整个问句最像的几条」——它没有「走一步」这个动作。

本模块的做法（简化版 HippoRAG，📄 arXiv:2405.14831）：

1. **建图**：从每条记忆抽实体，实体成为节点；记忆也是节点，连向它包含的实体；
   同一条记忆里共现的实体之间连边。
2. **检索**：从查询里抽出实体作为**种子**，跑 Personalized PageRank，
   让分数沿着边传播出去；最后读取记忆节点的分数。

PPR 的关键性质是**多跳传播**：种子的分数会流到一跳邻居，再流到二跳邻居，
每跳按阻尼系数衰减。这正是「顺着关系走」的数学形式。

**代价必须说清楚**（本章正文会量化）：

- 写入时每条记忆要抽一次实体（规则版免费，LLM 版每条一次调用）
- 图会随记忆增长，PPR 每次要在整张图上迭代
- **实体消解**是脏活：同一个实体的不同写法会变成不同节点，图就断了
- 规则抽取的召回有限——它抽不到「毛毛」这种没有标志的专有名词
"""

from __future__ import annotations

from typing import Any, Literal

import networkx as nx

from minimem.base import (
    CrossUserAccessError,
    MemoryItem,
    MemoryNotFoundError,
    MemoryStore,
    SearchResult,
)
from minimem.utils.embedding import Embedder, get_embedder
from minimem.utils.entities import EntityExtractor, RuleExtractor
from minimem.vector import reciprocal_rank_fusion

__all__ = ["GraphMemory"]

GraphMode = Literal["ppr", "dense", "hybrid"]

MEM = "mem"
ENT = "ent"


class GraphMemory(MemoryStore):
    """实体图 + Personalized PageRank 检索。

    Args:
        extractor: 实体抽取器。默认规则版（零成本、离线）。
            换成 ``LLMExtractor`` 召回更好，但每条记忆一次 LLM 调用。
        embedder: 句向量模型，用于 ``dense`` / ``hybrid`` 模式。
        mode: ``"ppr"`` 纯图检索、``"dense"`` 纯向量、``"hybrid"`` 两路 RRF 融合。
            默认 hybrid——**纯图检索有个致命弱点**：查询里抽不出实体时它什么也返回不了，
            而这种情况比想象中常见（「我平时业余时间干什么？」一个实体都没有）。
        damping: PPR 的阻尼系数。越大传播越远（多跳能力越强），也越容易跑偏。
            0.85 是 PageRank 的经典取值。
        max_iter: 幂迭代上限。
        ppr_floor_ratio: PPR 候选的相对门槛——分数低于本通道 top1 的这个比例时丢弃。
            **这是第 3 章那个 RRF 陷阱的复现**：PPR 会给图上每个连通节点一个非零分，
            哪怕小到 1e-8。这些噪声候选在 RRF 里享有和真结果同等的名次权重，
            会把真正相关的第 2、3 名挤出去。第 4 章实验二把这个现象测出来了。
            默认 0.01 —— PPR 分数按跳数指数衰减，所以阈值比 BM25 的 0.5 小得多。

    Note:
        图按用户隔离——每个 ``user_id`` 一张独立的图。这既是隐私要求，
        也让 PPR 的计算规模保持可控。
    """

    name = "graph"

    def __init__(
        self,
        *,
        extractor: EntityExtractor | None = None,
        embedder: Embedder | None = None,
        mode: GraphMode = "hybrid",
        damping: float = 0.85,
        max_iter: int = 100,
        ppr_floor_ratio: float = 0.01,
        rrf_k: int = 60,
        weights: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.extractor = extractor if extractor is not None else RuleExtractor()
        self.mode: GraphMode = mode
        self.damping = damping
        self.max_iter = max_iter
        self.ppr_floor_ratio = ppr_floor_ratio
        self.rrf_k = rrf_k
        self.weights = weights or {"ppr": 1.0, "dense": 1.0}

        self._embedder = embedder
        self._graphs: dict[str, nx.Graph] = {}
        self._items: dict[str, dict[str, MemoryItem]] = {}
        self._owner: dict[str, str] = {}
        self._vectors: dict[str, Any] = {}
        self._entities: dict[str, list[str]] = {}

    @property
    def embedder(self) -> Embedder:
        """惰性加载：纯 ppr 模式下不该为了一个用不上的模型付加载时间。"""
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    # ------------------------------------------------------------------
    # 图的构建
    # ------------------------------------------------------------------

    def _add(self, item: MemoryItem, *, user_id: str) -> str:
        graph = self._graphs.setdefault(user_id, nx.Graph())
        self._items.setdefault(user_id, {})[item.id] = item
        self._owner[item.id] = user_id

        mem_node = (MEM, item.id)
        graph.add_node(mem_node, kind=MEM)

        # 只抽一次。早期版本这里既调 extract_triples 又调 extract，
        # 用 LLM 抽取器时等于每条记忆付两次钱——实测 30 条记忆发了 84 次请求。
        # 这类重复调用在成本表出来之前完全看不出来。
        triples = self.extractor.extract_triples(item.content)
        entities = {t.subject for t in triples} | {t.object for t in triples}
        if not entities:
            # 句子里只有一个实体时抽不出三元组，这时才单独抽一次实体
            entities = {e.name for e in self.extractor.extract(item.content)}

        for name in entities:
            ent_node = (ENT, name)
            graph.add_node(ent_node, kind=ENT)
            # 记忆 → 实体：这条边让 PPR 的分数最终能落回记忆节点
            graph.add_edge(mem_node, ent_node, weight=1.0)

        for t in triples:
            a, b = (ENT, t.subject), (ENT, t.object)
            if a == b:
                continue
            graph.add_node(a, kind=ENT)
            graph.add_node(b, kind=ENT)
            # 实体 → 实体：共现越多，边越重，PPR 越容易走过去
            w = graph.get_edge_data(a, b, {}).get("weight", 0.0) + 1.0
            graph.add_edge(a, b, weight=w, predicate=t.predicate)

        # 抽出的实体是**派生索引数据**，不写进 item.metadata——
        # 那里属于用户，往里塞实现细节会污染 update 的浅合并语义
        # （契约测试 test_更新走浅合并 正是这么抓到这个 bug 的）。
        # 和 VectorMemory 把向量放在 _vectors 里是同一个理由。
        self._entities[item.id] = sorted(entities)

        if self.mode in ("dense", "hybrid"):
            self._vectors[item.id] = self.embedder.encode_one(item.content)
        return item.id

    def add_many(self, items: list[MemoryItem | str], *, user_id: str) -> list[str]:
        """批量写入：向量一次性编码，图逐条构建。"""
        self._check_user_id(user_id)
        objs = [MemoryItem(it) if isinstance(it, str) else it for it in items]
        objs = [o.copy_with(user_id=user_id) for o in objs]

        with self._meter.measure(op="add", store=self.name, user_id=user_id) as rec:
            if self.mode in ("dense", "hybrid"):
                vecs = self.embedder.encode([o.content for o in objs])
                for obj, vec in zip(objs, vecs, strict=True):
                    self._vectors[obj.id] = vec
            for obj in objs:
                self._add_no_vector(obj, user_id=user_id)
            rec.n_items = len(objs)
        return [o.id for o in objs]

    def _add_no_vector(self, item: MemoryItem, *, user_id: str) -> None:
        """add_many 里已经批量算过向量了，这里只建图。"""
        saved_mode, self.mode = self.mode, "ppr"
        try:
            self._add(item, user_id=user_id)
        finally:
            self.mode = saved_mode

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def _search(self, query: str, *, user_id: str, k: int, **kwargs: Any) -> list[SearchResult]:
        items = self._items.get(user_id)
        if not items:
            return []

        mode: GraphMode = kwargs.get("mode", self.mode)
        cand_k = max(k * 4, 20)

        channels: dict[str, list[str]] = {}
        raw: dict[str, dict[str, float]] = {}

        if mode in ("ppr", "hybrid"):
            ids, scores = self._ppr_rank(query, user_id=user_id, cand_k=cand_k)
            if ids:
                channels["ppr"] = ids
                raw["ppr"] = dict(zip(ids, scores, strict=True))

        if mode in ("dense", "hybrid"):
            ids, scores = self._dense_rank(query, user_id=user_id, cand_k=cand_k)
            channels["dense"] = ids
            raw["dense"] = dict(zip(ids, scores, strict=True))

        if not channels:
            # 纯 ppr 模式下查询抽不出实体，只能返回空。
            # 这是图检索的一个真实弱点，正文会展开——也是默认用 hybrid 的原因。
            return []

        if len(channels) == 1:
            ch = next(iter(channels))
            fused = {i: raw[ch][i] for i in channels[ch]}
        else:
            fused = reciprocal_rank_fusion(channels, k=self.rrf_k, weights=self.weights)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        results = []
        for mid, score in ranked:
            debug: dict[str, Any] = {f"{ch}_score": round(raw[ch].get(mid, 0.0), 6) for ch in raw}
            debug["entities"] = self._entities.get(mid, [])
            results.append(
                SearchResult(item=items[mid], score=float(score), source=mode, debug=debug)
            )
        return results

    def _ppr_rank(self, query: str, *, user_id: str, cand_k: int) -> tuple[list[str], list[float]]:
        graph = self._graphs.get(user_id)
        if graph is None or graph.number_of_nodes() == 0:
            return [], []

        seeds = [(ENT, e.name) for e in self.extractor.extract(query) if (ENT, e.name) in graph]
        if not seeds:
            return [], []

        personalization = dict.fromkeys(seeds, 1.0 / len(seeds))
        scores = nx.pagerank(
            graph,
            alpha=self.damping,
            personalization=personalization,
            max_iter=self.max_iter,
            weight="weight",
        )

        mem_scores = [(node[1], s) for node, s in scores.items() if node[0] == MEM and s > 0]
        mem_scores.sort(key=lambda kv: kv[1], reverse=True)
        if not mem_scores:
            return [], []

        # 门槛：PPR 给图上每个连通节点都会分到一点分数，哪怕小到 1e-8。
        # 不滤掉的话，这些噪声会以「第 3 名」的身份进入 RRF，挤掉真结果。
        floor = mem_scores[0][1] * self.ppr_floor_ratio
        kept = [(m, s) for m, s in mem_scores[:cand_k] if s >= floor]
        return [m for m, _ in kept], [s for _, s in kept]

    def _dense_rank(
        self, query: str, *, user_id: str, cand_k: int
    ) -> tuple[list[str], list[float]]:
        import numpy as np

        items = self._items.get(user_id, {})
        ids = [i for i in items if i in self._vectors]
        if not ids:
            return [], []
        matrix = np.vstack([self._vectors[i] for i in ids])
        sims = matrix @ self.embedder.encode_one(query)
        order = np.argsort(-sims)[:cand_k]
        return [ids[i] for i in order], [float(sims[i]) for i in order]

    # ------------------------------------------------------------------
    # 观察工具（教学用）
    # ------------------------------------------------------------------

    def explain(self, query: str, *, user_id: str, top_n: int = 8) -> str:
        """把一次 PPR 检索的中间过程摊开：种子是谁、分数流到了哪。"""
        graph = self._graphs.get(user_id)
        if graph is None:
            return "（图是空的）"

        seeds = [(ENT, e.name) for e in self.extractor.extract(query) if (ENT, e.name) in graph]
        lines = [f"查询：{query}", f"抽到的种子实体：{[s[1] for s in seeds] or '（无）'}"]
        if not seeds:
            lines.append("→ 抽不出实体，PPR 无从下手。这正是需要 hybrid 兜底的原因。")
            return "\n".join(lines)

        scores = nx.pagerank(
            graph,
            alpha=self.damping,
            personalization=dict.fromkeys(seeds, 1.0 / len(seeds)),
            max_iter=self.max_iter,
            weight="weight",
        )
        ent_scores = sorted(
            ((n[1], s) for n, s in scores.items() if n[0] == ENT), key=lambda kv: -kv[1]
        )[:top_n]
        seed_names = {s[1] for s in seeds}
        lines.append(f"\n分数传播到的实体（前 {top_n} 个）：")
        for name, s in ent_scores:
            hop = "种子" if name in seed_names else "传播"
            lines.append(f"  {s:.5f}  [{hop}] {name}")
        return "\n".join(lines)

    def stats(self, *, user_id: str) -> dict[str, int]:
        graph = self._graphs.get(user_id, nx.Graph())
        ents = [n for n in graph.nodes if n[0] == ENT]
        mems = [n for n in graph.nodes if n[0] == MEM]
        no_entity = sum(1 for n in mems if graph.degree(n) == 0)
        return {
            "记忆节点": len(mems),
            "实体节点": len(ents),
            "边": graph.number_of_edges(),
            "没抽到实体的记忆": no_entity,
        }

    # ------------------------------------------------------------------

    def _update(self, memory_id: str, patch: dict[str, Any], *, user_id: str) -> None:
        items = self._locate(memory_id, user_id)
        item = items[memory_id]
        new_item = self.apply_patch(item, patch)
        items[memory_id] = new_item

        if "content" in patch:
            # 内容变了，实体也会变——必须重建这条记忆的边，否则图里留着旧关系
            self._detach(memory_id, user_id=user_id)
            items[memory_id] = new_item
            self._add(new_item, user_id=user_id)

    def _delete(self, memory_id: str, *, user_id: str) -> None:
        items = self._locate(memory_id, user_id)
        self._detach(memory_id, user_id=user_id)
        items.pop(memory_id, None)
        self._owner.pop(memory_id, None)
        self._vectors.pop(memory_id, None)
        self._entities.pop(memory_id, None)

    def _detach(self, memory_id: str, *, user_id: str) -> None:
        """把记忆节点从图里摘掉，并清理因此变成孤儿的实体节点。

        不清理孤儿实体的话，图会越用越脏：删掉所有提到某实体的记忆后，
        那个实体仍然留在图里当种子，把 PPR 的分数引向不存在的地方。
        """
        graph = self._graphs.get(user_id)
        if graph is None:
            return
        mem_node = (MEM, memory_id)
        if mem_node not in graph:
            return
        neighbors = list(graph.neighbors(mem_node))
        graph.remove_node(mem_node)
        for ent in neighbors:
            if ent[0] == ENT and not any(n[0] == MEM for n in graph.neighbors(ent)):
                graph.remove_node(ent)

    def _all(self, *, user_id: str) -> list[MemoryItem]:
        return list(self._items.get(user_id, {}).values())

    def _locate(self, memory_id: str, user_id: str) -> dict[str, MemoryItem]:
        owner = self._owner.get(memory_id)
        if owner is None:
            raise MemoryNotFoundError(memory_id)
        if owner != user_id:
            raise CrossUserAccessError(f"记忆 {memory_id} 不属于用户 {user_id}")
        return self._items[user_id]
