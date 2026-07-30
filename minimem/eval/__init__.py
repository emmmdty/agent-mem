"""评测：把「效果」和「代价」放进同一张表。

- ``dataset``：MiniBench，一个刻意做得很小的中文记忆评测集
- ``metrics``：召回率 / 命中率 / 准确率 / MRR，以及各自的盲区
- ``harness``：第 10 章的评测执行器（把检索指标与 Meter 的成本数据合并）
"""

from minimem.eval.dataset import (
    CAPABILITIES,
    MINI_MEMORIES,
    MINI_QUERIES,
    MemoryRecord,
    Query,
    as_memory_items,
    load_mini_bench,
)
from minimem.eval.metrics import aggregate, hit_rate, mrr, precision_at_k, recall_at_k

__all__ = [
    "MemoryRecord",
    "Query",
    "MINI_MEMORIES",
    "MINI_QUERIES",
    "CAPABILITIES",
    "load_mini_bench",
    "as_memory_items",
    "recall_at_k",
    "hit_rate",
    "precision_at_k",
    "mrr",
    "aggregate",
]
