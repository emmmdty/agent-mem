# offline-ok
"""第 3 章实验四：规模墙推倒了吗？

    python docs/chapter3/code/04_scale_and_index.py

第 1 章测出 BufferMemory 的检索延迟随记忆条数线性增长。
换成向量检索之后呢？

答案是：**复杂度没变，常数变了**。

VectorMemory 的暴力检索仍然要算查询向量与全部 N 条记忆的内积，还是 O(N)。
但这一步是一次 numpy 矩阵乘法，走 BLAS，而不是 Python 循环。
在几万到几十万条的量级上，这个常数因子的改进就已经够用了。

真正的次线性检索需要 ANN（近似最近邻）索引，代价是**牺牲召回**。
本脚本最后会说明什么时候才该付这个代价。

（写这个实验时踩到过两个坑，都保留在了代码里：一是用 ``argsort`` 对全部候选
排序而不是 ``argpartition`` 取前 k；二是纯 dense 模式下也无条件构建了 BM25 索引。
两处修完，50000 条的检索延迟从 35ms 降到 10ms。
**检索的瓶颈经常不在你以为的地方**——所以才要测。）
"""

from __future__ import annotations

import random
import time

from minimem import BufferMemory
from minimem.base import MemoryItem
from minimem.utils.embedding import FakeEmbedder
from minimem.utils.metering import Meter
from minimem.vector import VectorMemory

USER = "u1"
SIZES = [1_000, 5_000, 20_000, 50_000]
N_QUERIES = 20

TOPICS = ["项目进展", "客户拜访", "风控模型", "周末计划", "读书笔记", "系统告警", "面试安排"]


def make_corpus(n: int) -> list[MemoryItem]:
    random.seed(42)
    return [
        MemoryItem(f"{random.choice(TOPICS)}：第 {i} 条记录，涉及若干细节与后续安排。")
        for i in range(n)
    ]


def bench(store, corpus: list[MemoryItem]) -> tuple[float, float]:
    """返回 (写入总耗时秒, 平均检索毫秒)。"""
    meter = store._meter  # noqa: SLF001 —— 教学脚本，直接读计量器
    meter.reset()

    t0 = time.perf_counter()
    store.add_many(list(corpus), user_id=USER)
    write_s = time.perf_counter() - t0

    meter.reset()
    for i in range(N_QUERIES):
        store.search(f"{TOPICS[i % len(TOPICS)]}的进展如何", user_id=USER, k=5)
    return write_s, meter.summary(op="search")["latency_ms_mean"]


def main() -> None:
    print("\n第 3 章实验四：规模与延迟")
    print("为排除模型推理时间的干扰，这里统一用 FakeEmbedder——")
    print("我们要比的是**索引与检索结构**，不是嵌入模型的速度。\n")

    print(
        f"  {'条数':<10}{'Buffer 检索(ms)':>18}{'Vector 检索(ms)':>18}{'加速':>10}{'Vector 写入(s)':>16}"
    )
    print("  " + "-" * 74)

    for n in SIZES:
        corpus = make_corpus(n)

        buf = BufferMemory(meter=Meter())
        _, buf_ms = bench(buf, corpus)

        vec = VectorMemory(embedder=FakeEmbedder(), mode="dense", meter=Meter())
        vec_write, vec_ms = bench(vec, corpus)

        speedup = f"{buf_ms / vec_ms:.0f}×" if vec_ms > 0 else "—"
        print(f"  {n:<10,}{buf_ms:>18.2f}{vec_ms:>18.3f}{speedup:>10}{vec_write:>16.2f}")

    print(
        """
  读法：

  · 两条曲线**都是线性的**——向量检索并没有把复杂度降下来。
    它降的是常数：一次 numpy 矩阵乘法（BLAS、SIMD、连续内存）
    对上一个 Python for 循环，本机实测约一个数量级以上的差距。
    具体倍数取决于你的 CPU 和 BLAS 实现，但方向是稳定的。

  · 别忽略最后一列：**写入变贵了**。
    每条记忆都要过一次嵌入模型（这里用的是零成本的假模型，
    真实模型在 CPU 上大约每条几毫秒），还要维护索引。
    第 1 章的 BufferMemory 写入几乎免费。

    这是本书会反复出现的模式：**检索侧的省，往往是写入侧的花**。
    判断一个方案划不划算，要看你的**读写比**。

  · 什么时候需要 ANN（HNSW / IVF）？
    经验分界大约在**百万条**：暴力检索在这个量级上单次要几十到几百毫秒，
    已经超出交互式应用的预算。ANN 能把它压到毫秒级，
    代价是**召回不再是 100%**——它是近似最近邻，会漏。

    对记忆系统来说这个取舍尤其要小心：漏掉一条文档，用户可能不察觉；
    漏掉「我对花生过敏」，后果是另一个量级。
    所以常见做法是分层：热层精确、冷层近似（第 7 章）。

  · 单用户五万条记忆是什么概念？一个每天聊 100 轮的重度用户，大约一年半。
    多数产品在触碰到 ANN 的必要性之前，会先撞上另外两堵墙：
    **记忆质量**（一堆没整理过的原始消息）和**成本**（每次检索的上下文越来越贵）。
    先解决那两个。
"""
    )


if __name__ == "__main__":
    main()
