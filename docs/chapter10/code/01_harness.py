# offline-ok
"""第 10 章实验一：把效果和代价放进同一张表。

    python docs/chapter10/code/01_harness.py

这是全书的收口实验。第 1 章定的 `MemoryStore` 接口、`Meter` 计量器，
到这里兑现：任意两种实现在同一份数据上跑出七列可比的数字。

注意表里同时有「召回」和「ctx token」。**只报前者不报后者的对比，
在本书里视为无效结论**——因为你总可以把 k 调大、把粒度合粗，
用更多的 token 换更高的召回率。
"""

from __future__ import annotations

import warnings

from minimem import BufferMemory, VectorMemory
from minimem.eval import EvalHarness
from minimem.utils.embedding import FakeEmbedder, get_embedder


def main() -> None:
    print("\n第 10 章实验一：统一评测")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        embedder = get_embedder()
        degraded = any(issubclass(w.category, RuntimeWarning) for w in caught)

    if degraded:
        print("  ⚠️  未加载到真实句向量模型，向量方案的结果不具参考价值。")
        print('     国内：pip install -e ".[vector]" && python -m minimem.utils.fetch_model\n')
    else:
        print(f"  句向量模型：{embedder.name}\n")

    harness = EvalHarness(k=5)
    results = harness.compare(
        {
            "buffer（第 1 章）": lambda m: BufferMemory(meter=m),
            "vector-fake": lambda m: VectorMemory(embedder=FakeEmbedder(), meter=m),
            "vector-dense": lambda m: VectorMemory(embedder=embedder, mode="dense", meter=m),
            "vector-hybrid": lambda m: VectorMemory(embedder=embedder, mode="hybrid", meter=m),
        }
    )

    print(harness.table(results))
    print("\n  分能力召回率")
    print(harness.capability_table(results))

    print("\n  拒答题：检索层返回空结果的比例")
    for r in results:
        print(f"    {r.label:<20}{r.abstain_rate:>6.0%}")

    print(
        """
  读法：

  · **先看最后那组数字**：所有方案在拒答题上的「返回空」比例都是 0%。
    也就是说，当用户问「我的血型是什么」——一个记忆里完全没有的信息——
    每种实现都老老实实返回了 5 条不相关的记忆，然后把它们塞进 prompt。

    **检索层没有拒答能力**。它的工作是「找最像的」，而「最像的」永远存在。
    拒答必须由别的地方来做：一个相似度阈值、一个「够不够回答」的判断、
    或者生成侧的指令。多数系统这三样都没有，于是模型看着 5 条无关记忆
    编出一个答案——这是记忆系统里最常见的幻觉来源。

  · 「可答」列（answerable）**不是端到端准确率**。
    它只回答「答案的关键词有没有出现在召回的文本里」，
    不回答「模型有没有用好这些文本」。真实的端到端准确率总是更低，
    差距来自：模型没注意到、被噪声带偏、或者召回里有互相矛盾的内容。

  · 把「召回」和「ctx token」并排看。如果一个方案召回率高 3 个百分点，
    但 token 多一倍，那不是改进，是换了个花钱方式。

  · 成本列在这里全是 0，因为本章的实现都不调 LLM。
    到第 6~8 章，这一列会变成主角——那时你会看到某些「聪明」的方案
    每写一条记忆要花掉几千 token。
"""
    )


if __name__ == "__main__":
    main()
