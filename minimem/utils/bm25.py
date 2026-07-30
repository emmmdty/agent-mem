"""BM25：一个仍然打不过的老基线。

为什么在 2026 年的记忆教程里还要写 BM25？因为它在两类查询上稳定优于稠密向量：

- **专有名词**：人名、订单号、型号、代码标识符。向量模型会把「XR-2049」和
  「XR-2050」编码到几乎同一个点上，BM25 不会。
- **罕见词**：训练语料里没怎么出现的词，向量表示往往是噪声。

而这两类恰恰是记忆场景的高频需求——用户问的常常是「我上次说的那个项目编号」。
所以第 3 章的做法不是「向量取代关键词」，而是**两路都跑，再融合**。

本实现约 60 行，够教学用。生产环境请用 Elasticsearch / OpenSearch / rank_bm25，
它们在分词、增量索引、持久化上做了本实现完全没做的事。
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

__all__ = ["BM25", "tokenize"]

_WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-.]*")
_CJK_RE = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    """中英混合分词。

    - 英文与数字按词切（保留 ``XR-2049`` 这样的标识符不被拆开）；
    - 中文切成**字符 bigram**（「信贷风控」→ 信贷 / 贷风 / 风控）。

    为什么中文用 bigram 而不是 jieba 之类的分词器？两个原因：
    一是零依赖，二是 bigram 对未登录词更鲁棒——新出现的产品名、人名不会被切错。
    代价是词表更大、有一定噪声。真实项目里建议评估后再决定。
    """
    text = text.lower()
    tokens = _WORD_RE.findall(text)
    cjk = _CJK_RE.findall(text)
    # 相邻的中文字符组成 bigram；单字也保留，否则单字查询会完全无法匹配
    cjk_str = "".join(cjk)
    tokens.extend(cjk)
    tokens.extend(cjk_str[i : i + 2] for i in range(len(cjk_str) - 1))
    return tokens


class BM25:
    """Okapi BM25。

    评分公式：

    .. math::

        \\text{score}(q, d) = \\sum_{t \\in q} \\text{IDF}(t) \\cdot
        \\frac{f(t, d) \\cdot (k_1 + 1)}{f(t, d) + k_1 \\cdot (1 - b + b \\cdot
        \\frac{|d|}{\\text{avgdl}})}

    两个参数的直观含义：

    - ``k1`` 控制词频饱和。一个词在文档里出现 10 次，不该比出现 5 次「相关两倍」。
      k1 越小饱和越快。
    - ``b`` 控制长度惩罚。b=1 时完全按长度归一化，b=0 时不惩罚长文档。

    默认值 (1.5, 0.75) 是检索领域的经验值，对大多数语料够用。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs_tokens: list[list[str]] = []
        self.doc_freqs: list[Counter[str]] = []
        self.idf: dict[str, float] = {}
        self.doc_lens: np.ndarray = np.zeros(0)
        self.avgdl: float = 0.0

    def fit(self, docs: list[str]) -> BM25:
        """重建索引。

        注意这是**全量重建**：每加一条记忆就重算一次的话，写入复杂度是 O(N)。
        真实系统用倒排索引做增量更新。第 3 章的实验会把这个代价测出来，
        这也是 `VectorMemory` 采用惰性重建（写入时打脏标记，检索时按需重建）的原因。
        """
        self.docs_tokens = [tokenize(d) for d in docs]
        self.doc_freqs = [Counter(t) for t in self.docs_tokens]
        self.doc_lens = np.array([len(t) for t in self.docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0

        n = len(docs)
        df: Counter[str] = Counter()
        for tokens in self.docs_tokens:
            df.update(set(tokens))
        # 带平滑的 IDF，保证非负：罕见词权重高，几乎每篇都有的词权重趋近 0
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        return self

    def score(self, query: str) -> np.ndarray:
        """返回查询对每篇文档的得分（未归一化）。"""
        if not self.docs_tokens:
            return np.zeros(0, dtype=np.float32)

        scores = np.zeros(len(self.docs_tokens), dtype=np.float32)
        q_tokens = tokenize(query)
        if not q_tokens:
            return scores

        for token in set(q_tokens):
            idf = self.idf.get(token)
            if idf is None:
                continue
            freqs = np.array([f.get(token, 0) for f in self.doc_freqs], dtype=np.float32)
            denom = freqs + self.k1 * (1 - self.b + self.b * self.doc_lens / max(self.avgdl, 1e-9))
            scores += idf * (freqs * (self.k1 + 1)) / np.maximum(denom, 1e-9)
        return scores
