"""句向量：把「意思」变成可计算的东西。

本模块提供一个统一的 ``Embedder`` 接口和三个实现：

=========================  ==================  ==================================
实现                        依赖                用途
=========================  ==================  ==================================
``SentenceTransformer…``   torch + st          **国内推荐**：配合 ModelScope 下载
``FastEmbedEmbedder``      fastembed           轻量：onnxruntime，无需 torch，
                                               但模型走 HuggingFace，国内常不通
``FakeEmbedder``           无                  离线兜底 / 对照实验
=========================  ==================  ==================================

**模型从哪来**：国内 HuggingFace 通常不可达，因此本书默认走 **ModelScope**。
先用内置命令把模型拉到本地，再让 ``AGENT_MEM_EMBED_MODEL`` 指向那个目录::

    python -m minimem.utils.fetch_model        # 从 ModelScope 下载到 ./models
    export AGENT_MEM_EMBED_MODEL=./models/...  # 命令会打印出确切的路径

``get_embedder()`` 按以下顺序选择，任一步成功即返回：

1. 取值为 ``"fake"`` → ``FakeEmbedder``
2. 取值是一个**存在的本地目录** → 用 sentence-transformers 加载它
3. 已装 modelscope → 自动下载到本地缓存再加载
4. fastembed（onnx，走 HuggingFace）
5. sentence-transformers 直连 HuggingFace
6. 全部失败 → ``FakeEmbedder`` **并打印警告**

最后一步是降级而不是抛异常，为的是让教程脚本在任何环境下都能跑完流程；
警告则是为了让你知道——**FakeEmbedder 下的检索效果没有任何参考价值**。

FakeEmbedder 本身也是教具：第 3 章实验一用它做对照，
量化「向量质量」到底贡献了多少，而不是空谈「embedding 很重要」。
"""

from __future__ import annotations

import hashlib
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

__all__ = [
    "Embedder",
    "FakeEmbedder",
    "FastEmbedEmbedder",
    "SentenceTransformerEmbedder",
    "get_embedder",
    "cosine_similarity",
    "download_from_modelscope",
    "DEFAULT_MODEL",
]

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
#: 模型的本地存放目录，已在 .gitignore 中忽略
LOCAL_MODEL_DIR = Path(os.getenv("AGENT_MEM_MODEL_DIR", "./models"))


class Embedder(ABC):
    """把文本编码成单位长度向量。

    约定所有实现都返回 **L2 归一化**后的向量，这样余弦相似度就等于内积，
    检索时一次矩阵乘法即可，不必反复算范数。
    """

    name: str = "embedder"
    dim: int = 0

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """返回形状 ``(len(texts), dim)`` 的归一化向量矩阵。"""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} dim={self.dim}>"


class FakeEmbedder(Embedder):
    """哈希向量：确定性、零依赖、**没有语义**。

    做法是把文本切成字符 n-gram，每个 n-gram 用哈希映射到一个固定维度上累加。
    它保留了「字面重叠」的信息（共享 n-gram 越多，向量越接近），
    但完全不理解同义——所以它的行为接近第 1 章的 `BufferMemory`，
    只是换了一种表示方式。

    这正是它作为对照组的价值：如果某个改进在 FakeEmbedder 下也能生效，
    那这个改进就跟语义无关。
    """

    name = "fake"

    def __init__(self, dim: int = 256, ngram: int = 2) -> None:
        self.dim = dim
        self.ngram = ngram

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for gram in self._grams(text):
                h = hashlib.md5(gram.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] % 2 == 0 else -1.0
                mat[i, idx] += sign
        return self._normalize(mat)

    def _grams(self, text: str) -> list[str]:
        text = text.strip().lower()
        if len(text) < self.ngram:
            return [text] if text else []
        return [text[j : j + self.ngram] for j in range(len(text) - self.ngram + 1)]


class FastEmbedEmbedder(Embedder):
    """基于 fastembed（onnxruntime）的实现。

    相比 sentence-transformers 的优势是**不需要 torch**：安装体积从约 2GB 降到
    约 100MB，CPU 推理速度也更快。代价是可选模型少一些。
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name)
        self.dim = len(next(iter(self._model.embed(["探测维度"]))))

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        return self._normalize(mat)


class SentenceTransformerEmbedder(Embedder):
    """基于 sentence-transformers 的实现（需要 torch）。"""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return self._normalize(np.asarray(mat, dtype=np.float32))


def download_from_modelscope(
    model_name: str = DEFAULT_MODEL, *, cache_dir: str | Path | None = None
) -> Path:
    """从 ModelScope 下载模型，返回本地快照目录。

    国内访问 ModelScope 通常比 HuggingFace 快一到两个数量级，
    所以本书把它作为默认下载源。

    Raises:
        ImportError: 未安装 modelscope。
    """
    from modelscope import snapshot_download

    cache_dir = Path(cache_dir) if cache_dir is not None else LOCAL_MODEL_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(snapshot_download(model_name, cache_dir=str(cache_dir)))


def _find_local_snapshot(model_name: str, root: Path = LOCAL_MODEL_DIR) -> Path | None:
    """在本地模型目录里找已下载好的快照，避免每次都联网询问。"""
    if not root.exists():
        return None
    # modelscope 的目录结构：<cache>/models/<org>--<name>/snapshots/<rev>
    marker = model_name.replace("/", "--")
    for path in root.rglob("*"):
        if path.is_dir() and marker in str(path) and (path / "config.json").exists():
            return path
    return None


def get_embedder(model_name: str | None = None, *, quiet: bool = False) -> Embedder:
    """按配置返回一个可用的 Embedder，必要时降级。

    选择顺序见模块文档。降级而不是抛异常，是为了让教程脚本在任何环境下都能跑完；
    警告则是为了让你知道自己看到的结果不能当真。
    """
    model_name = model_name or os.getenv("AGENT_MEM_EMBED_MODEL", DEFAULT_MODEL)

    if model_name == "fake":
        return FakeEmbedder()

    errors: list[str] = []

    # 1) 本地目录：用户已经用 fetch_model 或 git lfs 下载好了
    local = Path(model_name)
    if local.is_dir():
        try:
            return SentenceTransformerEmbedder(str(local))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"本地目录加载失败: {type(exc).__name__}")

    # 2) 本地缓存里已有的 ModelScope 快照
    cached = _find_local_snapshot(model_name)
    if cached is not None:
        try:
            return SentenceTransformerEmbedder(str(cached))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"本地快照加载失败: {type(exc).__name__}")

    # 3) ModelScope 在线下载（国内首选）
    try:
        path = download_from_modelscope(model_name)
        return SentenceTransformerEmbedder(str(path))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"ModelScope: {type(exc).__name__}")

    # 4~5) HuggingFace 系（海外网络可用时）
    for cls in (FastEmbedEmbedder, SentenceTransformerEmbedder):
        try:
            return cls(model_name)
        except Exception as exc:  # noqa: BLE001 —— 依赖缺失、下载失败都要能降级
            errors.append(f"{cls.__name__}: {type(exc).__name__}")

    if not quiet:
        warnings.warn(
            "无法加载真实句向量模型（" + "；".join(errors) + "），已降级为 FakeEmbedder。\n"
            "  流程能跑通，但**检索效果不具参考价值**。\n"
            "  国内推荐：pip install modelscope sentence-transformers\n"
            "            python -m minimem.utils.fetch_model\n"
            "  海外网络：pip install fastembed",
            RuntimeWarning,
            stacklevel=2,
        )
    return FakeEmbedder()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """余弦相似度。``a`` 可以是单个向量或矩阵，``b`` 是矩阵。

    因为 ``Embedder`` 保证输出已归一化，这里直接做内积；
    但为了让这个函数也能用在未归一化的向量上，仍然除了一次范数。
    """
    a = np.atleast_2d(a)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    b_norm[b_norm == 0] = 1.0
    return (a / a_norm) @ (b / b_norm).T
