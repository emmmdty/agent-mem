"""token 估算：让「这一步花多少 token」随时可算。

优先使用 ``tiktoken`` 精确计数；未安装时退回一个启发式估算，误差通常在 ±15% 以内，
足够用来判断量级——而判断量级正是本书需要的：你要分辨的是「2 千 token」和
「2 万 token」的差别，不是「2000」和「2050」的差别。
"""

from __future__ import annotations

import os
from functools import lru_cache

__all__ = ["estimate_tokens", "count_tokens", "estimate_cost", "TokenCost"]


@lru_cache(maxsize=1)
def _get_encoder():
    """尝试加载 tiktoken 编码器；失败返回 None（不抛异常）。"""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 —— 没装 / 下载失败都退回启发式
        return None


def estimate_tokens(text: str) -> int:
    """启发式 token 估算，不依赖任何外部包。

    经验规则：CJK 字符约 1 token 一个（有时 1 个字会被切成 2 个 token，
    但取 1 已足够判断量级），其余按 4 字符 1 token 折算。

    >>> estimate_tokens("你好") >= 2
    True
    >>> estimate_tokens("hello world") >= 2
    True
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿" or "　" <= ch <= "〿")
    other = len(text) - cjk
    return cjk + max(1, round(other / 4)) if other else cjk


def count_tokens(text: str, *, exact: bool = True) -> int:
    """token 计数。``exact=True`` 且 tiktoken 可用时给出精确值。

    Note:
        cl100k_base 是 GPT-3.5/4 系列的编码。不同模型的分词器不同，
        换模型时绝对数值会变（中文差异尤其明显），但量级结论通常仍成立。
    """
    if exact:
        enc = _get_encoder()
        if enc is not None:
            return len(enc.encode(text))
    return estimate_tokens(text)


class TokenCost:
    """把 token 数换算成钱。

    价格从环境变量读取（单位：美元 / 每百万 token），默认值仅为占位，
    请按你实际使用的模型改 ``.env``。价格变动频繁，本书正文不写死任何单价。
    """

    def __init__(self, price_in: float | None = None, price_out: float | None = None) -> None:
        self.price_in = (
            price_in if price_in is not None else float(os.getenv("AGENT_MEM_PRICE_IN", "0.15"))
        )
        self.price_out = (
            price_out if price_out is not None else float(os.getenv("AGENT_MEM_PRICE_OUT", "0.60"))
        )

    def __call__(self, tokens_in: int, tokens_out: int = 0) -> float:
        return (tokens_in * self.price_in + tokens_out * self.price_out) / 1_000_000


def estimate_cost(tokens_in: int, tokens_out: int = 0) -> float:
    """用默认价格估算成本（美元）。"""
    return TokenCost()(tokens_in, tokens_out)
