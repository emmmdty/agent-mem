"""LLM 客户端：第 6 章起，记忆系统开始自己调用模型。

从第 6 章开始，记忆不再只是被动存取——它要**判断**：这条消息里有什么事实？
新信息和旧记忆冲突吗？这段经历值得总结成一条技能吗？这些判断需要 LLM。

本模块提供两个实现：

======================  ==========================================================
``OpenAIClient``        真实调用。任何 OpenAI 兼容接口（含国内各家）
``ScriptedLLM``         离线模拟。按 prompt 里的标记路由到预设逻辑
======================  ==========================================================

``ScriptedLLM`` 的定位必须说清楚，否则会误导：

- 它**是**一个让流程跑通、让实验可复现、让 CI 能跑的工具；
- 它**不是**对真实 LLM 能力的模拟。它不会推理、不会处理意料之外的输入、
  不会犯 LLM 那些有意思的错。

所以每一章用到它时都会明确标注，并且会指出「换成真模型后哪些结论会变」。
凡是结论依赖 LLM 判断力的地方，本书都要求你用真模型跑一遍。

**成本计量**：``OpenAIClient`` 会把每次调用的 token 与估算金额记进 ``Meter``，
第 10 章的 harness 直接消费这些数据。这是「代价必须一等呈现」在代码层的落实——
到第 6~8 章，成本那一列会从 0 变成主角。
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from minimem.utils.metering import Meter, get_default_meter
from minimem.utils.tokens import TokenCost, count_tokens

__all__ = [
    "LLMResult",
    "LLMClient",
    "OpenAIClient",
    "ScriptedLLM",
    "get_llm",
    "extract_json",
]


@dataclass
class LLMResult:
    """一次调用的结果与用量。"""

    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model: str = ""
    raw: Any = None

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def json(self, default: Any = None) -> Any:
        """把回复解析成 JSON。**失败返回 default 而不是抛异常**。

        这不是偷懒——LLM 返回不合法 JSON 是常态而非例外（多一段解释、
        包在 markdown 代码块里、尾随逗号）。记忆系统的写入路径上，
        一次解析失败不该让整条消息丢失。第 6 章会专门讨论这个失败模式的代价。
        """
        return extract_json(self.text, default=default)


def extract_json(text: str, default: Any = None) -> Any:
    """从可能夹带解释文字或 markdown 围栏的回复里抠出 JSON。"""
    if not text:
        return default

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)

    # 再试一次「第一个 { 到最后一个 }」——覆盖模型在 JSON 前后加话的情况
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])
    first, last = text.find("["), text.rfind("]")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])

    for cand in candidates:
        try:
            return json.loads(cand.strip())
        except (json.JSONDecodeError, ValueError):
            continue
    return default


class LLMClient(ABC):
    """LLM 客户端接口。"""

    name: str = "llm"

    def __init__(self, *, meter: Meter | None = None, store_label: str = "llm") -> None:
        self._meter = meter if meter is not None else get_default_meter()
        self._store_label = store_label
        self.call_count = 0

    @abstractmethod
    def _complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResult: ...

    def complete(
        self,
        messages: list[dict[str, str]] | str,
        *,
        op: str = "llm",
        **kwargs: Any,
    ) -> LLMResult:
        """发起一次调用并计量。

        Args:
            messages: OpenAI 风格的消息列表，或直接给一个字符串（当作 user 消息）。
            op: 记进计量器的操作名，如 ``"extract"`` / ``"reflect"``。
                **给它一个有意义的名字**——第 10 章的报表按 op 分组，
                「写入侧的抽取花了多少」和「检索侧的重排花了多少」需要分开看。
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        self.call_count += 1
        result = self._complete(messages, **kwargs)
        self._meter.add_llm_usage(
            op=op,
            store=self._store_label,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
        )
        return result

    @property
    def is_real(self) -> bool:
        """是否是真实模型。教学脚本据此决定要不要打免责提示。"""
        return True


class OpenAIClient(LLMClient):
    """任何 OpenAI 兼容接口。

    配置从环境变量读（见 ``.env.example``）：
    ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` / ``AGENT_MEM_MODEL``。
    换供应商（DeepSeek、通义千问、Kimi、本地 Ollama）只需改后两个。
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        from openai import OpenAI

        self.model = model or os.getenv("AGENT_MEM_MODEL", "gpt-4o-mini")
        self.name = self.model
        self.temperature = temperature
        self._cost = TokenCost()
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=timeout,
        )

    def _complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResult:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=kwargs.pop("temperature", self.temperature),
            **kwargs,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        tin = usage.prompt_tokens if usage else sum(count_tokens(m["content"]) for m in messages)
        tout = usage.completion_tokens if usage else count_tokens(text)
        return LLMResult(
            text=text,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=self._cost(tin, tout),
            model=self.model,
            raw=resp,
        )


@dataclass
class _Handler:
    marker: str
    fn: Callable[[str], str]


class ScriptedLLM(LLMClient):
    """离线模拟：按 prompt 里的标记路由到一段预设逻辑。

    用法：各章注册自己的处理器，标记通常就是 prompt 模板里的一个固定串。

        llm = ScriptedLLM()
        llm.register("抽取事实", lambda prompt: json.dumps(...))

    没有任何标记命中时，返回 ``fallback``（默认一个空 JSON 对象），
    **并且计入 ``unmatched`` 计数**——这个计数很重要：
    如果某个实验的 unmatched 很高，说明它其实没在测你以为的东西。

    Args:
        strict: 为 True 时，未命中直接抛异常。写测试时建议开启，
            避免「模拟器悄悄返回空值，测试却通过了」这种假绿灯。
    """

    name = "scripted"

    def __init__(self, *, fallback: str = "{}", strict: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fallback = fallback
        self.strict = strict
        self.unmatched = 0
        self._handlers: list[_Handler] = []
        self.history: list[str] = []

    def register(self, marker: str, fn: Callable[[str], str]) -> ScriptedLLM:
        self._handlers.append(_Handler(marker=marker, fn=fn))
        return self

    def _complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResult:
        prompt = "\n".join(m["content"] for m in messages)
        self.history.append(prompt)

        text = None
        for handler in self._handlers:
            if handler.marker in prompt:
                text = handler.fn(prompt)
                break

        if text is None:
            self.unmatched += 1
            if self.strict:
                raise LookupError(
                    f"ScriptedLLM 没有匹配的处理器。prompt 开头：{prompt[:80]!r}\n"
                    "写测试时这通常意味着你改了 prompt 模板但忘了同步 register 的标记。"
                )
            text = self.fallback

        tin = sum(count_tokens(m["content"]) for m in messages)
        # 模拟器不花钱，但仍如实记录 token——否则第 10 章的成本表会低估真实开销
        return LLMResult(
            text=text, tokens_in=tin, tokens_out=count_tokens(text), cost_usd=0.0, model="scripted"
        )

    @property
    def is_real(self) -> bool:
        return False


def get_llm(*, meter: Meter | None = None, store_label: str = "llm", quiet: bool = False):
    """有 API Key 就用真模型，否则返回一个空的 ``ScriptedLLM`` 并提示。

    注意返回的 ScriptedLLM **没有注册任何处理器**——调用方需要自己注册，
    或者接受它对所有请求都返回 ``{}``。各章的脚本会处理这件事。
    """
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIClient(meter=meter, store_label=store_label)
        except Exception as exc:  # noqa: BLE001 —— 没装 openai 或配置错都要能降级
            if not quiet:
                print(f"  ⚠️  无法创建 OpenAI 客户端（{type(exc).__name__}），改用离线模拟。")
    elif not quiet:
        print("  ⚠️  未配置 OPENAI_API_KEY，使用离线模拟（ScriptedLLM）。")
        print("     流程能跑通，但**凡依赖模型判断力的结论都不可当真**。")
        print("     配置方式见 .env.example，本地 Ollama 也可以。")
    return ScriptedLLM(meter=meter, store_label=store_label)
