"""MiniMem —— agent-mem 教程贯穿全书的最小可用 Agent 记忆系统。

各章逐步加入的实现：

============  ========================  ==================================
章节           模块                      机制
============  ========================  ==================================
第 1 章        ``BufferMemory``          朴素缓冲 + 词重叠检索
第 2 章        ``WindowMemory``          滑动窗口 + sink 保留
第 3 章        ``VectorMemory``          向量 + BM25 混合检索 + 重排
第 4 章        ``GraphMemory``           实体图 + Personalized PageRank
第 5 章        ``TemporalGraphMemory``   双时间轴 + 事实失效
第 6 章        ``AgenticMemory``         自组织 note + 动态链接
第 7 章        ``LayeredMemory``         三层调度 + 热度换页
第 8 章        ``SkillMemory``           反思 + 技能库
第 10 章       ``minimem.eval``          评测 harness + 成本核算
============  ========================  ==================================

**依赖可选的实现走惰性导入。** 本书按章拆分依赖——只装核心依赖的读者
应该能正常用第 1~3 章，而不是因为第 4 章的 networkx 没装就整个 import 失败。

所以 ``GraphMemory`` / ``TemporalGraphMemory`` 在**真正被访问时**才导入，
缺依赖时给出「装哪个 extras」的明确提示，而不是一串 ImportError 堆栈。

（这个设计是被 CI 打脸后补上的：早期版本在顶层 import GraphMemory，
结果只装 ``.[dev]`` 的环境连 ``from minimem import BufferMemory`` 都会崩。
**核心包不该因为可选依赖缺失而整体不可用**。）
"""

from typing import TYPE_CHECKING

from minimem.agentic import AgenticMemory
from minimem.base import (
    CrossUserAccessError,
    MemoryItem,
    MemoryNotFoundError,
    MemoryStore,
    SearchResult,
)
from minimem.buffer import BufferMemory
from minimem.layered import LayeredMemory
from minimem.skill import SkillMemory
from minimem.utils.metering import Meter, OpRecord, get_default_meter, reset_default_meter
from minimem.vector import VectorMemory
from minimem.window import WindowMemory

if TYPE_CHECKING:  # 让类型检查器与 IDE 仍能看到它们
    from minimem.graph import GraphMemory
    from minimem.temporal import TemporalGraphMemory

#: 名字 -> (模块路径, 缺失的第三方包, 该装哪个 extras)
_LAZY: dict[str, tuple[str, str, str]] = {
    "GraphMemory": ("minimem.graph", "networkx", "graph"),
    "TemporalGraphMemory": ("minimem.temporal", "networkx", "graph"),
}


def __getattr__(name: str):
    """惰性导入依赖可选包的实现（PEP 562）。"""
    if name not in _LAZY:
        raise AttributeError(f"module 'minimem' has no attribute {name!r}")

    import importlib

    module_path, dep, extra = _LAZY[name]
    try:
        return getattr(importlib.import_module(module_path), name)
    except ImportError as exc:
        raise ImportError(
            f"{name} 需要 {dep}，而它属于可选依赖。\n"
            f'  安装：pip install -e ".[{extra}]"\n'
            f"  本书按章拆分依赖，读到第 4 章时再装即可。"
        ) from exc


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__version__ = "0.1.0"

__all__ = [
    "MemoryItem",
    "SearchResult",
    "MemoryStore",
    "MemoryNotFoundError",
    "CrossUserAccessError",
    "BufferMemory",
    "AgenticMemory",
    "GraphMemory",
    "LayeredMemory",
    "TemporalGraphMemory",
    "SkillMemory",
    "VectorMemory",
    "WindowMemory",
    "Meter",
    "OpRecord",
    "get_default_meter",
    "reset_default_meter",
    "__version__",
]
