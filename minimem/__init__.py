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

尚未实现的模块导入时会给出明确提示，而不是 ImportError 的堆栈。
"""

from minimem.base import (
    CrossUserAccessError,
    MemoryItem,
    MemoryNotFoundError,
    MemoryStore,
    SearchResult,
)
from minimem.buffer import BufferMemory
from minimem.utils.metering import Meter, OpRecord, get_default_meter, reset_default_meter
from minimem.vector import VectorMemory

__version__ = "0.1.0"

__all__ = [
    "MemoryItem",
    "SearchResult",
    "MemoryStore",
    "MemoryNotFoundError",
    "CrossUserAccessError",
    "BufferMemory",
    "VectorMemory",
    "Meter",
    "OpRecord",
    "get_default_meter",
    "reset_default_meter",
    "__version__",
]
