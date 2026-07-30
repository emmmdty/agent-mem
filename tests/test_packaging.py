"""打包与依赖边界的测试。

本书按章拆分依赖：读者只装核心依赖时，第 1~3 章必须能正常用。
这条约定曾经被破坏过——早期版本在 ``minimem/__init__.py`` 顶层
``from minimem.graph import GraphMemory``，于是只装 ``.[dev]`` 的环境里
连 ``from minimem import BufferMemory`` 都会崩（CI 是这么发现的）。

这些测试把那条约定钉住。
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _reload_minimem(monkeypatch) -> None:
    """清掉 minimem 的模块缓存，让下一次 import 重新执行。"""
    for name in list(sys.modules):
        if name == "minimem" or name.startswith("minimem."):
            monkeypatch.delitem(sys.modules, name, raising=False)


class TestOptionalDependencies:
    def test_缺networkx时核心包仍可导入(self, monkeypatch):
        """第 4 章的依赖没装，不应该影响第 1~3 章。"""
        _reload_minimem(monkeypatch)
        monkeypatch.setitem(sys.modules, "networkx", None)  # 让 import networkx 失败

        minimem = importlib.import_module("minimem")
        assert minimem.BufferMemory is not None
        assert minimem.VectorMemory is not None
        assert minimem.WindowMemory is not None

    def test_缺networkx时访问图记忆给出可操作的提示(self, monkeypatch):
        _reload_minimem(monkeypatch)
        monkeypatch.setitem(sys.modules, "networkx", None)

        minimem = importlib.import_module("minimem")
        with pytest.raises(ImportError) as exc:
            _ = minimem.GraphMemory

        msg = str(exc.value)
        assert "networkx" in msg
        assert ".[graph]" in msg, "错误信息必须告诉读者装哪个 extras"

    def test_依赖齐全时惰性导入照常工作(self):
        import minimem

        assert minimem.GraphMemory.__name__ == "GraphMemory"
        assert minimem.TemporalGraphMemory.__name__ == "TemporalGraphMemory"

    def test_未知属性仍抛AttributeError(self):
        import minimem

        with pytest.raises(AttributeError):
            _ = minimem.根本不存在的东西

    def test_dir包含惰性导出的名字(self):
        import minimem

        names = dir(minimem)
        assert "GraphMemory" in names
        assert "BufferMemory" in names


class TestPublicSurface:
    def test_all里的名字都能取到(self):
        import minimem

        for name in minimem.__all__:
            assert getattr(minimem, name) is not None, f"__all__ 里的 {name} 取不到"

    def test_八种实现都在公开接口里(self):
        import minimem

        expected = {
            "BufferMemory",
            "WindowMemory",
            "VectorMemory",
            "GraphMemory",
            "TemporalGraphMemory",
            "AgenticMemory",
            "LayeredMemory",
            "SkillMemory",
        }
        assert expected <= set(minimem.__all__)

    def test_版本号可读(self):
        import minimem

        assert minimem.__version__
