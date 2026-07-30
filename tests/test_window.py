"""WindowMemory 的测试。

重点覆盖三条容易写错的地方：
1. sink 必须先占预算，否则会安静地退化成 recent；
2. 窗口外的记忆必须检索不到（这是窗口方案的定义性质，不是 bug）；
3. 「移出窗口」不等于「删除」——第 10 章的删除权依赖这个区分。
"""

from __future__ import annotations

import pytest

from minimem.base import MemoryItem
from minimem.utils.metering import Meter
from minimem.utils.tokens import count_tokens
from minimem.window import WindowMemory

USER = "u1"


def _long_dialog(n: int = 30) -> list[MemoryItem]:
    return [
        MemoryItem("我对花生过敏，点餐要注意。"),
        MemoryItem("我在星辰银行工作。"),
        *[MemoryItem(f"第 {i} 条日常闲聊，没有什么特别的内容。") for i in range(n)],
    ]


class TestConstruction:
    def test_预算必须为正(self):
        with pytest.raises(ValueError):
            WindowMemory(budget_tokens=0)

    def test_n_sink不能为负(self):
        with pytest.raises(ValueError):
            WindowMemory(n_sink=-1)


class TestVisibility:
    def test_预算充足时全部可见(self):
        s = WindowMemory(budget_tokens=100_000, meter=Meter())
        s.add_many(_long_dialog(5), user_id=USER)
        assert len(s.visible(user_id=USER)) == len(s.all(user_id=USER))

    def test_recent策略丢掉开头(self):
        s = WindowMemory(budget_tokens=120, policy="recent", meter=Meter())
        s.add_many(_long_dialog(20), user_id=USER)
        texts = [i.content for i in s.visible(user_id=USER)]
        assert not any("过敏" in t for t in texts), "recent 策略应该已经丢掉了开头的过敏记录"

    def test_sink策略保住开头(self):
        s = WindowMemory(budget_tokens=120, policy="sink", n_sink=2, meter=Meter())
        s.add_many(_long_dialog(20), user_id=USER)
        texts = [i.content for i in s.visible(user_id=USER)]
        assert any("过敏" in t for t in texts), "sink 策略必须保住开头几条"

    def test_sink优先占预算(self):
        """如果先填最近的，预算会在轮到 sink 之前用光——这个退化很安静。"""
        s = WindowMemory(budget_tokens=60, policy="sink", n_sink=1, meter=Meter())
        s.add_many(_long_dialog(30), user_id=USER)
        first = s.all(user_id=USER)[0]
        assert first.id in {i.id for i in s.visible(user_id=USER)}

    def test_pinned策略保住被标记的(self):
        s = WindowMemory(budget_tokens=110, policy="pinned", n_sink=0, meter=Meter())
        s.add(MemoryItem("我对花生过敏。", metadata={"pinned": True}), user_id=USER)
        s.add_many([MemoryItem(f"闲聊 {i}，内容不重要。") for i in range(20)], user_id=USER)
        texts = [i.content for i in s.visible(user_id=USER)]
        assert any("过敏" in t for t in texts)

    def test_pin方法生效(self):
        s = WindowMemory(budget_tokens=110, policy="pinned", n_sink=0, meter=Meter())
        mid = s.add("我对花生过敏。", user_id=USER)
        s.pin(mid, user_id=USER)
        s.add_many([MemoryItem(f"闲聊 {i}，内容不重要。") for i in range(20)], user_id=USER)
        assert mid in {i.id for i in s.visible(user_id=USER)}

    def test_窗口内不超预算(self):
        s = WindowMemory(budget_tokens=100, policy="sink", meter=Meter())
        s.add_many(_long_dialog(30), user_id=USER)
        total = sum(count_tokens(i.content) for i in s.visible(user_id=USER))
        assert total <= 100

    def test_保持时间顺序(self):
        s = WindowMemory(budget_tokens=100_000, meter=Meter())
        s.add_many(["第一", "第二", "第三"], user_id=USER)
        assert [i.content for i in s.visible(user_id=USER)] == ["第一", "第二", "第三"]

    def test_evicted列出被丢弃的(self):
        s = WindowMemory(budget_tokens=100, policy="recent", meter=Meter())
        s.add_many(_long_dialog(30), user_id=USER)
        assert len(s.evicted(user_id=USER)) > 0
        vis = {i.id for i in s.visible(user_id=USER)}
        assert all(i.id not in vis for i in s.evicted(user_id=USER))

    def test_usage报告(self):
        s = WindowMemory(budget_tokens=100, meter=Meter())
        s.add_many(_long_dialog(20), user_id=USER)
        u = s.usage(user_id=USER)
        assert u["窗口内条数"] <= u["总条数"]
        assert u["窗口内 token"] <= u["预算"]


class TestSearch:
    def test_窗口外检索不到(self):
        """这是窗口方案的定义性质，不是缺陷。"""
        s = WindowMemory(budget_tokens=110, policy="recent", meter=Meter())
        s.add_many(_long_dialog(25), user_id=USER)
        hits = s.search("过敏", user_id=USER, k=5)
        assert all("过敏" not in h.content for h in hits)

    def test_窗口内可检索(self):
        s = WindowMemory(budget_tokens=110, policy="sink", n_sink=2, meter=Meter())
        s.add_many(_long_dialog(25), user_id=USER)
        hits = s.search("过敏", user_id=USER, k=5)
        assert any("过敏" in h.content for h in hits)


class TestDeletionSemantics:
    def test_默认不真删除(self):
        """「移出窗口」≠「删除」。第 10 章的删除权依赖这个区分。"""
        s = WindowMemory(budget_tokens=100, policy="recent", drop_from_store=False, meter=Meter())
        s.add_many(_long_dialog(30), user_id=USER)
        assert len(s.all(user_id=USER)) == 32
        assert len(s.visible(user_id=USER)) < 32

    def test_显式开启后真删除(self):
        s = WindowMemory(budget_tokens=100, policy="recent", drop_from_store=True, meter=Meter())
        s.add_many(_long_dialog(30), user_id=USER)
        assert len(s.all(user_id=USER)) == len(s.visible(user_id=USER))
