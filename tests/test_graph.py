"""GraphMemory 与实体抽取的测试。

重点覆盖三个真实踩过的坑：
1. 贪婪匹配把「我在星辰银行」当成实体，导致同一实体在图上断成两个节点；
2. `_add` 重复调用抽取器，用 LLM 抽取时每条记忆付两次钱；
3. `LLMExtractor.extract` 从三元组反推实体，单实体的**查询**因此静默返回空。
"""

from __future__ import annotations

import json

import pytest

from minimem.eval import as_memory_items
from minimem.graph import GraphMemory
from minimem.utils.embedding import FakeEmbedder
from minimem.utils.entities import LLMExtractor, RuleExtractor
from minimem.utils.llm import ScriptedLLM
from minimem.utils.metering import Meter

USER = "u1"


@pytest.fixture
def store():
    return GraphMemory(mode="ppr", embedder=FakeEmbedder(), meter=Meter())


class TestRuleExtractor:
    def test_抽出英文标识符(self):
        names = {e.name for e in RuleExtractor().extract("我负责的项目代号是 XR-2049。")}
        assert "xr-2049" in names

    def test_抽出带后缀的机构名(self):
        names = {e.name for e in RuleExtractor().extract("我在星辰银行工作。")}
        assert "星辰银行" in names

    def test_同一实体在不同句子里必须同名(self):
        """贪婪匹配会抽出「我在星辰银行」和「就从星辰银行」——图会因此断开。"""
        ex = RuleExtractor()
        a = {e.name for e in ex.extract("我在星辰银行工作，是一名信贷风控工程师。")}
        b = {e.name for e in ex.extract("我下周就要从星辰银行离职了，交接已经开始。")}
        assert "星辰银行" in a
        assert "星辰银行" in b, "同一实体在不同句子里必须归一化成同一个名字"

    def test_剥离定语(self):
        names = {e.name for e in RuleExtractor().extract("我住在城西的枫林小区。")}
        assert "枫林小区" in names

    def test_称呼必须以姓氏打头(self):
        """否则「调去总行」会被抽成「调去总」。"""
        names = {e.name for e in RuleExtractor().extract("李姐下个季度要调去总行了。")}
        assert "李姐" in names
        assert not any("调去" in n for n in names)

    def test_泛称不会被抽成实体(self):
        names = {e.name for e in RuleExtractor().extract("我的领导要去哪？")}
        assert "领导" not in names

    def test_抽不到裸专名_这是规则版的已知局限(self):
        """「毛毛」没有任何标志，规则抽不到。这条断言记录的是缺陷，不是能力。"""
        names = {e.name for e in RuleExtractor().extract("我们家的猫叫毛毛，今年三岁了。")}
        assert "毛毛" not in names

    def test_单实体句子的三元组为空(self):
        assert RuleExtractor().extract_triples("我喜欢周末去爬山。") == []


class TestLLMExtractor:
    @pytest.fixture
    def llm(self):
        llm = ScriptedLLM(strict=True)
        llm.register(
            "抽取知识三元组",
            lambda p: json.dumps(
                {"triples": [{"subject": "小明", "predicate": "养", "object": "毛毛"}]},
                ensure_ascii=False,
            ),
        )
        return llm

    def test_抽出规则版抓不到的裸专名(self, llm):
        names = {e.name for e in LLMExtractor(llm).extract("我们家的猫叫毛毛。")}
        assert "毛毛" in names

    def test_非法json时退回规则抽取(self):
        llm = ScriptedLLM()
        llm.register("抽取知识三元组", lambda p: "这不是 JSON，模型今天心情不好")
        triples = LLMExtractor(llm).extract_triples("我在星辰银行工作，做信贷风控。")
        assert triples, "解析失败必须退回规则版，而不是丢掉整条记忆"

    def test_单实体查询也能抽出实体(self):
        """三元组为空时必须退回规则抽取，否则图检索连种子都没有。"""
        llm = ScriptedLLM()
        llm.register("抽取知识三元组", lambda p: '{"triples": []}')
        names = {e.name for e in LLMExtractor(llm).extract("我负责的那个项目带来了什么效果？")}
        assert "项目" in names

    def test_每条文本只调用一次(self, llm):
        LLMExtractor(llm).extract_triples("我们家的猫叫毛毛。")
        assert llm.call_count == 1


class TestGraphConstruction:
    def test_建图后有记忆节点与实体节点(self, store):
        store.add("我在星辰银行工作。", user_id=USER)
        stats = store.stats(user_id=USER)
        assert stats["记忆节点"] == 1
        assert stats["实体节点"] >= 1

    def test_共享实体的记忆通过实体相连(self, store):
        store.add("我负责的项目代号是 XR-2049。", user_id=USER)
        store.add("XR-2049 上线后效果不错。", user_id=USER)
        hits = store.search("XR-2049 怎么样", user_id=USER, k=5)
        assert len(hits) == 2, "两条记忆应通过 XR-2049 这个实体连通"

    def test_抽不到实体的记忆是孤点(self, store):
        store.add("今天天气不错。", user_id=USER)
        assert store.stats(user_id=USER)["没抽到实体的记忆"] == 1

    def test_每条记忆只抽一次(self):
        """曾经既调 extract_triples 又调 extract，等于每条记忆付两次钱。"""
        llm = ScriptedLLM()
        llm.register("抽取知识三元组", lambda p: '{"triples": []}')
        s = GraphMemory(extractor=LLMExtractor(llm), mode="ppr", meter=Meter())
        s.add("我在星辰银行工作，做信贷风控。", user_id=USER)
        assert llm.call_count == 1


class TestPPRSearch:
    def test_多跳召回(self, store):
        store.add_many(as_memory_items(), user_id=USER)
        rids = [
            h.item.metadata["rid"]
            for h in store.search("我负责的那个项目带来了什么效果？", user_id=USER, k=5)
        ]
        assert "m06" in rids and "m07" in rids, "PPR 应能顺着 XR-2049 走到第二跳"

    def test_抽不出种子时返回空(self, store):
        store.add_many(as_memory_items(), user_id=USER)
        assert store.search("我的领导要去哪？", user_id=USER, k=5) == []

    def test_hybrid模式兜住无实体查询(self):
        s = GraphMemory(mode="hybrid", embedder=FakeEmbedder(), meter=Meter())
        s.add_many(as_memory_items(), user_id=USER)
        assert s.search("我的领导要去哪？", user_id=USER, k=5), "向量通道应兜底"

    def test_门槛过滤近零分候选(self):
        """不过滤的话，这些噪声会在 RRF 里挤掉真结果。"""
        loose = GraphMemory(mode="ppr", ppr_floor_ratio=0.0, meter=Meter())
        strict = GraphMemory(mode="ppr", ppr_floor_ratio=0.01, meter=Meter())
        for s in (loose, strict):
            s.add_many(as_memory_items(), user_id=USER)
        q = "我负责的那个项目带来了什么效果？"
        assert len(strict.search(q, user_id=USER, k=10)) < len(loose.search(q, user_id=USER, k=10))

    def test_explain输出可读(self, store):
        store.add_many(as_memory_items(), user_id=USER)
        out = store.explain("XR-2049 是什么项目？", user_id=USER)
        assert "种子" in out
        assert "抽不出实体" in store.explain("完全没有实体的问句", user_id=USER)


class TestMutation:
    def test_更新内容会重建边(self, store):
        mid = store.add("我在星辰银行工作。", user_id=USER)
        store.update(mid, {"content": "我在长风科技工作。"}, user_id=USER)
        hits = store.search("长风科技", user_id=USER, k=3)
        assert hits and hits[0].item.id == mid
        assert store.search("星辰银行", user_id=USER, k=3) == [], "旧实体的边必须清掉"

    def test_删除会清理孤儿实体(self, store):
        mid = store.add("我在星辰银行工作。", user_id=USER)
        store.add("今天开了一天会。", user_id=USER)
        before = store.stats(user_id=USER)["实体节点"]
        store.delete(mid, user_id=USER)
        after = store.stats(user_id=USER)["实体节点"]
        assert after < before, "没有记忆引用的实体应从图里移除"

    def test_删除后不再被检索(self, store):
        mid = store.add("我负责的项目代号是 XR-2049。", user_id=USER)
        store.delete(mid, user_id=USER)
        assert store.search("XR-2049", user_id=USER, k=5) == []

    def test_用户隔离(self, store):
        store.add("用户1在星辰银行工作。", user_id="u1")
        store.add("用户2在长风科技工作。", user_id="u2")
        hits = store.search("星辰银行", user_id="u2", k=5)
        assert all(h.item.user_id == "u2" for h in hits)
