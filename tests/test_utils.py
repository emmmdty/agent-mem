"""计量器、token 估算与 MockLLM 的测试。"""

from __future__ import annotations

import pytest

from minimem.utils.metering import Meter
from minimem.utils.mock_llm import MockLLM
from minimem.utils.tokens import TokenCost, count_tokens, estimate_tokens


class TestTokens:
    def test_空串为零(self):
        assert estimate_tokens("") == 0

    def test_中文按字计(self):
        assert estimate_tokens("你好世界") == 4

    def test_英文按四字符折算(self):
        assert estimate_tokens("hello world") == 3  # 11 字符 / 4 ≈ 3

    def test_中英混合(self):
        assert estimate_tokens("使用 RAG") > 0

    def test_count_tokens在无tiktoken时退回估算(self):
        # 无论 tiktoken 是否安装，都必须返回一个正整数而不是抛异常
        assert count_tokens("测试文本") > 0
        assert count_tokens("测试文本", exact=False) == estimate_tokens("测试文本")

    def test_成本按百万token计价(self):
        cost = TokenCost(price_in=1.0, price_out=2.0)
        assert cost(1_000_000, 0) == pytest.approx(1.0)
        assert cost(0, 1_000_000) == pytest.approx(2.0)
        assert cost(500_000, 500_000) == pytest.approx(1.5)


class TestMeter:
    def test_记录耗时(self):
        meter = Meter()
        with meter.measure(op="search", store="s") as rec:
            rec.n_items = 3
        assert len(meter.records) == 1
        assert meter.records[0].n_items == 3
        assert meter.records[0].duration_ms >= 0

    def test_异常时仍记录并标记(self):
        meter = Meter()
        with pytest.raises(RuntimeError), meter.measure(op="add", store="s"):
            raise RuntimeError("boom")
        assert meter.records[0].extra["failed"] == "RuntimeError"

    def test_禁用后不记录(self):
        meter = Meter(enabled=False)
        with meter.measure(op="add", store="s"):
            pass
        assert meter.records == []

    def test_summary区分op与store(self):
        meter = Meter()
        for _ in range(3):
            with meter.measure(op="add", store="a"):
                pass
        with meter.measure(op="search", store="b"):
            pass

        assert meter.summary(op="add")["count"] == 3
        assert meter.summary(store="b")["count"] == 1
        assert meter.summary()["count"] == 4

    def test_空summary不报错(self):
        assert Meter().summary()["count"] == 0

    def test_llm用量单独记录(self):
        meter = Meter()
        meter.add_llm_usage(
            op="extract", store="agentic", tokens_in=100, tokens_out=20, cost_usd=0.001
        )
        s = meter.summary(op="extract")
        assert s["tokens_total"] == 120
        assert s["llm_calls"] == 1

    def test_report可读(self):
        meter = Meter()
        with meter.measure(op="add", store="buffer"):
            pass
        assert "buffer" in meter.report()
        assert "（无计量记录）" in Meter().report()


class TestMockLLM:
    @pytest.fixture
    def facts(self):
        return [
            {"role": "user", "content": "你好，我叫小明。"},
            {"role": "user", "content": "我在星辰银行工作，是一名信贷风控工程师。"},
            {"role": "user", "content": "我对花生过敏。"},
        ]

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("我叫什么？", "小明"),
            ("我在哪工作？", "星辰银行"),
            ("我是做什么的？", "信贷风控工程师"),
            ("我对什么过敏？", "花生"),
        ],
    )
    def test_上下文里有就能答(self, facts, question, expected):
        llm = MockLLM()
        resp = llm.chat([*facts, {"role": "user", "content": question}])
        assert expected in resp.content

    def test_上下文里没有就说不知道(self):
        llm = MockLLM()
        resp = llm.chat([{"role": "user", "content": "我叫什么？"}])
        assert "不知道" in resp.content

    def test_提问句不污染事实(self):
        """「我叫什么来着」里的「什么来着」不能被当成姓名。"""
        llm = MockLLM()
        resp = llm.chat(
            [
                {"role": "user", "content": "我叫小明。"},
                {"role": "user", "content": "对了，我叫什么来着？"},
            ]
        )
        assert "小明" in resp.content

    def test_后说的覆盖先说的(self):
        llm = MockLLM()
        resp = llm.chat(
            [
                {"role": "user", "content": "我在星辰银行工作。"},
                {"role": "user", "content": "我在长风科技工作。"},
                {"role": "user", "content": "我在哪工作？"},
            ]
        )
        assert "长风科技" in resp.content

    def test_超出上下文限制时丢最旧的(self):
        llm = MockLLM(context_limit=30)
        msgs = [
            {"role": "user", "content": "我叫小明。"},
            *[{"role": "user", "content": f"这是第 {i} 条无关的填充消息内容。"} for i in range(10)],
            {"role": "user", "content": "我叫什么？"},
        ]
        resp = llm.chat(msgs)
        assert "不知道" in resp.content, "开头的信息应已被截断"
        assert int(resp.debug["被截断的消息数"]) > 0

    def test_非提问只作确认且不谎称记住(self):
        llm = MockLLM()
        resp = llm.chat([{"role": "user", "content": "今天天气不错。"}])
        assert "记住" not in resp.content

    def test_用量累计(self):
        llm = MockLLM()
        llm.chat([{"role": "user", "content": "我叫小明。"}])
        llm.chat([{"role": "user", "content": "我叫什么？"}])
        usage = llm.usage()
        assert usage["调用次数"] == 2
        assert usage["输入 token"] > 0

        llm.reset()
        assert llm.usage()["调用次数"] == 0
