# 附录 A：数据集与 API 清单

> **用途**：找评测集、找模型、配 API 时的速查页。
>
> **更新政策**：🕐 本页信息**截至 2026-07**。基准的已知问题会随第三方审计更新，
> API 与模型清单会随供应商变化。提 PR 时请同步更新日期。
>
> **一个前提**：读这一页之前，建议先读第 10 章 10.3~10.4 节——
> 它解释了为什么本书不推荐直接用公开基准做选型。

---

## A.1 记忆与长对话基准

| 基准 | 规模与形式 | 考察什么 | 已知问题 |
| :--- | :--- | :--- | :--- |
| **LoCoMo** | 约 10 段长对话，1500+ 问题 | 多轮对话中的信息提取与推理 | ⚠️ 有第三方审计报告称其 1540 题中存在约 6.4% 的答案键错误（见第 10 章 10.3.2）。样本量小——只有 10 段对话 |
| **LongMemEval** 📄 arXiv:2410.10813 | 500 题，5 类能力；S 子集约 115K token/题，M 子集约 1.5M | 信息提取、多会话推理、时序推理、知识更新、**拒答** | ⚠️ S 子集的上下文在现代窗口内放得下，有「更像上下文窗口测试」的批评（10.3.4）。分到 5 类后单类样本偏少 |
| **MSC**（Multi-Session Chat） | 多会话对话 | 跨会话的人设一致性 | 年代较早，难度对当前模型偏低 |
| **PerLTQA** | 个性化长期问答 | 个人信息的长期记忆 | — |
| **DialSim** | 对话模拟 | 长期交互中的一致性 | — |
| **MemoryAgentBench** | agent 场景 | 记忆驱动的任务完成 | — |
| 🕐 **BEAM / MemoryCD / MEMTRACK** | 2026 年新出 | 各异 | ⚠️ **本书未深入验证其方法学**，仅作存在性提及。新基准的答案键质量、评判方式、样本量都需要时间与第三方复现来检验 |

**LongMemEval 的分类值得单独看**：它把能力分成五类，其中 **abstention（拒答）** 是最容易被忽略的一类——「记忆里没有的东西，系统会不会承认不知道」。这一类在总分里几乎看不出来，但在实践中出问题的频率极高（见第 10 章 10.7 节：检索层结构上没有拒答能力）。

本书 MiniBench 的八类能力划分受它启发，并额外加了「多跳」与「归纳」两类。

---

## A.2 长上下文与多跳基准

### 长上下文

| 基准 | 一句话 |
| :--- | :--- |
| **NIAH**（Needle in a Haystack） | 长文本里藏一句话再问出来。最简单，也最容易刷 |
| **RULER** | NIAH 的加强版：多针、多跳、聚合等变体 |
| **∞Bench** / **HELMET** / **LongBench / v2** | 更综合的长上下文能力评估 |

这一类测的是**模型**，不是记忆系统。记忆方案常拿它做对照。

### 多跳与知识密集

| 基准 | 一句话 |
| :--- | :--- |
| **HotpotQA** | 经典多跳问答 |
| **2WikiMultihopQA** | 第 4 章引用的 HippoRAG 数字就出自它 |
| **MuSiQue** | 构造更严格的多跳，难度更高 |

---

## A.3 为什么本书不推荐直接用公开基准做选型

这一节是本附录最重要的部分。

第 10 章给出了完整论证，这里只列结论：

1. **基准本身可能不可靠**——答案键错误、LLM 评判宽松、样本量不足。
2. **公开数字之间不可比**——同一基准同两个系统，公开可查的分数有四个互斥版本（第 10 章 10.4 节）。差异来自评测子集、k 值、评判方式、底座模型、是否含写入处理五项，而这五项在公开报告里很少被完整说明。
3. **基准测不到的差异不代表不存在**——第 3 章的实例：MiniBench 上混合检索没赢过纯 dense，但换成 12 个相似编号的压力测试，dense 掉到 7/12 而 BM25 全中。

### 替代做法

**在你自己的业务数据上建一个小评测集。** 参考 MiniBench 的做法（`minimem/eval/dataset.py`）：

- **规模可以很小**（30 条记忆 + 24 个查询），但要**按能力分类**
- 八类能力：直接检索、同义改写、专有名词、多跳、时序、知识更新、归纳、**拒答**
- 每个查询标注 gold（应召回哪几条）与 answer_keywords（正确回答该含什么）
- **拒答类不能少**——它是最容易出问题也最容易被忽略的一类

然后用第 10 章的 harness 跑：

```bash
python docs/chapter10/code/01_harness.py
```

**一个小评测集 + 一张分能力表，比任何公开榜单都更能指导你的决策。**

---

## A.4 LLM API

本书所有代码走 **OpenAI 兼容接口**，换供应商只需改两个环境变量。以下配置与 `.env.example` 保持一致。

| 供应商 | `OPENAI_BASE_URL` | 常用 `AGENT_MEM_MODEL` |
| :--- | :--- | :--- |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 通义千问（阿里云百炼） | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| 本地 Ollama | `http://localhost:11434/v1` | `qwen2.5:7b`（`OPENAI_API_KEY` 随便填） |

配置方式：

```bash
cp .env.example .env    # 填入你的 key
python -m minimem.utils.check_env   # 验证连通性
```

> 🕐 各家的模型名与价格变动频繁，本书**不写死任何单价**。
> 成本估算用 `.env` 里的 `AGENT_MEM_PRICE_IN` / `AGENT_MEM_PRICE_OUT`（单位：美元/百万 token），
> 请按你实际使用的模型填。

### 哪些章节需要 API

| 章节 | 需要吗 |
| :--- | :--- |
| 第 1~5 章 | ❌ 全部离线可跑 |
| 第 6~8 章 | ⚠️ 脚本用 `ScriptedLLM` 模拟，**离线可跑**；想看真实效果需要 API |
| 第 9 章 | ❌ 概念演示离线；GPU 脚本另说 |
| 第 10 章 | ❌ 离线可跑 |

**关于 `ScriptedLLM`**：它模拟的是**流程与用量，不是判断力**。第 6、8 章都明确标注了「哪些结论换成真模型会变」——通常是质量类结论会变，成本结构不变。

---

## A.5 中文句向量模型

| 模型 | 维度 | 大小 | 说明 |
| :--- | :--- | :--- | :--- |
| **`BAAI/bge-small-zh-v1.5`** | 512 | 约 95MB | **本书默认**。CPU 上编码一句几毫秒 |
| `BAAI/bge-base-zh-v1.5` | 768 | 约 400MB | 效果更好，成本更高 |
| `BAAI/bge-large-zh-v1.5` | 1024 | 约 1.3GB | — |
| `BAAI/bge-reranker-base` | — | 约 1.1GB | 交叉编码重排器（第 3 章 3.7 节的可选项） |
| `moka-ai/m3e-base` / `shibing624/text2vec-*` | 768 | 约 400MB | 中文社区常用的其他选择 |

### 下载：走 ModelScope，不走 HuggingFace

**国内 HuggingFace 通常不可达**，本书默认走 ModelScope。

```bash
pip install -e ".[vector]"              # 含 modelscope
python -m minimem.utils.fetch_model     # 下载默认模型，并打印要设的环境变量
python -m minimem.utils.fetch_model --list   # 看本书用到哪些模型
```

命令结束会提示你把类似这行加进 `.env`：

```bash
AGENT_MEM_EMBED_MODEL=./models/models/BAAI--bge-small-zh-v1.5/snapshots/master
```

**备选：git lfs**

```bash
git lfs install
git clone https://www.modelscope.cn/BAAI/bge-small-zh-v1.5.git models/bge-small-zh-v1.5
export AGENT_MEM_EMBED_MODEL=./models/bge-small-zh-v1.5
```

**海外网络可用时**：`pip install -e ".[vector-onnx]"`（fastembed，不需要 torch，约 100MB），模型自动从 HuggingFace 拉取。

**完全离线**：`export AGENT_MEM_EMBED_MODEL=fake`。流程能跑通，但**检索效果不具参考价值**——第 3 章实验一专门用它做了对照。

### 选型建议

🕐 MTEB / C-MTEB 榜单可以用来**筛候选**，但不要用它做决定：

- 榜单任务与你的任务未必相关
- 榜单变动频繁
- 差几个百分点在你的数据上可能完全反转

**做法**：从榜单挑 2~3 个候选，用 A.3 节说的自建评测集实测，看分能力表。

---

## A.6 数据集与模型的存放约定

本书的 `.gitignore` 已配置：

```
models/          # 模型权重，不入库
data/            # 实验数据
.learn/          # 交互式课程的个人进度
```

模型默认下载到 `./models`，可用 `AGENT_MEM_MODEL_DIR` 改。

---

← 返回 [第 10 章](../chapter10/第十章%20评测、安全与工程落地.md)　|　[附录 B 选型对比表](附录B-选型对比表.md) →
