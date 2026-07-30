<div align="center">

# agent-mem

**从零开始学 Agent 记忆：把「记忆」当成一个可实现、可评测、可攻击的工程系统**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/内容许可-CC%20BY--NC--SA%204.0-lightgrey.svg)](./LICENSE.txt)
[![Code License: MIT](https://img.shields.io/badge/代码许可-MIT-green.svg)](./LICENSE-CODE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Docs](https://img.shields.io/badge/在线阅读-GitHub%20Pages-brightgreen.svg)](https://emmmdty.github.io/agent-mem/)

[在线阅读](https://emmmdty.github.io/agent-mem/) · [开发路线图](./ROADMAP.md) · [参与贡献](./CONTRIBUTING.md) · [English](./README_en.md)

</div>

---

## 这是什么

`agent-mem` 是一本**面向工程实现的中文 Agent 记忆（Agent Memory）开源教程**。

它不满足于罗列「MemGPT、Mem0、Zep、A-MEM……」这些名字，而是要求你在读完后能回答三个问题：

1. 一个记忆系统由哪几种**表征**和哪几种**操作**构成？为什么这个切分比「工作记忆/情景记忆/语义记忆/程序记忆」的心理学四分法更适合写代码？
2. 你选的方案，**每写入一条记忆、每检索一次，分别花多少 token、多少毫秒、多少钱**？
3. 别人往你的记忆库里投一条毒，会发生什么？你怎么测出来？

全书围绕一条主线展开：从零手写一个名为 **MiniMem** 的记忆系统，从最朴素的对话缓冲，一路长成带向量检索、图结构、双时间轴、自组织链接、分层调度、技能库和评测 harness 的完整体。每一章新增一个模块，接口自始至终统一。

## 为什么再写一本

现有中文材料大多停在两类：翻译综述，或某个框架的 API 教程。而这个领域当前最真实的状况是：

- **分类混乱**。三篇主流综述用三套不同的分类轴；「activation memory 算不算独立一类」至今没有共识。
- **数字不可信**。厂商在 LoCoMo / LongMemEval 上互报的分数彼此矛盾且难以复现，独立审计还在这些基准的答案键里查出了成规模的错误。
- **口号盖过机制**。「memory OS」「brain-inspired」「context lake」这类词汇在论文和产品页里的密度，远高于其对应的技术实质。

本教程的编写原则因此定为三条：**讲机制不讲口号**、**所有第三方数字标注来源与不确定性**、**每个方案都必须给出成本-延迟-token 的代价**。

## 全书目录

| 章节 | 主题 | 主线新增模块 | 算力要求 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| [第 1 章](./docs/chapter1/) | 导论：为什么 Agent 需要记忆 | `MemoryStore` 接口 / `BufferMemory` | CPU | ✅ |
| [第 2 章](./docs/chapter2/) | 上下文与长上下文记忆 | `WindowMemory` | CPU / 单卡 | 🚧 |
| [第 3 章](./docs/chapter3/) | 检索增强记忆基础 | `VectorMemory` | CPU | ✅ |
| [第 4 章](./docs/chapter4/) | 结构化与图记忆 | `GraphMemory` | CPU | 🚧 |
| [第 5 章](./docs/chapter5/) | 时间感知与知识演化 | `TemporalGraphMemory` | CPU | 🚧 |
| [第 6 章](./docs/chapter6/) | agentic memory 与自组织 | `AgenticMemory` | CPU + API | 🚧 |
| [第 7 章](./docs/chapter7/) | OS 式分层记忆与调度 | `LayeredMemory` | CPU + API | 🚧 |
| [第 8 章](./docs/chapter8/) | 经验与技能记忆 | `SkillMemory` | CPU + API | 🚧 |
| [第 9 章](./docs/chapter9/) | 参数化记忆与持续学习 | （对照实验，不入主线） | **需 GPU** | 🚧 |
| [第 10 章](./docs/chapter10/) | 评测、安全与工程落地 | `EvalHarness` | CPU + API | ✅ |
| [附录](./docs/附录/) | 数据集清单 / 选型对比 / 参考文献 / 术语表 | — | — | 🚧 |

> 写作顺序不是 1→10，而是先打通 **1 / 3 / 10** 这条闭环主线（有抽象、有实现、有评测），
> 让后续每一章都能立刻被同一个 harness 检验。详见 [ROADMAP](./ROADMAP.md)。

> 算力硬约束：除第 9 章的模型编辑实验外，**所有实验都能在纯 CPU + 一个 LLM API Key 的条件下跑通**；标注「单卡」的实验在 24GB 显存内可完成。

## 全书统一的分析框架

我们采用 **表征 × 操作 × 载体** 三轴，而不是心理学四分法：

```
表征（记忆存在哪）          操作（对记忆做什么）           载体（用什么存）
├─ 参数化                  ├─ 编码 Encoding              ├─ 内存 / 文件
│  （权重里，第 9 章）       ├─ 巩固 Consolidation         ├─ 向量库
├─ 上下文-非结构化           ├─ 索引 Indexing              ├─ 图库
│  （原文/摘要，第 2、3 章）  ├─ 检索 Retrieval             ├─ 关系库
└─ 上下文-结构化            ├─ 更新 Updating              └─ 模型权重
   （图/表/技能，第 4~8 章）  ├─ 遗忘 Forgetting
                           ├─ 压缩 Compression
                           └─ 反思 Reflection
```

采用它的理由、以及为什么把心理学四分法降格为「导论里的类比」而非模块边界，见[第 1 章](./docs/chapter1/)。

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/emmmdty/agent-mem.git
cd agent-mem

# 2. 建环境（Python 3.10+）
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. 安装主线包（第 1~2 章只需核心依赖）
pip install -e .

# 4. 跑通第一个例子：让一个无记忆的对话「失忆」，再给它装上记忆
python docs/chapter1/code/01_goldfish_agent.py
```

## 交互式学习

除了读正文，你还可以让终端带着你学：

```bash
python -m minimem.learn          # 总览与进度
python -m minimem.learn next     # 开始/继续
python -m minimem.learn review   # 今日复习（间隔重复）
```

它不是把正文念一遍，而是把学习科学里几条最扎实的结论做成流程：

| 环节 | 做什么 | 依据 |
| :--- | :--- | :--- |
| 💡 **一句话** | 每章压成一句能记住的话 | 组块化 |
| 🤔 **先猜一猜** | 给出结论**前**先出选择题，选项里必有一个「合理但错误」 | 生成效应 —— **猜错比猜对更有价值** |
| ▶️ **跑一下** | 几秒钟的脚本，先说清该盯哪一处 | 主动实践 |
| ✅ **检查点** | 问答题，答案默认折叠 | 提取练习优于重读 |
| 🗣️ **讲一遍** | 用自己的话解释，对照清单自评 | 费曼技巧 |
| 🔧 **动手挑战** | 🟢 改参数 / 🟡 补实现 / 🔴 开放设计 | 合意困难 |
| 🚪 **下一章** | 留一个本章解决不了的问题 | 蔡格尼克效应 |

正文里的同名标记就是这些环节，所以**只读 markdown 也能得到完整体验**。进度存在 `.learn/`（不入库），随时 `q` 退出、`next` 接上。

复习卡片可以导出到 Anki：`python -m minimem.learn export-anki`。

> 为什么把这些原理明写出来而不是偷偷用？因为元认知本身就是有效的学习干预。
> 完整说明见 [docs/学习方法.md](./docs/学习方法.md)。

各章依赖**按章拆分**，需要时再装：

```bash
pip install -e ".[vector]"   # 第 3 章
pip install -e ".[graph]"    # 第 4~5 章
pip install -e ".[llm]"      # 第 6~8 章
pip install -e ".[eval]"     # 第 10 章
```

详细的环境、API Key 与国内镜像配置见 [docs/学习与环境准备.md](./docs/学习与环境准备.md)。

## 本地预览文档站

文档站基于 docsify，**无需构建**：

```bash
npm i -g docsify-cli
docsify serve docs      # 打开 http://localhost:3000
```

## 学习路径建议

- **想快速建立全局观（约 1 周）**：第 1 章 → 第 3 章 → 第 10 章。这条线能让你把「记忆系统」和「评测它的方法」同时建立起来，避免陷入只堆功能不看代价的坑。
- **做产品/落地选型（约 2 周）**：第 1 → 3 → 5 → 7 → 10 章，重点看每章末尾的「代价与选型」小节。
- **做研究/想读论文（约 3 周）**：按顺序通读，配合[附录 C 的必读论文清单](./docs/附录/)。

## 关于本教程引用的数字

本领域大量流传的性能数字来自厂商自述或单篇论文的特定实验条件。本教程中**凡是第三方数字，一律标注来源与可信度等级**：

| 标记 | 含义 |
| :--- | :--- |
| 📄 **论文原文** | 出自同行评议或 arXiv 论文，标注编号与具体表格 |
| 🏢 **厂商自述** | 出自公司博客/文档，无独立第三方复现 |
| ⚠️ **存在争议** | 不同来源互相矛盾，或已被独立审计质疑 |
| 🕐 **时效性内容** | 会随版本迭代过时，标注截止日期 |

例如：LoCoMo、LongMemEval 上的「谁是 SOTA」之争涉及至少三方互斥的数字，且基准本身被审计出答案键错误与评判宽松的问题——这类结论在本教程中一律降级为「⚠️ 存在争议」，并在[第 10 章](./docs/chapter10/)展开分析。

## 参与贡献

欢迎提 Issue 与 PR：内容勘误、论文补充、实验复现结果、框架实测数据都非常需要。请先读 [CONTRIBUTING.md](./CONTRIBUTING.md)。

## 致谢

本教程的形式设计（docsify 文档站、按章拆依赖、中文命名章节、内容 CC BY-NC-SA + 代码 MIT 双许可）参考了 [Datawhale](https://github.com/datawhalechina) 的开源教程实践，尤其是 [happy-llm](https://github.com/datawhalechina/happy-llm)。

内容框架主要参考 Du et al. 的记忆分类综述与 Zhang et al. 的 agent memory 综述，完整参考文献见附录 C。

## 许可

- 教程内容（`docs/`、README 等）：[CC BY-NC-SA 4.0](./LICENSE.txt)
- 源代码（`minimem/`、`tests/`、`docs/**/code/`）：[MIT](./LICENSE-CODE)
