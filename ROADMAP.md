# agent-mem 开发路线图

> 最后更新：2026-07-30

本文件是 `agent-mem` 的总体规划与阶段划分。它同时充当**验收标准**：每个阶段列出的「完成定义（DoD）」不满足，就不打对应 tag。

---

## 0. 项目目标

做一本**工程导向的中文 Agent 记忆教程**，形式对标 Datawhale 系开源教程（docsify 站点 + 分章 markdown + 按章代码），内容对标当前最严肃的综述与基准批评工作。

三条不可妥协的编写原则：

1. **讲机制，不讲口号**。凡是「memory OS」「brain-inspired」「first-class resource」这类营销词汇，必须还原成它对应的具体机制（调度策略？分页？self-editing block？），否则不写。
2. **数字必须标来源**。第三方性能数字一律带 📄/🏢/⚠️/🕐 四级标记（见 README）。厂商自述与不可复现的对比数字不得作为结论使用。
3. **代价必须一等呈现**。每个方案除「效果」外，必须给出 token 消耗、延迟、金钱成本三项中至少两项的量级估计。

## 1. 读者假设

- 会 Python，能读懂类与抽象方法。
- 知道什么是 embedding、什么是 RAG，但不要求做过。
- 有一个可用的 LLM API Key（OpenAI 兼容接口即可），或本地能跑 7B 以下小模型。
- **不假设**有 GPU。第 9 章是唯一例外，且已明确标注。

## 2. 阶段划分

### P0 —— 骨架与规范（tag: `v0.1.0`）

| 项 | 状态 |
| :--- | :--- |
| 目录结构（docs/chapterN + code + images） | ✅ |
| docsify 站点（index.html / _sidebar / _coverpage / .nojekyll） | ✅ |
| 双许可（内容 CC BY-NC-SA 4.0 + 代码 MIT） | ✅ |
| README / README_en / ROADMAP / CONTRIBUTING / CITATION.cff | ✅ |
| CI：ruff lint + pytest + markdown 死链检查 | ✅ |
| Issue / PR 模板 | ✅ |
| pyproject + 按章可选依赖 | ✅ |
| 前言 + 学习与环境准备 | ✅ |

**DoD**：`git clone` 后 `pip install -e ".[dev]"` 与 `pytest` 均能通过；`docsify serve docs` 能打开站点且侧边栏完整。

### P1 —— 主线打通（tag: `v0.2.0`）

先写第 1、3、10 章。理由：这三章合起来构成一个**闭环最小系统**——有记忆抽象、有能用的检索实现、有能测出好坏与代价的 harness。后续每章都能立刻被 harness 检验，而不是写完十章才发现无法比较。

| 章节 | 主线产出 |
| :--- | :--- |
| 第 1 章 导论 | `minimem/base.py`（`MemoryStore` 抽象）、`minimem/buffer.py` |
| 第 3 章 检索增强记忆 | `minimem/vector.py`（chunk + embed + hybrid + rerank） |
| 第 10 章 评测与安全 | `minimem/eval/`（数据集适配、成本-延迟-token 记录器、投毒回归） |

**DoD**：能用一条命令在同一份小规模数据上，对 `BufferMemory` 与 `VectorMemory` 跑出一张包含「准确率 / 平均延迟 / 每次调用 token / 估算成本」的对比表。

### P2 —— 上下文与结构化（tag: `v0.3.0`）

| 章节 | 主线产出 |
| :--- | :--- |
| 第 2 章 长上下文 | `minimem/window.py`（滑窗 + attention sink 保留策略） |
| 第 4 章 图记忆 | `minimem/graph.py`（实体图 + Personalized PageRank 检索） |
| 第 5 章 时间感知 | `minimem/temporal.py`（bi-temporal 事实表 + 失效 + 时点查询） |

**DoD**：第 4、5 章的实现能接入 P1 的 harness 并产出可比对的成本表；第 5 章能正确回答「三个月前他在哪家公司」这类时点问题。

### P3 —— agentic 与系统式（tag: `v0.4.0`）

| 章节 | 主线产出 |
| :--- | :--- |
| 第 6 章 自组织记忆 | `minimem/agentic.py`（note 生成 + 动态链接 + 演化 + extract-update） |
| 第 7 章 分层调度 | `minimem/layered.py`（core/archival/recall 三层 + heat 换页 + self-editing） |
| 第 8 章 技能记忆 | `minimem/skill.py`（reflection 触发 + 技能库 + 轨迹归纳） |

**DoD**：三种记忆均可与前序实现在 harness 中横向比较；第 7 章能演示「上下文塞满时自动换页且不丢关键事实」。

### P4 —— 参数化、附录与发布（tag: `v1.0.0`）

| 项 | 内容 |
| :--- | :--- |
| 第 9 章 | 模型编辑（ROME/MEMIT）实验 + 副作用测量，明确标注需 GPU |
| 附录 A | 数据集与 API 清单 |
| 附录 B | 向量库 / 图库 / 记忆框架选型对比表（带更新日期） |
| 附录 C | 分组参考文献（必读 / 选读） |
| 附录 D | 术语表（中英对照） |
| 发布 | PDF release + 完整目录校对 + 全书死链检查 |

**DoD**：全书 10 章 + 4 附录完成；CI 全绿；PDF 可下载。

## 3. 主线项目 MiniMem 的接口契约

全书所有记忆实现共享同一个接口，这样才能横向比较：

```python
class MemoryStore(ABC):
    def add(self, item: MemoryItem, *, user_id: str) -> str: ...
    def search(self, query: str, *, user_id: str, k: int = 5) -> list[SearchResult]: ...
    def update(self, memory_id: str, patch: dict, *, user_id: str) -> None: ...
    def delete(self, memory_id: str, *, user_id: str) -> None: ...
    def all(self, *, user_id: str) -> list[MemoryItem]: ...
```

两条硬约束：

- **多用户隔离**：所有方法带 `user_id`，跨用户读写必须失败。这既是工程刚需，也是第 10 章隐私一节的前提。
- **可观测**：每次 `add` / `search` 都经过统一的计量装饰器，记录耗时、token、调用次数，供 `EvalHarness` 直接消费。

## 4. 质量阈值（触发返工的红线）

| 触发条件 | 处理动作 |
| :--- | :--- |
| 某章实验无法在「纯 CPU + API」或「单卡 24GB」内复现 | 改用 API 实现或缩小数据规模，不得保留跑不动的示例 |
| 引用的基准数字仅来自单一厂商博客且无第三方复现 | 降级为 🏢 厂商自述，并加免责说明 |
| 两个来源的数字互相矛盾 | 标 ⚠️，两边都列出，写明分歧原因，不选边 |
| 某框架超过 6 个月无提交或已归档 | 移出教学载体候选，仅在生态章节作历史提及 |
| 章节出现无法还原为具体机制的营销术语 | 重写该段 |

## 5. Git 与发布管理

- **分支**：`main` 为可发布分支。开发用 `feat/chNN-<slug>`（章节）、`docs/<slug>`（文档）、`fix/<slug>`（修订）、`chore/<slug>`（工程）。
- **提交信息**：Conventional Commits，作用域用章节号。例如 `feat(ch03): 实现 BM25 与 dense 的混合检索`。
- **合并**：章节分支以 `--no-ff` 合入 main，保留章节开发的历史边界。
- **打 tag**：每阶段 DoD 满足后打 `vX.Y.0`，并同步 GitHub Release。
- **CI 门禁**：ruff、pytest、markdown 死链检查全绿方可合并。

详见 [CONTRIBUTING.md](./CONTRIBUTING.md)。

## 6. 明确不做的事

- 不做各框架的 API 使用手册（会过时，且官方文档更准）。框架只作为「机制的一个实例」出现。
- 不排「记忆框架天梯榜」。理由见第 10 章对现有基准的分析。
- 不复现完整规模的 LongMemEval / LoCoMo 评测（成本高且结论不可靠），只提供**可在几分钟内跑完的子集 harness** 并说明其局限。
