# 附录 C：参考文献

> **用途**：全书引用的完整清单，按主题分组，每组分「必读」与「选读」。
>
> **更新政策**：正文新增引用时同步更新此页。arXiv 编号一经核实不再变动；
> 会议/期刊信息若有更新（如 arXiv 预印本被正式收录）请提 PR。
>
> **必读清单刻意控制在 12 篇以内**——列一百篇等于没列。

---

## 怎么用这份清单

如果你只有一个周末，读**必读清单的前 4 篇**（标 ⭐）。它们分别给你：一套分类框架、一个长上下文的边界认知、一个检索机制、一个评测的怀疑视角。

如果你要做研究，按主题分组通读，并注意每篇后面标注的「它解决了什么、没解决什么」。

**关于数字**：本清单里的论文都会报告一些性能数字。本书的立场见第 10 章——这些数字受评测子集、k 值、评判方式、底座模型、是否含写入处理五项影响，**跨论文不可直接比较**。

---

## 一、综述与分类体系

### 必读

⭐ **1. Du, E. Y., et al. _Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions_. arXiv:2505.00675 (2025).**

> **为什么必读**：本书的分类骨架（表征 × 操作）就来自这篇。它把表征分成参数化 / 上下文-结构化 / 上下文-非结构化，并定义了 Consolidation、Updating、Indexing、Forgetting、Retrieval、Compression 六种原子操作。
> 配套仓库：`Elvin-Yiming-Du/Survey_Memory_in_AI`。
> 首次出现：第 1 章 1.5 节。

⭐ **2. Zhang, Z., Bo, X., Ma, C., Li, R., Chen, X., Dai, Q., Zhu, J., Dong, Z., Wen, J.-R. _A Survey on the Memory Mechanism of Large Language Model based Agents_. arXiv:2404.13501 (2024). 后收入 ACM TOIS, DOI: 10.1145/3748302.**

> **为什么必读**：第一篇专门面向 agent memory 的系统综述，按 sources / forms / operations 三维切分。和上一篇对照读，你会直观看到「这个领域连分类轴都没有共识」。
> 首次出现：第 1 章 1.5 节。

### 选读

3. _From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs_. arXiv:2504.15965 (2025). —— 人类记忆到 AI 记忆的映射视角。读它时请对照第 1 章 1.5 节关于「心理学四分法在代码里大多只是标签」的讨论。

4. Li, Z., et al. _MemOS: A Memory OS for AI System_. arXiv:2507.03724 (2025)；短文版 _MemOS: An Operating System for Memory-Augmented Generation (MAG)_, arXiv:2505.22101. —— 提出 MemCube 抽象，把明文/激活/参数三类记忆统一封装。⚠️ 注意与 MemoryOS 是不同项目（见第 7 章 7.1 节）。

---

## 二、长上下文与 KV cache

### 必读

⭐ **5. Liu, N. F., et al. _Lost in the Middle: How Language Models Use Long Contexts_. arXiv:2307.03172, TACL 2024.**

> **为什么必读**：它把「窗口塞得下 ≠ 模型用得好」变成了可测量的事实——关键信息位于长输入中部时，检索准确率明显低于位于首尾时，形成 U 形曲线。这个结论支撑了本书「不要靠加长窗口解决问题」的立场。
> 首次出现：第 1 章 1.3 节。

### 选读

6. Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. _Efficient Streaming Language Models with Attention Sinks_. arXiv:2309.17453, ICLR 2024. —— attention sink 的出处。**读它时务必注意**：它解决的是 KV cache 的数值稳定性，**不扩展模型的有效上下文长度**（第 2 章 2.3 节）。

7. Zhang, Z., et al. _H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models_. NeurIPS 2023. —— KV cache 驱逐。

8. Li, Y., et al. _SnapKV: LLM Knows What You are Looking for Before Generation_. NeurIPS 2024.

9. Behrouz, A., et al. _Titans: Learning to Memorize at Test Time_. arXiv:2501.00663, NeurIPS 2025. —— 用「surprise」（关联记忆损失的梯度）决定记什么。思路值得借鉴（第 8 章讨论「什么值得记」时会回到它）。⚠️ 部分 SOTA 声称在社区中有呼吁更严格对比的声音。

---

## 三、检索增强

### 必读

⭐ **10. Cormack, G. V., Clarke, C. L. A., Buettcher, S. _Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods_. SIGIR 2009.**

> **为什么必读**：RRF 的原始论文，$k=60$ 的出处。本书第 3、4 章都在用它，而且都踩到了同一个坑（弱通道的噪声候选按名次劫持融合结果）。理解 RRF「只看名次」的设计，才能理解那个坑为什么必然存在。
> 首次出现：第 3 章 3.4.2 节。

### 选读

11. Robertson, S., Zaragoza, H. _The Probabilistic Relevance Framework: BM25 and Beyond_. Foundations and Trends in IR, 2009. —— BM25 的权威综述。

12. Gao, L., Ma, X., Lin, J., Callan, J. _Precise Zero-Shot Dense Retrieval without Relevance Labels_. arXiv:2212.10496. —— HyDE：先让 LLM 编一个假想答案，再用它检索。

13. Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H. _Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection_. arXiv:2310.11511, ICLR 2024. —— 它提出的那个问题值得记住：**不是每个查询都需要检索**。

---

## 四、结构化与图记忆

### 必读

⭐ **14. Gutiérrez, B. J., et al. _HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models_. arXiv:2405.14831, NeurIPS 2024.**

> **为什么必读**：本书第 4 章 PPR 检索的直接来源。它把「顺着关系走」变成了一次幂迭代，这是解决多跳问题最干净的一个思路。
> 📄 论文报告相比当时基线「up to 20%」提升、「10-30 times cheaper and 6-13 times faster」，2WikiMultiHopQA 的 Recall@5 从 ColBERTv2 的 68.2 提升到 89.1。
> 首次出现：第 4 章 4.3 节。

### 选读

15. Sarthi, P., et al. _RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval_. ICLR 2024. —— GMM 软聚类 + LLM 摘要建树。

16. Edge, D., et al. _From Local to Global: A Graph RAG Approach to Query-Focused Summarization_. Microsoft, 2024. —— 实体图 + 社区检测 + 社区摘要。⚠️ 注意其 prompt 长度可达 10⁴ token 量级。

17. Guo, Z., et al. _LightRAG: Simple and Fast Retrieval-Augmented Generation_. 2024. —— 双层检索 + 增量更新。

18. HippoRAG 2 —— dense-sparse 编码 + query-to-triple 匹配，锚定更准。

19. Page, L., Brin, S., et al. _The PageRank Citation Ranking_. 1998. —— PPR 的源头。

---

## 五、时间感知与知识演化

### 必读

⭐ **20. Rasmussen, P., et al. _Zep: A Temporal Knowledge Graph Architecture for Agent Memory_. arXiv:2501.13956.**

> **为什么必读**：bi-temporal（生效时间 vs 记录时间）与 provenance 的设计范例，本书第 5 章的直接来源。
> ⚠️ **它还有一个额外的阅读价值**：论文自己指出所用的 DMR 基准「each conversation contains only 60 messages, easily fitting within current LLM context windows」——**这种自我限定比大多数技术报告都诚实**，而这类自承恰恰是判断一份报告可信度的好信号。
> 首次出现：第 5 章 5.2.1 节。

---

## 六、agentic 记忆与自组织

### 选读

21. Xu, W., et al. _A-MEM: Agentic Memory for LLM Agents_. arXiv:2502.12110, NeurIPS 2025. —— Zettelkasten 式的 note 生成与动态链接。

22. Chhikara, P., et al. _Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory_. arXiv:2504.19413, ECAI 2025. —— extract-update 流水线。

> ⚠️ **关于 21 与 22 的效果数字**：Mem0 与 Zep 在 LoCoMo/LongMemEval 上互报的分数存在
> **四个互相矛盾的版本**（58.44% / 65.99% / 75.14% / 84%），且多个仓库报告无法复现。
> 本书不引用任何一方的数字作为结论，完整拆解见第 10 章 10.4 节。

---

## 七、OS 式分层

### 必读

⭐ **23. Packer, C., et al. _MemGPT: Towards LLMs as Operating Systems_. arXiv:2310.08560.**

> **为什么必读**：core / recall / archival 三层与 self-editing memory blocks 的出处。
> 读它时请带着第 7 章 7.2 节的问题：**OS 隐喻里哪些是机制、哪些是叙事**。
> 答案是：self-editing memory blocks 与异步 ingestion 是实打实的机制，其余主要是类比。
> 产品化为 Letta。首次出现：第 7 章 7.1 节。

### 选读

24. MemoryOS —— 分层 STM/MTM/LPM 与热度换页。⚠️ 与 MemOS 是**不同项目**。

---

## 八、经验与技能记忆

### 必读

⭐ **25. Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., Bernstein, M. S. _Generative Agents: Interactive Simulacra of Human Behavior_. UIST 2023, DOI: 10.1145/3586183.3606763.**

> **为什么必读**：recency / relevance / importance 三项打分与 reflection 的出处，被后续大量工作沿用。
> 📄 论文报告去掉 reflection 后，agent 在 48 模拟小时内退化为重复的、无上下文的响应。
> ⚠️ 但请注意第 8 章的实测：**三项打分是为「模拟有连续生活的 agent」设计的**，
> 照搬到问答系统会伤害多跳能力。
> 首次出现：第 1 章 1.7 节。

### 选读

26. Shinn, N., et al. _Reflexion: Language Agents with Verbal Reinforcement Learning_. NeurIPS 2023. —— 📄 报告 HumanEval 的 pass@1 从 0.80 提升到 0.91。核心思想：**把失败的教训写进记忆**。

27. Wang, G., et al. _Voyager: An Open-Ended Embodied Agent with Large Language Models_. 2023. —— 技能库存**可执行代码**。📄 报告去掉技能库后 tech-tree 里程碑速度损失约 15.3×。⚠️ **这是 Minecraft 环境的结果（动作空间明确、成功可自动验证），不可外推。**

28. Wang, Z., et al. _Agent Workflow Memory_. 2024. —— 带 trigger 与参数槽的 workflow，本书第 8 章走的路线。

29. ExpeL、Synapse、ReasoningBank —— 同一族的其他做法。

---

## 九、参数化记忆与模型编辑

### 选读

30. Meng, K., et al. _Locating and Editing Factual Associations in GPT_ (ROME). NeurIPS 2022.

31. Meng, K., et al. _Mass-Editing Memory in a Transformer_ (MEMIT). ICLR 2023. —— ROME 的批量版。

32. Mitchell, E., et al. _Fast Model Editing at Scale_ (MEND). ICLR 2022.

33. Wang, Y., et al. _MemoryLLM: Towards Self-Updatable Large Language Models_. arXiv:2402.04624；后续工作 M+（ICML 2025）。

34. Memory³ / Larimar / TTT layers —— 各自给模型加一块显式或隐式的记忆结构。

35. EasyEdit —— 把上述编辑方法统一在同一 API 下的工具箱。🕐 API 在版本间变动较大。

---

## 十、评测基准

### 必读

⭐ **36. Wu, D., et al. _LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory_. arXiv:2410.10813, ICLR 2025.**

> **为什么必读**：它把记忆能力拆成五类——信息提取、多会话推理、时序推理、知识更新、**abstention（拒答）**。最后一类在实践中出问题的频率极高，而在总分里几乎看不出来。本书 MiniBench 的八类能力划分受它启发。
> ⚠️ 关于其 S 子集「更像上下文窗口测试」的批评，见第 10 章 10.3.4 节。
> 首次出现：第 10 章 10.2.1 节。

### 选读

37. LoCoMo —— 长对话记忆基准。⚠️ 其答案键质量与评判宽松问题见第 10 章 10.3 节。

38. RULER / HELMET / ∞Bench / LongBench 系列 —— 长上下文基准。🕐 迭代较快。

39. HotpotQA / 2WikiMultihopQA / MuSiQue —— 多跳问答。

---

## 十一、安全与隐私

### 必读

⭐ **40. Dong, S., et al. _MINJA: Memory INJection Attack on LLM Agents_. arXiv:2503.03704, NeurIPS 2025.**

> **为什么必读**：它描述的威胁模型最反直觉也最重要——**攻击者不需要访问你的记忆库，只需要能正常对话**。说过的话被写进记忆，记忆在将来被检索时发作。这让「一次对话」变成「一次持久化的注入」。
> ⚠️ 论文报告的高注入成功率是特定数据集、特定 agent 配置、攻击者可多轮交互条件下的结果，**不可外推**。
> 首次出现：第 10 章 10.8 节。

### 选读

41. AgentPoison —— 假设攻击者能直接访问记忆库，用优化过的触发器植入后门。

42. MEXTRA —— 记忆内容泄漏。

---

## 十二、⚠️ 待核实的引用

以下 arXiv 编号出现在本项目所依据的调研报告中，**本书未独立核实**。它们在正文中出现时均已标注 ⚠️，此处单列以免与已核实清单混淆。

| 编号 | 声称的内容 | 状态 |
| :--- | :--- | :--- |
| arXiv:2603.07670 | _Memory for Autonomous LLM Agents_，被引用于「巩固很少自动发生」这一判断 | ⚠️ 编号形如 2026 年论文，未核实。第 1 章引用它时已注明「仅作线索列出」，且该判断在本书中另有独立的实证支撑（第 6、8 章） |
| arXiv:2511.10523 | _ConvoMem_，被引用于「多数基准类别样本量 < 100」 | ⚠️ 未核实 |
| arXiv:2603.25973 | _MemoryCD_ | ⚠️ 未核实，仅作存在性提及 |
| arXiv:2510.01353 | _MEMTRACK_ | ⚠️ 未核实，仅作存在性提及 |

**另有一类未核实来源**：第 10 章引用的「Penfield Labs 2026 审计」（称 LoCoMo 1540 题中有 99 处答案键错误、gpt-4o-mini judge 对主题相邻的错误答案接受率 62.81%），以及 Mem0 与 Zep 在 GitHub issue 与厂商博客上的互斥数字。这些均**转引自公开渠道，本书未独立复现**，第 10 章已明确标注。

> **为什么要单列这一节**：一份参考文献清单的价值，一半在于它列了什么，
> 另一半在于它**诚实交代了哪些没核实**。把未核实的引用混进主清单，
> 等于把不确定性偷偷转嫁给读者。

---

## 十三、非论文来源

🕐 以下内容变动频繁，标注为「截至 2026-07」。

- **Anthropic**. _Effective Context Engineering for AI Agents_（2025-09）—— 主张 just-in-time 加载与轻量持久化。
- **Anthropic**. Claude memory tool（2025-09-29 beta，`memory_20250818`）与 context editing（`clear_tool_uses_20250919`）。🏢 官方自述的 token 节省数字见第 7、10 章的可信度讨论。
- **阿里**. MemoryScope（`modelscope/MemoryScope`）—— 中文生态里少见的开源分层记忆实现，详见附录 B.5。
- **MTEB / C-MTEB** —— 嵌入模型榜单。🕐 变动频繁，且榜单排名与你的具体任务相关性有限。**用它筛候选，不要用它做决定。**

---

← [附录 B 选型对比表](附录B-选型对比表.md)　|　[附录 D 术语表](附录D-术语表.md) →
