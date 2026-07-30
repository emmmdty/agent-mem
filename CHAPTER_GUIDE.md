# 章节写作指南

> 给贡献者（和给 AI 协作者）的规范。写新章节前请完整读一遍。
> 已完成的第 1、2、3、10 章是参考实现——**有疑问时以它们为准**。

---

## 一、三条不可妥协的原则

### 1. 讲机制，不讲口号

凡「memory OS」「brain-inspired」「first-class resource」「context lake」这类词汇，必须还原成它对应的具体机制（调度策略？分页？self-editing block？），否则不写。

**反例**：「MemOS 把记忆提升为一等资源。」
**正例**：「MemOS 提出 MemCube 抽象，把明文/激活/参数三类记忆统一封装，并附带 provenance 与版本元数据。『一等资源』是这个封装的说法，落地的机制是统一的读写接口与调度器。」

### 2. 数字必须带来源标记

| 标记 | 用法 |
| :--- | :--- |
| 📄 | 论文原文，写明 arXiv 编号与具体表格/小节 |
| 🏢 | 厂商自述，无第三方复现，必须显式写「厂商自述」 |
| ⚠️ | 存在争议或未独立核实，两方数字都要列，写明分歧原因，不选边 |
| 🕐 | 时效性内容，必须写「截至 2026-07」 |

**本书自己的实验数据不用标记，但必须是真跑出来的。** 严禁编造表格数字。

### 3. 代价必须一等呈现

每个方案至少给出 token / 延迟 / 成本 三项中的两项。每章末必须有「代价与选型」小节。

**判断划不划算的核心问题是读写比**——这句话在第 2、3 章各出现过一次，后续章节该说时继续说。

---

## 二、诚实呈现失败

这是本书最重要的编写习惯，也是它与其他教程的主要差别。

**如果实验结果和你的预期不符，改叙述，不要改实验。** 已有的三个例子：

- 第 1 章：`BufferMemory` 的检索方案答错了一题。原叙述说「同样全对」，改成如实呈现并把它作为第 3 章的动机。
- 第 3 章：混合检索在 MiniBench 上**没有赢过**纯 dense。如实报告，并补一个可控的压力测试来展示 BM25 真正的价值场景。
- 第 3 章：写实验时踩了两个性能坑（`argsort` 而非 `argpartition`、纯 dense 模式也建 BM25 索引），把定位过程写进了正文。

**把踩过的坑写进教材，比写一个从未出错的教程有用得多。**

---

## 三、章节文件结构

```
docs/chapterN/
├── 第N章 中文标题.md          # 正文
├── code/
│   ├── 01_xxx.py             # 实验脚本，序号开头
│   ├── 02_xxx.py
│   └── requirements.txt      # 说明依赖与运行方式，即使无额外依赖
└── images/                   # 插图（可选）
```

正文文件名含空格，在 markdown 链接里用 `%20` 编码。

### 脚本要求

- 文件第一行 `# offline-ok` 表示「无网络、无 API Key 也能跑完」，CI 会真的执行它。做不到就别加这个标记。
- 模块 docstring 写清：运行命令、这个实验要回答什么问题、预计耗时。
- 输出末尾必须有「读法」段落，解释每个数字意味着什么、以及**它不能说明什么**。
- 不硬编码 API Key，一律从环境变量读。
- 需要 GPU 或联网的脚本，在文件头注明前置条件与预计耗时。

---

## 四、正文模板

```markdown
# 第N章 标题

> 💡 **一句话**：（一句能记住的话，也是复习锚点）
>
> ⏱ 约 X 分钟　|　难度 ●○○　|　前置：第 M 章　|　💻 `python -m minimem.learn chNN`
>
> **学完你能**
> 1. …（可检验的能力，不是「了解 X」）
> 2. …
>
> **主线产出**：`minimem/xxx.py` 里的 `XxxMemory`。
>
> **本章实验**：几个，各多久，是否需要 API Key。

## N.1 先把问题定准
（承接上一章留下的问题。别急着讲方案。）

## N.2 …

> 🤔 **先猜一猜**
>
> （在给出结论**之前**出一道选择题）
>
> A. …（一个「听起来非常合理但实际是错的」选项）
> B. …
>
> <details>
> <summary>想好了再展开</summary>
>
> **B。** …解释为什么直觉会错…
>
> </details>

（实验命令 + 实测输出 + 解读）

> ✅ **检查点 N.2**
>
> （问答题）
>
> <details>
> <summary>参考答案</summary>
> …
> </details>

## N.x 代价与选型
（表格：方案 / 写入代价 / 检索代价 / 依赖 / 什么时候选。外加给实践者的建议。）

## N.y 常见误区
（4~6 条，每条指回本章的具体小节或实测反例。）

## N.z 小结
（要点列表。最后一段承接到下一章。）

## 🔧 动手挑战
（🟢 五分钟 / 🟡 半小时到一小时 / 🔴 半天以上，各一道。
每道写明：起点文件、做对了会看到什么、可选的验证命令。）

## 🧠 复习卡片
（3~5 张，用 <details> 折叠。与 lessons/chNN.py 里的 cards 保持一致。）

## 🚪 下一章
（留一个本章解决不了的问题。）

## 参考文献
（带 📄/🏢/⚠️/🕐 标记。）

> **本章数字的可信度说明**
> （哪些是本机实测可复现，哪些是转引未核实。）
```

**篇幅参考**：第 1 章约 1.1 万字，第 2 章约 9 千字，第 3 章约 1.2 万字，第 10 章约 1.3 万字。

---

## 五、主线代码规范

### `MemoryStore` 契约

所有记忆实现继承 `minimem.base.MemoryStore`，只实现五个下划线钩子：

```python
def _add(self, item: MemoryItem, *, user_id: str) -> str: ...
def _search(self, query: str, *, user_id: str, k: int, **kwargs) -> list[SearchResult]: ...
def _update(self, memory_id: str, patch: dict, *, user_id: str) -> None: ...
def _delete(self, memory_id: str, *, user_id: str) -> None: ...
def _all(self, *, user_id: str) -> list[MemoryItem]: ...
```

基类负责参数校验、用户隔离、计量。**不要绕过基类的公开方法自己计时。**

硬约束：

- `user_id` 必填；越权访问抛 `CrossUserAccessError`（不是返回空）。
- 新字段一律进 `item.metadata`，不要往 `MemoryItem` 加字段。
- 删除时清理所有派生数据（向量缓存、索引、图边），并写一条断言测试。
- 设置类属性 `name`，harness 报表按它分组。

### 新实现完成后

在 `tests/test_base.py` 的 `ALL_STORES` 里加一行，全部契约测试自动生效：

```python
pytest.param(functools.partial(YourMemory, embedder=FakeEmbedder()), id="your"),
```

### LLM 调用

用 `minimem.utils.llm` 的 `LLMClient`：

```python
from minimem.utils.llm import LLMClient, ScriptedLLM, get_llm

result = llm.complete(prompt, op="extract")   # op 要有意义，报表按它分组
data = result.json(default={})                # 解析失败返回 default，不抛异常
```

**离线可跑是硬约束。** 需要 LLM 的实现必须接受注入的 `LLMClient`，并在章节脚本里用 `ScriptedLLM` 注册预设逻辑，保证 `# offline-ok` 成立。同时在正文里明确：**哪些结论换成真模型后会变**。

### 依赖

新依赖加进 `pyproject.toml` 的对应 extras。模型下载走 **ModelScope**（`python -m minimem.utils.fetch_model`），**不要走 HuggingFace**——国内不可达。

---

## 六、交互式课程

每章配一个 `minimem/learn/lessons/chNN.py`，导出 `LESSON`：

```python
LESSON = Lesson(
    chapter=N, title="…", minutes=X, difficulty=1..3, prerequisites=[…],
    steps=[
        Motto("一句话"),
        Predict(question=…, options={"A":…,"B":…}, answer="B", reveal=…, trap=…),
        Run(script="docs/chapterN/code/01_xxx.py", why=…, look_for=…, minutes=1),
        Recall(question=…, answer=…),
        Explain(task=…, checklist=[…]),
        Apply(level="🟢"/"🟡"/"🔴", task=…, starting_point=…, success_looks_like=…, verify=…),
        Cliff(text=…, next_chapter=…),
    ],
    cards=[Card(front=…, back=…, chapter=N), …],   # 至少 3 张
)
```

`tests/test_learn.py` 会自动校验：答案在选项内、脚本路径存在、三级难度齐全、成功标准非空、卡片正面唯一、每章有 motto 和 cliff。

**预测题的质量要求**：选项里必须有一个「大多数人的直觉」且它是错的。没有陷阱的预测题等于没有预测题。

**卡片只问「为什么」和「什么时候」，不问「是什么」**——定义查得到，判断查不到。

---

## 七、并行协作时的文件所有权

多人（或多个 AI 任务）同时写不同章节时，**以下共享文件不要碰**，由集成者统一修改：

```
minimem/__init__.py                 # 新类的导出
minimem/learn/lessons/__init__.py   # LESSON 的注册
tests/test_base.py                  # ALL_STORES 的登记
docs/_sidebar.md
README.md
ROADMAP.md
pyproject.toml
```

需要改这些文件时，在你的交付说明里写清楚「需要加哪一行」，集成者会统一处理。

**不要执行任何 git 操作**（commit / branch / merge / push）。

---

## 八、自检清单

```bash
ruff check . && ruff format --check .
pytest -q
python docs/chapterN/code/01_xxx.py        # 每个脚本都跑一遍
python -m minimem.learn chNN < /dev/null   # 课程在非交互模式下不卡住
```

内容层面：

- [ ] 每个第三方数字都有 📄/🏢/⚠️/🕐 标记与出处
- [ ] 每个方案都写了代价
- [ ] 正文里的每个表格都是真跑出来的
- [ ] 实验结果与预期不符的地方，改的是叙述而不是实验
- [ ] 有「常见误区」，且每条指向本章的具体反例
- [ ] 预测题有陷阱选项
- [ ] 挑战三级齐全，且都有可观察的成功标准
- [ ] 章末留了悬念
