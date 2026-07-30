# offline-ok
"""第 9 章实验二：模型编辑难在哪（概念演示，不涉及真实模型）。

    python docs/chapter9/code/02_why_editing_is_hard.py

**重要声明**：这个脚本**不是**真实的模型编辑。它用一个玩具知识库 + 推理规则，
模拟模型编辑的三个失败模式，好让你在没有 GPU 的情况下也能理解困难在哪。

真实的模型编辑（ROME / MEMIT / MEND）修改的是 Transformer 里特定层的权重，
行为比这个玩具复杂得多。但这三个失败模式是真实存在的，
而且是当前这条路线尚不成熟的核心原因。

三个失败模式：
  1. **涟漪效应**：改了「A 的首都是 B」，模型未必知道「B 是 A 的首都」
  2. **多编辑冲突**：编辑之间互相干扰，越改越乱
  3. **跨域泛化失败**：改了一个说法，换个问法又打回原形
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToyModel:
    """一个玩具「模型」：事实表 + 几条推理规则。

    它和真实 LLM 唯一的共同点是：**知识以某种纠缠的方式存着**，
    改一处会牵动别处。真实模型里这种纠缠发生在权重矩阵中，
    这里用显式的规则模拟。
    """

    facts: dict[tuple[str, str], str] = field(default_factory=dict)
    #: 反向规则：知道 (A, 首都) = B，就该知道 (B, 属于) = A
    inverse: dict[str, str] = field(default_factory=dict)
    #: 同义问法：「首都」和「首府」问的是同一件事
    aliases: dict[str, str] = field(default_factory=dict)
    edits: int = 0

    def ask(self, subject: str, relation: str) -> str:
        relation = self.aliases.get(relation, relation)
        if (subject, relation) in self.facts:
            return self.facts[(subject, relation)]
        # 反向推理
        inv = self.inverse.get(relation)
        if inv:
            for (s, r), o in self.facts.items():
                if r == relation and o == subject:
                    return s
        return "（不知道）"

    def edit(self, subject: str, relation: str, new_value: str, *, ripple: bool = False) -> None:
        """编辑一条事实。

        Args:
            ripple: 是否同时更新受影响的推论。
                **真实的模型编辑默认做不到这件事**——这正是涟漪效应的来源。
        """
        self.edits += 1
        self.facts[(subject, relation)] = new_value
        if not ripple:
            return
        # 「理想中」的编辑：把所有派生结论一起更新
        inv = self.inverse.get(relation)
        if inv:
            self.facts[(new_value, inv)] = subject


def build() -> ToyModel:
    return ToyModel(
        facts={
            ("法国", "首都"): "巴黎",
            ("巴黎", "位于"): "法国",
            ("巴黎", "人口"): "两百万",
        },
        inverse={"首都": "位于"},
        aliases={"首府": "首都", "capital": "首都"},
    )


def demo_ripple() -> None:
    print("\n  失败模式一：涟漪效应")
    print("  " + "-" * 66)

    m = build()
    print(f"\n    编辑前：法国的首都是 {m.ask('法国', '首都')}")
    print(f"            巴黎位于 {m.ask('巴黎', '位于')}")

    m.edit("法国", "首都", "里昂")
    print("\n    执行编辑：法国的首都 → 里昂")
    print(f"    编辑后：法国的首都是 {m.ask('法国', '首都')}  ✅")
    print(f"            **里昂**位于 {m.ask('里昂', '位于')}  ← 问题在这")
    print(f"            巴黎位于 {m.ask('巴黎', '位于')}  ← 旧推论还在")

    print(
        """
    改了一条事实，但**由它推出的结论没跟着改**。
    模型会同时相信「法国的首都是里昂」和「巴黎位于法国（且是首都）」。

    📄 这个现象在模型编辑文献里叫 **ripple effect（涟漪效应）**，
    是评估编辑质量的核心指标之一。要处理它，你得知道
    「哪些结论依赖这条事实」——而在真实模型里，**没人知道**。
"""
    )


def demo_multi_edit() -> None:
    print("\n  失败模式二：多编辑冲突")
    print("  " + "-" * 66)

    m = build()
    edits = [
        ("法国", "首都", "里昂"),
        ("里昂", "位于", "德国"),  # 与第一条冲突
        ("法国", "首都", "马赛"),  # 推翻第一条
    ]
    print()
    for subject, relation, value in edits:
        m.edit(subject, relation, value)
        print(f"    编辑 {m.edits}：{subject} 的 {relation} → {value}")

    print("\n    三次编辑后的状态：")
    for q in [("法国", "首都"), ("里昂", "位于"), ("马赛", "位于")]:
        print(f"      {q[0]} 的 {q[1]}：{m.ask(*q)}")

    print(
        """
    三条编辑互相干扰，得到一个自相矛盾的知识状态。

    真实系统里这个问题更严重：**编辑是累积的，且不可逐条撤销**。
    ROME 类方法每次编辑都在改权重，改了十次之后，
    你既说不清模型现在相信什么，也没法回退到第五次编辑前的状态。

    MEMIT 一类的批量编辑方法在这方面有改进（一次性求解多条编辑），
    但 ⚠️ 编辑数量上到成百上千时，副作用仍然显著。
"""
    )


def demo_generalization() -> None:
    print("\n  失败模式三：跨域泛化失败")
    print("  " + "-" * 66)

    m = build()
    m.edit("法国", "首都", "里昂")

    print("\n    编辑：法国的首都 → 里昂")
    print("    换几种问法：")
    for relation in ["首都", "首府", "capital"]:
        print(f"      问「法国的{relation}」→ {m.ask('法国', relation)}  ✅（别名表命中）")

    print("\n    但这个玩具模型有一张**显式的别名表**。真实模型没有。")
    m.aliases.clear()
    print("    清空别名表后：")
    for relation in ["首都", "首府", "capital"]:
        answer = m.ask("法国", relation)
        mark = "✅" if answer == "里昂" else "❌"
        print(f"      问「法国的{relation}」→ {answer}  {mark}")

    print(
        """
    ⚠️ 这正是模型编辑评估里的 **generalization** 维度：
    编辑必须对**同义改写、换语言、间接提问**都生效，否则用户换个问法
    就会得到旧答案。

    文献报告的编辑成功率通常在「原样重问」这个最容易的设定下最高，
    在改写和跨语言设定下明显下降。看到「编辑成功率 99%」这类数字时，
    第一个要问的是：**在哪个设定下？**
"""
    )


def main() -> None:
    print("\n第 9 章实验二：模型编辑难在哪")
    print("\n  ⚠️  这是**概念演示**，用玩具知识库模拟，不涉及任何真实模型。")
    print("     它的作用是让你在没有 GPU 时也能理解困难所在。")

    demo_ripple()
    demo_multi_edit()
    demo_generalization()

    print(
        """
  三个失败模式合起来说明了什么：

  · 知识在模型里是**纠缠**的，改一处会牵动别处，而你不知道牵动了哪些。
  · 编辑是**累积且难以撤销**的，多次编辑后的状态无法审计。
  · 编辑的**生效范围难以界定**，换个问法可能就失效。

  对比前八章的上下文类记忆：

      改一条记忆 → 改数据库里的一行 → 可审计、可回滚、可解释

  这就是本书把参数化记忆放在第 9 章而不是主线的原因。
  **不是它不重要，是它目前还不适合作为「用户告诉你什么，你记住什么」
  这条主路径的实现方式。**

  它合适的场景在正文 9.5 节：稳定的领域知识、风格适配、不常变的事实。
"""
    )


if __name__ == "__main__":
    main()
