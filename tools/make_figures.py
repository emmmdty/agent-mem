#!/usr/bin/env python3
"""生成全书的数据图表。

用法：``python tools/make_figures.py``

设计约定，三条：

1. **所有数字硬编码自章节正文的实测表格**，每个数据块上方标注它抄自哪一节。
   这些数字不在这里计算——它们由各章的实验脚本跑出来，本文件只负责画。
   改数字之前先去改那一节，然后把两边对齐。
2. **每张图出两份**：``-light.svg`` 与 ``-dark.svg``。站点有白天/夜间两套主题，
   单份图总有一套底下看不清。正文用两个 ``<img>`` 配 CSS 切换。
3. **画布透明 + 文字转路径**。透明是为了贴合两套主题的背景色；
   转路径是为了读者机器上没有中文字体时也能正确显示
   （matplotlib 的 SVG 默认把文字写成 <text>，缺字体就变方框）。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

OUT_DIR = "docs/images"

# 文字转路径：读者没装中文字体也不会变成方框
matplotlib.rcParams["svg.fonttype"] = "path"
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Microsoft YaHei",
    "Droid Sans Fallback",
]
matplotlib.rcParams["axes.unicode_minus"] = False


class Theme:
    """一套配色。取值与 docs/index.html 里的 --am-* 变量保持一致。"""

    def __init__(self, name, fg, muted, grid, accent, accent2, warn, ok):
        self.name = name
        self.fg = fg
        self.muted = muted
        self.grid = grid
        self.accent = accent
        self.accent2 = accent2
        self.warn = warn
        self.ok = ok


LIGHT = Theme("light", "#34495e", "#7f8c9a", "#e4e8ee", "#3f7ef0", "#8e44ad", "#c0392b", "#27ae60")
DARK = Theme("dark", "#c3ccd8", "#8794a5", "#2b3340", "#6ea8ff", "#c89bf0", "#f08d7c", "#7fd6c4")


def new_fig(theme, figsize=(7.2, 4.0)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)  # 透明画布，贴合两套主题
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.grid)
    ax.tick_params(colors=theme.muted, labelsize=9)
    ax.yaxis.label.set_color(theme.fg)
    ax.xaxis.label.set_color(theme.fg)
    ax.grid(True, color=theme.grid, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, name, theme):
    path = f"{OUT_DIR}/{name}-{theme.name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  ✓ {path}")


# ---------------------------------------------------------------------------
# F1.4　四种策略的累计输入 token
# 数据抄自：第 1 章 1.3「第一笔：成本是二次增长的」的表格
# ---------------------------------------------------------------------------
ROUNDS = [10, 50, 100, 500]
COST_SERIES = [
    ("无记忆", [570, 2850, 5700, 28500], "o"),
    ("全历史注入", [4755, 116775, 466050, 11630250], "s"),
    ("滑动窗口", [4755, 76320, 172170, 938970], "^"),
    ("检索 top-k", [1674, 9474, 19224, 97224], "D"),
]


def fig_token_growth(theme):
    fig, ax = new_fig(theme)
    colors = [theme.muted, theme.warn, theme.accent2, theme.accent]
    for (label, ys, marker), color in zip(COST_SERIES, colors, strict=True):
        lw = 2.4 if label in ("全历史注入", "检索 top-k") else 1.6
        ax.plot(ROUNDS, ys, marker=marker, label=label, color=color, linewidth=lw, markersize=5)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(ROUNDS)
    ax.set_xticklabels([str(r) for r in ROUNDS])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("对话轮数", fontsize=10)
    ax.set_ylabel("累计输入 token（对数轴）", fontsize=10)
    ax.set_title("轮数翻倍，全历史方案的账单翻四倍", color=theme.fg, fontsize=12, pad=12)

    # 500 轮处标出两条主线的倍数关系（11,630,250 / 97,224 ≈ 120）
    ax.annotate(
        "500 轮时约为检索方案的 120 倍",
        xy=(500, 11630250),
        xytext=(12, 2.6e6),
        color=theme.warn,
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=theme.warn, linewidth=1.2),
    )
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for text in leg.get_texts():
        text.set_color(theme.fg)
    fig.text(
        0.5,
        -0.06,
        "数据：第 1 章 1.3 节实测。仅计输入侧，假设每轮等长，未计 prompt 缓存折扣——请当量级参考。",
        ha="center",
        color=theme.muted,
        fontsize=7.5,
    )
    save(fig, "fig-1-4-token-growth", theme)


# ---------------------------------------------------------------------------
# F1.5　Lost in the Middle 的 U 形曲线
# 第 1 章 1.3「第三笔」。📄 Liu et al. arXiv:2307.03172 (TACL 2024)。
# 本书未复现其数值，因此**纵轴不标任何刻度**，只画趋势。
# ---------------------------------------------------------------------------
def fig_lost_in_middle(theme):
    fig, ax = new_fig(theme, figsize=(6.4, 3.4))
    xs = [i / 100 for i in range(101)]
    # 纯示意曲线：两端高、中部低。不代表任何实测数值。
    ys = [0.55 + 0.45 * (2 * (x - 0.5)) ** 2 for x in xs]
    ax.plot(xs, ys, color=theme.accent, linewidth=2.6)
    ax.fill_between(xs, ys, 0.4, color=theme.accent, alpha=0.10)

    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["输入开头", "输入中部", "输入结尾"])
    ax.set_yticks([])
    ax.set_ylim(0.4, 1.08)
    ax.set_xlabel("关键信息在长输入中的位置", fontsize=10)
    ax.set_ylabel("检索准确率（仅示意趋势）", fontsize=10)
    ax.set_title("窗口装得下，不等于模型用得好", color=theme.fg, fontsize=12, pad=12)
    ax.annotate(
        "中部最差",
        xy=(0.5, 0.55),
        xytext=(0.5, 0.72),
        ha="center",
        color=theme.warn,
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=theme.warn, linewidth=1.2),
    )
    fig.text(
        0.5,
        -0.08,
        "论文来源：Liu et al., Lost in the Middle, arXiv:2307.03172 (TACL 2024)。"
        "本书未复现其数值，此图仅示意趋势，纵轴无刻度。",
        ha="center",
        color=theme.muted,
        fontsize=7.5,
    )
    save(fig, "fig-1-5-lost-in-the-middle", theme)


# ---------------------------------------------------------------------------
# F3.3　dense / bm25 / hybrid 三方对比
# 数据抄自：第 3 章 3.4.3 的结果表
# ---------------------------------------------------------------------------
RETRIEVAL_MODES = ["dense", "bm25", "hybrid"]
RECALL = [88.2, 78.5, 86.8]
MRR = [0.896, 0.868, 0.938]
LATENCY_MS = [2.15, 0.34, 2.40]


def fig_hybrid_compare(theme):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.8))
    fig.patch.set_alpha(0.0)
    for ax in (ax1, ax2):
        ax.set_facecolor("none")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(theme.grid)
        ax.tick_params(colors=theme.muted, labelsize=9)
        ax.grid(True, axis="y", color=theme.grid, linewidth=0.8)
        ax.set_axisbelow(True)

    xs = range(len(RETRIEVAL_MODES))
    bars = ax1.bar(
        [x - 0.2 for x in xs], RECALL, width=0.38, label="召回率@5 (%)", color=theme.accent
    )
    ax1.bar_label(bars, fmt="%.1f", color=theme.fg, fontsize=8, padding=2)
    ax1b = ax1.twinx()
    ax1b.set_facecolor("none")
    for side in ("top", "right", "left"):
        ax1b.spines[side].set_visible(False)
    ax1b.tick_params(colors=theme.muted, labelsize=9)
    bars2 = ax1b.bar([x + 0.2 for x in xs], MRR, width=0.38, label="MRR", color=theme.accent2)
    ax1b.bar_label(bars2, fmt="%.3f", color=theme.fg, fontsize=8, padding=2)
    ax1b.set_ylim(0, 1.15)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(RETRIEVAL_MODES)
    ax1.set_ylabel("召回率@5 (%)", color=theme.fg, fontsize=10)
    ax1b.set_ylabel("MRR", color=theme.fg, fontsize=10)
    ax1.set_title("hybrid 的召回没超过 dense，赢的是 MRR", color=theme.fg, fontsize=11, pad=10)
    leg = ax1.legend(
        [bars, bars2],
        ["召回率@5 (%)", "MRR"],
        frameon=False,
        fontsize=9,
        loc="lower center",
        ncol=2,
    )
    for text in leg.get_texts():
        text.set_color(theme.fg)

    bars3 = ax2.bar(list(xs), LATENCY_MS, width=0.5, color=theme.muted)
    ax2.bar_label(bars3, fmt="%.2f", color=theme.fg, fontsize=8, padding=2)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(RETRIEVAL_MODES)
    ax2.set_ylabel("检索延迟 (ms)", color=theme.fg, fontsize=10)
    ax2.set_ylim(0, 3.0)
    ax2.set_title("代价：混合要多付一次向量搜索", color=theme.fg, fontsize=11, pad=10)

    fig.text(
        0.5,
        -0.04,
        "数据：第 3 章 3.4.3 节在 MiniBench 上的实测。",
        ha="center",
        color=theme.muted,
        fontsize=7.5,
    )
    fig.tight_layout()
    save(fig, "fig-3-3-hybrid-compare", theme)


# ---------------------------------------------------------------------------
# F7.2　摘要退化曲线
# 数据抄自：第 7 章 7.4 的实验二输出
# ---------------------------------------------------------------------------
SUMMARY_ROUNDS = ["原文", "第 1 次", "第 2 次", "第 3 次", "第 4 次", "第 5 次"]
SUMMARY_TOKENS = [125, 45, 45, 45, 34, 25]
SUMMARY_COVERAGE = [100, 62, 62, 62, 38, 25]


def fig_summary_decay(theme):
    fig, ax = new_fig(theme, figsize=(7.6, 4.0))
    xs = range(len(SUMMARY_ROUNDS))

    ax.plot(
        xs, SUMMARY_COVERAGE, marker="o", color=theme.warn, linewidth=2.6, label="关键点覆盖率 (%)"
    )
    ax.set_ylim(0, 112)
    ax.set_ylabel("关键点覆盖率 (%)", color=theme.fg, fontsize=10)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(SUMMARY_ROUNDS)
    for x, y in zip(xs, SUMMARY_COVERAGE, strict=True):
        ax.annotate(
            f"{y}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            color=theme.warn,
            fontsize=8.5,
        )

    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)
    ax2.tick_params(colors=theme.muted, labelsize=9)
    ax2.plot(
        xs,
        SUMMARY_TOKENS,
        marker="s",
        color=theme.muted,
        linewidth=1.6,
        linestyle="--",
        label="token 数",
    )
    ax2.set_ylabel("token 数", color=theme.fg, fontsize=10)
    ax2.set_ylim(0, 140)

    ax.set_title("摘要退化是单调的，而且不可见", color=theme.fg, fontsize=12, pad=12)
    # 标注放右上空白区，避开 62% 那三个数据点标签
    ax.annotate(
        "「XR-2049」被截成「XR-2」\n看起来正确、实际错误的值",
        xy=(5, 25),
        xytext=(3.15, 97),
        color=theme.warn,
        fontsize=8.5,
        ha="left",
        va="top",
        arrowprops=dict(
            arrowstyle="->", color=theme.warn, linewidth=1.2, connectionstyle="arc3,rad=0.18"
        ),
    )

    lines = ax.get_lines() + ax2.get_lines()
    labels = [str(ln.get_label()) for ln in lines]
    leg = ax.legend(lines, labels, frameon=False, fontsize=9, loc="lower left")
    for text in leg.get_texts():
        text.set_color(theme.fg)
    fig.text(
        0.5,
        -0.06,
        "数据：第 7 章 7.4 节实验二实测（规则式摘要）。LLM 摘要退化更慢，但方向相同。",
        ha="center",
        color=theme.muted,
        fontsize=7.5,
    )
    save(fig, "fig-7-2-summary-decay", theme)


# ---------------------------------------------------------------------------
# F10.2　投毒条数与 top-5 占领
# 数据抄自：第 10 章 10.8.2 的实测表
# ---------------------------------------------------------------------------
POISON_N = [1, 2, 3, 5]
POISON_OCCUPANCY = [20, 40, 60, 100]
POISON_EVICTED = [1, 2, 3, 5]


def fig_poisoning(theme):
    # 只画一组柱子。占比与「被挤掉的条数」是同一个量的两种说法
    # （top-5 里每多一条投毒，就正好挤掉一条真实记忆），
    # 画成两组会得到两根等高的柱子，等于没给信息。挤掉数改标在柱内。
    fig, ax = new_fig(theme, figsize=(7.0, 3.9))
    xs = list(range(len(POISON_N)))
    bars = ax.bar(xs, POISON_OCCUPANCY, width=0.5, color=theme.warn)
    ax.bar_label(bars, fmt="%d%%", color=theme.fg, fontsize=9.5, padding=3)

    ax.axhline(100, color=theme.muted, linewidth=1.0, linestyle=":")
    # 标注放最左边：右边会撞上 100% 那根柱子的数值标签
    ax.annotate(
        "top-5 已无任何真实记忆",
        (-0.42, 103),
        color=theme.muted,
        fontsize=8.5,
        ha="left",
    )

    ax.set_xticks(xs)
    # 「挤掉几条」并进刻度标签，不塞进柱子里——最矮那根柱子装不下两行字
    ax.set_xticklabels(
        [f"投毒 {n} 条\n挤掉 {e} 条真实记忆" for n, e in zip(POISON_N, POISON_EVICTED, strict=True)]
    )
    ax.set_ylim(0, 122)
    ax.set_ylabel("top-5 中投毒占比 (%)", color=theme.fg, fontsize=10)
    ax.set_title("五条就能完全占领 top-5", color=theme.fg, fontsize=12, pad=12)
    fig.text(
        0.5,
        -0.07,
        "数据：第 10 章 10.8.2 节实测。occupancy 衡量的是攻击面，不是攻击成功率——模型仍可能识破。\n"
        "注意：MINJA 论文报告的高成功率是在特定数据集与 agent 配置下测得，不能外推。",
        ha="center",
        color=theme.muted,
        fontsize=7.5,
    )
    save(fig, "fig-10-2-memory-poisoning", theme)


FIGURES = [
    fig_token_growth,
    fig_lost_in_middle,
    fig_hybrid_compare,
    fig_summary_decay,
    fig_poisoning,
]


def main() -> None:
    print(f"生成数据图表到 {OUT_DIR}/ …")
    for make in FIGURES:
        for theme in (LIGHT, DARK):
            make(theme)
    print(f"完成：{len(FIGURES)} 张图 × 2 套主题 = {len(FIGURES) * 2} 个文件。")


if __name__ == "__main__":
    main()
