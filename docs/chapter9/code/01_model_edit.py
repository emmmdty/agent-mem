# needs-gpu
"""第 9 章实验一：用 ROME 做一次模型编辑，并测量它的副作用。

    ⚠️  这个脚本需要 GPU，且**本书作者未在 GPU 上验证过它的输出**。
        正文引用的所有数字均来自论文，不来自这个脚本。
        它写在这里是为了让有条件的读者能亲手跑一次——
        如果你跑通了（或跑不通），欢迎提 Issue 告诉我们。

前置条件
--------
* GPU 显存 ≥ 16GB（GPT-2 XL 约需 12GB；换更大的模型要相应增加）
* 单独的 Python 环境。模型编辑工具链对 torch / transformers 版本很敏感，
  与本书其他章节容易冲突：

      python -m venv .venv-ch9 && source .venv-ch9/bin/activate
      pip install -r docs/chapter9/code/requirements.txt

* 模型从 **ModelScope** 下载（国内 HuggingFace 不可达）：

      python -m minimem.utils.fetch_model gpt2-xl

预计耗时
--------
下载约 6GB，编辑本身几十秒到几分钟（取决于显存与模型大小）。

这个脚本做什么
--------------
1. 记录编辑前模型对三类问题的回答：
   - **目标问题**：要改的那条事实
   - **改写问题**：同一件事换个问法（考察 generalization）
   - **邻近问题**：与目标相关但不该被改的事实（考察 specificity / 涟漪效应）
2. 执行一次 ROME 编辑
3. 重新问一遍，对比三类问题的变化

**三类问题必须一起看。** 只报告目标问题的成功率，等于只报告
「我改的地方改成功了」——那几乎总是成功的。真正的难点在后两类。
"""

from __future__ import annotations

import sys

# ----------------------------------------------------------------------
# 编辑请求：把「法国的首都」从巴黎改成里昂
# ----------------------------------------------------------------------

EDIT_REQUEST = {
    "prompt": "The capital of France is",
    "subject": "France",
    "target_new": {"str": " Lyon"},
}

PROBES = {
    "目标问题": [
        "The capital of France is",
    ],
    "改写问题（考察泛化）": [
        "France's capital city is",
        "What is the capital of France? It is",
        "If you visit the capital of France, you are in",
    ],
    "邻近问题（不该被改）": [
        "The capital of Germany is",
        "Paris is a city in",
        "The population of Paris is about",
    ],
}


def load_model(model_path: str):
    """加载模型与分词器。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tok


def generate(model, tok, prompt: str, max_new_tokens: int = 8) -> str:
    import torch

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 贪心解码，保证可复现
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prompt) :].strip()


def probe_all(model, tok) -> dict[str, list[tuple[str, str]]]:
    results = {}
    for group, prompts in PROBES.items():
        results[group] = [(p, generate(model, tok, p)) for p in prompts]
    return results


def apply_rome(model, tok, request: dict):
    """执行一次 ROME 编辑。

    这里用 EasyEdit 的接口。它把 ROME / MEMIT / MEND 等方法包在同一个 API 下，
    是目前最省事的入口。

    ⚠️ EasyEdit 的 API 在不同版本间变动较大。如果下面这段跑不通，
    请对照你安装的版本的文档调整——**这本身就是这条技术路线不成熟的一个信号**。
    """
    try:
        from easyeditor import BaseEditor, ROMEHyperParams
    except ImportError:
        print("❌ 未安装 easyeditor。见 requirements.txt", file=sys.stderr)
        raise

    hparams = ROMEHyperParams.from_hparams("./hparams/ROME/gpt2-xl.yaml")
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=[request["prompt"]],
        target_new=[request["target_new"]["str"]],
        subject=[request["subject"]],
        keep_original_weight=False,
    )
    return edited_model, metrics


def report(before: dict, after: dict) -> None:
    print("\n" + "=" * 74)
    print("  编辑前后对比")
    print("=" * 74)

    for group in PROBES:
        print(f"\n  {group}")
        print("  " + "-" * 70)
        for (prompt, old), (_, new) in zip(before[group], after[group], strict=True):
            changed = "已改变" if old != new else "未变"
            print(f"    「{prompt}」")
            print(f"       编辑前：{old}")
            print(f"       编辑后：{new}   [{changed}]")

    print(
        """
  怎么读这张表：

  · **目标问题**改变了 → 编辑生效（这一项几乎总是成功）
  · **改写问题**没改变 → 泛化失败，用户换个问法就能拿到旧答案
  · **邻近问题**改变了 → 副作用，编辑污染了不该动的知识

  论文报告的「编辑成功率」通常主要反映第一项。
  **后两项才是这条路线的真实难度所在。**
"""
    )


def main() -> int:
    import os

    model_path = os.getenv("AGENT_MEM_EDIT_MODEL", "./models/gpt2-xl")
    print(f"\n第 9 章实验一：ROME 模型编辑\n模型：{model_path}")
    print("⚠️  需要 GPU。本书作者未在 GPU 上验证过本脚本的输出。\n")

    model, tok = load_model(model_path)

    print("编辑前，先问一遍三类问题…")
    before = probe_all(model, tok)

    print("执行 ROME 编辑：The capital of France is → Lyon")
    edited_model, metrics = apply_rome(model, tok, EDIT_REQUEST)
    print(f"EasyEdit 返回的指标：{metrics}")

    print("编辑后，再问一遍…")
    after = probe_all(edited_model, tok)

    report(before, after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
