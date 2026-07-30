# needs-gpu
"""第 9 章实验三：一次真实的权重编辑（不依赖 EasyEdit）。

    python docs/chapter9/code/03_minimal_edit.py

为什么有这个脚本
----------------
`01_model_edit.py` 用的是 EasyEdit——标准工具，但依赖锁定严格、API 在版本间
变动大，装不上的概率不低。这个脚本换一条路：**用一百来行代码自己实现一次
权重编辑**，只依赖 torch + transformers。

它实现的不是 ROME，而是 ROME 论文里的 **constrained fine-tuning（FT+L∞）
基线**：只微调某一层 MLP 的输出投影，并把权重改动量限制在一个 L∞ 球内。

这个选择是有代价的，必须说清楚：
* FT 基线的 **specificity（特异性）通常比 ROME 差**——它更容易伤到无关知识。
* 所以本脚本测出的副作用，是这类方法的**上界**，不是模型编辑能达到的最好水平。

但它足以验证第 9 章的核心论点，因为那三个失败模式（涟漪效应、泛化失败、
特异性副作用）是**这一整类方法共有的**，只是程度不同。

怎么读结果
----------
脚本会报告三个维度，**必须一起看**：

    Reliability    目标问题改对了吗          ← 几乎总是成功
    Generalization 换个问法还对吗            ← 难点之一
    Specificity    不该改的地方没被改吧      ← 难点之二

只报告第一项，等于只报告「我改的地方改成功了」。

环境
----
* GPU 显存：Qwen2.5-0.5B 约需 4GB；GPT-2 XL 约需 8GB
* 模型走 ModelScope 下载（国内 HuggingFace 不可达）
* 用 AGENT_MEM_EDIT_MODEL 指定模型目录，AGENT_MEM_EDIT_LAYER 指定编辑第几层
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class EditCase:
    """一次编辑请求，以及用来评估它的三类探针。"""

    name: str
    prompt: str
    old_answer: str
    new_answer: str
    paraphrases: list[str]  # 同一件事的其他问法 → 考察 generalization
    neighbors: list[tuple[str, str]]  # (提示, 期望仍然出现的词) → 考察 specificity


CASE = EditCase(
    name="法国的首都：巴黎 → 里昂",
    prompt="法国的首都是",
    old_answer="巴黎",
    new_answer="里昂",
    paraphrases=[
        "法国的首府是",
        "请问法国的首都是哪座城市？答：",
        "如果你去法国的首都旅游，你会到达",
        "The capital of France is",
    ],
    neighbors=[
        ("德国的首都是", "柏林"),
        ("意大利的首都是", "罗马"),
        ("英国的首都是", "伦敦"),
        ("巴黎是一座位于", "法国"),
    ],
)


def resolve_model() -> str:
    """拿到本地模型目录，必要时从 ModelScope 下载。"""
    configured = os.getenv("AGENT_MEM_EDIT_MODEL", DEFAULT_MODEL)
    if os.path.isdir(configured):
        return configured

    print(f"从 ModelScope 下载 {configured} …")
    from modelscope import snapshot_download

    return snapshot_download(configured, cache_dir=os.getenv("AGENT_MEM_MODEL_DIR", "./models"))


def load(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # 编辑要算梯度，fp32 更稳
        trust_remote_code=True,
    ).cuda()
    model.eval()
    return model, tok


def greedy(model, tok, prompt: str, max_new_tokens: int = 6) -> str:
    """贪心解码，保证可复现。"""
    import torch

    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def find_edit_layer(model, layer_idx: int):
    """定位要编辑的那一层 MLP 输出投影。

    不同架构的模块名不一样，这里覆盖 Qwen/Llama 系与 GPT-2 系。
    **这个函数本身就说明了一件事**：模型编辑方法与具体架构强耦合，
    换一个模型家族就要改代码——这是这条路线工程化程度低的表现之一。
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):  # Qwen / Llama
        return model.model.layers[layer_idx].mlp.down_proj
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):  # GPT-2
        return model.transformer.h[layer_idx].mlp.c_proj
    raise RuntimeError(f"不认识的模型结构：{type(model).__name__}")


def edit(
    model,
    tok,
    case: EditCase,
    *,
    layer_idx: int,
    steps: int = 30,
    lr: float = 5e-4,
    eps: float = 5e-4,
):
    """constrained fine-tuning：只改一层，且改动量限制在 L∞ 球内。

    L∞ 约束是关键。不加约束的话，模型会为了背下这一条事实
    把权重改得面目全非，其他能力一起崩掉——那就不叫「编辑」，
    叫「用一个样本做灾难性微调」。
    """
    import torch

    module = find_edit_layer(model, layer_idx)
    weight = module.weight

    for p in model.parameters():
        p.requires_grad_(False)
    weight.requires_grad_(True)

    w0 = weight.detach().clone()
    opt = torch.optim.Adam([weight], lr=lr)

    full = case.prompt + case.new_answer
    enc = tok(full, return_tensors="pt").to(model.device)
    prompt_len = tok(case.prompt, return_tensors="pt")["input_ids"].shape[1]

    labels = enc["input_ids"].clone()
    labels[:, :prompt_len] = -100  # 只对答案部分算损失

    print(f"\n  编辑第 {layer_idx} 层（{type(module).__name__}，权重 {tuple(weight.shape)}）")
    print(f"  目标：「{case.prompt}」→「{case.new_answer}」")
    print(f"  {'step':>6}{'loss':>10}{'‖Δw‖∞':>12}")

    for step in range(steps):
        opt.zero_grad()
        loss = model(**enc, labels=labels).loss
        loss.backward()
        opt.step()

        with torch.no_grad():  # 投影回 L∞ 球
            delta = (weight - w0).clamp_(-eps, eps)
            weight.copy_(w0 + delta)

        if step % 5 == 0 or step == steps - 1:
            linf = (weight - w0).abs().max().item()
            print(f"  {step:>6}{loss.item():>10.4f}{linf:>12.2e}")

        if loss.item() < 0.02:
            print(f"  {step:>6}{loss.item():>10.4f}   已收敛，提前停止")
            break

    weight.requires_grad_(False)
    return (weight - w0).abs().max().item()


def probe_all(model, tok, case: EditCase) -> dict[str, list[tuple[str, str]]]:
    return {
        "目标问题": [(case.prompt, greedy(model, tok, case.prompt))],
        "改写问题": [(p, greedy(model, tok, p)) for p in case.paraphrases],
        "邻近问题": [(p, greedy(model, tok, p)) for p, _ in case.neighbors],
    }


def report(case: EditCase, before: dict, after: dict, linf: float) -> None:
    print("\n" + "=" * 78)
    print("  编辑前后对比")
    print("=" * 78)

    for group in ("目标问题", "改写问题", "邻近问题"):
        print(f"\n  {group}")
        print("  " + "-" * 74)
        for (prompt, old), (_, new) in zip(before[group], after[group], strict=True):
            flag = "变了" if old != new else "未变"
            print(f"    「{prompt}」")
            print(f"       前：{old[:40]}")
            print(f"       后：{new[:40]}   [{flag}]")

    # ---- 三个维度的量化 ----
    reliability = case.new_answer in after["目标问题"][0][1]

    gen_hits = sum(1 for _, ans in after["改写问题"] if case.new_answer in ans)
    generalization = gen_hits / len(after["改写问题"])

    intact = sum(
        1
        for (_, old), (_, new) in zip(before["邻近问题"], after["邻近问题"], strict=True)
        if old == new
    )
    specificity = intact / len(after["邻近问题"])

    print("\n" + "=" * 78)
    print("  三个维度")
    print("=" * 78)
    print(f"\n  Reliability     目标问题改对了吗          {'✅ 是' if reliability else '❌ 否'}")
    print(
        f"  Generalization  换个问法还对吗            {generalization:.0%}  ({gen_hits}/{len(after['改写问题'])})"
    )
    print(
        f"  Specificity     不该改的地方没被改吧      {specificity:.0%}  ({intact}/{len(after['邻近问题'])} 保持不变)"
    )
    print(f"\n  权重改动量 ‖Δw‖∞ = {linf:.2e}")

    print(
        """
  怎么读：

  · Reliability 高是意料之中的——你改的地方当然改得动。
    **论文报告的「编辑成功率」主要反映这一项。**

  · Generalization 和 Specificity 才是难点。如果 Generalization 低，
    用户换个问法就能拿到旧答案；如果 Specificity 低，
    编辑污染了不该动的知识。

  · 注意这是 **constrained FT 基线**，不是 ROME。
    ROME 一类方法的 specificity 通常更好。所以这里测出的副作用
    是这类方法的**上界**，不是模型编辑的最好水平。

  · 但三个失败模式是这**一整类方法共有的**，只是程度不同。
    第 9 章正文的结论——「参数化记忆目前不适合作为
    『用户说什么你记什么』这条主路径」——不依赖于具体是哪种方法。
"""
    )


def main() -> int:
    print("\n第 9 章实验三：一次真实的权重编辑")
    print("（constrained fine-tuning，不依赖 EasyEdit）")

    try:
        import torch
    except ImportError:
        print("❌ 需要 torch。见 docs/chapter9/code/requirements.txt", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("❌ 需要 GPU。这个脚本标记为 # needs-gpu。", file=sys.stderr)
        return 1

    free, total = torch.cuda.mem_get_info()
    print(
        f"\nGPU：{torch.cuda.get_device_name(0)}，空闲 {free / 1024**3:.1f}GB / {total / 1024**3:.1f}GB"
    )

    model_path = resolve_model()
    print(f"模型：{model_path}")
    model, tok = load(model_path)

    n_layers = len(
        getattr(
            getattr(model, "model", model),
            "layers",
            getattr(getattr(model, "transformer", model), "h", []),
        )
    )
    layer_idx = int(os.getenv("AGENT_MEM_EDIT_LAYER", str(max(0, n_layers // 2))))
    print(f"共 {n_layers} 层，编辑第 {layer_idx} 层")

    print("\n编辑前，先问一遍三类问题…")
    before = probe_all(model, tok, CASE)

    linf = edit(model, tok, CASE, layer_idx=layer_idx)

    print("\n编辑后，再问一遍…")
    after = probe_all(model, tok, CASE)

    report(CASE, before, after, linf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
