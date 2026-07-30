"""环境自检：`python -m minimem.utils.check_env`

检查 Python 版本、按章可选依赖、LLM API 连通性、embedding 模型缓存，
并对缺失项给出"装什么、怎么装"的具体命令，而不是让你去猜。
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys

OK = "  ✅"
WARN = "  ⚠️ "
FAIL = "  ❌"

# 章节 -> (extras 名, [(模块名, pip 包名)])
CHAPTER_DEPS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "第 3 章 向量检索": (
        "vector",
        [
            ("sentence_transformers", "sentence-transformers"),
            ("chromadb", "chromadb"),
            ("rank_bm25", "rank-bm25"),
        ],
    ),
    "第 4~5 章 图记忆": (
        "graph",
        [
            ("networkx", "networkx"),
            ("kuzu", "kuzu"),
        ],
    ),
    "第 6~8 章 LLM 驱动": (
        "llm",
        [
            ("openai", "openai"),
            ("tiktoken", "tiktoken"),
        ],
    ),
    "第 10 章 评测": (
        "eval",
        [
            ("datasets", "datasets"),
            ("pandas", "pandas"),
            ("matplotlib", "matplotlib"),
        ],
    ),
}


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def check_python() -> bool:
    v = sys.version_info
    print("\nPython 环境")
    print(f"  版本：{platform.python_version()}  ({platform.system()} {platform.machine()})")
    if v >= (3, 10):
        print(f"{OK} 版本满足要求（>= 3.10）")
        return True
    print(f"{FAIL} 需要 Python 3.10 及以上，本书用到了 match 与 X | None 语法")
    return False


def check_core() -> bool:
    print("\n主线包 minimem")
    try:
        import minimem

        print(f"{OK} minimem {minimem.__version__} 已安装")
    except ImportError:
        print(f"{FAIL} 未安装。在仓库根目录执行：pip install -e .")
        return False

    if not _has("numpy"):
        print(f"{FAIL} numpy 缺失，执行：pip install numpy")
        return False
    print(f"{OK} numpy 可用")
    return True


def check_chapters() -> None:
    print("\n按章可选依赖")
    for chapter, (extra, deps) in CHAPTER_DEPS.items():
        missing = [pkg for mod, pkg in deps if not _has(mod)]
        if not missing:
            print(f"{OK} {chapter}：齐全")
        else:
            print(f"{WARN}{chapter}：缺 {', '.join(missing)}")
            print(f'      读到该章时执行：pip install -e ".[{extra}]"')


def check_llm_api() -> None:
    print("\nLLM API 配置（第 6 章起需要）")
    key = os.getenv("OPENAI_API_KEY", "")
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("AGENT_MEM_MODEL", "(未设置)")

    if not key:
        print(f"{WARN}未设置 OPENAI_API_KEY")
        print("      复制 .env.example 为 .env 并填入配置；第 1~5 章无需 API 也能跑通")
        return

    print(f"{OK} OPENAI_API_KEY 已设置（{key[:6]}…，长度 {len(key)}）")
    print(f"      BASE_URL = {base}")
    print(f"      MODEL    = {model}")

    if not _has("openai"):
        print(f'{WARN}openai 包未安装，无法测试连通性：pip install -e ".[llm]"')
        return

    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=base, timeout=15.0)
        resp = client.chat.completions.create(
            model=os.getenv("AGENT_MEM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": "ping，只回复 pong"}],
            max_tokens=5,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        print(f"{OK} API 连通，模型回复：{text!r}")
        if usage:
            print(f"      本次消耗 {usage.prompt_tokens} + {usage.completion_tokens} token")
    except Exception as exc:  # noqa: BLE001 —— 自检脚本要显示所有失败原因
        print(f"{FAIL} API 调用失败：{type(exc).__name__}: {exc}")
        print("      检查 BASE_URL 是否为 OpenAI 兼容路径（通常以 /v1 结尾）与模型名是否正确")


def check_embedding() -> None:
    print("\nEmbedding 模型（第 3 章起需要）")
    configured = os.getenv("AGENT_MEM_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

    if configured == "fake":
        print(f"{WARN}当前使用内置 FakeEmbedder（哈希向量，质量很差）")
        print("      可跑通全部流程，但检索效果不具参考性；第 3 章有专门的对照实验")
        return

    print(f"      配置的模型：{configured}")
    if not _has("sentence_transformers"):
        print(f'{WARN}sentence-transformers 未安装：pip install -e ".[vector]"')
        return

    from pathlib import Path

    cache = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = cache / "hub"
    marker = configured.replace("/", "--")
    if hub.exists() and any(marker in p.name for p in hub.iterdir()):
        print(f"{OK} 模型已在本地缓存：{hub}")
    else:
        print(f"{WARN}模型尚未下载，首次使用会自动下载（约 400MB）")
        print("      国内建议先设置：export HF_ENDPOINT=https://hf-mirror.com")


def main() -> int:
    print("=" * 60)
    print("agent-mem 环境自检")
    print("=" * 60)

    ok = check_python()
    ok = check_core() and ok
    check_chapters()
    check_llm_api()
    check_embedding()

    print("\n" + "=" * 60)
    if ok:
        print("核心环境就绪。跑第一个例子：")
        print("  python docs/chapter1/code/01_goldfish_agent.py")
    else:
        print("核心环境有问题，请先处理上面标 ❌ 的项。")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
