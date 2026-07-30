"""从 ModelScope 预下载模型：``python -m minimem.utils.fetch_model``

国内访问 HuggingFace 通常不可达，本书因此把 **ModelScope** 作为默认下载源。
这个脚本把需要的模型拉到本地 ``./models``，并打印出该设置的环境变量。

用法::

    python -m minimem.utils.fetch_model                    # 下载默认的句向量模型
    python -m minimem.utils.fetch_model BAAI/bge-base-zh-v1.5
    python -m minimem.utils.fetch_model --list             # 看本书用到哪些模型

如果连 ModelScope 也不方便，可以用 git lfs 手动克隆::

    git lfs install
    git clone https://www.modelscope.cn/BAAI/bge-small-zh-v1.5.git models/bge-small-zh-v1.5
    export AGENT_MEM_EMBED_MODEL=./models/bge-small-zh-v1.5
"""

from __future__ import annotations

import argparse
import sys

from minimem.utils.embedding import DEFAULT_MODEL, LOCAL_MODEL_DIR, download_from_modelscope

# 本书各章用到的模型。写明用途和大小，避免读者一次性下载几个 GB。
MODELS: dict[str, tuple[str, str]] = {
    DEFAULT_MODEL: ("第 3 章起：中文句向量，512 维", "约 95 MB"),
    "BAAI/bge-base-zh-v1.5": ("可选：更强的中文句向量，768 维", "约 400 MB"),
    "BAAI/bge-reranker-base": ("第 3 章可选：交叉编码重排器", "约 1.1 GB"),
}


def do_list() -> int:
    print("\n本书用到的模型（按需下载，不必一次装齐）\n")
    print(f"  {'模型':<32}{'用途':<38}{'大小'}")
    print("  " + "-" * 84)
    for name, (purpose, size) in MODELS.items():
        mark = " ← 默认" if name == DEFAULT_MODEL else ""
        print(f"  {name:<32}{purpose:<38}{size}{mark}")
    print(f"\n  下载到：{LOCAL_MODEL_DIR.resolve()}")
    print("  用法：python -m minimem.utils.fetch_model <模型名>\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从 ModelScope 下载本书需要的模型（国内推荐）",
    )
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL, help="ModelScope 模型 ID")
    parser.add_argument("--list", action="store_true", help="列出本书用到的模型")
    args = parser.parse_args(argv)

    if args.list:
        return do_list()

    print(f"\n从 ModelScope 下载：{args.model}")
    info = MODELS.get(args.model)
    if info:
        print(f"用途：{info[0]}    大小：{info[1]}")
    print(f"目标目录：{LOCAL_MODEL_DIR.resolve()}\n")

    try:
        path = download_from_modelscope(args.model)
    except ImportError:
        print("❌ 未安装 modelscope。执行：pip install modelscope", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 —— 下载失败要给出可操作的提示
        print(f"❌ 下载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\n可以试试 git lfs 手动克隆：\n"
            "  git lfs install\n"
            f"  git clone https://www.modelscope.cn/{args.model}.git models/{args.model.split('/')[-1]}\n"
            f"  export AGENT_MEM_EMBED_MODEL=./models/{args.model.split('/')[-1]}",
            file=sys.stderr,
        )
        return 1

    print(f"\n✅ 下载完成：{path}")
    print("\n把下面这行加进 .env 或当前 shell：")
    print(f"  export AGENT_MEM_EMBED_MODEL={path}")
    print("\n验证：python -m minimem.utils.check_env\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
