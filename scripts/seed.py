"""把 data/kb 下的知识库文件灌入向量库。

用法：
    python scripts/seed.py            # 增量（内容未变则跳过）
    python scripts/seed.py --force    # 强制全部重新入库
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.container import build_container  # noqa: E402
from app.utils.logging import configure_logging  # noqa: E402


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()

    container = build_container(settings)
    force = "--force" in sys.argv

    summary = container.knowledge.load_directory(force=force)

    print("-" * 62)
    print(f"知识库目录 : {summary['directory']}")
    print(f"扫描文件   : {summary['files']}")
    print(f"本次入库   : {summary['ingested']}")
    print(f"跳过未变   : {summary['skipped']}")
    print(f"新增知识块 : {summary['chunks']}")
    print(f"向量库后端 : {container.store.name}")
    print(f"Embedding  : {container.embedder.name} (dim={container.embedder.dim})")
    print(f"当前总块数 : {container.knowledge.count()}")

    for note in container.notes:
        print(f"[降级提示] {note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
