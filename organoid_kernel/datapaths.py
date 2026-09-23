"""本机数据镜像定位。

镜像的规范位置是 ~/Projects/data/<name>(本地开发机);服务器部署保持 ~/<name>
(见 mcp/deploy.sh 的 rsync 目标)。同一脚本两端都要跑,故按此顺序解析,
两处都不存在时报错并写明找过哪里——不静默猜测。
"""
from __future__ import annotations

from pathlib import Path

_CANDIDATE_BASES = (Path.home() / "Projects/data", Path.home())


def data_dir(name: str) -> Path:
    for base in _CANDIDATE_BASES:
        p = base / name
        if p.exists():
            return p
    raise SystemExit(
        f"数据镜像不存在: {name}(已找 " +
        " 与 ".join(str(b / name) for b in _CANDIDATE_BASES) + ")")
