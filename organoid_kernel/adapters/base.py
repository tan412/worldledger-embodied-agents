"""Adapter 公共件:原始文件清单与格式探测。Adapter 只做事实提取,不做裁决。"""
from __future__ import annotations

import json
from pathlib import Path

from ..hashing import sha256_file


def file_record(path: Path, with_hash: bool = True) -> dict:
    p = Path(path)
    return {"path": str(p), "bytes": p.stat().st_size,
            "sha256": sha256_file(p) if with_hash else None}


def detect_format(path: Path) -> str:
    """按目录/文件特征探测数据集格式。"""
    p = Path(path)
    if p.is_file():
        if p.suffix == ".bag":
            return "rosbag"
        if p.suffix in (".h5", ".hdf5"):
            return "hdf5"
        if p.suffix == ".mp4":
            return "ego_mp4"
    if p.is_dir():
        if (p / "trajectory.json").exists() and (p / "trajectory.npz").exists():
            return "organoid_trajectory_v2"
        info = p / "meta/info.json"
        if info.exists():
            ver = json.loads(info.read_text()).get("codebase_version", "")
            return f"lerobot_{ver}" if ver else "lerobot"
        if (p / "trajectory.csv").exists() and (p / "segments.json").exists():
            return "legacy_organoid"          # 老 organoid episode 产物目录
        if (p / "proprio_stats/proprio_stats.hdf5").exists():
            return "fourier"
        if (p / "data").is_dir() and list((p / "data").glob("trackingData_*.txt")):
            return "pico_vr"
        if list(p.glob("robot0_vio_eef_pose.csv")):
            return "gendas"
        if (p / ".zgroup").exists() or (p / "zarr.json").exists() or p.suffix == ".zarr":
            return "umi_zarr"
        if any(p.glob("*.bag")):
            return "rosbag"
        if any(p.glob("*.h5")) or any(p.glob("*.hdf5")):
            return "hdf5"
        if any(p.glob("*.mp4")):
            return "ego_mp4"
    return "unknown"
