"""HDF5 通用 Adapter(阶段 7):按常见命名约定映射,未识别数据集原样登记。

约定优先级:显式 mapping JSON > 常见键名探测(qpos/joint_positions/observations.qpos 等)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..evidence import EvidencePackage, Stream
from .base import file_record

JOINT_KEYS = ("qpos", "joint_positions", "observations/qpos", "observation/joint_position",
              "obs/qpos", "joint_position")
ACTION_KEYS = ("action", "actions")
TIME_KEYS = ("timestamp", "timestamps", "time")


def load(h5_path: Path, mapping: dict = None) -> EvidencePackage:
    import h5py
    h5_path = Path(h5_path)
    pkg = EvidencePackage(episode_id=h5_path.stem, dataset_format="hdf5",
                          raw_files=[file_record(h5_path)])
    mapping = mapping or {}

    with h5py.File(h5_path, "r") as f:
        flat = {}
        f.visititems(lambda name, obj: flat.__setitem__(name, obj.shape)
                     if hasattr(obj, "shape") else None)
        pkg.meta["datasets"] = {k: list(v) for k, v in flat.items()}

        def pick(explicit, candidates):
            if explicit and explicit in f:
                return explicit
            for k in candidates:
                if k in f:
                    return k
            return None

        jk = pick(mapping.get("joint_position"), JOINT_KEYS)
        if jk:
            arr = np.asarray(f[jk], dtype=float)
            tk = pick(mapping.get("timestamp"), TIME_KEYS)
            ts = np.asarray(f[tk], dtype=float) if tk else None
            fps = mapping.get("fps") or (
                round(1.0 / float(np.median(np.diff(ts))), 3) if ts is not None and len(ts) > 2
                else 0.0)
            pkg.fps = float(fps or 0.0)
            names = mapping.get("joint_names") or [f"joint_{i}" for i in range(arr.shape[1])]
            pkg.add(Stream("robot.joint_position", "observed", data=arr,
                           columns=list(names), unit=mapping.get("unit", "rad"),
                           timestamps=ts, source_file=str(h5_path), source_field=jk))
        ak = pick(mapping.get("action"), ACTION_KEYS)
        if ak:
            pkg.add(Stream("robot.action", "observed",
                           data=np.asarray(f[ak], dtype=float),
                           source_file=str(h5_path), source_field=ak))
        # 未识别数据集原样引用,不丢弃
        known = {jk, ak, pick(mapping.get("timestamp"), TIME_KEYS)}
        for k in flat:
            if k not in known:
                pkg.add(Stream(f"extra.{k}", "observed", source_file=str(h5_path),
                               source_field=k, note=f"shape={flat[k]},未映射,原样引用"))
    return pkg
