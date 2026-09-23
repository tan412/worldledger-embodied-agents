"""UMI / Zarr Adapter(阶段 7):末端 SE(3) 轨迹 + 夹爪宽度 + RGB。

UMI 源数据没有机器人关节 —— 不产 robot.joint_position,
走 umi_source_v1 策略(末端运动学 + 夹爪通道),不声称机器人可执行。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..evidence import EvidencePackage, Stream

POSE_KEYS = ("robot0_eef_pos", "eef_pos", "pose", "tcp_pose", "end_effector_pose")
ROT_KEYS = ("robot0_eef_rot_axis_angle", "eef_rot", "rotation")
GRIPPER_KEYS = ("robot0_gripper_width", "gripper_width", "gripper")


def load(store_path: Path, episode: int = 0) -> EvidencePackage:
    import zarr
    store_path = Path(store_path)
    root = zarr.open(str(store_path), mode="r")
    pkg = EvidencePackage(episode_id=f"{store_path.stem}-ep{episode}",
                          dataset_format="umi_zarr",
                          raw_files=[{"path": str(store_path), "bytes": None, "sha256": None}])

    def arrays(g, prefix=""):
        out = {}
        for k in g.array_keys():
            out[prefix + k] = g[k]
        for k in getattr(g, "group_keys", lambda: [])():
            out.update(arrays(g[k], prefix + k + "/"))
        return out
    flat = arrays(root)
    pkg.meta["arrays"] = {k: list(v.shape) for k, v in flat.items()}

    # episode 切片:episode_ends 约定(replay buffer)或整段
    lo, hi = 0, None
    if "meta/episode_ends" in flat:
        ends = np.asarray(flat["meta/episode_ends"])
        hi = int(ends[episode])
        lo = int(ends[episode - 1]) if episode > 0 else 0

    def find(keys):
        for name, arr in flat.items():
            base = name.split("/")[-1]
            if base in keys:
                return name, np.asarray(arr[lo:hi])
        return None, None

    pos_key, pos = find(POSE_KEYS)
    rot_key, rot = find(ROT_KEYS)
    if pos is not None:
        if rot is not None and rot.shape[1] >= 3:
            data = np.concatenate([pos, rot], axis=1)
            cols = ["x", "y", "z"] + [f"rot_{i}" for i in range(rot.shape[1])]
            field = f"{pos_key}+{rot_key}"
        else:
            data, cols, field = pos, ["x", "y", "z"], pos_key
        pkg.add(Stream("umi.gripper_pose", "observed", data=data, columns=cols,
                       unit="m/axis-angle", source_file=str(store_path),
                       source_field=field))
    gk, grip = find(GRIPPER_KEYS)
    if grip is not None:
        pkg.add(Stream("umi.gripper_width", "observed", data=grip,
                       unit="m", source_file=str(store_path), source_field=gk))
    for name, arr in flat.items():
        base = name.split("/")[-1]
        if base in POSE_KEYS + ROT_KEYS + GRIPPER_KEYS or name.startswith("meta/"):
            continue
        if "rgb" in base or "camera" in base or "img" in base:
            pkg.add(Stream("camera.rgb", "observed", source_file=str(store_path),
                           source_field=name, note=f"shape={list(arr.shape)},帧数组原样引用"))
        else:
            pkg.add(Stream(f"extra.{name}", "observed", source_file=str(store_path),
                           source_field=name, note=f"shape={list(arr.shape)}"))
    return pkg
