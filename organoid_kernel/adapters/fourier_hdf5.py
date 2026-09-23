"""傅利叶专有 HDF5 Adapter(按门户 example_data 的 proprio_stats.hdf5 实测布局)。

层级:state/{joint,end,effector,head,waist,robot}/*、action/*、timestamps(ns)。
亮点:state/robot/position+orientation 是**观测到的基座位姿**(VIO/里程),
state/end/wrench 是双腕六维力 —— 这两样在 LeRobot 交付里通常丢失。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..evidence import EvidencePackage, Stream
from .base import file_record


def load(episode_dir: Path) -> EvidencePackage:
    import h5py
    episode_dir = Path(episode_dir)
    h5 = episode_dir / "proprio_stats/proprio_stats.hdf5"
    pkg = EvidencePackage(episode_id=episode_dir.name, dataset_format="fourier_hdf5",
                          raw_files=[file_record(h5)],
                          meta={"vendor": "fourier"})
    with h5py.File(h5, "r") as f:
        ts = np.asarray(f["timestamps"], dtype=np.int64) / 1e9
        ts = ts - ts[0]
        pkg.fps = round(1.0 / float(np.median(np.diff(ts))), 2) if len(ts) > 2 else 0.0

        joint = np.asarray(f["state/joint/position"], dtype=float)
        pkg.add(Stream("robot.joint_position", "observed", data=joint,
                       columns=[f"arm_joint_{i}" for i in range(joint.shape[1])],
                       unit="rad", timestamps=ts, source_file=str(h5),
                       source_field="state/joint/position",
                       note="双臂 14 关节;厂商未给关节名,占位命名待 Profile 映射"))
        if "state/robot/position" in f:
            pos = np.asarray(f["state/robot/position"], dtype=float)
            quat = np.asarray(f["state/robot/orientation"], dtype=float)
            pkg.add(Stream("robot.base_pose", "observed",
                           data=np.concatenate([pos, quat], axis=1),
                           columns=["x", "y", "z", "qx", "qy", "qz", "qw"],
                           timestamps=ts, source_file=str(h5),
                           source_field="state/robot/position+orientation",
                           note="观测基座位姿(专有交付保留了它 —— LeRobot 转换通常丢)"))
        if "state/end/wrench" in f:
            w = np.asarray(f["state/end/wrench"], dtype=float).reshape(len(ts), -1)
            pkg.add(Stream("sensor.force_torque", "observed", data=w,
                           timestamps=ts, source_file=str(h5),
                           source_field="state/end/wrench", note="双腕六维力"))
        for field, name in (("state/effector/position (dexhand)", "extra.dexhand_position"),
                            ("state/waist/position", "extra.waist_position"),
                            ("state/head/position", "extra.head_position"),
                            ("action/joint/position", "robot.action")):
            if field in f:
                pkg.add(Stream(name, "observed",
                               data=np.asarray(f[field], dtype=float),
                               timestamps=ts, source_file=str(h5), source_field=field))
    cam_dir = episode_dir / "camera"
    vids = sorted(cam_dir.rglob("*.mp4")) if cam_dir.exists() else []
    if vids:
        pkg.add(Stream("camera.rgb", "observed", files=[str(v) for v in vids]))
    return pkg
