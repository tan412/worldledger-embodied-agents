"""受控合成变体(移植自老 organoid synthesis 能力,收窄为确定性口径)。

对一个证据包生成扰动变体:time_scale(时间缩放)、root_offset(基座平移)、
joint_noise(关节高斯噪声)。三条纪律:
  * 确定性:同 seed 同输入 → 逐位相同的变体(numpy RandomState);
  * 变体是 derived revision:原始包不动,新包 origin=derived、provenance 记
    变体名/参数/seed/源内容哈希;
  * 合成产物永不冒充实采:episode_id 加 @variant 后缀,dataset_format 加 +synthetic。
"""
from __future__ import annotations

import copy

import numpy as np

from .evidence import EvidencePackage, Stream
from .hashing import sha256_array


def make_variant(pkg: EvidencePackage, name: str, seed: int = 0,
                 time_scale: float = 1.0, root_offset=(0.0, 0.0, 0.0),
                 joint_noise_rad: float = 0.0) -> EvidencePackage:
    rng = np.random.RandomState(seed)
    joint = pkg.get("robot.joint_position")
    if joint is None or joint.data is None:
        raise ValueError("无关节流,无法生成运动学变体")
    src_hash = sha256_array(np.asarray(joint.data))

    out = EvidencePackage(
        episode_id=f"{pkg.episode_id}@{name}",
        dataset_format=pkg.dataset_format + "+synthetic",
        fps=pkg.fps, raw_files=list(pkg.raw_files),
        meta={**pkg.meta, "synthetic_variant": name})

    prov = {"variant": name, "seed": seed, "time_scale": time_scale,
            "root_offset_m": list(root_offset), "joint_noise_rad": joint_noise_rad,
            "source_joint_hash": src_hash}

    arr = np.asarray(joint.data, dtype=float).copy()
    if joint_noise_rad > 0:
        arr = arr + rng.normal(0.0, joint_noise_rad, arr.shape)
    ts = np.asarray(joint.timestamps, dtype=float) if joint.timestamps is not None else None
    if ts is not None and time_scale != 1.0:
        ts = ts[0] + (ts - ts[0]) * time_scale
    out.add(Stream("robot.joint_position", "derived", data=arr,
                   columns=list(joint.columns), unit=joint.unit, timestamps=ts,
                   source_file=joint.source_file, source_field=joint.source_field,
                   provenance={**joint.provenance, "synthesis": prov}))

    base = pkg.get("robot.base_pose")
    if base is not None and base.data is not None:
        pose = np.asarray(base.data, dtype=float).copy()
        pose[:, 0:3] = pose[:, 0:3] + np.asarray(root_offset)
        bts = ts if ts is not None else base.timestamps
        out.add(Stream("robot.base_pose", "derived", data=pose,
                       columns=list(base.columns), timestamps=bts,
                       provenance={**base.provenance, "synthesis": prov}))

    # 其余流原样引用(标注时间随 time_scale 缩放)
    for sname, s in pkg.streams.items():
        if sname in out.streams:
            continue
        c = copy.copy(s)
        if sname == "annotation.language_segments" and s.data and time_scale != 1.0:
            c.data = [{**seg, "start_s": seg["start_s"] * time_scale,
                       "end_s": seg["end_s"] * time_scale} for seg in s.data]
            c.origin = "derived"
            c.provenance = {**s.provenance, "synthesis": prov}
        out.add(c)
    if ts is not None and time_scale != 1.0:
        out.fps = pkg.fps / time_scale
    out.adapter_notes = pkg.adapter_notes + [f"合成变体 {name}: {prov}"]
    return out
