"""通用运动学预检(§7.2):从 validate-g1 抽象出的轨迹级 Validator。

五个子检查与上游同口径:关节覆盖、限位、限速(差分)、根四元数、唯一帧。
重复帧冲突量先做四元数 q/−q 半球归一再比对。
"""
from __future__ import annotations

import numpy as np

from ..fk import load_urdf
from ..ledger import Claim, ACCEPTED, REJECTED

REQUIRES = ["robot.joint_position", "robot.model"]
CLAIMS = ["kinematic_precheck"]
QUAT_NORM_P95_MAX = 1e-2


def run(pkg, inventory: dict, ctx: dict) -> tuple:
    profile = ctx["profile"]
    model = ctx.get("urdf_model") or load_urdf(profile.urdf_path(), profile.mesh_path())
    ctx["urdf_model"] = model
    limits = model.limits()
    joint = pkg.get("robot.joint_position")
    arr = np.asarray(joint.data, dtype=float)
    cols = joint.columns
    fps = pkg.fps or 30.0
    frames = (np.asarray(joint.provenance.get("frame_index"), dtype=float)
              if joint.provenance.get("frame_index") is not None
              else np.arange(len(arr), dtype=float))

    checks, pos_v, vel_v, p99 = {}, {}, {}, {}

    # 1) 关节覆盖:轨迹每列都能在模型限位表里找到
    missing = [c for c in cols if c not in limits]
    checks["model_joint_coverage"] = not missing

    # 2) 限位
    for i, c in enumerate(cols):
        if c not in limits:
            continue
        lo, hi, _ = limits[c]
        if lo == hi == 0:
            continue
        v = arr[:, i]
        over = (v < lo) | (v > hi)
        if over.any():
            excess = np.maximum(lo - v, v - hi)
            pos_v[c] = {"count": int(over.sum()),
                        "max_excess_rad": round(float(excess[over].max()), 6),
                        "limit": [lo, hi]}
    checks["position_limits"] = not pos_v

    # 3) 限速:差分角速度 |Δq|·fps/Δ帧号,采样间隙如实表现为帧号跳变
    dframe = np.diff(frames)
    valid = dframe > 0
    for i, c in enumerate(cols):
        if c not in limits:
            continue
        _, _, vlim = limits[c]
        if vlim <= 0:
            continue
        dv = np.abs(np.diff(arr[:, i]))[valid] * fps / dframe[valid]
        over = dv > vlim
        if over.any():
            vel_v[c] = {"count": int(over.sum()),
                        "max_observed_rad_s": round(float(dv[over].max()), 4),
                        "limit_rad_s": vlim}
        p99[c] = round(float(np.percentile(dv, 99)), 4) if len(dv) else 0.0
    checks["velocity_limits"] = not vel_v

    # 4) 根四元数模长(有基座位姿流才查;派生的常量基座恒过)
    base = pkg.get("robot.base_pose")
    quat_p95 = None
    if base is not None and base.origin != "missing" and base.data is not None:
        q = np.asarray(base.data, dtype=float)[:, 3:7]
        err = np.abs(np.linalg.norm(q, axis=1) - 1.0)
        quat_p95 = float(np.percentile(err, 95))
        checks["root_quaternion"] = quat_p95 <= QUAT_NORM_P95_MAX
    else:
        checks["root_quaternion"] = True

    # 5) 唯一帧 + 重复帧冲突(四元数半球归一后再比冲突量)
    uniq, first_seen, conflict = True, {}, 0.0
    for k, f in enumerate(frames.astype(int)):
        if f in first_seen:
            uniq = False
            other = first_seen[f]
            d = float(np.abs(arr[k] - arr[other]).max())
            if base is not None and base.data is not None:
                qa = np.asarray(base.data[k][3:7], dtype=float)
                qb = np.asarray(base.data[other][3:7], dtype=float)
                if np.dot(qa, qb) < 0:
                    qb = -qb                      # q 与 −q 是同一姿态
                d = max(d, float(np.abs(qa - qb).max()))
            conflict = max(conflict, d)
        else:
            first_seen[f] = k
    checks["unique_source_frames"] = uniq

    passed = all(checks.values())
    receipt = {
        "schema": "organoid-kernel.kinematic-precheck.v1",
        "episode_id": pkg.episode_id,
        "robot_model": profile.name,
        "checks": checks,
        "position_violations": pos_v or None,
        "velocity_violations": vel_v or None,
        "observed_velocity_p99_rad_s": p99,
        "root_quaternion_norm_error_p95": quat_p95,
        "duplicate_frame_conflict_rad": round(conflict, 6) if not uniq else None,
        "kinematic_precheck_passed": passed,
        "frames": len(arr), "fps": fps,
    }
    failed = [k for k, v in checks.items() if not v]
    return receipt, [Claim("kinematic_precheck", ACCEPTED if passed else REJECTED,
                           "" if passed else f"未过: {failed}",
                           detail={"failed": failed})]
