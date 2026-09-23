"""能力清单与派生规则(阶段 1 + §6.1)。

在任何 Validator 之前运行:
  * 盘点每个流的存在性与证据等级;
  * 识别"数据会骗人"的两类情况 —— 合成时间基(timestamp ≡ 帧号/fps)与
    占位通道(schema 有、内容为常量/全零),两者都降级,不得冒充测量;
  * 应用受条件约束的派生规则(静止基座推导):条件由数据自检、误差上界入收据。
"""
from __future__ import annotations

import numpy as np

from .evidence import EvidencePackage, Stream

SYNTHETIC_DT_EPS = 1e-4        # dt 逐点全等(容 float32 表示误差)→ 合成栅格
PLACEHOLDER_ACC_MAX = 0.1      # m/s^2:加速度计全程近零 → 占位
LEG_FROZEN_EPS = 0.05          # rad:腿关节全程摆幅小于此视为"站桩"
QUAT_SWAY_EPS = 0.025          # quat 分量摆幅 ≈ 2.9° 晃动
BASE_HEIGHT_GUESS_M = 0.9      # 忽略平移上界 = 基座高 × sin(晃动角)


def detect_synthetic_timebase(pkg: EvidencePackage) -> dict:
    """timestamp 列若为严格等距栅格,丢帧/重复帧/配对时长失去独立见证价值。"""
    joint = pkg.get("robot.joint_position")
    if joint is None or joint.timestamps is None:
        return {"applicable": False}
    ts = np.asarray(joint.timestamps, dtype=float)
    if len(ts) < 3:
        return {"applicable": False}
    dt = np.diff(ts)
    synthetic = bool(float(dt.max() - dt.min()) < SYNTHETIC_DT_EPS)
    return {"applicable": True, "synthetic": synthetic,
            "dt_median_s": round(float(np.median(dt)), 6),
            "dt_spread_s": round(float(dt.max() - dt.min()), 9)}


def detect_placeholder_imu(pkg: EvidencePackage) -> dict:
    """IMU 四元数恒为单位元且加速度计全零 → 占位值,降级 missing。"""
    imu = pkg.get("robot.imu")
    if imu is None or imu.origin == "missing" or imu.data is None:
        return {"applicable": False}
    arr = np.asarray(imu.data, dtype=float)   # 列: quat xyzw (+ 可选 acc xyz)
    cols = {c: i for i, c in enumerate(imu.columns)}
    quat_idx = [cols[c] for c in ("quat_x", "quat_y", "quat_z", "quat_w") if c in cols]
    acc_idx = [cols[c] for c in ("acc_x", "acc_y", "acc_z") if c in cols]
    if len(quat_idx) < 4:
        return {"applicable": False}
    quat = arr[:, quat_idx]
    identity = bool(np.abs(quat[:, :3]).max() < 1e-9)
    acc_zero = bool(len(acc_idx) == 3 and np.abs(arr[:, acc_idx]).max() < PLACEHOLDER_ACC_MAX)
    placeholder = identity and (acc_zero or not acc_idx)
    if placeholder:
        imu.origin = "missing"
        imu.note = "占位通道:quat 恒单位元" + (" 且加速度计全零" if acc_zero else "")
    return {"applicable": True, "placeholder": placeholder,
            "quat_identity": identity, "acc_all_zero": acc_zero}


def derive_static_base(pkg: EvidencePackage, leg_columns: list) -> dict:
    """静止基座推导:腿关节全程冻结 + IMU 晃动有界 ⇒ 基座平移取常数。

    这不是顶替:腿焊死 + 脚踩地 ⇒ 基座不动是运动学结论;
    条件不满足(机器人在走)时写静态根才是伪造 —— 那种情况这里什么都不做。
    """
    if pkg.has("robot.base_pose", "observed"):
        return {"applied": False, "reason": "已有观测基座位姿"}
    joint = pkg.get("robot.joint_position")
    imu = pkg.get("robot.imu")
    if joint is None or imu is None or imu.origin == "missing":
        return {"applied": False, "reason": "缺关节流或 IMU 不可用(占位/缺失)"}
    legs = [c for c in leg_columns if c in joint.columns]
    if not legs:
        return {"applied": False, "reason": "腿关节列不在轨迹中"}
    arr = np.asarray(joint.data, dtype=float)
    idx = [joint.columns.index(c) for c in legs]
    leg_range = float((arr[:, idx].max(0) - arr[:, idx].min(0)).max())

    cols = {c: i for i, c in enumerate(imu.columns)}
    quat = np.asarray(imu.data, dtype=float)[:, [cols[c] for c in
                                                 ("quat_x", "quat_y", "quat_z", "quat_w")]]
    quat = quat * np.sign(quat[:, 3:4] + 1e-12)           # 半球归一,防 q/-q 假摆幅
    quat_range = float((quat.max(0) - quat.min(0)).max())

    ok = leg_range < LEG_FROZEN_EPS and quat_range < QUAT_SWAY_EPS
    result = {"applied": bool(ok), "rule": "static_base_v1",
              "leg_range_rad": round(leg_range, 6), "quat_range": round(quat_range, 6),
              "thresholds": {"leg_frozen_rad": LEG_FROZEN_EPS, "quat_sway": QUAT_SWAY_EPS}}
    if not ok:
        result["reason"] = "腿有活动或 IMU 晃动超界,静止基座推导不成立"
        return result

    n = len(arr)
    sway_deg = float(np.degrees(2 * np.arcsin(min(1.0, quat_range / 2))))
    bound_m = float(BASE_HEIGHT_GUESS_M * np.sin(np.radians(sway_deg)))
    pose = np.zeros((n, 7))
    pose[:, 3:] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    pkg.add(Stream(
        name="robot.base_pose", origin="derived", data=pose,
        columns=["x", "y", "z", "qx", "qy", "qz", "qw"], unit="m/quat",
        frame="world(基座起点为原点)", timestamps=joint.timestamps,
        provenance={"rule": "static_base_v1",
                    "conditions_measured": {"leg_range_rad": leg_range,
                                            "quat_range": quat_range},
                    "neglected_translation_bound_m": round(bound_m, 4),
                    "orientation_source": "robot.imu quat(逐帧,半球归一)"},
        note=f"静止基座推导:忽略的平移上界≈{bound_m*100:.1f} cm"))
    result["neglected_translation_bound_m"] = round(bound_m, 4)
    return result


def build_inventory(pkg: EvidencePackage, leg_columns: list = ()) -> dict:
    """产出 capability-inventory 收据,并就地完成降级与派生。顺序:先识破,再派生。"""
    timebase = detect_synthetic_timebase(pkg)
    placeholder = detect_placeholder_imu(pkg)
    derivation = derive_static_base(pkg, list(leg_columns))
    return {
        "schema": "organoid-kernel.capability-inventory.v1",
        "episode_id": pkg.episode_id,
        "streams": {name: {"origin": s.origin,
                           "rows": (len(s.data) if s.data is not None
                                    and hasattr(s.data, "__len__") else None)}
                    for name, s in sorted(pkg.streams.items())},
        "timebase": timebase,
        "placeholder_imu": placeholder,
        "derivations": {"static_base": derivation},
    }
