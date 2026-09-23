"""源媒介与 UMI 源数据 Validator(阶段 7):视频完整性、末端运动学、夹爪通道。"""
from __future__ import annotations

import numpy as np

from ..ledger import Claim, ACCEPTED, REJECTED, NOT_EVALUATED, INCONCLUSIVE

REQUIRES = []          # 按流存在性各自决定;整体无硬性输入
CLAIMS = ["video_integrity", "end_effector_kinematics", "gripper_channel"]
EE_SPEED_LIMIT = 3.0       # m/s:手持夹爪合理速度上界
EE_ACC_LIMIT = 50.0        # m/s^2
EE_TELEPORT_SPEED = 8.0    # m/s:超过人手可能的速度 → 追踪伪影
EE_TELEPORT_STEP_M = 0.3   # m:单帧位移超过此值 → 瞬移
EE_TELEPORT_RATE_MAX = 6.0 # 次/分钟:瞬移频次超过此值判负,以下记附注


def run(pkg, inventory: dict, ctx: dict) -> tuple:
    receipt = {"schema": "organoid-kernel.source-media.v1", "episode_id": pkg.episode_id}
    claims = []

    rgb = pkg.get("camera.rgb")
    if rgb is None:
        claims.append(Claim("video_integrity", NOT_EVALUATED, "无视频流"))
    elif rgb.files:
        prov = rgb.provenance
        seq = prov.get("sequential_decodable_of_3")
        if seq is None:
            import cv2
            ok_all = True
            for f in rgb.files:
                cap = cv2.VideoCapture(str(f))
                ok, _ = cap.read()
                cap.release()
                ok_all &= bool(ok)
            receipt["video"] = {"files": len(rgb.files), "first_frame_decodable": ok_all}
            claims.append(Claim("video_integrity", ACCEPTED if ok_all else REJECTED,
                                "" if ok_all else "存在无法解码的视频"))
        else:
            seekable = prov.get("seekable", True)
            receipt["video"] = {"sequential_decodable_of_3": seq, "seekable": seekable,
                                **{k: prov.get(k) for k in ("fps", "frames", "resolution")}}
            if seq < 3:
                claims.append(Claim("video_integrity", REJECTED,
                                    f"顺序解码 3 帧仅成功 {seq} 帧"))
            elif not seekable:
                claims.append(Claim("video_integrity", INCONCLUSIVE,
                                    "顺序解码正常,但随机寻址失败(容器索引/时间基异常,"
                                    "抽帧类下游需按顺序读)", detail=receipt["video"]))
            else:
                claims.append(Claim("video_integrity", ACCEPTED))
    else:
        claims.append(Claim("video_integrity", INCONCLUSIVE, "视频为数组引用,未做解码检查"))

    ee = pkg.get("umi.gripper_pose")
    if ee is None or ee.data is None:
        claims.append(Claim("end_effector_kinematics", NOT_EVALUATED, "无末端位姿流"))
    else:
        arr = np.asarray(ee.data, dtype=float)
        fps = pkg.fps or 30.0
        # NaN 行 = 该帧未追踪到(适配器的诚实编码):速度只在相邻两帧都有效时计算
        valid = np.isfinite(arr[:, :3]).all(axis=1)
        coverage = float(valid.mean()) if len(valid) else 0.0
        pair_ok = valid[:-1] & valid[1:]
        dv = np.diff(arr[:, :3], axis=0)
        v = np.linalg.norm(dv[pair_ok], axis=1) * fps if pair_ok.any() else np.array([])
        a = np.abs(np.diff(v)) * fps if len(v) > 1 else np.array([])
        tracker_ratio = ee.provenance.get("tracker_accurate_ratio")
        receipt["end_effector"] = {
            "frames": len(arr), "tracking_coverage": round(coverage, 4),
            "tracker_accurate_ratio": tracker_ratio,
            "speed_p99_m_s": round(float(np.percentile(v, 99)), 3) if len(v) else None,
            "speed_max_m_s": round(float(v.max()), 3) if len(v) else None,
            "acc_p99_m_s2": round(float(np.percentile(a, 99)), 2) if len(a) else None}
        # 快动作(3~8 m/s)是人手可能的;瞬移(>8 m/s 或单步 >0.3 m)是追踪伪影
        disp = np.linalg.norm(dv[pair_ok], axis=1) if pair_ok.any() else np.array([])
        n_fast = int(((v > EE_SPEED_LIMIT) & (v <= EE_TELEPORT_SPEED)).sum()) if len(v) else 0
        n_tele = int(((v > EE_TELEPORT_SPEED) | (disp > EE_TELEPORT_STEP_M)).sum()) if len(v) else 0
        span_min = max((ts_span := (len(arr) / fps / 60.0)), 1e-6)
        tele_rate = n_tele / span_min
        receipt["end_effector"].update(fast_motion_samples=n_fast,
                                       teleport_samples=n_tele,
                                       teleports_per_minute=round(tele_rate, 2))
        bad = tele_rate > EE_TELEPORT_RATE_MAX
        reason = ""
        if n_tele:
            reason = (f"追踪瞬移 {n_tele} 处/{span_min:.1f} 分钟(单步>0.3 m 或 >8 m/s,"
                      f"峰值 {float(v.max()):.0f} m/s)")
            if tracker_ratio is not None and tracker_ratio < 0.5:
                reason += f";追踪状态 accurate 占比仅 {tracker_ratio:.0%},互证"
            if not bad:
                reason += " —— 频次在手部追踪常见范围,记为附注不判负"
        if not bad and coverage < 0.9:
            claims.append(Claim("end_effector_kinematics", INCONCLUSIVE,
                                f"追踪覆盖仅 {coverage:.0%}(缺帧为未追踪),运动学在有效帧上正常",
                                detail=receipt["end_effector"]))
        else:
            claims.append(Claim("end_effector_kinematics", REJECTED if bad else ACCEPTED,
                                reason, detail=receipt["end_effector"]))

    grip = pkg.get("umi.gripper_width")
    if grip is None or grip.data is None:
        claims.append(Claim("gripper_channel", NOT_EVALUATED, "无夹爪宽度流"))
    else:
        arr = np.asarray(grip.data, dtype=float).reshape(len(grip.data), -1)
        flat = bool(np.abs(arr - arr[0]).max() < 1e-9)
        neg = bool((arr < -1e-6).any())
        receipt["gripper"] = {"constant": flat, "negative_values": neg,
                              "range_m": [round(float(arr.min()), 4),
                                          round(float(arr.max()), 4)]}
        ok = not flat and not neg
        claims.append(Claim("gripper_channel", ACCEPTED if ok else INCONCLUSIVE,
                            "" if ok else ("夹爪通道全程常量" if flat else "夹爪宽度出现负值")))
    return receipt, claims
