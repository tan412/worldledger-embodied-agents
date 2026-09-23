"""传感器健康(阶段 7):力/触觉/深度通道的独立 Validator,不套用人手规则。"""
from __future__ import annotations

import numpy as np

from ..ledger import Claim, ACCEPTED, INCONCLUSIVE, NOT_EVALUATED

REQUIRES = ["robot.joint_position"]        # 至少有主时间轴;各通道存在才各自查
CLAIMS = ["sensor_health"]
FT_SATURATION = 500.0                      # N / Nm:六维力饱和阈(保守)


def run(pkg, inventory: dict, ctx: dict) -> tuple:
    receipt = {"schema": "organoid-kernel.sensor-health.v1",
               "episode_id": pkg.episode_id, "channels": {}}
    findings = []

    ft = pkg.get("sensor.force_torque")
    if ft is not None and ft.data is not None:
        arr = np.asarray(ft.data, dtype=float)
        flat = not bool(np.abs(np.diff(arr, axis=0)).max() > 1e-9) if len(arr) > 1 else True
        sat = float(np.abs(arr).max())
        receipt["channels"]["force_torque"] = {
            "dims": int(arr.shape[1]) if arr.ndim > 1 else 1,
            "constant": flat, "abs_max": round(sat, 2),
            "zero_bias_estimate": [round(float(v), 3) for v in
                                   np.median(arr, axis=0)][:6] if arr.ndim > 1 else None}
        if flat:
            findings.append("六维力全程常量(疑似未接或占位)")
        if sat > FT_SATURATION:
            findings.append(f"六维力峰值 {sat:.0f} 疑似饱和")

    tac = pkg.get("sensor.tactile")
    if tac is not None and tac.data is not None:
        arr = np.asarray(tac.data, dtype=float)
        live = float((np.abs(np.diff(arr, axis=0)).max(axis=0) > 1e-9).mean()) if len(arr) > 1 else 0.0
        receipt["channels"]["tactile"] = {"dims": int(arr.shape[1]),
                                          "live_cell_ratio": round(live, 4)}
        if live < 0.01:
            findings.append(f"触觉阵列活跃单元仅 {live:.1%}(疑似未接)")

    depth = pkg.get("camera.depth")
    if depth is not None and depth.files:
        receipt["channels"]["depth"] = {"files": len(depth.files),
                                        "note": "仅登记存在性;有效率检查需解码,按需扩展"}

    if not receipt["channels"]:
        return receipt, [Claim("sensor_health", NOT_EVALUATED, "无力/触觉/深度通道")]
    status = ACCEPTED if not findings else INCONCLUSIVE
    return receipt, [Claim("sensor_health", status,
                           "" if not findings else "; ".join(findings))]
