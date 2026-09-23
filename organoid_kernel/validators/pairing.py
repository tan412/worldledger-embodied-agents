"""配对门禁:标注段总时长 vs 轨迹墙钟时长。

轨迹时长永远取时间戳墙钟跨度,不用帧数÷fps ——
丢帧不会被误判成"时长对不上"。时间基为合成栅格时如实降级独立见证价值。
"""
from __future__ import annotations

import numpy as np

from ..ledger import Claim, ACCEPTED, REJECTED, NOT_EVALUATED

REQUIRES = ["annotation.language_segments"]
CLAIMS = ["pairing"]
TOLERANCE = 0.12          # 与上游收据一致的相对时长容差


def run(pkg, inventory: dict, ctx: dict) -> tuple:
    seg = pkg.get("annotation.language_segments")
    joint = pkg.get("robot.joint_position") or pkg.get("umi.gripper_pose")
    receipt = {"schema": "organoid-kernel.pairing.v1", "episode_id": pkg.episode_id,
               "tolerance": TOLERANCE}
    if seg is None or not seg.data:
        return receipt, [Claim("pairing", NOT_EVALUATED, "无标注段")]
    if joint is None or joint.timestamps is None:
        return receipt, [Claim("pairing", NOT_EVALUATED, "无带时间轴的轨迹流")]

    seg_span = max(s["end_s"] for s in seg.data) - min(s["start_s"] for s in seg.data)
    ts = np.asarray(joint.timestamps, dtype=float)
    traj_span = float(ts[-1] - ts[0])          # 墙钟跨度
    delta = abs(seg_span - traj_span) / max(seg_span, traj_span) if max(seg_span, traj_span) else 1.0
    synthetic = bool(inventory.get("timebase", {}).get("synthetic"))
    receipt.update({
        "durations_sec": {"segments": round(seg_span, 3), "trajectory": round(traj_span, 3)},
        "trajectory_duration_source": "wall_clock",
        "relative_duration_delta": round(delta, 6),
        "independent_witness": not synthetic,
        "note": ("时间基为合成栅格,标注帧号与轨迹帧号同源,时长比对退化为结构自洽检查"
                 if synthetic else None),
        "semantic_identity_verified": False,
    })
    ok = delta <= TOLERANCE
    return receipt, [Claim("pairing", ACCEPTED if ok else REJECTED,
                           "" if ok else f"时长偏差 {delta:.1%} 超容差",
                           detail={"delta": round(delta, 6)})]
