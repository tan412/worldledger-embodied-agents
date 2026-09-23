"""通用数据质量门禁(§7.1):时间基、占位通道、有限值、标注覆盖。"""
from __future__ import annotations

import numpy as np

from ..ledger import Claim, ACCEPTED, REJECTED, NOT_EVALUATED, INCONCLUSIVE

REQUIRES = []          # 主时间轴自适应:关节流或末端流,都没有则相应 claim 降级
CLAIMS = ["timebase_integrity", "stream_liveness", "annotation_coverage"]


def run(pkg, inventory: dict, ctx: dict) -> tuple:
    receipt = {"schema": "organoid-kernel.quality.v1", "episode_id": pkg.episode_id}
    claims = []

    joint = pkg.get("robot.joint_position") or pkg.get("umi.gripper_pose")
    if joint is None or joint.data is None:
        claims.append(Claim("timebase_integrity", NOT_EVALUATED, "无带时间轴的数值流"))
        dead = [n for n, s in pkg.streams.items() if s.origin == "missing" and s.note]
        claims.append(Claim("stream_liveness", ACCEPTED if not dead else INCONCLUSIVE,
                            "" if not dead else f"占位通道: {dead}"))
        claims.append(Claim("annotation_coverage", NOT_EVALUATED, "无数值主轴,覆盖率不计算"))
        return receipt, claims
    arr = np.asarray(joint.data, dtype=float)
    finite = bool(np.isfinite(arr).all())
    receipt["finite_values"] = finite

    tb = inventory.get("timebase", {})
    receipt["timebase"] = tb
    if not finite:
        claims.append(Claim("timebase_integrity", REJECTED, "关节流含非有限值"))
    elif tb.get("synthetic"):
        # 合成栅格不是判负:时间基完好可用,但丢帧/重复帧检查失去可见性 —— 记入收据
        receipt["synthetic_note"] = ("timestamp 为合成栅格(帧号/fps),真实采样时刻已在"
                                     "转换中丢失;丢帧/重复帧/采样间隙检查对该数据不可见")
        claims.append(Claim("timebase_integrity", ACCEPTED,
                            "时间基为合成栅格,结构完好;独立见证能力受限(见收据)",
                            detail={"synthetic": True}))
    else:
        ts = np.asarray(joint.timestamps, dtype=float) if joint.timestamps is not None else None
        if ts is None:
            claims.append(Claim("timebase_integrity", INCONCLUSIVE, "无时间戳"))
        else:
            mono = bool((np.diff(ts) >= 0).all())
            dt = np.diff(ts)
            gaps = int((dt > 0.05).sum())
            receipt.update(monotonic=mono, gaps_over_50ms=gaps,
                           max_gap_s=round(float(dt.max()), 4) if len(dt) else None)
            claims.append(Claim("timebase_integrity", ACCEPTED if mono else REJECTED,
                                "" if mono else "时间戳非单调"))

    # 占位通道(能力清单已降级,这里入账)
    ph = inventory.get("placeholder_imu", {})
    dead = [n for n, s in pkg.streams.items() if s.origin == "missing" and s.note]
    receipt["placeholder_streams"] = dead
    claims.append(Claim("stream_liveness", ACCEPTED if not dead else INCONCLUSIVE,
                        "" if not dead else f"占位通道: {dead}",
                        detail={"placeholder_imu": ph.get("placeholder", False)}))

    # 标注覆盖
    seg = pkg.get("annotation.language_segments")
    n_axis = len(arr)
    if seg is None or not seg.data:
        claims.append(Claim("annotation_coverage", NOT_EVALUATED, "无标注段"))
        receipt["annotation"] = {"segments": 0}
    else:
        fps = pkg.fps or 30.0
        total = n_axis / fps
        covered = sum(min(s["end_s"], total) - s["start_s"] for s in seg.data
                      if s["end_s"] > s["start_s"])
        ratio = covered / total if total else 0.0
        receipt["annotation"] = {"segments": len(seg.data),
                                 "coverage_ratio": round(ratio, 4)}
        claims.append(Claim("annotation_coverage",
                            ACCEPTED if ratio >= 0.5 else REJECTED,
                            "" if ratio >= 0.5 else f"标注只覆盖录制的 {ratio:.0%}",
                            detail={"coverage_ratio": round(ratio, 4)}))
    return receipt, claims
