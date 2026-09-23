"""修复回路(§8 分级):判负后的定向处置。病因三分类,互不越界:

  测量侧(可证明) → 修数据,逐格留痕,修复前后都留哈希;
  稀疏单样本台阶  → 不修(台阶后的位置是真实信号,抹平=篡改),降级带警示;
  规格表侧/行为侧 → 单条 episode 内不动(改表须跨 episode 立案,由批量驱动器做)。
"""
from __future__ import annotations

import numpy as np

from .hashing import sha256_array
from .ledger import Claim, ACCEPTED

OFFSET_MIN_RATIO = 0.99   # 全程恒越界才认零点偏置
WARN_MAX_RATIO = 0.01     # 限速判负降警示:逐关节违规占比 ≤1% 且 p99 在限内
EDGE_EPS = 2e-6           # 平移多留一点,防峰值恰落边界被舍入再挤出去


def attempt(pkg, ctx, ledger, out_dir) -> dict:
    pre = ctx["receipts"].get("kinematics") or {}
    joint = pkg.get("robot.joint_position")
    arr = np.asarray(joint.data, dtype=float)
    model = ctx.get("urdf_model")
    limits = model.limits() if model else {}
    patch = {"schema": "organoid-kernel.repair-patch.v1", "offsets": [], "declined": []}
    changed = False

    # ---- 零点回正:全程恒越界 → 整列平移常数
    for jname, item in (pre.get("position_violations") or {}).items():
        ratio = item["count"] / max(1, pre.get("frames", 1))
        if ratio < OFFSET_MIN_RATIO:
            patch["declined"].append({"joint": jname, "reason":
                                      f"越界帧占 {ratio:.0%},非全程恒越界,不符合零点偏置特征,不修"})
            continue
        lo, hi, _ = limits.get(jname, (0, 0, 0))
        i = joint.columns.index(jname)
        v = arr[:, i]
        excess_hi, excess_lo = float((v - hi).max()), float((lo - v).max())
        shift = -(excess_hi + EDGE_EPS) if excess_hi >= excess_lo else (excess_lo + EDGE_EPS)
        before = sha256_array(arr)
        arr[:, i] = v + shift
        patch["offsets"].append({"joint": jname, "shift_rad": round(shift, 9),
                                 "hash_before": before, "hash_after": sha256_array(arr),
                                 "reason": "全程恒越界(零点偏置特征),整列平移使峰值落回限位内"})
        changed = True
    if changed:
        joint.data = arr          # 写回修复后的数组:asarray(dtype=float) 对 float32 源是副本,不写回则修复为空操作
        joint.origin = "derived"
        joint.provenance["repair"] = patch["offsets"]
        ledger.repairs["zero_offset"] = [o["joint"] for o in patch["offsets"]]

    # ---- 稀疏台阶 → 警示单(数据一格不改,判负改记 accepted_with_warnings 候选)
    velv = pre.get("velocity_violations") or {}
    p99 = pre.get("observed_velocity_p99_rad_s") or {}
    frames_total = pre.get("frames") or 1
    if velv:
        sparse = all(item["count"] / frames_total <= WARN_MAX_RATIO
                     and p99.get(j, float("inf")) <= item.get("limit_rad_s", float("inf"))
                     for j, item in velv.items())
        posv_left = {j: it for j, it in (pre.get("position_violations") or {}).items()
                     if not any(o["joint"] == j for o in patch["offsets"])}
        if sparse and not posv_left:
            ledger.warnings["velocity_steps"] = {
                j: {"count": it["count"],
                    "ratio": round(it["count"] / frames_total, 5)}
                for j, it in velv.items()}
            # 直接以警示放行该 claim(数据未动,收据链留有原判负)
            ledger.claims = [c for c in ledger.claims if c.name != "kinematic_precheck"]
            ledger.add(Claim("kinematic_precheck", ACCEPTED,
                             "限速判负全为稀疏单样本台阶(占比≤1% 且 p99 在限内),降级为警示",
                             receipt="receipt-kinematics.json"))

    import json
    from pathlib import Path
    (Path(out_dir) / "repair-patch.json").write_text(
        json.dumps(patch, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"changed": changed, "patch": patch}
