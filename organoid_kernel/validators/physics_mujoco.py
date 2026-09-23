"""MuJoCo 平台物理验证(§7.3):自由根双足五项 + 固定基口径。

预检收据是必需输入 —— 缺失时 kinematic_precheck 联判
输出 not_evaluated 并写明原因,绝不把该检查从清单上静默划掉。
所有收据明确验证范围:准静态姿态审计,不证明动态可执行性。
"""
from __future__ import annotations

import numpy as np

from .. import mjworld
from ..fk import load_urdf
from ..ledger import Claim, ACCEPTED, REJECTED, NOT_EVALUATED

REQUIRES = ["robot.joint_position", "robot.model"]
CLAIMS = ["foot_ground_contact", "balance", "self_collision", "ground_penetration",
          "kinematic_precheck_gate", "world_binding"]
TARGET_SAMPLES = 400


def _hull_2d(points: np.ndarray) -> np.ndarray:
    """Andrew 单调链凸包;点数 < 3 原样返回(退化支撑)。"""
    pts = np.unique(np.round(points, 6), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and np.cross(out[-1] - out[-2], p - out[-2]) <= 0:
                out.pop()
            out.append(p)
        return out
    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _margin(com_xy: np.ndarray, hull: np.ndarray) -> float:
    """质心到支撑域边界的有符号距离:内正外负;退化支撑(点/线)恒为负。"""
    if len(hull) == 0:
        return -np.inf
    if len(hull) == 1:
        return -float(np.linalg.norm(com_xy - hull[0]))
    if len(hull) == 2:
        a, b = hull
        t = np.clip(np.dot(com_xy - a, b - a) / (np.dot(b - a, b - a) or 1.0), 0, 1)
        return -float(np.linalg.norm(com_xy - (a + t * (b - a))))
    dmin, inside = np.inf, True
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        edge, rel = b - a, com_xy - a
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        if cross < 0:
            inside = False
        t = np.clip(np.dot(rel, edge) / (np.dot(edge, edge) or 1.0), 0, 1)
        dmin = min(dmin, float(np.linalg.norm(com_xy - (a + t * edge))))
    return dmin if inside else -dmin


def run(pkg, inventory: dict, ctx: dict) -> tuple:
    import mujoco
    profile = ctx["profile"]
    free_root = profile.base_type == "free_root"
    receipt = {"schema": "organoid-kernel.mujoco-physics.v1",
               "episode_id": pkg.episode_id, "engine": f"mujoco {mujoco.__version__}",
               "scope": "quasi_static_pose_audit", "dynamic_executability": "not_proven"}
    claims = []

    # ---- 预检收据必传,缺失即拒绝执行联判(且如实记录,不静默)
    pre = ctx.get("receipts", {}).get("kinematics")
    if pre is None:
        claims.append(Claim("kinematic_precheck_gate", NOT_EVALUATED,
                            "缺少运动学预检收据 —— 按门禁纪律拒绝联判,不静默移除该检查"))
        gate_ok = None
    else:
        gate_ok = bool(pre.get("kinematic_precheck_passed"))
        claims.append(Claim("kinematic_precheck_gate", ACCEPTED if gate_ok else REJECTED,
                            "" if gate_ok else "预检存在判负项(联判)"))

    joint = pkg.get("robot.joint_position")
    arr = np.asarray(joint.data, dtype=float)
    n = len(arr)

    base = pkg.get("robot.base_pose")
    if free_root and (base is None or base.origin == "missing" or base.data is None
                      or not np.isfinite(np.asarray(base.data)).all()):
        for c in ("foot_ground_contact", "balance", "ground_penetration"):
            claims.append(Claim(c, NOT_EVALUATED, "浮动基位姿不可得(观测缺失且派生条件不成立)"))
        claims.append(Claim("self_collision", NOT_EVALUATED, "浮动基位姿不可得"))
        receipt["not_evaluated_reason"] = (base.provenance if base else None) or "无基座位姿流"
        return receipt, claims
    base_arr = np.asarray(base.data, dtype=float) if base is not None else None

    # ---- 世界编译(先零地面估地面高,再按标定高度正式编译)
    model_u = ctx.get("urdf_model") or load_urdf(profile.urdf_path(), profile.mesh_path())
    ctx["urdf_model"] = model_u
    info0 = mjworld.build_world(model_u, profile, 0.0, free_root=free_root, with_floor=False)
    m0 = mjworld.compile_model(info0)
    d0 = mujoco.MjData(m0)
    jadr = {mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, i):
            int(m0.jnt_qposadr[i]) for i in range(m0.njnt)}
    col_map = [(i, jadr[c]) for i, c in enumerate(joint.columns) if c in jadr]
    foot_gid = {mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_GEOM, g): g
                for g in range(m0.ngeom)}
    foot_ids = {gid: (side, r) for name, (side, r) in info0.foot_geoms.items()
                for gname, gid in foot_gid.items() if gname == name}

    stride = max(1, n // TARGET_SAMPLES)
    sample = list(range(0, n, stride))

    def set_frame(m, d, k):
        d.qpos[:] = 0
        if free_root:
            d.qpos[0:3] = base_arr[k, 0:3]
            q = base_arr[k, 3:7]
            d.qpos[3:7] = [q[3], q[0], q[1], q[2]]      # xyzw → wxyz
        for ci, adr in col_map:
            d.qpos[adr] = arr[k, ci]
        mujoco.mj_forward(m, d)

    # 地面标定:官方脚球最低点的众数(与上游 modal_minimum 口径同族)
    ground = 0.0
    if free_root and foot_ids:
        lows = []
        for k in sample[:: max(1, len(sample) // 60)]:
            set_frame(m0, d0, k)
            lows.append(min(float(d0.geom_xpos[g][2]) - r for g, (s, r) in foot_ids.items()))
        vals, counts = np.unique(np.round(lows, 3), return_counts=True)
        ground = float(vals[np.argmax(counts)])
    receipt["ground"] = {"estimated_height_m": round(ground, 6),
                         "method": "modal_minimum_foot_sphere_height"}

    info = mjworld.build_world(model_u, profile, ground, free_root=free_root)
    m = mjworld.compile_model(info)

    # 世界绑定审计(老 audit-binding 的核心项):编译出的接触几何要与 Profile 声明一致
    receipt["world_binding"] = {"geometry_counts": info.counts,
                                "expected_foot_spheres": profile.expected_foot_spheres}
    if free_root and profile.expected_foot_spheres:
        binding_ok = info.counts.get("foot_contact") == profile.expected_foot_spheres
        claims.append(Claim("world_binding", ACCEPTED if binding_ok else REJECTED,
                            "" if binding_ok else
                            f"脚部接触球 {info.counts.get('foot_contact')} 个,"
                            f"Profile 声明 {profile.expected_foot_spheres} 个 —— "
                            "URDF 与档案不一致,支撑域判定不可信"))
        if not binding_ok:
            receipt["not_evaluated_reason"] = "world_binding 失败"
            for c in ("foot_ground_contact", "balance", "ground_penetration",
                      "self_collision"):
                claims.append(Claim(c, NOT_EVALUATED, "世界绑定审计未过"))
            return receipt, claims
    else:
        claims.append(Claim("world_binding", ACCEPTED, "固定基或未声明脚球数,仅记录几何计数"))
    d = mujoco.MjData(m)
    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    gname = {g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)}
    jadr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): int(m.jnt_qposadr[i])
            for i in range(m.njnt)}
    col_map = [(i, jadr[c]) for i, c in enumerate(joint.columns) if c in jadr]
    foot_ids = {g: (info.foot_geoms[nm][0], info.foot_geoms[nm][1])
                for g, nm in gname.items() if nm in info.foot_geoms}
    margin_cfg = float(profile.foot_contact_margin or 0.003)

    frames_rec = []
    counts = {"support": 0, "balanced": 0, "pen_ok": 0, "selfcol_free": 0, "finite": 0}
    worst = {"balance": (None, np.inf), "pen": (None, 0.0), "self": (None, 0.0)}
    for k in sample:
        set_frame(m, d, k)
        finite = bool(np.isfinite(d.qpos).all() and np.isfinite(d.xpos).all())
        rec = {"frame": int(k), "finite": finite}
        if finite:
            counts["finite"] += 1
            support = []
            pen = 0.0
            for g, (side, r) in foot_ids.items():
                z = float(d.geom_xpos[g][2])
                if z - r <= ground + margin_cfg:
                    support.append([float(d.geom_xpos[g][0]), float(d.geom_xpos[g][1])])
                pen = max(pen, ground - (z - r))
            com = d.subtree_com[1 if free_root else 0].copy() if free_root else d.subtree_com[0].copy()
            hull = _hull_2d(np.array(support)) if support else np.empty((0, 2))
            marg = _margin(np.array(com[:2]), hull) if len(hull) else -np.inf
            balanced = bool(marg > 0)
            selfcols = []
            mujoco.mj_collision(m, d)
            for ci in range(d.ncon):
                con = d.contact[ci]
                g1, g2 = int(con.geom1), int(con.geom2)
                if floor_id in (g1, g2):
                    continue
                depth = -float(con.dist)
                if depth > mjworld.SELF_COLLISION_MIN_DEPTH_M:
                    selfcols.append({"pair": f"{gname[g1]} <-> {gname[g2]}",
                                     "depth_m": round(depth, 5)})
            rec.update(support_points=len(support),
                       balance_margin_m=round(float(marg), 6) if np.isfinite(marg) else None,
                       balanced=balanced,
                       max_ground_penetration_m=round(float(max(pen, 0)), 6),
                       self_collisions=selfcols)
            counts["support"] += bool(support)
            counts["balanced"] += balanced
            counts["pen_ok"] += (max(pen, 0) <= mjworld.GROUND_PENETRATION_LIMIT_M)
            counts["selfcol_free"] += (not selfcols)
            if np.isfinite(marg) and marg < worst["balance"][1]:
                worst["balance"] = (k, float(marg))
            if pen > worst["pen"][1]:
                worst["pen"] = (k, float(pen))
            for sc in selfcols:
                if sc["depth_m"] > worst["self"][1]:
                    worst["self"] = (k, sc["depth_m"])
        frames_rec.append(rec)

    total = len(sample)
    cov = {k: round(v / total, 4) for k, v in counts.items()}
    checks = {
        "finite_state": counts["finite"] == total,
        "self_collision": counts["selfcol_free"] == total,
    }
    if free_root:
        checks.update({
            "foot_ground_contact": counts["support"] == total,
            "balance": counts["balanced"] == total,
            "ground_penetration": counts["pen_ok"] == total,
        })
    receipt.update({
        "evaluated_frames": total, "sample_stride": stride,
        "coverage": cov, "checks": checks,
        "worst": {"min_balance_margin_m": worst["balance"][1] if worst["balance"][0] is not None else None,
                  "min_balance_margin_frame": worst["balance"][0],
                  "max_ground_penetration_m": worst["pen"][1],
                  "max_self_penetration_m": worst["self"][1]},
        "frames": frames_rec,
    })

    def claim_of(name, ok, why):
        claims.append(Claim(name, ACCEPTED if ok else REJECTED, "" if ok else why,
                            detail={"coverage": cov}))
    claim_of("self_collision", checks["self_collision"],
             f"自碰撞最深 {worst['self'][1]*1000:.1f} mm @frame {worst['self'][0]}")
    if free_root:
        claim_of("foot_ground_contact", checks["foot_ground_contact"],
                 f"支撑覆盖 {cov['support']:.1%}")
        claim_of("balance", checks["balance"],
                 f"平衡覆盖 {cov['balanced']:.1%},最差裕度 {worst['balance'][1]:.3f} m")
        claim_of("ground_penetration", checks["ground_penetration"],
                 f"最大穿透 {worst['pen'][1]*1000:.1f} mm")
    else:
        for c in ("foot_ground_contact", "balance", "ground_penetration"):
            claims.append(Claim(c, "not_applicable", "固定基平台无双足门禁"))
    return receipt, claims
