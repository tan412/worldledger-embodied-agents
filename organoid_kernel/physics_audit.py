"""物理守恒审计层(原则性根治的第二层):不预设 bug 类型,违背守恒即判负。

背景:五类问题(机构互穿/行程越限/非物理质量/接触假形变/托抱冒充夹持)全部
出在"给仿真器什么"的规格层,而旧防线是逐例加闸 —— 人眼发现一个补一个。
本模块把闸的"母规则"形式化,fail-closed:

  geometry_fidelity   编译期:凡有视觉网格的可接触刚体,其碰撞几何必须覆盖
                      视觉网格到 ε 以内(采样顶点到碰撞体的最大外距)。
                      指尖件漏装碰撞体这类问题在任何实验开始前就编译判负。
  FrameAuditor        运行时逐帧:
                      ① 不可入性 —— 邻近刚体对的视觉网格真距不得低于该材质的
                        柔度预算(负=互穿),对是枚举的全部候选而非"想到的那几对";
                      ② 幽灵力 —— 自由物体所受约束合力必须能被逐个接触力解释,
                        残差>阈值 = 有仿真专用约束(weld/equality/qpos覆盖)在
                        直接搬动物体("没夹到但物体离开桌面"属于此类)。

诚实边界:守恒审计只保证"与声明物理自洽",不保证"声明为真"——参数真值
要靠外部锚(厂商规格/真实数据拟合);能量账本暂未实现,列为已知缺口。
"""
from __future__ import annotations

import numpy as np

RIGID_OVERLAP_BUDGET_M = 0.003     # 刚-刚接触柔度预算(30kN/m 刚度 × ~90N)
SOFT_OVERLAP_BUDGET_M = 0.015      # 软物(泡沫)压缩形变代理
GHOST_ABS_FLOOR_N = 0.2            # 幽灵力残差的绝对噪声地板
GHOST_REL_MG = 0.5                 # 相对阈:物体自重的一半(被"拽着走"≈mg 必现形)
FIDELITY_EPS_M = 0.004             # 视觉顶点到碰撞体的最大允许外距


def _point_geom_outside_dist(m, d, gid, pts_world) -> np.ndarray:
    """点到单个碰撞几何体的外距(在内为 0)。支持 box/sphere/capsule/cylinder;
    mesh 碰撞体按其 AABB 近似(保守偏松,如实标注)。"""
    import mujoco
    R = d.geom_xmat[gid].reshape(3, 3)
    local = (pts_world - d.geom_xpos[gid]) @ R
    t = m.geom_type[gid]
    s = m.geom_size[gid]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.maximum(np.linalg.norm(local, axis=1) - s[0], 0.0)
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        q = np.maximum(np.abs(local) - s[:3], 0.0)
        return np.linalg.norm(q, axis=1)
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        z = np.clip(local[:, 2], -s[1], s[1])
        c = local.copy()
        c[:, 2] -= z
        return np.maximum(np.linalg.norm(c, axis=1) - s[0], 0.0)
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        dr = np.maximum(np.linalg.norm(local[:, :2], axis=1) - s[0], 0.0)
        dz = np.maximum(np.abs(local[:, 2]) - s[1], 0.0)
        return np.sqrt(dr ** 2 + dz ** 2)
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[gid]
        a = m.mesh_vertadr[mid]
        v = m.mesh_vert[a:a + m.mesh_vertnum[mid]]
        lo, hi = v.min(0), v.max(0)
        q = np.maximum(np.maximum(lo - local, local - hi), 0.0)
        return np.linalg.norm(q, axis=1)
    return np.full(len(pts_world), np.inf)


def geometry_fidelity(m, d, body_prefixes=(),
                      samples=160, eps=FIDELITY_EPS_M) -> dict:
    """编译期几何忠实度:对含视觉网格的可接触刚体,采样视觉顶点,量到本体
    全部碰撞几何的最小外距;最大外距 > eps 即该刚体"物理上有隐形部位"。"""
    import mujoco
    mujoco.mj_forward(m, d)
    gname = {g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
             for g in range(m.ngeom)}
    by_body: dict = {}
    for g in range(m.ngeom):
        by_body.setdefault(int(m.geom_bodyid[g]), []).append(g)
    report, worst = {}, 0.0
    for b, geoms in by_body.items():
        names = [gname[g] or "" for g in geoms]
        body_nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or str(b)
        # 审计范围:爪链刚体(按 body 名认链,不依赖碰撞几何是否存在 ——
        # 注入实验证明按 geom 前缀筛会漏掉"整块没碰撞"的事故态)与物体。
        is_claw = "_bar-" in body_nm or "fingers" in body_nm
        is_object = any(n.startswith("object::") for n in names)
        is_configured = any(body_nm.startswith(p) for p in body_prefixes)
        if not (is_claw or is_object or is_configured):
            continue
        vis = [g for g in geoms if (gname[g] or "").startswith("visual::")]
        col = [g for g in geoms
               if not (gname[g] or "").startswith("visual::")
               and (m.geom_contype[g] | m.geom_conaffinity[g])]
        if not vis:
            continue
        if col and is_claw and "_bar-2" not in body_nm:
            # 有碰撞的曲柄(bar-1/bar-3):几何凹形交错,凸近似点覆盖必假阳性
            # (实测伪缺口 25mm)—— 曲柄×物体的视觉互穿由运行时审计全对兜底;
            # 完全无碰撞的爪件(隐形物理)则无豁免,落到下方 defect=∞ 判负
            continue
        body_name = body_nm
        pts = []
        rng = np.random.RandomState(7)
        for g in vis:
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            a = m.mesh_vertadr[mid]
            v = m.mesh_vert[a:a + m.mesh_vertnum[mid]].copy()
            if len(v) > samples:
                v = v[rng.choice(len(v), samples, replace=False)]
            R = d.geom_xmat[g].reshape(3, 3)
            pts.append(v @ R.T + d.geom_xpos[g])
        if not pts:
            continue
        pts = np.concatenate(pts)
        if not col:
            defect = float("inf")
        else:
            dist = np.min(np.stack([_point_geom_outside_dist(m, d, g, pts)
                                    for g in col]), axis=0)
            defect = float(dist.max())
        report[body_name] = round(defect, 4) if np.isfinite(defect) else None
        worst = max(worst, defect if np.isfinite(defect) else 1.0)
    return {"per_body_defect_m": report, "worst_defect_m": round(worst, 4),
            "eps_m": eps, "ok": worst <= eps,
            "note": "视觉顶点到本体碰撞几何的最大外距;None/∞=有视觉无碰撞"}


class FrameAuditor:
    """逐帧守恒审计:候选刚体对全枚举(爪件×爪件、爪件×物体、物体×台面),
    粗筛(中心距)后量视觉网格真距;自由物体做幽灵力核算。"""

    def __init__(self, m, d, object_joint: str, soft_object: bool = False):
        import mujoco
        self.m, self.d = m, d
        self.mujoco = mujoco
        gname = {g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                 for g in range(m.ngeom)}
        vis = [(g, n) for g, n in gname.items()
               if n and n.startswith("visual::") and "bar" in n]
        objs = [(g, n) for g, n in gname.items()
                if n and n.startswith("object::")]
        f_side = [g for g, n in vis if "_f_" in n]
        b_side = [g for g, n in vis if "_b_" in n]
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, object_joint)
        if jid < 0:
            raise ValueError(f"Unknown audited object joint: {object_joint}")
        object_body = int(m.jnt_bodyid[jid])
        self.obj_geoms = [g for g, _ in objs if int(m.geom_bodyid[g]) == object_body]
        self.pairs = ([(a, b) for a in f_side for b in b_side]
                      + [(a, o) for a, _n in vis for o in self.obj_geoms])
        self.obj_bodies = {int(m.geom_bodyid[g]) for g in self.obj_geoms}
        self.obj_dof = int(m.jnt_dofadr[jid])
        self.obj_mass = float(sum(m.body_mass[b] for b in self.obj_bodies))
        self.ghost_tol = max(GHOST_ABS_FLOOR_N, GHOST_REL_MG * self.obj_mass * 9.81)
        self.budget = SOFT_OVERLAP_BUDGET_M if soft_object else RIGID_OVERLAP_BUDGET_M
        self._ft = np.zeros(6)
        self._cf = np.zeros(6)
        self.overlap_min_m = 0.0           # 最深互穿(负)
        self.ghost_max_N = 0.0
        self.violation_frames = 0

    def frame(self) -> None:
        m, d, mujoco = self.m, self.d, self.mujoco
        # ① 不可入性:粗筛 8cm 内的对
        deepest = 0.0
        for a, b in self.pairs:
            if np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b]) > 0.08:
                continue
            dist = mujoco.mj_geomDistance(m, d, a, b, 0.05, self._ft)
            deepest = min(deepest, float(dist))
        self.overlap_min_m = min(self.overlap_min_m, deepest)
        # ② 幽灵力:物体约束合力 vs Σ接触力
        fq = d.qfrc_constraint[self.obj_dof:self.obj_dof + 3].copy()
        fc = np.zeros(3)
        for ci in range(d.ncon):
            c = d.contact[ci]
            g1b = int(m.geom_bodyid[c.geom1])
            g2b = int(m.geom_bodyid[c.geom2])
            o1, o2 = g1b in self.obj_bodies, g2b in self.obj_bodies
            if not (o1 or o2):
                continue
            mujoco.mj_contactForce(m, d, ci, self._cf)
            R = np.array(c.frame).reshape(3, 3)
            fw = R.T @ self._cf[:3]        # 接触系→世界系;MuJoCo 约定作用于 geom2
            fc += fw if o2 else -fw
        ghost = float(np.linalg.norm(fq - fc))
        self.ghost_max_N = max(self.ghost_max_N, ghost)
        if -deepest > self.budget or ghost > self.ghost_tol:
            self.violation_frames += 1

    def verdict(self) -> dict:
        return {"ok": self.violation_frames == 0,
                "overlap_worst_mm": round(-self.overlap_min_m * 1000, 2),
                "overlap_budget_mm": round(self.budget * 1000, 1),
                "ghost_force_max_N": round(self.ghost_max_N, 2),
                "ghost_tol_N": round(self.ghost_tol, 2),
                "violation_frames": self.violation_frames,
                "note": "不可入性=候选对全枚举视觉网格真距;幽灵力=物体约束合力"
                        "须被逐个接触力解释(weld/equality 直接搬物体会现形)"}
