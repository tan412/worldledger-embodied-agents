"""共享的动态物理判据。

这里定义的是实验协议层的硬约束，不替代 MuJoCo 的动力学积分：
额定速度由 URDF 给出，仿真中若超过它就必须判为动态不可执行，
不能把超速轨迹继续当成成功样本。
"""
from __future__ import annotations

import numpy as np


DEFAULT_DYNAMIC_RULES = {
    "root_drop_m": 0.25,
    "tilt_deg": 25.0,
    "root_drift_m": 0.5,
    "velocity_limit_ratio": 1.0,
}


def dynamic_rules(profile) -> dict:
    rules = dict(DEFAULT_DYNAMIC_RULES)
    rules.update(getattr(profile, "dynamic_rules", {}) or {})
    return rules


def joint_velocity_status(model_u, mujoco_model, mujoco_data, joint_names,
                          limit_ratio: float = 1.0) -> dict:
    """返回指定关节的当前速度/额定速度状态，单位沿用 URDF。"""
    import mujoco

    by_name = {j.name: j for j in model_u.joints}
    out = {}
    for name in joint_names:
        joint = by_name.get(name)
        if joint is None or not joint.velocity:
            continue
        jid = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        dof = int(mujoco_model.jnt_dofadr[jid])
        actual = abs(float(mujoco_data.qvel[dof]))
        limit = float(joint.velocity)
        out[name] = {
            "velocity": actual,
            "limit": limit,
            "ratio": actual / limit,
            "violated": actual > limit * limit_ratio,
        }
    return out


def dynamic_rule_failures(root_drop_m: float, tilt_deg: float,
                          root_drift_m: float, velocity_status: dict,
                          rules: dict, finite_state: bool = True) -> list[str]:
    """以稳定顺序返回当前帧违反的动态规则。"""
    failures = []
    if not finite_state:
        failures.append("nonfinite_state")
    if root_drop_m > float(rules["root_drop_m"]):
        failures.append("root_drop")
    if tilt_deg > float(rules["tilt_deg"]):
        failures.append("tilt")
    if root_drift_m > float(rules["root_drift_m"]):
        failures.append("root_drift")
    if any(v["violated"] for v in velocity_status.values()):
        failures.append("joint_velocity")
    return failures


def free_root_tilt_deg(qpos) -> float:
    """MuJoCo freejoint qpos 的 wxyz 四元数对应的根倾角。"""
    w, x, y, z = np.asarray(qpos[3:7], dtype=float)
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))


class DynamicMonitor:
    """子步级动态协议监视器。

    监视器只观察 MuJoCo 状态，不修改状态；它把根降/倾角/漂移/额定速度
    和非有限数值统一成一个判定入口，避免脚本在输出帧边界漏掉子步峰值。
    """

    def __init__(self, model_u, mujoco_model, mujoco_data, joint_names,
                 root_z0: float, root_xy0, rules: dict,
                 root_mode: str = "free_root"):
        import mujoco
        self.model_u = model_u
        self.mujoco_model = mujoco_model
        self.mujoco_data = mujoco_data
        self.joint_names = list(joint_names)
        self.root_z0 = float(root_z0)
        self.root_xy0 = np.asarray(root_xy0, dtype=float).copy()
        self.rules = dict(rules)
        if root_mode not in ("free_root", "fixed_root"):
            raise ValueError(f"未知 root_mode: {root_mode}")
        self.root_mode = root_mode
        by_name = {j.name: j for j in model_u.joints}
        self._joint_limits = []
        for name in self.joint_names:
            joint = by_name.get(name)
            if joint is None or not joint.velocity:
                continue
            jid = mujoco.mj_name2id(
                mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self._joint_limits.append(
                    (name, int(mujoco_model.jnt_dofadr[jid]), float(joint.velocity)))
        self.fell_at_s = None
        self.invalid_at_s = None
        self.invalid_reasons = set()
        self.max_tilt_deg = 0.0
        self.max_root_drift_m = 0.0
        self.max_joint_velocity = {}
        self.first_velocity_violation = None
        self.root_z = []

    def observe(self, time_s: float) -> dict:
        d = self.mujoco_data
        finite = bool(np.isfinite(d.qpos).all()
                      and np.isfinite(d.qvel).all()
                      and np.isfinite(d.xpos).all())
        if self.root_mode == "free_root":
            tilt = free_root_tilt_deg(d.qpos) if finite else float("inf")
            drop = self.root_z0 - float(d.qpos[2]) if finite else float("inf")
            drift = (float(np.linalg.norm(d.qpos[0:2] - self.root_xy0))
                     if finite else float("inf"))
        else:
            # 固定根操作协议没有自由根 qpos；根部失稳判据在该协议中
            # 不适用，但有限状态和关节额定速度仍然有效。
            tilt = 0.0 if finite else float("inf")
            drop = 0.0 if finite else float("inf")
            drift = 0.0 if finite else float("inf")
        velocity_status = {}
        if finite:
            for name, dof, limit in self._joint_limits:
                actual = abs(float(d.qvel[dof]))
                velocity_status[name] = {
                    "velocity": actual,
                    "limit": limit,
                    "ratio": actual / limit,
                    "violated": actual > limit * self.rules["velocity_limit_ratio"],
                }
        for name, status in velocity_status.items():
            self.max_joint_velocity[name] = max(
                status["velocity"], self.max_joint_velocity.get(name, 0.0))
            if status["violated"] and self.first_velocity_violation is None:
                self.first_velocity_violation = {
                    "joint": name,
                    "time_s": round(float(time_s), 4),
                    "velocity": round(float(status["velocity"]), 4),
                    "limit": round(float(status["limit"]), 4),
                    "ratio": round(float(status["ratio"]), 4),
                }
        failures = dynamic_rule_failures(
            drop, tilt, drift, velocity_status, self.rules,
            finite_state=finite)
        geometric_failures = {"nonfinite_state", "root_drop", "tilt"} & set(failures)
        if self.fell_at_s is None and geometric_failures:
            self.fell_at_s = round(float(time_s), 4)
        if failures:
            self.invalid_reasons.update(failures)
            if self.invalid_at_s is None:
                self.invalid_at_s = round(float(time_s), 4)
        if finite:
            self.max_tilt_deg = max(self.max_tilt_deg, tilt)
            self.max_root_drift_m = max(self.max_root_drift_m, drift)
            self.root_z.append(float(d.qpos[2]))
        return {
            "finite": finite,
            "root_drop_m": drop,
            "tilt_deg": tilt,
            "root_drift_m": drift,
            "velocity_status": velocity_status,
            "failures": failures,
        }

    def velocity_limit_violations(self) -> dict:
        by_name = {j.name: j for j in self.model_u.joints}
        out = {}
        for name, max_velocity in self.max_joint_velocity.items():
            joint = by_name.get(name)
            if joint is None or not joint.velocity:
                continue
            ratio = max_velocity / float(joint.velocity)
            if ratio > self.rules["velocity_limit_ratio"]:
                out[name] = {
                    "max_velocity": round(max_velocity, 4),
                    "limit": float(joint.velocity),
                    "ratio": round(ratio, 4),
                }
        return out

    @property
    def survived(self) -> bool:
        return self.fell_at_s is None and self.invalid_at_s is None
