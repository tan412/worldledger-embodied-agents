"""RL 环境:把 organoid 的世界编译、验证判据、扰动协议包装成 reset/step 接口。

两个入口,对应回路里两个可学习的位置:

  BalanceEnv        逐步控制环境(平衡/抗扰)。全身动力学与 fullbody_replay 同一
                    编译路径(自由根 + 力受限位置伺服 + implicitfast);策略输出
                    **残差** —— 叠加在录制轨迹的伺服目标上,而不是从零学走路。
                    奖励 = 存活 + 直立/驻留裕度 − 动作正则,判倒阈值与
                    fullbody_replay 完全一致(根降 0.25m / 倾角 25°)。
  GraspVariantBandit 变体级黑盒目标(bandit / CEM / CMA-ES 用)。每次 evaluate
                    以子进程原样运行受信的 grasp_transplant 协议(基线闸照常
                    先行),shaped reward 来自 organoid_kernel.reward ——
                    学习者与验证器隔离,骗不过物理作证。

设计约束:
  * 不依赖 gymnasium —— reset()/step() 鸭子类型即可接任何 RL 库;要注册成
    gymnasium.Env 只需在外面包一层 spaces 声明。
  * 确定性 —— 随机扰动仅由显式 seed 驱动(np.random.default_rng),扰动参数
    写进 info,同 seed 同轨迹可复现。
  * 诚实口径 —— 与重放脚本一致:结论限定在"该伺服近似下",PD 增益是选择
    不是真机标定;obs_labels/协议参数全部可查。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import mjworld
from .physics_rules import DynamicMonitor, dynamic_rules
from .reward import grasp_shaped_reward

ROOT = Path(__file__).resolve().parents[1]
FPS = 30.0
FALL_ROOT_DROP_M = 0.25            # 与 scripts/fullbody_replay.py 一致
SETTLE_S = 1.5

# 随机扰动采样范围(reset(push="random") 用;逐项可被 push_ranges 覆盖)
PUSH_RANGES = {"force_N": (20.0, 80.0), "duration_s": (0.25, 0.25),
               "dir_deg": (0.0, 360.0), "at_frac": (0.15, 0.7)}


def _quat_tilt_deg(q_wxyz):
    w, x, y, z = q_wxyz
    up_z = 1 - 2 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(up_z, -1, 1))))


class BalanceEnv:
    """录制轨迹跟踪 + 残差控制的平衡环境。

    action ∈ [-1,1]^n_act:残差关节(默认腿+腰)伺服目标的偏移,量程
    ±max_delta_rad;其余关节严格跟录制轨迹。obs 为本体感知(根高/姿态/速度 +
    残差关节位置速度)+ 轨迹相位,不含扰动力先验 —— 策略必须靠反应,不靠预告。
    """

    def __init__(self, model_u, profile, ground: float, traj: np.ndarray,
                 columns: list, base_quat_xyzw, push=None,
                 residual_prefixes=("leg", "waist"), max_delta_rad: float = 0.15,
                 action_penalty: float = 0.01, max_frames: int = None):
        import mujoco
        self._mujoco = mujoco
        self._rules = dynamic_rules(profile)
        self.traj = np.asarray(traj, dtype=float)
        self.columns = list(columns)
        self.n_frames = min(len(self.traj), max_frames or len(self.traj))
        self.push_cfg = push
        self.max_delta_rad = float(max_delta_rad)
        self.action_penalty = float(action_penalty)
        self._base_quat_wxyz = [base_quat_xyzw[3], base_quat_xyzw[0],
                                base_quat_xyzw[1], base_quat_xyzw[2]]
        self._model_u = model_u

        info = mjworld.build_world(model_u, profile, ground, free_root=True,
                                   claw_collision=True, full_body_actuators=True,
                                   with_visual_meshes=False)
        self.m = mjworld.compile_model(info)
        self.d = mujoco.MjData(self.m)
        self._jadr = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j):
                      int(self.m.jnt_qposadr[j]) for j in range(self.m.njnt)}
        self._actid = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, a): a
                      for a in range(self.m.nu)}
        self._base_bid = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, profile.root_link)
        # 有执行器的录制列;残差子集按前缀筛
        self._driven = [(i, c, self._actid[f"fb::{c}"])
                        for i, c in enumerate(self.columns)
                        if f"fb::{c}" in self._actid]
        self._residual = [t for t in self._driven
                          if t[1].startswith(tuple(residual_prefixes))]
        self._res_qadr = [self._jadr[c] for _, c, _ in self._residual]
        self._res_dofadr = [int(self.m.jnt_dofadr[
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, c)])
            for _, c, _ in self._residual]
        self.action_size = len(self._residual)
        self.obs_labels = (["root_z", "root_qw", "root_qx", "root_qy", "root_qz"]
                           + [f"root_vel_{k}" for k in "xyz"]
                           + [f"root_angvel_{k}" for k in "xyz"]
                           + [f"qpos::{c}" for _, c, _ in self._residual]
                           + [f"qvel::{c}" for _, c, _ in self._residual]
                           + ["phase"])
        self.observation_size = len(self.obs_labels)
        self._spf = max(1, int(round(1 / FPS / self.m.opt.timestep)))
        self._f = 0
        self._push = None
        self._settled = None            # 首次 reset 落地稳定后的 (qpos,qvel) 缓存

    # ---- gym 鸭子接口 ----
    def reset(self, push=None, seed=None):
        """push: None=沿用构造配置;dict={force_N,duration_s,at_s,dir_deg};
        "random"=按 PUSH_RANGES 用 seed 采样(参数入 info,可复现)。

        落地稳定段(1.5s)完全确定 —— 首次 reset 后缓存终态,后续 reset 直接
        恢复,训练时省掉每回合 ~750 步物理。缓存含求解器 warmstart(qacc_warmstart)
        与执行器状态:恢复必须与"刚跑完 settle"逐位等价,否则确定性重执行的
        复现比对(rollout_closed_loop.py B 段)会在混沌放大下失配。"""
        mujoco = self._mujoco
        if self._settled is None:
            mujoco.mj_resetData(self.m, self.d)
            self.d.qpos[0:3] = [0, 0, 0.005]
            self.d.qpos[3:7] = self._base_quat_wxyz
            for i, c, _ in self._driven:
                self.d.qpos[self._jadr[c]] = self.traj[0, i]
            for i, _, a in self._driven:
                self.d.ctrl[a] = self.traj[0, i]
            mujoco.mj_forward(self.m, self.d)
            for _ in range(int(SETTLE_S / self.m.opt.timestep)):
                mujoco.mj_step(self.m, self.d)
            self._settled = (self.d.qpos.copy(), self.d.qvel.copy(),
                             self.d.ctrl.copy(), self.d.act.copy(),
                             self.d.qacc_warmstart.copy())
        else:
            mujoco.mj_resetData(self.m, self.d)
            (self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:],
             self.d.act[:], self.d.qacc_warmstart[:]) = self._settled
            mujoco.mj_forward(self.m, self.d)
        self._z0 = float(self.d.qpos[2])
        self._xy0 = self.d.qpos[0:2].copy()
        self._f = 0
        self._monitor = DynamicMonitor(
            self._model_u, self.m, self.d, self.columns,
            self._z0, self._xy0, self._rules)

        push = self.push_cfg if push is None else push
        if push == "random":
            rng = np.random.default_rng(seed)
            lo_hi = lambda k: PUSH_RANGES[k]
            at = float(rng.uniform(*lo_hi("at_frac"))) * self.n_frames / FPS
            push = {"force_N": float(rng.uniform(*lo_hi("force_N"))),
                    "duration_s": float(rng.uniform(*lo_hi("duration_s"))),
                    "dir_deg": float(rng.uniform(*lo_hi("dir_deg"))),
                    "at_s": round(at, 2)}
        self._push = push
        if push:
            rad = float(push.get("dir_deg", 0.0)) * np.pi / 180.0
            self._push_f = (float(push["force_N"]) * np.sin(rad),
                            float(push["force_N"]) * np.cos(rad))
            self._push_t = (float(push.get("at_s", 4.0)),
                            float(push.get("at_s", 4.0))
                            + float(push.get("duration_s", 0.25)))
        return self._obs(), {"push": push, "settle_root_z": self._z0}

    def step(self, action):
        mujoco = self._mujoco
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if a.shape != (self.action_size,):
            raise ValueError(f"action 形状应为 ({self.action_size},),收到 {a.shape}")
        f = self._f
        for k, (i, _, act) in enumerate(self._driven):
            self.d.ctrl[act] = self.traj[f, i]
        for k, (i, _, act) in enumerate(self._residual):
            self.d.ctrl[act] = self.traj[f, i] + a[k] * self.max_delta_rad
        t_now = f / FPS
        pushing = bool(self._push) and self._push_t[0] <= t_now < self._push_t[1]
        self.d.xfrc_applied[self._base_bid, 0] = self._push_f[0] if pushing else 0.0
        self.d.xfrc_applied[self._base_bid, 1] = self._push_f[1] if pushing else 0.0
        last_state = None
        for _ in range(self._spf):
            mujoco.mj_step(self.m, self.d)
            last_state = self._monitor.observe(
                (f + (_ + 1) / self._spf) / FPS)
        # 监视器按子步记首次失效，但完整走完当前输出帧，保证导出的
        # qpos/ctrl 与独立验证器使用的固定帧边界完全一致。

        tilt = float(last_state["tilt_deg"])
        drop = float(last_state["root_drop_m"])
        drift = float(last_state["root_drift_m"])
        failures = list(self._monitor.invalid_reasons)
        fell = not self._monitor.survived
        reward = (1.0
                  - 0.5 * min(tilt / self._rules["tilt_deg"], 1.0)
                  - 0.25 * min(drift / self._rules["root_drift_m"], 1.0)
                  - self.action_penalty * float(np.mean(a * a)))
        self._f += 1
        truncated = self._f >= self.n_frames and not fell
        info = {"tilt_deg": round(tilt, 2), "root_drop_m": round(drop, 4),
                "root_drift_m": round(drift, 4), "fell": fell,
                "failure_reasons": sorted(failures),
                "velocity_limit_violations": {
                    name: v for name, v in self._monitor.velocity_limit_violations().items()},
                "frame": self._f, "pushing": pushing}
        return self._obs(), reward, bool(fell), truncated, info

    def _obs(self):
        d = self.d
        return np.concatenate([
            [d.qpos[2]], d.qpos[3:7], d.qvel[0:6],
            d.qpos[self._res_qadr], d.qvel[self._res_dofadr],
            [self._f / self.n_frames]]).astype(np.float64)


def make_balance_env(push=None, max_frames=None, **kwargs) -> BalanceEnv:
    """以 fullbody_replay 同款数据源(乐聚 流水线理瓶 ep0)构造 BalanceEnv。

    需要本机 leju_vendor 数据镜像;地面高度取 runs_leju 已验证收据,
    与重放脚本同一回退值。"""
    from .adapters import lerobot_v21
    from .datapaths import data_dir
    from .fk import load_urdf
    from .inventory import build_inventory
    from .profile import load_profile

    task = next(data_dir("leju_vendor").glob("raw/**/WL_01_01(流水线理瓶)"))
    pkg = lerobot_v21.load(task, 0)
    prof = load_profile("biped_s200049")
    build_inventory(pkg, prof.leg_columns)
    joint = pkg.get("robot.joint_position")
    base_pose = np.asarray(pkg.get("robot.base_pose").data, dtype=float)[0]
    ground = -0.824
    for p in (ROOT / "runs_leju").glob("*流水线理瓶*-ep0/receipt-physics.json"):
        ground = json.loads(p.read_text())["ground"]["estimated_height_m"]
    model_u = load_urdf(prof.urdf_path(), prof.mesh_path())
    return BalanceEnv(model_u, prof, ground,
                      np.asarray(joint.data, dtype=float), list(joint.columns),
                      base_pose[3:7], push=push, max_frames=max_frames, **kwargs)


class GraspVariantBandit:
    """变体级黑盒目标:spec → 受信抓取移植协议 → shaped reward。

    每次 evaluate 是一次完整实验(基线闸先行 + 目标变体,含渲染,~20-40s):
    优化器(CEM/CMA-ES/bandit)在物体参数空间搜索,永远隔着完整验证协议拿分,
    无法直接触碰仿真状态。产物目录可保留,证据链与手工实验同构。
    """

    def __init__(self, repo_root: Path = None, keep_artifacts: bool = False):
        self.root = Path(repo_root or ROOT)
        self.keep = keep_artifacts

    def evaluate(self, spec: dict, timeout: int = 900) -> dict:
        """spec: {label, name, kind(box/cylinder/sphere), size, mass,
        friction?, soft?, rgba?}(摆位/朝向由协议标定给出,不可指定)。"""
        for k in ("label", "name", "kind", "size", "mass"):
            if k not in spec:
                raise ValueError(f"spec 缺少必填字段: {k}")
        tmp = Path(tempfile.mkdtemp(prefix="grasp_bandit_"))
        (tmp / "variants.json").write_text(
            json.dumps([spec], ensure_ascii=False), encoding="utf-8")
        env = dict(os.environ,
                   GRASP_VARIANTS_JSON=str(tmp / "variants.json"),
                   GRASP_OUT=str(tmp / "out"))
        r = subprocess.run([sys.executable, str(self.root / "scripts/grasp_transplant.py")],
                           cwd=self.root, env=env, capture_output=True, text=True,
                           timeout=timeout)
        receipt_p = tmp / "out" / "grasp-transplant.json"
        if r.returncode != 0 or not receipt_p.exists():
            raise RuntimeError(f"抓取协议运行失败: {r.stderr[-400:]}")
        receipt = json.loads(receipt_p.read_text())
        res = next((x for x in receipt["results"]
                    if x["variant"] == spec["label"]), None)
        if res is None:
            out = {"reward": 0.0, "accepted": False, "verdict": None,
                   "baseline_gate_passed": receipt.get("baseline_gate_passed"),
                   "note": "基线闸未过或变体未运行"}
        else:
            shaped = grasp_shaped_reward(res["criteria_values"], res["numerics"])
            out = {"reward": shaped["reward"], "reward_breakdown": shaped,
                   "accepted": bool(res["success"]), "verdict": res,
                   "baseline_gate_passed": receipt.get("baseline_gate_passed")}
        if self.keep:
            out["artifacts_dir"] = str(tmp)
        else:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        return out
