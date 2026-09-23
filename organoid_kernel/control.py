"""控制输入提取工具。

重放时必须优先使用 action(控制器目标),不能把 observation state
(机器人已经执行出来的状态)悄悄当成命令。后者只用于对比跟踪误差。
"""
from __future__ import annotations

import numpy as np


def action_target_diagnostics(command: np.ndarray, joint_columns: list[str],
                             joint_limits: dict[str, tuple] | None = None) -> dict:
    """检查动作列是否真的是对应关节的可执行位置目标。

    LeRobot 的 ``action`` 可能同时包含关节角、夹爪百分比或缺失通道哨兵值。
    列名匹配只能解决排列，不能证明数值物理有效；这里把这两件事分开。
    """
    command = np.asarray(command, dtype=float)
    if command.ndim != 2 or command.shape[1] != len(joint_columns):
        raise ValueError(
            f"command/joint_columns shape 不一致: {command.shape} vs "
            f"{len(joint_columns)}")
    per_joint = {}
    invalid = np.zeros(command.shape, dtype=bool)
    for i, name in enumerate(joint_columns):
        values = command[:, i]
        finite = np.isfinite(values)
        bad = ~finite
        entry = {
            "finite": bool(np.all(finite)),
            "min": round(float(np.nanmin(values)), 6) if values.size else None,
            "max": round(float(np.nanmax(values)), 6) if values.size else None,
            "invalid_frames": 0,
        }
        if joint_limits and name in joint_limits:
            lower, upper, _ = joint_limits[name]
            bad |= finite & ((values < float(lower)) | (values > float(upper)))
            entry["limit_lower"] = float(lower)
            entry["limit_upper"] = float(upper)
        invalid[:, i] = bad
        entry["invalid_frames"] = int(np.sum(bad))
        entry["valid_fraction"] = round(
            float(1.0 - np.mean(bad)) if len(bad) else 0.0, 6)
        per_joint[name] = entry
    bad_cols = [name for name, v in per_joint.items() if v["invalid_frames"]]
    return {
        "n_frames": int(command.shape[0]),
        "n_joints": int(command.shape[1]),
        "invalid_frames_total": int(np.sum(invalid)),
        "invalid_values_total": int(np.sum(invalid)),
        "invalid_joints": bad_cols,
        "per_joint": per_joint,
        "_invalid_mask": invalid,
    }


def raw_joint_position_targets(pkg, joint_columns: list[str]) -> np.ndarray:
    """按列名抽取未经物理校验的本体 action，供审计记录原始证据。"""
    action = pkg.get("robot.action")
    if action is None or action.data is None:
        raise ValueError("缺少 robot.action，不能进行 action 驱动重放")
    columns = list(action.columns)
    index = {name: i for i, name in enumerate(columns)}
    missing = [name for name in joint_columns if name not in index]
    if missing:
        raise ValueError(f"robot.action 缺少本体关节列: {missing}")
    arr = np.asarray(action.data, dtype=float)
    return arr[:, [index[name] for name in joint_columns]]


def joint_position_targets(
        pkg, joint_columns: list[str],
        joint_limits: dict[str, tuple] | None = None,
        observed: np.ndarray | None = None,
        invalid_policy: str = "raise") -> tuple[np.ndarray, str]:
    """按关节名从 robot.action 抽取本体目标。

    返回 (N x len(joint_columns), source)。动作流缺失或列不全时直接报错，
    避免降级成伪造的“动作重放”。若给出 joint_limits，越界值同样被视为
    无效动作；只有显式指定 ``invalid_policy="hold_observed"`` 才允许用观测
    状态保持该通道，并在 source 中留下这个事实。
    """
    command = raw_joint_position_targets(pkg, joint_columns)
    diag = action_target_diagnostics(command, joint_columns, joint_limits)
    invalid = diag.pop("_invalid_mask")
    if np.any(invalid):
        if invalid_policy == "raise":
            bad = ", ".join(diag["invalid_joints"])
            raise ValueError(
                f"robot.action 含越过关节限位/非有限值的本体目标: {bad}; "
                "如需诊断性重放必须显式使用 invalid_policy='hold_observed'")
        if invalid_policy != "hold_observed":
            raise ValueError(f"未知 invalid_policy: {invalid_policy}")
        if observed is None:
            raise ValueError(
                "invalid_policy='hold_observed' 需要同时提供 observed")
        observed = np.asarray(observed, dtype=float)
        if observed.shape != command.shape:
            raise ValueError(
                f"observed shape 不匹配: {observed.shape} vs {command.shape}")
        command = command.copy()
        command[invalid] = observed[invalid]
        return command, "robot.action+observed_hold_invalid"
    return command, "robot.action"


def joint_velocity_observation(pkg, joint_columns: list[str]) -> np.ndarray | None:
    """按关节名提取观测速度；缺失时返回 None。"""
    velocity = pkg.get("robot.joint_velocity")
    if velocity is None or velocity.data is None:
        return None
    index = {name: i for i, name in enumerate(velocity.columns)}
    if any(name not in index for name in joint_columns):
        return None
    arr = np.asarray(velocity.data, dtype=float)
    return arr[:, [index[name] for name in joint_columns]]


def fit_action_alignment(command: np.ndarray, observed: np.ndarray,
                         max_delay_frames: int = 12,
                         valid_mask: np.ndarray | None = None,
                         fps: float = 30.0) -> list[dict]:
    """拟合 action→observed 的采样延迟、线性比例和偏置。

    这是控制接口对齐诊断，不是电机参数辨识。对每个候选延迟 d，
    用 observed[t] ≈ scale * command[t-d] + offset 的最小二乘误差选优。
    """
    command = np.asarray(command, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if command.shape != observed.shape:
        raise ValueError(
            f"command/observed shape 不一致: {command.shape} vs {observed.shape}")
    n, joints = command.shape
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != command.shape:
            raise ValueError(
                f"valid_mask shape 不一致: {valid_mask.shape} vs {command.shape}")
    out = []
    for j in range(joints):
        best = None
        for delay in range(min(max_delay_frames, n - 2) + 1):
            # 正 delay 的定义是 observed[t] ≈ command[t-delay]。
            # 因而要丢掉观测序列前面的 delay 帧，而不是动作序列前面的帧。
            if delay:
                x = command[:-delay, j]
                y = observed[delay:, j]
                valid = (valid_mask[:-delay, j]
                         if valid_mask is not None else np.ones(len(x), bool))
            else:
                x = command[:, j]
                y = observed[:, j]
                valid = (valid_mask[:, j]
                         if valid_mask is not None else np.ones(len(x), bool))
            valid &= np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            if len(x) < 3:
                continue
            A = np.column_stack((x, np.ones(len(x))))
            scale, offset = np.linalg.lstsq(A, y, rcond=None)[0]
            rmse = float(np.sqrt(np.mean((scale * x + offset - y) ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, delay, scale, offset)
        if best is None:
            out.append({"note": "有效动作样本不足"})
            continue
        rmse, delay, scale, offset = best
        out.append({
            "delay_frames": int(delay), "delay_s": round(delay / float(fps), 4),
            "scale": round(float(scale), 6),
            "offset_rad": round(float(offset), 6),
            "rmse_rad": round(rmse, 6),
        })
    return out
