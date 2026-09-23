"""Shaped reward:把裁决证据里的连续判据裕度折算成 [0,1] 标量,供 RL 消费。

设计约束(与 fail-closed 信条的关系):
  * reward 与 accepted 正交 —— accepted 仍是布尔裁决(六判据全过),照旧守链;
    reward 只是把"离阈值多远"的信息交给学习者,不参与任何接受判定。
  * 单一事实源 —— kabuki 桥、kabuki MCP 服务、RL 环境都从这里取公式;
    公式带版本号入收据,改公式必须升版本,旧 trace 的 reward 语义不漂移。
  * 纯标准库 —— 消费方(桥/服务)不因此背上 numpy 依赖。
  * 防 reward hacking —— BADQACC(物理数值爆炸)直接判 reward=0:
    数值不可信时给部分分,等于奖励把仿真弄炸的策略。

阈值与 scripts/grasp_transplant.py 的 CRIT、scripts/fullbody_replay.py 的
FALL_* 保持一致(此处为消费侧镜像,实验协议侧仍以脚本内定义为准)。
"""
from __future__ import annotations

GRASP_REWARD_VERSION = "grasp_shaped_v1"
FULLBODY_REWARD_VERSION = "fullbody_shaped_v1"

# 抓取六判据阈值(镜像 grasp_transplant.CRIT)
GRASP_CRIT = {"bilateral_cov": 0.8, "min_airborne_frames": 30, "min_lift_m": 0.05,
              "max_slip_m": 0.03, "settle_table_cov": 0.6, "settle_lift_tol_m": 0.02,
              "pick_lift_m": 0.03}

# 全身倒地判据(镜像 fullbody_replay.FALL_*)
FULLBODY_TILT_LIMIT_DEG = 25.0
FULLBODY_DRIFT_SCALE_M = 0.5


def _clip01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def grasp_shaped_reward(criteria_values: dict, numerics: dict = None,
                        crit: dict = None) -> dict:
    """抓取判据值 → shaped reward。

    五个分量各归一到 [0,1](达到成功阈值即满分,线性给部分分),取均值:
      grip   双侧接触覆盖 / 0.8
      carry  连续腾空帧数 / 30
      lift   最大抬升 / 5cm
      place  放回台面覆盖 / 0.6 × 末帧高度回归因子 —— 仅在物体确实被拿起过
             (max_lift ≥ pick 阈)时计分;没拿起来的物体"稳坐台面"不是放回
      hold   爪内滑移裕度 (3cm − slip)/3cm;slip 无值 = 从未建立持取,0 分

    BADQACC > 0 时 reward 强制 0(分量保留供诊断)。
    """
    c = dict(GRASP_CRIT)
    if crit:
        c.update(crit)
    cv = criteria_values
    max_lift = float(cv.get("max_lift_m") or 0.0)
    slip = cv.get("slip_in_gripper_m")
    end_lift = abs(float(cv.get("end_lift_m") or 0.0))
    tol = c["settle_lift_tol_m"]
    picked = max_lift >= c["pick_lift_m"]
    components = {
        "grip": _clip01(float(cv.get("bilateral_cov") or 0.0) / c["bilateral_cov"]),
        "carry": _clip01(float(cv.get("airborne_run_frames") or 0)
                         / c["min_airborne_frames"]),
        "lift": _clip01(max_lift / c["min_lift_m"]),
        "place": (_clip01(float(cv.get("settle_table_cov") or 0.0)
                          / c["settle_table_cov"])
                  * _clip01(tol / max(end_lift, tol))) if picked else 0.0,
        "hold": 0.0 if slip is None else _clip01(
            (c["max_slip_m"] - float(slip)) / c["max_slip_m"]),
    }
    badqacc = int((numerics or {}).get("badqacc_warnings") or 0)
    reward = 0.0 if badqacc > 0 else sum(components.values()) / len(components)
    return {"reward": round(reward, 4),
            "components": {k: round(v, 4) for k, v in components.items()},
            "gated_by_badqacc": badqacc > 0,
            "version": GRASP_REWARD_VERSION}


def fullbody_shaped_reward(survived: bool, fell_at_s, max_tilt_deg: float,
                           max_root_drift_m: float, duration_s: float,
                           rules: dict = None) -> dict:
    """全身重放裁决 → shaped reward。

    0.6×存活时长占比 + 0.25×直立裕度(1 − 最大倾角/25°) + 0.15×驻留裕度
    (1 − 根漂移/0.5m)。倒地时倾角必然触顶,直立裕度自然归零 ——
    "晚倒"与"勉强站住"的分差主要由存活占比拉开。
    """
    rules = rules or {}
    tilt_limit = float(rules.get("tilt_deg", FULLBODY_TILT_LIMIT_DEG))
    drift_limit = float(rules.get("root_drift_m", FULLBODY_DRIFT_SCALE_M))
    if survived:
        frac = 1.0
    else:
        frac = _clip01(float(fell_at_s or 0.0) / duration_s) if duration_s > 0 else 0.0
    upright = 1.0 - _clip01(float(max_tilt_deg) / tilt_limit)
    steady = 1.0 - _clip01(float(max_root_drift_m) / drift_limit)
    reward = 0.6 * frac + 0.25 * upright + 0.15 * steady
    return {"reward": round(reward, 4),
            "components": {"survival_frac": round(frac, 4),
                           "upright_margin": round(upright, 4),
                           "steady_margin": round(steady, 4)},
            "version": FULLBODY_REWARD_VERSION}
