"""任务级场景物理(阶段 8):有 scene.object_pose 流时的联合验证。

诚实边界:真实数据集几乎不带物体位姿,本层主要服务仿真数据(RoboTwin2 类)
与合成回归。当前实现两项:物体运动连续性;抓取标注段内末端-物体邻近合理性。
机器人-物体联合 MuJoCo 接触世界在物体几何齐备时才有意义,接口预留。
"""
from __future__ import annotations

import numpy as np

from ..fk import load_urdf
from ..ledger import Claim, ACCEPTED, REJECTED, NOT_EVALUATED

REQUIRES = ["robot.joint_position", "scene.object_pose"]
CLAIMS = ["object_motion_continuity", "task_contact_plausibility"]
JUMP_LIMIT_M = 0.10        # 单帧位置跳变阈
GRASP_NEAR_M = 0.35        # 抓取段内末端与物体的邻近半径(粗口径)
GRASP_WORDS = ("抓", "拿", "grab", "pick", "grasp", "capture")


def run(pkg, inventory: dict, ctx: dict) -> tuple:
    obj = pkg.get("scene.object_pose")
    arr = np.asarray(obj.data, dtype=float)       # (n, k*7) 或 (n, 7)
    fps = pkg.fps or 30.0
    if arr.ndim == 2 and arr.shape[1] % 7 == 0:
        objs = arr.reshape(len(arr), -1, 7)
    else:
        return {"schema": "organoid-kernel.task-scene.v1"}, [
            Claim("object_motion_continuity", NOT_EVALUATED, "object_pose 形状不识别"),
            Claim("task_contact_plausibility", NOT_EVALUATED, "object_pose 形状不识别")]

    receipt = {"schema": "organoid-kernel.task-scene.v1", "episode_id": pkg.episode_id,
               "objects": objs.shape[1]}
    claims = []

    # 1) 连续性:单帧位置跳变
    jumps = []
    for k in range(objs.shape[1]):
        d = np.linalg.norm(np.diff(objs[:, k, :3], axis=0), axis=1)
        bad = np.where(d > JUMP_LIMIT_M)[0]
        for f in bad[:10]:
            jumps.append({"object": k, "frame": int(f), "jump_m": round(float(d[f]), 4)})
    receipt["position_jumps"] = jumps
    ok = not jumps
    claims.append(Claim("object_motion_continuity", ACCEPTED if ok else REJECTED,
                        "" if ok else f"{len(jumps)} 处单帧位置跳变 > {JUMP_LIMIT_M} m"))

    # 2) 抓取段邻近:标注说在抓,末端与至少一个物体应邻近
    segs = pkg.get("annotation.language_segments")
    profile = ctx.get("profile")
    if segs is None or not segs.data or profile is None:
        claims.append(Claim("task_contact_plausibility", NOT_EVALUATED,
                            "无标注段或无模型,邻近性不评估"))
        return receipt, claims
    model = ctx.get("urdf_model") or load_urdf(profile.urdf_path(), profile.mesh_path())
    ctx["urdf_model"] = model
    joint = pkg.get("robot.joint_position")
    jarr = np.asarray(joint.data, dtype=float)
    ee_links = [l for l in model.links if l.endswith("_end_effector")] or \
               [j.child for j in model.movable if "arm" in j.name][-1:]
    checks = []
    for seg in segs.data:
        text = (seg.get("text") or "").lower()
        if not any(w in text for w in GRASP_WORDS):
            continue
        mid = min(len(jarr) - 1, int((seg["start_s"] + seg["end_s"]) / 2 * fps))
        T = model.fk({c: float(jarr[mid, i]) for i, c in enumerate(joint.columns)})
        dists = []
        for l in ee_links:
            if l not in T:
                continue
            ee = T[l][:3, 3]
            for k in range(objs.shape[1]):
                dists.append(float(np.linalg.norm(ee - objs[mid, k, :3])))
        near = bool(dists and min(dists) <= GRASP_NEAR_M)
        checks.append({"segment_id": seg["id"], "min_ee_object_dist_m":
                       round(min(dists), 4) if dists else None, "plausible": near})
    receipt["grasp_proximity"] = checks
    if not checks:
        claims.append(Claim("task_contact_plausibility", NOT_EVALUATED, "无抓取类标注段"))
    else:
        bad = [c for c in checks if not c["plausible"]]
        claims.append(Claim("task_contact_plausibility",
                            ACCEPTED if not bad else REJECTED,
                            "" if not bad else
                            f"{len(bad)} 个抓取段末端与所有物体距离 > {GRASP_NEAR_M} m"))
    return receipt, claims
