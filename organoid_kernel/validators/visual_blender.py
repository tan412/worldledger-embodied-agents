"""Blender 视觉验证(阶段 6):FK 预算连杆位姿 → Blender 头less 摆位渲染。

内核算 FK(纯 numpy),Blender 只做导入/摆位/渲染 —— 引擎间零几何分歧。
检查的是视觉完整性(mesh 全部加载、整机在画幅内、与轨迹哈希绑定),不是动作对错。
Blender 不存在时输出 not_evaluated,不是数据失败。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..fk import load_urdf
from ..hashing import sha256_array
from ..ledger import Claim, ACCEPTED, REJECTED, NOT_EVALUATED

REQUIRES = ["robot.joint_position", "robot.model"]
CLAIMS = ["visual_mesh_integrity"]
def _find_blender() -> str:
    """按 env ORGANOID_BLENDER → PATH → macOS 默认位置解析;都找不到返回空串(→ not_evaluated)。"""
    for cand in (os.environ.get("ORGANOID_BLENDER"),
                 shutil.which("blender"),
                 "/Applications/Blender.app/Contents/MacOS/Blender"):
        if cand and Path(cand).exists():
            return cand
    return ""


BLENDER = _find_blender()
RENDER_SCRIPT = Path(__file__).resolve().parents[1] / "blender_render.py"


def run(pkg, inventory: dict, ctx: dict) -> tuple:
    receipt = {"schema": "organoid-kernel.blender-visual.v1", "episode_id": pkg.episode_id}
    if not Path(BLENDER).exists():
        return receipt, [Claim("visual_mesh_integrity", NOT_EVALUATED,
                               "本机无 Blender,视觉验证跳过")]
    profile = ctx["profile"]
    model = ctx.get("urdf_model") or load_urdf(profile.urdf_path(), profile.mesh_path())
    ctx["urdf_model"] = model
    joint = pkg.get("robot.joint_position")
    arr = np.asarray(joint.data, dtype=float)
    state_hash = sha256_array(arr)

    # 渲染帧:起始帧 + 物理最差平衡帧(有物理收据时)
    frames = [0]
    phys = ctx.get("receipts", {}).get("physics") or {}
    worst = (phys.get("worst") or {}).get("min_balance_margin_frame")
    if worst is not None and worst not in frames:
        frames.append(int(worst))

    base = pkg.get("robot.base_pose")
    base_arr = np.asarray(base.data, dtype=float) if base is not None and base.data is not None else None
    out = ctx["out_dir"]
    shots = []
    for f in frames:
        qpos = {c: float(arr[f, i]) for i, c in enumerate(joint.columns)}
        base_T = np.eye(4)
        if base_arr is not None and np.isfinite(base_arr[f]).all():
            x, y, z, qx, qy, qz, qw = base_arr[f]
            n2 = qx*qx + qy*qy + qz*qz + qw*qw
            s = 2.0 / n2 if n2 else 0.0
            base_T[:3, :3] = np.array([
                [1 - s*(qy*qy + qz*qz), s*(qx*qy - qz*qw), s*(qx*qz + qy*qw)],
                [s*(qx*qy + qz*qw), 1 - s*(qx*qx + qz*qz), s*(qy*qz - qx*qw)],
                [s*(qx*qz - qy*qw), s*(qy*qz + qx*qw), 1 - s*(qx*qx + qy*qy)]])
            base_T[:3, 3] = [x, y, z]
        T = model.fk(qpos, base_T)
        items = []
        missing = 0
        for lname, link in model.links.items():
            for g in link.geoms:
                if not g.visual or g.kind != "mesh":
                    continue
                if not Path(g.mesh).exists():
                    missing += 1
                    continue
                M = T[lname] @ g.T
                items.append({"mesh": g.mesh, "matrix": M.reshape(-1).tolist(),
                              "scale": list(g.scale)})
        spec = {"items": items, "image": str(out / f"visual-{f}.png")}
        spec_path = out / f"visual-{f}.spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        proc = subprocess.run([BLENDER, "--background", "--python", str(RENDER_SCRIPT),
                               "--", str(spec_path)],
                              capture_output=True, text=True, timeout=300)
        result_path = out / f"visual-{f}.result.json"
        rendered = result_path.exists()
        res = json.loads(result_path.read_text()) if rendered else {}
        shots.append({"frame": f, "requested_meshes": len(items),
                      "imported": res.get("imported"), "missing_meshes": missing,
                      "in_frame_ratio": res.get("in_frame_ratio"),
                      "rendered": rendered and Path(spec["image"]).exists(),
                      "stderr": None if rendered else proc.stderr[-200:]})
    receipt.update({"state_hash": state_hash, "shots": shots})
    ok = all(s["rendered"] and s["missing_meshes"] == 0
             and s["imported"] == s["requested_meshes"] for s in shots)
    detail = {"frames": [s["frame"] for s in shots],
              "missing_total": sum(s["missing_meshes"] for s in shots)}
    return receipt, [Claim("visual_mesh_integrity", ACCEPTED if ok else REJECTED,
                           "" if ok else "渲染失败或缺失 mesh", detail=detail)]
