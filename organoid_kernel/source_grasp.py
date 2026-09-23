"""Audited captured grasp priors, with explicit task-space retargeting semantics."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np

from .adapters.lerobot_v21 import load
from .cli import _run_one
from .control import raw_joint_position_targets
from .datapaths import data_dir
from .hashing import sha256_bytes, sha256_file, sha256_json
from .humanoid_tasks import robot_assets

PORTAL = "http://47.95.13.189:7007/data-viz/datasets/leju_sample_data/"


def normalized_curve(points):
    points = np.asarray(points, dtype=float)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    progress = distance / distance[-1] if distance[-1] > 1e-8 else np.linspace(0, 1, len(points))
    t = np.linspace(0, 1, len(points))
    grid = np.linspace(0, 1, 33)
    residual = points - (points[0] + progress[:, None] * (points[-1] - points[0]))
    scale = max(float(np.max(np.linalg.norm(residual, axis=1))), 1e-9)
    return {"progress": np.interp(grid, t, progress).tolist(),
            "residual": np.stack([np.interp(grid, t, residual[:, i] / scale)
                                  for i in range(3)], axis=1).tolist()}


def prepare_library(output, *, remote=True, source=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mirror = data_dir("leju_vendor") / "raw"
    source = Path(source) if source else next(mirror.glob("**/WL_01_01(搭建简单积木)"))
    profile, urdf = robot_assets()
    rows = []
    info = json.loads((source / "meta/info.json").read_text())
    for ep in range(info["total_episodes"]):
        directory = output / f"episode_{ep:06d}"
        directory.mkdir()
        parquet = source / f"data/chunk-000/episode_{ep:06d}.parquet"
        url = PORTAL + quote(str(parquet.relative_to(mirror)))
        remote_hash = None
        if remote:
            with urlopen(url, timeout=60) as response:
                remote_hash = sha256_bytes(response.read())
            if remote_hash != sha256_file(parquet):
                raise ValueError("Portal/local captured episode content mismatch")
        shutil.copy2(parquet, directory / "captured.parquet")
        shutil.copy2(source / "meta/info.json", directory / "info.json")
        pkg, ledger = _run_one(source, ep, directory / "audit", None, "biped_s200049", [])
        ledger = ledger.to_json()
        joint = pkg.get("robot.joint_position")
        command = raw_joint_position_targets(pkg, list(joint.columns))
        observed = np.asarray(joint.data, dtype=float)
        eff = np.asarray(pkg.get("extra.observation.state.effector.position").data)
        side_index = int(np.argmax(np.ptp(eff, axis=0)))
        side = ("l", "r")[side_index]
        closed = np.where(eff[:, side_index] > 50)[0]
        if not len(closed):
            rows.append({"episode": ep, "status": "not_evaluated", "reason": "no_grasp_cycle"})
            continue
        close = int(closed[0])
        releases = np.where(eff[close:, side_index] < 50)[0]
        if not len(releases):
            rows.append({"episode": ep, "status": "not_evaluated", "reason": "no_release"})
            continue
        release = close + int(releases[0])
        start, end = max(0, close - 45), min(len(command) - 1, release + 15)
        columns = list(joint.columns)
        arm_names = [f"zarm_{side}{i}_joint" for i in range(1, 8)]
        indices = [columns.index(n) for n in arm_names]
        bounds = np.array([urdf.limits()[n][:2] for n in columns])
        clip = command[start:end + 1]
        excess = np.maximum(bounds[:, 0] - clip, clip - bounds[:, 1])
        checks = {
            "source_ledger_accepted": ledger["grade"] == "accepted",
            "finite_commands": bool(np.isfinite(clip).all()),
            "joint_limits_with_1e_6_rounding_tolerance": bool(np.max(excess) <= 1e-6),
            "finite_observations": bool(np.isfinite(observed[start:end + 1]).all()),
            "right_arm_supported": side == "r",
            "ordered_grasp_release": start < close < release < end,
        }
        raw_video = next(p for p in pkg.get("camera.rgb").files if "camera_top" in str(p))
        path = []
        for row in command:
            configuration = dict(zip(columns, np.clip(row, bounds[:, 0], bounds[:, 1])))
            transform = urdf.fk(configuration)[f"zarm_{side}7_link"]
            point = transform @ np.array([0, .0008 if side == "r" else -.0008, -.197, 1])
            path.append(point[:3])
        path = np.asarray(path)
        peak = close + int(np.argmax(path[close:release + 1, 2]))
        checks["lift_then_release"] = close < peak < release
        provenance = {
            "portal_url": url, "portal_sha256": remote_hash, "local_sha256": sha256_file(parquet),
            "source_directory": str(source), "episode": ep, "source_robot_type": info["robot_type"],
            "source_video": str(raw_video), "source_video_sha256": sha256_file(raw_video),
            "urdf_sha256": sha256_file(profile.urdf_path()),
            "timebase": "dataset_30Hz_grid_not_independent_hardware_clock",
            "ledger_grade": ledger["grade"], "checks": checks,
            "scope": "kinematic_source_admission_not_source_scene_dynamic_success",
            "source_size_and_scene_pose": "not_measured_by_dataset",
            "source_command_limit_rounding_max_rad": max(0., float(np.max(excess))),
        }
        prior = {
            "schema": "organoid.captured-grasp-prior.v1", "source": provenance,
            "frames": {"start": start, "close": close, "peak": peak, "release": release, "end": end},
            "source_fps": info["fps"], "active_arm": side,
            "curves": {name: normalized_curve(path[a:b + 1]) for name, a, b in (
                ("descend", start, close), ("lift", close, peak),
                ("transport", close, release), ("lower", peak, release),
                ("retract", release, end))},
            "source_closure_fraction": float(np.median(eff[close + 5:release - 5, side_index]) / 100),
            "retargeting": {
                "preserved": ["source_phase_progress", "bounded_source_path_curvature", "grasp_release_order"],
                "replaced": ["world_anchors", "top_down_wrist_orientation", "safe_phase_durations",
                             "model_gripper_gap_control"],
                "method": "recorded_action_FK_phase_curves_retargeted_to_scene_anchors",
                "not_raw_joint_command_replay": True,
            },
        }
        np.savez_compressed(directory / "captured-motion.npz", command=command, observed=observed,
                            effector_observed=eff, tcp_from_commands=path,
                            time_s=np.arange(len(command)) / info["fps"],
                            joint_names=np.asarray(columns), active_arm_columns=indices)
        prior["captured_motion_sha256"] = sha256_file(directory / "captured-motion.npz")
        prior["status"] = "accepted" if all(checks.values()) else "rejected"
        (directory / "prior.json").write_text(json.dumps(prior, indent=2, allow_nan=False))
        rows.append({"episode": ep, "status": prior["status"], "prior": str(directory / "prior.json"),
                     "prior_sha256": sha256_json(prior), "checks": checks})
    (output / "library.json").write_text(json.dumps(rows, indent=2))
    return rows
