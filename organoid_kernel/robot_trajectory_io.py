"""Explicit external joint schemas and opt-in Cartesian task-space transfer."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from .hashing import sha256_file
from .multi_robot_tasks import (
    ArmRunner, SceneSpec, initial_features, scene_model, verified_profile, write_json,
)
from .trajectory import read_trajectory, write_trajectory


def validate_joint_input(profile_name, columns, unit, timestamps, positions):
    profile = verified_profile(profile_name)
    permutation = profile.map_joint_columns(list(columns), unit)
    timestamps = np.asarray(timestamps)
    positions = np.asarray(positions)
    if timestamps.ndim != 1 or len(timestamps) < 2 or not np.isfinite(timestamps).all():
        raise ValueError("At least two finite timestamps are required")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Timestamp clock must be strictly increasing")
    if positions.shape != (len(timestamps), len(columns)) or not np.isfinite(positions).all():
        raise ValueError("Finite state matrix must match timestamps and named joints")
    positions = positions[:, permutation]
    model = mujoco.MjModel.from_xml_path(str(profile.mjcf_path()))
    ids = [model.joint(name).id for name in profile.joint_names]
    bounds = model.jnt_range[ids]
    excess = np.maximum(bounds[:, 0] - positions, positions - bounds[:, 1]).clip(min=0)
    return positions, {
        "profile": profile_name, "joint_names": profile.joint_names, "rows": len(positions),
        "unit": "rad", "time_unit": "s", "mapping": permutation,
        "joint_limit_max_excess_rad": float(excess.max()),
        "within_model_joint_limits": bool(excess.max() <= .001),
        "peak_finite_difference_velocity_rad_s": np.max(
            np.abs(np.diff(positions, axis=0) / np.diff(timestamps)[:, None]), axis=0).tolist(),
        "collision": "not_evaluated_no_scene",
        "dynamic_replay": "not_evaluated_state_not_control",
        "hardware_execution": "not_evaluated",
    }


def import_joint_file(source, sidecar, output):
    source, sidecar, output = Path(source), Path(sidecar), Path(output)
    metadata = json.loads(sidecar.read_text())
    if metadata.get("time_unit") != "s" or metadata.get("origin") not in ("observed", "simulated"):
        raise ValueError("Declare time_unit=s and origin=observed or simulated")
    if metadata.get("state_semantics") != "joint_position":
        raise ValueError("Only explicit joint_position state is supported")
    with np.load(source, allow_pickle=False) as archive:
        time = archive[metadata["time_key"]]
        state = archive[metadata["state_key"]]
    mapped, report = validate_joint_input(
        metadata["profile"], metadata["joint_names"], metadata["joint_unit"], time, state)
    manifest = write_trajectory(output, {"timestamp_s": time, "joint_position": mapped}, {
        "timestamp_s": {"unit": "s", "origin": metadata["origin"], "clock": "timestamp_s"},
        "joint_position": {"unit": "rad", "origin": metadata["origin"], "clock": "timestamp_s",
                           "names": report["joint_names"]},
    }, {"robot_profile": metadata["profile"], "origin": metadata["origin"],
        "source_file": str(source.resolve()), "source_sha256": sha256_file(source),
        "sidecar_sha256": sha256_file(sidecar), "source_metadata": metadata,
        "hardware_execution": "not_evaluated"})
    report["trajectory_sha256"] = manifest["sha256"]
    write_json(output / "joint-audit.json", report)
    return report


def transfer_episode(source, target_robot, output):
    """Translate task-space waypoints between declared model-world TCP origins.

    Positions only, fixed scale/world axes. This re-solves IK and reruns physics;
    it never copies actuator controls between robots or validates a grasp.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest, arrays = read_trajectory(source)
    metadata = manifest["metadata"]
    if metadata.get("cartesian_frame") != "model_world":
        raise ValueError("Explicit model_world Cartesian coordinates required")
    for key in ("candidate_waypoints", "task_targets", "tcp_position"):
        if manifest["streams"].get(key, {}).get("unit") != "m":
            raise ValueError(f"Explicit metre-valued {key} required")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing nonempty transfer output")
    source_robot = metadata["robot_profile"]
    if source_robot == target_robot:
        raise ValueError("Source and destination profiles must differ")
    verified_profile(target_robot)
    source_start = arrays["tcp_position"][0]
    task_targets = arrays["task_targets"] - source_start
    waypoints = arrays["candidate_waypoints"] - source_start
    source_scene = metadata["scene"]
    spec = SceneSpec(
        family=source_scene["family"], robot=target_robot, task=source_scene["task"],
        seed=source_scene["seed"], initial_jitter=0,
        goal_offset=tuple(task_targets[-1]), obstacle_offset=tuple(source_scene["obstacle_offset"]),
        obstacle_half=tuple(source_scene["obstacle_half"]), control_noise=source_scene["control_noise"])
    model, initial, xml, start, goal, obstacle = scene_model(spec)
    features = arrays["initial_features"]
    names = metadata["feature_names"]
    candidate = {
        "name": "transferred", "waypoints": start + waypoints,
        "lift": float(features[names.index("lift_m")]),
        "duration": float(features[names.index("segment_duration_s")]),
    }
    targets = start + task_targets
    runner = ArmRunner(spec, model, initial)
    vector = initial_features(spec, model, initial, start, goal, obstacle, candidate)
    receipt = runner.run(candidate, targets)
    runner.save(output, xml, candidate, targets, vector, receipt)
    transfer = {
        "schema": "organoid.cartesian-transfer.v1",
        "source_robot": source_robot, "target_robot": target_robot,
        "source_trajectory_sha256": manifest["sha256"],
        "source_directory": str(source), "target_scene": asdict(spec),
        "mapping": {"kind": "translation", "from_origin_m": source_start.tolist(),
                    "to_origin_m": start.tolist(), "axes": "model_world_xyz", "scale": 1.0},
        "source_commands_copied": False, "target_ik_resolved": True,
        "target_status": receipt["status"], "labels": receipt["labels"],
        "replay": receipt["replay"], "scope": "position-only task transfer with independent target physics",
        "hardware_test": "not_evaluated",
    }
    write_json(output / "transfer.json", transfer)
    return transfer
