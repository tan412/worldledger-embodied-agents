"""Kuavo sorting backend for the existing hash-bound Kabuki transaction gate."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

import cv2
import h5py
import mujoco
import numpy as np
import scipy

from .hashing import sha256_file
from .humanoid_sorting import (
    SortingRunner, SortingScene, default_policy, validate_policy, CAMERAS,
)
from .humanoid_tasks import robot_assets, DT
from .profile import PKG_ROOT
from .trajectory import read_trajectory

PROTOCOL_SCHEMA = "organoid.kuavo-sorting-protocol.v1"
CHECKS = (
    "task_sequence_complete", "both_objects_in_assigned_bins", "both_objects_settled",
    "both_objects_lifted_40mm", "bilateral_transport", "slip_below_20mm",
    "no_forbidden_contact", "penetration_below_3mm", "self_penetration_below_3mm",
    "joint_limits", "joint_speed_limits", "effort_limits", "numerically_valid",
    "no_external_forces", "geometry_fidelity", "contact_conservation",
)


def is_sorting(context):
    return context.get("protocol", {}).get("schema") == PROTOCOL_SCHEMA


def prepare_sorting_context(raw):
    from .kabuki_validation import digest, bind_candidate

    if set(raw) != {"robot", "task", "parameters"}:
        raise ValueError("Sorting spec needs robot, task and parameters")
    if raw["robot"] != "biped_s200049" or raw["task"] != "visual_sorting":
        raise ValueError("Unsupported humanoid task")
    spec = SortingScene(**raw["parameters"])
    robot_assets.cache_clear()
    runner = SortingRunner(spec, cameras=False)
    try:
        meshes = {}
        for mesh in ET.fromstring(runner.xml).findall("./asset/mesh"):
            path = Path(mesh.get("file"))
            meshes[path.name] = sha256_file(path)
        context = {
            "schema": "organoid.robot-context.v1",
            "robot": {"profile": runner.profile.raw,
                      "urdf_sha256": sha256_file(runner.profile.urdf_path()), "mesh_sha256": meshes},
            "initial_state": {"state_spec": int(runner.state_spec), "values": runner.initial_state.tolist()},
            "scene": {"spec": {"robot": raw["robot"], "task": raw["task"], "parameters": asdict(spec)},
                      "xml": runner.xml, "calibration": runner.calibration},
            "protocol": {
                "schema": PROTOCOL_SCHEMA, "required_checks": list(CHECKS),
                "dt_s": DT, "sensor_hz": 10, "replay_error_max": 1e-10,
                "maximum_penetration_m": .003, "joint_limit_excess_rad": .01,
                "minimum_lift_m": .04, "minimum_bilateral_transport_fraction": .8,
                "maximum_transport_slip_m": .02,
                "runtime": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                            "numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__},
                "code_sha256": {name: sha256_file(PKG_ROOT / "organoid_kernel" / name) for name in (
                    "humanoid_sorting.py", "humanoid_tasks.py", "humanoid_dataset.py",
                    "kuavo_sorting_validation.py", "physics_audit.py", "mjworld.py",
                    "trajectory.py", "profile.py", "fk.py", "hashing.py", "kabuki_validation.py")},
                "worker_sha256": sha256_file(PKG_ROOT / "scripts/verify_kabuki_candidate.py"),
                "observation_contract": "RGBD_and_proprioception_plus_simulated_fingertip_force",
                "candidate_contract": "observation_conditioned_controller_policy_not_fixed_waypoints",
                "base": "fixed", "teacher": "geometric_RGBD_feedback_not_learned_policy",
            },
        }
        context = json.loads(json.dumps(context, allow_nan=False))
        candidates = [bind_candidate(context, default_policy()),
                      bind_candidate(context, default_policy(order=["blue", "orange"]))]
        return context, candidates, None
    finally:
        runner.close()


def rank_sorting(context, candidates):
    from .kabuki_validation import binding, digest
    rows = []
    for index, candidate in enumerate(candidates):
        row = {"input_index": index, "candidate": candidate, "candidate_sha256": digest(candidate),
               "score": None, "execution_authorized": False}
        try:
            row["binding"] = binding(context, candidate)
            row["critic"] = {"status": "not_evaluated_unsupported_profile_and_features",
                             "execution_authorized": False}
        except Exception as exc:
            row["critic"] = {"status": "not_evaluated", "reason": str(exc),
                             "execution_authorized": False}
        rows.append(row)
    return {"schema": "organoid.candidate-ranking.v1", "context_sha256": digest(context),
            "checkpoint_sha256": None, "execution_authorized": False, "ranking": rows}


def verify_sorting(context, candidate, directory, run_id):
    from .kabuki_validation import binding, reconstruct
    from .multi_robot_tasks import write_json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "request.json", {"context": context, "candidate": candidate, "run_id": run_id})
    result = {"schema": "organoid.bound-validation.v1", "run_id": run_id,
              "status": "not_evaluated", "binding": None, "physics": None, "fresh_rollout": False}
    runner = None
    try:
        result["binding"] = binding(context, candidate)
        reconstruct(context)
        spec = SortingScene(**context["scene"]["spec"]["parameters"])
        runner = SortingRunner(spec)
        if not np.array_equal(runner.initial_state, context["initial_state"]["values"]):
            raise ValueError("Rebuilt humanoid initial state changed")
        physics = runner.run_sorting(candidate["action"], directory / "episode")
        result.update(fresh_rollout=True, physics=physics)
        reconstruct(context)
        if physics["replay"]["verified"] is True:
            result["status"] = physics["status"]
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if runner is not None:
            runner.close()
    result["artifact_sha256"] = {
        str(p.relative_to(directory)): sha256_file(p)
        for p in sorted(directory.rglob("*")) if p.is_file()
    }
    write_json(directory / "validation.json", result)
    return result


def check_sorting_episode(directory, request, physics):
    from .kabuki_validation import digest
    episode = Path(directory) / "episode"
    context, candidate = request["context"], request["candidate"]
    if not is_sorting(context) or physics.get("schema") != "organoid.kuavo-visual-sorting.v1":
        raise ValueError("Wrong physics receipt for the locked task")
    manifest, arrays = read_trajectory(episode)
    metadata = manifest["metadata"]
    if digest(metadata["policy"]) != digest(candidate["action"]) or physics["policy"] != candidate["action"]:
        raise ValueError("Controller policy differs from bound candidate")
    if digest(metadata["scene"]) != digest(context["scene"]["spec"]["parameters"]):
        raise ValueError("Sorting scene differs from locked context")
    if not np.array_equal(arrays["initial_state"], context["initial_state"]["values"]):
        raise ValueError("Humanoid initial state mismatch")
    tree = ET.fromstring(context["scene"]["xml"])
    for mesh in tree.findall("./asset/mesh"):
        mesh.set("file", "robot_assets/" + Path(mesh.get("file")).name)
    if (episode / "scene.xml").read_text() != ET.tostring(tree, encoding="unicode"):
        raise ValueError("Portable humanoid scene does not match locked model")
    for name, expected in context["robot"]["mesh_sha256"].items():
        if sha256_file(episode / "robot_assets" / name) != expected:
            raise ValueError("Original robot mesh changed")
    if sha256_file(episode / "sensors.h5") != metadata["sensor_sha256"]:
        raise ValueError("Sensor stream hash mismatch")
    indices = arrays["sensor_control_index"]
    if indices[0] != 0 or indices[-1] != len(arrays["control"]):
        raise ValueError("Sensor endpoints do not cover the saved candidate")
    with h5py.File(episode / "sensors.h5", "r") as sensor:
        if not np.array_equal(sensor["control_index"][:], indices):
            raise ValueError("Sensor clocks disagree")
        for camera in CAMERAS:
            for name in ("rgb", "depth", "T_world_camera"):
                if len(sensor[camera][name]) != len(indices):
                    raise ValueError("Incomplete camera stream")
    for decision in json.loads((episode / "decisions.json").read_text()):
        if "sensor_index" in decision and indices[decision["sensor_index"]] > decision["control_index"]:
            raise ValueError("Decision consumed a future image")
