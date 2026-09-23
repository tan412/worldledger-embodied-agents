"""Hash-bound candidate ranking and fresh, fail-closed robot validation.

This module issues evidence, never authorizes a Kabuki state transition.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np
import scipy

from .hashing import sha256_file
from .multi_robot_tasks import (
    ArmRunner, DT, GOAL_TOL, IK_TOL, ROBOTS, TASKS, SceneSpec,
    candidate_paths, initial_features, scene_model, task_targets,
    verified_profile, write_json,
)
from .profile import PKG_ROOT

CONTEXT_SCHEMA = "organoid.robot-context.v1"
CANDIDATE_SCHEMA = "organoid.bound-candidate.v1"
RECEIPT_SCHEMA = "organoid.bound-validation.v1"
REQUIRED_CHECKS = (
    "completed_path", "ordered_targets_reached", "final_target_error",
    "no_obstacle_collision", "no_self_collision", "joint_limits",
    "effort_limits", "numerically_valid", "no_external_forces",
)


def digest(value):
    """Canonical JSON shared with kabuki.artifacts.sha256_payload."""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def protocol():
    return {
        "schema": "organoid.robot-acceptance-protocol.v1",
        "required_checks": list(REQUIRED_CHECKS),
        "dt_s": DT, "ik_tolerance_m": IK_TOL, "goal_tolerance_m": GOAL_TOL,
        "penetration_threshold_m": .0001, "joint_limit_excess_rad": .01,
        "effort_excess_native": 1e-6, "replay_error_max": 1e-10,
        "fresh_rollout_required": True, "saved_control_replay_required": True,
        "controller": "bounded_cartesian_ik_force_limited_position_servos",
        "runtime": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                    "numpy": np.__version__, "scipy": scipy.__version__},
        "code_sha256": {name: sha256_file(PKG_ROOT / "organoid_kernel" / name)
                        for name in ("multi_robot_tasks.py", "profile.py",
                                     "trajectory.py", "hashing.py", "kabuki_validation.py")},
        "worker_sha256": sha256_file(PKG_ROOT / "scripts/verify_kabuki_candidate.py"),
        "scope": {"base": "fixed", "task": "position_only",
                  "hardware": "not_evaluated", "grasp_stability": "not_evaluated",
                  "calibrated_velocity_safety": "not_evaluated"},
    }


def number(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a finite number")
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name}: outside [{low}, {high}]")
    return float(value)


def vector(value, low, high, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name}: expected XYZ")
    return tuple(number(x, low, high, name) for x in value)


def parse_scene(raw):
    fields = set(SceneSpec.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError("SceneSpec requires exactly its declared fields")
    if raw["robot"] not in ROBOTS or raw["task"] not in TASKS:
        raise ValueError("Unsupported robot or task")
    normalized = dict(raw)
    for key in ("family", "seed"):
        if type(raw[key]) is not int or not 0 <= raw[key] < 2**31:
            raise ValueError(f"{key}: expected nonnegative 31-bit integer")
    normalized["initial_jitter"] = number(raw["initial_jitter"], 0, .3, "initial_jitter")
    normalized["control_noise"] = number(raw["control_noise"], 0, .02, "control_noise")
    for key in ("goal_offset", "obstacle_offset"):
        normalized[key] = vector(raw[key], -2, 2, key)
    normalized["obstacle_half"] = vector(raw["obstacle_half"], .001, 1, "obstacle_half")
    return SceneSpec(**normalized)


def prepare_context(raw):
    if isinstance(raw, dict) and raw.get("task") == "autonomous_grasp":
        from .autonomous_grasp_validation import prepare_transfer_context
        return prepare_transfer_context(raw)
    if isinstance(raw, dict) and raw.get("task") == "visual_sorting":
        from .kuavo_sorting_validation import prepare_sorting_context
        return prepare_sorting_context(raw)
    spec = parse_scene(raw)
    # Every load/verification rechecks source bytes, not a previous cache hit.
    verified_profile.cache_clear()
    profile = verified_profile(spec.robot)
    model, initial, xml, start, goal, obstacle = scene_model(spec)
    state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.zeros(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, initial, state, state_spec)
    targets = task_targets(spec.task, start, goal)
    context = {
        "schema": CONTEXT_SCHEMA,
        "robot": {"profile": profile.raw,
                  "source_manifest_sha256": sha256_file(PKG_ROOT / profile.raw["source_manifest"])},
        "initial_state": {"state_spec": state_spec, "values": state.tolist()},
        "scene": {"spec": asdict(spec), "xml": xml, "targets": targets.tolist()},
        "protocol": protocol(),
    }
    # Roundtrip normalizes tuples before comparison and JSON transport.
    context = json.loads(json.dumps(context, allow_nan=False))
    candidates = [bind_candidate(context, action) for action in candidate_paths(targets, start)]
    return context, candidates, (spec, model, initial, xml, start, goal, obstacle, targets)


def normalize_action(raw):
    if not isinstance(raw, dict) or set(raw) != {"name", "lift", "duration", "waypoints"}:
        raise ValueError("Action requires name, lift, duration, waypoints only")
    if not isinstance(raw["name"], str) or not 1 <= len(raw["name"]) <= 80:
        raise ValueError("Invalid action name")
    duration = number(raw["duration"], .02, 10, "duration")
    lift = number(raw["lift"], 0, 2, "lift")
    points = raw["waypoints"]
    if isinstance(points, np.ndarray):
        points = points.tolist()
    if not isinstance(points, list) or not 1 <= len(points) <= 32:
        raise ValueError("Expected 1..32 waypoints")
    points = [list(vector(p, -5, 5, "waypoint")) for p in points]
    if len(points) * (duration + .2) + .5 > 60:
        raise ValueError("Candidate exceeds 60-second simulation budget")
    return {"name": raw["name"], "lift": lift, "duration": duration, "waypoints": points}


def bind_candidate(context, action):
    if context.get("protocol", {}).get("schema") == "organoid.autonomous-grasp-protocol.v1":
        from .autonomous_grasp import validate_action
        return {"schema": CANDIDATE_SCHEMA, "context_sha256": digest(context),
                "action": validate_action(action)}
    if context.get("protocol", {}).get("schema") == "organoid.kuavo-sorting-protocol.v1":
        from .humanoid_sorting import validate_policy
        return {"schema": CANDIDATE_SCHEMA, "context_sha256": digest(context),
                "action": validate_policy(action)}
    return {"schema": CANDIDATE_SCHEMA, "context_sha256": digest(context),
            "action": normalize_action(action)}


def check_candidate(context, candidate):
    if not isinstance(candidate, dict) or set(candidate) != {"schema", "context_sha256", "action"}:
        raise ValueError("Malformed bound candidate")
    if candidate["schema"] != CANDIDATE_SCHEMA or candidate["context_sha256"] != digest(context):
        raise ValueError("Candidate context hash mismatch")
    if context.get("protocol", {}).get("schema") == "organoid.autonomous-grasp-protocol.v1":
        from .autonomous_grasp import validate_action
        action = validate_action(candidate["action"])
    elif context.get("protocol", {}).get("schema") == "organoid.kuavo-sorting-protocol.v1":
        from .humanoid_sorting import validate_policy
        action = validate_policy(candidate["action"])
    else:
        action = normalize_action(candidate["action"])
    if digest(action) != digest(candidate["action"]):
        raise ValueError("Candidate action is not canonically normalized")
    return action


def binding(context, candidate):
    check_candidate(context, candidate)
    result = {f"{key}_sha256": digest(context[key])
              for key in ("robot", "initial_state", "scene", "protocol")}
    result.update(context_sha256=digest(context), action_sha256=digest(candidate["action"]),
                  candidate_sha256=digest(candidate))
    return result


def reconstruct(context):
    if context.get("schema") != CONTEXT_SCHEMA:
        raise ValueError("Unsupported robot context")
    fresh, candidates, simulation = prepare_context(context["scene"]["spec"])
    if digest(fresh) != digest(context):
        raise ValueError("Robot, initial state, scene, protocol or runtime drift")
    return candidates, simulation


def rank_candidates(context, candidates, checkpoint=None):
    templates, simulation = reconstruct(context)
    if context.get("protocol", {}).get("schema") in ("organoid.kuavo-sorting-protocol.v1",
                                                   "organoid.autonomous-grasp-protocol.v1"):
        from .kuavo_sorting_validation import rank_sorting
        return rank_sorting(context, candidates)
    spec, model, initial, _, start, goal, obstacle, _ = simulation
    checkpoint = Path(checkpoint) if checkpoint else None
    checkpoint_hash = sha256_file(checkpoint) if checkpoint and checkpoint.is_file() else None
    rows = []
    for index, candidate in enumerate(candidates):
        row = {"input_index": index, "candidate": candidate,
               "candidate_sha256": digest(candidate), "score": None,
               "execution_authorized": False}
        try:
            action = check_candidate(context, candidate)
            row["binding"] = binding(context, candidate)
            if not any(digest(t["action"]) == digest(action) for t in templates):
                row["critic"] = {"status": "not_evaluated_unsupported_path"}
            elif checkpoint_hash is None:
                row["critic"] = {"status": "not_evaluated_checkpoint_unavailable"}
            else:
                from .feasibility import score_candidate
                action = dict(action, waypoints=np.asarray(action["waypoints"]))
                features = initial_features(spec, model, initial, start, goal, obstacle, action)
                row["critic"] = score_candidate(checkpoint, spec.robot, features)
                if row["critic"]["status"] == "scored":
                    value = row["critic"]["probabilities"].get("success")
                    if value is not None:
                        row["score"] = number(value, 0, 1, "critic probability")
        except Exception as exc:
            row["critic"] = {"status": "not_evaluated", "reason": f"{type(exc).__name__}: {exc}"}
        row["critic"]["execution_authorized"] = False
        rows.append(row)
    if checkpoint_hash and sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError("Critic checkpoint changed during ranking")
    rows.sort(key=lambda row: (row["score"] is None, -(row["score"] or 0), row["input_index"]))
    return {"schema": "organoid.candidate-ranking.v1", "context_sha256": digest(context),
            "checkpoint_sha256": checkpoint_hash, "execution_authorized": False,
            "ranking": rows}


def verify_candidate(context, candidate, directory, run_id):
    """Fresh rollout and saved-control replay, preserving all partial evidence."""
    if context.get("protocol", {}).get("schema") == "organoid.autonomous-grasp-protocol.v1":
        from .autonomous_grasp_validation import verify_transfer
        return verify_transfer(context, candidate, directory, run_id)
    if context.get("protocol", {}).get("schema") == "organoid.kuavo-sorting-protocol.v1":
        from .kuavo_sorting_validation import verify_sorting
        return verify_sorting(context, candidate, directory, run_id)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "request.json", {"context": context, "candidate": candidate,
                                          "run_id": run_id})
    result = {"schema": RECEIPT_SCHEMA, "run_id": run_id, "status": "not_evaluated",
              "binding": None, "physics": None, "fresh_rollout": False}
    try:
        result["binding"] = binding(context, candidate)
        _, simulation = reconstruct(context)
        spec, model, initial, xml, start, goal, obstacle, targets = simulation
        action = dict(candidate["action"], waypoints=np.asarray(candidate["action"]["waypoints"]))
        runner = ArmRunner(spec, model, initial)
        features = initial_features(spec, model, initial, start, goal, obstacle, action)
        physics = runner.run(action, targets)
        result["fresh_rollout"] = True
        runner.save(directory / "episode", xml, action, targets, features, physics)
        result["physics"] = physics
        # Recheck inputs after simulation as well, before handing off evidence.
        reconstruct(context)
        if physics.get("replay", {}).get("verified") is True:
            result["status"] = physics["status"]
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    result["artifact_sha256"] = {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*")) if path.is_file()
    }
    write_json(directory / "validation.json", result)
    return result


def check_evidence(directory, expected_binding, run_id):
    """Verify receipt provenance and bytes; Kabuki independently applies gates."""
    directory = Path(directory)
    result = json.loads((directory / "validation.json").read_text())
    if result["schema"] != RECEIPT_SCHEMA or result["run_id"] != run_id:
        raise ValueError("Validation receipt identity mismatch")
    if result["binding"] != expected_binding:
        raise ValueError("Validation receipt binding mismatch")
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != set(result["artifact_sha256"]) | {"validation.json"}:
        raise ValueError("Validation artifact set mismatch")
    for relative, expected in result["artifact_sha256"].items():
        path = (directory / relative).resolve()
        if directory.resolve() not in path.parents or sha256_file(path) != expected:
            raise ValueError(f"Validation artifact hash mismatch: {relative}")
    request = json.loads((directory / "request.json").read_text())
    if request["run_id"] != run_id or binding(request["context"], request["candidate"]) != expected_binding:
        raise ValueError("Validation request binding mismatch")
    if result.get("physics") is not None:
        physics = json.loads((directory / "episode/receipt.json").read_text())
        if physics != result["physics"]:
            raise ValueError("Physics receipt mismatch")
        if request["context"].get("protocol", {}).get("schema") == "organoid.autonomous-grasp-protocol.v1":
            from .autonomous_grasp_validation import check_transfer_episode
            check_transfer_episode(directory, request, physics)
            return result
        if request["context"].get("protocol", {}).get("schema") == "organoid.kuavo-sorting-protocol.v1":
            from .kuavo_sorting_validation import check_sorting_episode
            check_sorting_episode(directory, request, physics)
            return result
        from .trajectory import read_trajectory
        _, arrays = read_trajectory(directory / "episode")
        if not np.array_equal(arrays["initial_state"], request["context"]["initial_state"]["values"]):
            raise ValueError("Episode initial state mismatch")
        if not np.array_equal(arrays["candidate_waypoints"], request["candidate"]["action"]["waypoints"]):
            raise ValueError("Episode candidate mismatch")
        if not np.array_equal(arrays["task_targets"], request["context"]["scene"]["targets"]):
            raise ValueError("Episode targets mismatch")
        if (directory / "episode/scene.xml").read_text() != request["context"]["scene"]["xml"]:
            raise ValueError("Episode scene mismatch")
    return result
