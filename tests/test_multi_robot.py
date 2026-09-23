"""Multi-robot provenance, non-leaky labels, clocks, mapping, and replay."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip("mujoco")

from organoid_kernel.evidence import EvidencePackage
from organoid_kernel.planner import _requirement_met
from organoid_kernel.profile import load_profile
from organoid_kernel.trajectory import write_trajectory, read_trajectory
from organoid_kernel.multi_robot_tasks import (
    ROBOTS, FEATURE_NAMES, ArmRunner, candidate_paths, initial_features,
    replay_episode, sample_scenes, scene_model, task_targets,
)
from organoid_kernel.feasibility import family_split, data_arrays, score_candidate
from organoid_kernel.robot_trajectory_io import validate_joint_input, transfer_episode, import_joint_file
from organoid_kernel.adapters.base import detect_format
from organoid_kernel.cli import _load_episode


def test_explicit_joint_schema_rejects_wrong_robot_and_units():
    profile = load_profile(ROBOTS[0])
    reverse = profile.joint_names[::-1]
    assert profile.map_joint_columns(reverse, "rad") == list(range(6, -1, -1))
    for columns, unit in ((reverse, "deg"), (["unknown"] * 7, "rad"), (reverse[:6], "rad")):
        with pytest.raises(ValueError):
            profile.map_joint_columns(columns, unit)


def test_mjcf_profiles_do_not_enter_urdf_only_validators():
    pkg = EvidencePackage("test", "synthetic")
    assert not _requirement_met("robot.model", pkg, {"profile": load_profile(ROBOTS[0])})
    assert _requirement_met("robot.model", pkg, {"profile": load_profile("biped_s200049")})


def test_trajectory_clocks_missing_values_and_integrity(tmp_path):
    arrays = {"time": np.array([0., .1]), "state": np.array([[1., np.nan], [2., 3.]])}
    streams = {
        "time": {"unit": "s", "origin": "observed", "clock": "time"},
        "state": {"unit": "rad", "origin": "observed", "clock": "time"},
    }
    with pytest.raises(ValueError, match="missing"):
        write_trajectory(tmp_path / "bad", arrays, streams, {})
    streams["state"]["allows_missing"] = True
    write_trajectory(tmp_path / "good", arrays, streams, {})
    _, loaded = read_trajectory(tmp_path / "good")
    np.testing.assert_array_equal(loaded["state"], arrays["state"])
    arrays["time"] = np.array([.1, .1])
    with pytest.raises(ValueError, match="Nonmonotonic"):
        write_trajectory(tmp_path / "badtime", arrays, streams, {})
    with (tmp_path / "good/trajectory.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        read_trajectory(tmp_path / "good")


def test_family_split_and_mask_do_not_use_receipt_metrics():
    records = [{"family": i, "initial_features": np.zeros(len(FEATURE_NAMES)).tolist(),
                "labels": {"success": i % 2 == 0, "collision_free": None, "ik_feasible": False}}
               for i in range(20) for _ in range(8)]
    split = family_split(records, 123)
    assert set(split) == set(range(20))
    assert set(split.values()) == {"train", "validation", "test"}
    before = data_arrays(records)
    for record in records:
        record["metrics"] = {"final_xy_error_norm_m": 999, "collision": True}
    after = data_arrays(records)
    np.testing.assert_array_equal(before[0], after[0])
    assert not before[2][:, 1].any()
    assert before[2][:, 2].all()
    assert not any("final" in name or "residual" in name for name in FEATURE_NAMES)


@pytest.mark.parametrize("robot", ROBOTS)
def test_real_model_episode_replays_and_saves_full_trace(tmp_path, robot):
    spec = next(s for s in sample_scenes(1, 20260911) if s.robot == robot)
    model, initial, xml, start, goal, obstacle = scene_model(spec)
    targets = task_targets(spec.task, start, goal)
    candidate = candidate_paths(targets, start)[0]
    runner = ArmRunner(spec, model, initial)
    vector = initial_features(spec, model, initial, start, goal, obstacle, candidate)
    receipt = runner.run(candidate, targets)
    assert receipt["labels"]["success"]
    assert receipt["labels"]["stable_grasp"] is None
    directory = tmp_path / robot
    runner.save(directory, xml, candidate, targets, vector, receipt)
    assert replay_episode(directory)["verified"]
    manifest, arrays = read_trajectory(directory)
    assert detect_format(directory) == "organoid_trajectory_v2"
    package = _load_episode(directory, 0)
    assert package.get("robot.joint_position").columns == load_profile(robot).joint_names
    assert package.get("robot.joint_position").origin == "derived"
    assert package.get("robot.action").provenance["synthetic"] is True
    assert len(arrays["qpos"]) == len(arrays["control"]) + 1
    assert arrays["control"].dtype == np.float64
    assert manifest["streams"]["joint_position"]["names"] == load_profile(robot).joint_names
    mapped, audit = validate_joint_input(
        robot, load_profile(robot).joint_names[::-1], "rad",
        arrays["state_time_s"], arrays["joint_position"][:, ::-1])
    np.testing.assert_array_equal(mapped, arrays["joint_position"])
    assert audit["dynamic_replay"].startswith("not_evaluated")
    sidecar = tmp_path / (robot + "-sidecar.json")
    sidecar.write_text(json.dumps({
        "profile": robot, "joint_names": load_profile(robot).joint_names,
        "joint_unit": "rad", "time_unit": "s", "state_semantics": "joint_position",
        "origin": "simulated", "time_key": "state_time_s", "state_key": "joint_position",
    }))
    imported = import_joint_file(directory / "trajectory.npz", sidecar,
                                 tmp_path / (robot + "-import"))
    assert imported["rows"] == len(arrays["qpos"])
    assert imported["within_model_joint_limits"]
    other = ROBOTS[1] if robot == ROBOTS[0] else ROBOTS[0]
    transferred = transfer_episode(directory, other, tmp_path / ("transfer-" + robot))
    assert transferred["replay"]["verified"]
    assert not transferred["source_commands_copied"]
    with (directory / "scene.xml").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Scene hash"):
        replay_episode(directory)


def test_solver_failure_masks_unobserved_collision_outcome():
    spec = sample_scenes(1, 20260911)[0]
    model, initial, _, start, goal, _ = scene_model(spec)
    runner = ArmRunner(spec, model, initial)
    candidate = {"waypoints": np.asarray([start + [0, 0, 10]]), "duration": 1.}
    # Force a bounded solver failure at the first waypoint, not a runtime error.
    runner.ik = lambda target, q: (q, .1)
    receipt = runner.run(candidate, np.asarray([goal]))
    assert receipt["labels"]["ik_feasible"] is False
    assert receipt["labels"]["collision_free"] is None
    assert receipt["labels"]["success"] is False


def test_checkpoint_load_and_ood_fail_closed():
    root = ROOT / "runs_multi_robot_480_v1"
    if not (root / "feasibility_critic.pt").exists():
        pytest.skip("Generated dataset not installed")
    records = json.loads((root / "records.json").read_text())
    record = records[0]
    scored = score_candidate(root / "feasibility_critic.pt", record["robot"], record["initial_features"])
    assert scored["execution_authorized"] is False
    outside = list(record["initial_features"])
    outside[0] = 1000
    assert score_candidate(root / "feasibility_critic.pt", record["robot"], outside)["status"] == "not_evaluated_ood"
    assert score_candidate(root / "feasibility_critic.pt", "unknown", record["initial_features"])["status"] == "not_evaluated_ood"
