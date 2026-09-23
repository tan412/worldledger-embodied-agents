"""Real Kabuki API and fresh MuJoCo gates, including fail-closed fault injection."""
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
KABUKI = Path(os.environ.get("KABUKI_PATH", ROOT.parents[1] / "3d_world_agent/kabuki"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(KABUKI))
pytest.importorskip("mujoco")
pytest.importorskip("fastapi")
if not (KABUKI / "kabuki/robot_control.py").exists():
    pytest.skip("Set KABUKI_PATH to the integrated control plane", allow_module_level=True)

from fastapi.testclient import TestClient
from kabuki.artifacts import sha256_payload
from kabuki.robot_control import RobotControl, create_world, transaction, get, put, decide
from organoid_kernel.kabuki_validation import (
    bind_candidate, binding, check_evidence, digest, prepare_context, rank_candidates,
    reconstruct, verify_candidate,
)
from organoid_kernel.multi_robot_tasks import ROBOTS, sample_scenes
from organoid_kernel.hashing import sha256_file
from organoid_kernel.trajectory import read_trajectory


@pytest.fixture(scope="module")
def scene_contexts():
    scenes = sample_scenes(2, 20260911)
    return {s.robot: prepare_context(asdict(s))[:2] for s in scenes if s.family == 1}


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("KABUKI_PATH", str(KABUKI))
    monkeypatch.setenv("ORGANOID_KERNEL", str(ROOT))
    monkeypatch.setenv("KABUKI_WORLDS", str(tmp_path / "worlds"))
    monkeypatch.setenv("ORGANOID_CRITIC", str(ROOT / "runs_multi_robot_480_v1/feasibility_critic.pt"))
    spec = importlib.util.spec_from_file_location("kabuki_test_service", KABUKI / "mcp/service/app.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return TestClient(module.app), module


def load(api, robot=ROBOTS[1], family=1):
    client, module = api
    scene = next(s for s in sample_scenes(family + 1, 20260911)
                 if s.robot == robot and s.family == family)
    response = client.post("/worlds/load_robot", json={"scene_spec": asdict(scene), "label": "test"})
    assert response.status_code == 200, response.text
    world = response.json()
    return world, "/worlds/" + world["world_id"], module.WORLDS / world["world_id"]


def submit(client, url, candidate, observation=None):
    return client.post(url + "/act", json={
        "patches": [{"op": "replace", "path": "/candidate", "value": candidate}],
        "observation": observation or {},
    })


def test_canonical_hash_matches_actual_kabuki(scene_contexts):
    context, candidates = scene_contexts[ROBOTS[0]]
    for value in (context, candidates[0], {"unicode": "机器人", "x": .2}):
        assert digest(value) == sha256_payload(value)
    first = binding(context, candidates[0])
    changed = copy.deepcopy(candidates[0])
    changed["action"]["duration"] += .1
    second = binding(context, changed)
    assert first["action_sha256"] != second["action_sha256"]
    assert first["initial_state_sha256"] == second["initial_state_sha256"]


def test_kuavo_visual_task_api_keeps_unknown_evidence(api):
    client, module = api
    response = client.post("/worlds/load_robot", json={
        "scene_spec": {"robot": "biped_s200049", "task": "visual_sorting",
                       "parameters": {"head_blackout": True}},
        "label": "kuavo-camera-unavailable",
    })
    assert response.status_code == 200, response.text
    world = response.json()
    url = "/worlds/" + world["world_id"]
    ranked = client.post(url + "/rank", json={"candidates": world["candidates"]}).json()
    assert all(row["score"] is None and not row["execution_authorized"]
               for row in ranked["ranking"])
    pending = submit(client, url, world["candidates"][0]).json()
    assert pending["status"] == "pending_verification"
    result = client.post(url + "/verify", json={"mode": "robot_trajectory"}).json()
    assert result["outcome"] == "not_evaluated_kept_evidence", result
    assert result["state_hash"] == world["state_hash"]
    verification = result["verification"]
    evidence = module.WORLDS / world["world_id"] / verification["evidence_directory"]
    receipt = check_evidence(evidence, pending["binding"], verification["run_id"])
    assert receipt["physics"]["metrics"]["control_steps"] == 0
    assert receipt["physics"]["replay"]["verified"]
    trace = client.get(url + "/trace").json()
    assert len(trace["not_evaluated"]) == 1 and not trace["accepted"]
    replay = client.post(url + "/replay").json()
    assert replay["matches_live_state"] and replay["evidence_hashes_checked"]


def test_captured_transfer_api_rejects_missing_vision_without_changing_world(api):
    source = ROOT / "runs_source_grasp_library_v2/episode_000000/prior.json"
    if not source.exists():
        pytest.skip("Requires the admitted captured-grasp integration fixture")
    client, _ = api
    response = client.post("/worlds/load_robot", json={"scene_spec": {
        "robot": "biped_s200049", "task": "autonomous_grasp",
        "parameters": {"head_blackout": True}, "source_prior_path": str(source)},
        "label": "captured-transfer-api"})
    assert response.status_code == 200, response.text
    world = response.json()
    url = "/worlds/" + world["world_id"]
    pending = submit(client, url, world["candidates"][0]).json()
    assert pending["status"] == "pending_verification"
    outcome = client.post(url + "/verify", json={"mode": "robot_trajectory"}).json()
    assert outcome["outcome"] == "not_evaluated_kept_evidence", outcome
    assert outcome["state_hash"] == world["state_hash"]
    trace = client.get(url + "/trace").json()
    assert len(trace["not_evaluated"]) == 1 and not trace["accepted"]
    assert client.post(url + "/replay").json()["evidence_hashes_checked"]


@pytest.mark.parametrize("robot", ROBOTS)
def test_real_api_rank_fresh_verify_and_replay(api, robot):
    client, _ = api
    world, url, directory = load(api, robot, family=0)
    ranked = client.post(url + "/rank", json={"candidates": world["candidates"]}).json()
    assert ranked["execution_authorized"] is False
    assert len(ranked["ranking"]) == 4
    assert all(r["execution_authorized"] is False for r in ranked["ranking"])
    candidate = world["candidates"][0]
    pending = submit(client, url, candidate).json()
    assert pending["status"] == "pending_verification"
    assert client.get(url + "/observe").json()["state_hash"] == world["state_hash"]
    outcome = client.post(url + "/verify", json={"mode": "robot_trajectory"}).json()
    assert outcome["outcome"] == "committed", outcome
    verification = outcome["verification"]
    assert all(verification["gates"].values())
    evidence = directory / verification["evidence_directory"]
    receipt = check_evidence(evidence, pending["binding"], verification["run_id"])
    assert receipt["fresh_rollout"] is True
    manifest, arrays = read_trajectory(evidence / "episode")
    assert len(arrays["control"]) > 500
    assert len(arrays["qpos"]) == len(arrays["control"]) + 1
    assert manifest["metadata"]["synthetic"] is True
    result = client.post(url + "/replay").json()
    assert result["matches_live_state"] and result["evidence_hashes_checked"]
    assert result["diff_count"] == 1
    # Durable state survives a new controller instance.
    assert RobotControl(directory, ROOT).observe()["state_hash"] == outcome["state_hash"]
    with (evidence / "episode/trajectory.npz").open("ab") as handle:
        handle.write(b"tamper")
    assert client.post(url + "/replay").status_code == 409


def test_high_critic_score_cannot_accept_collision(api, monkeypatch):
    from organoid_kernel import feasibility
    monkeypatch.setattr(feasibility, "score_candidate", lambda *args: {
        "status": "scored", "execution_authorized": True,
        "probabilities": {"success": 1., "collision_free": 1., "ik_feasible": 1.}})
    client, _ = api
    world, url, _ = load(api)
    ranked = client.post(url + "/rank", json={"candidates": world["candidates"]}).json()
    assert ranked["ranking"][0]["score"] == 1.
    assert ranked["ranking"][0]["critic"]["execution_authorized"] is False
    submit(client, url, world["candidates"][0], {"accepted": True, "critic_score": 1.})
    result = client.post(url + "/verify", json={"mode": "robot_trajectory"}).json()
    assert result["outcome"] == "rejected_kept_evidence", result
    assert "no_obstacle_collision" in result["verification"]["failed_checks"]
    assert result["state_hash"] == world["state_hash"]
    submit(client, url, world["candidates"][1])
    repaired = client.post(url + "/verify", json={"mode": "robot_trajectory"}).json()
    assert repaired["outcome"] == "committed", repaired
    trace = client.get(url + "/trace").json()
    assert len(trace["accepted"]) == len(trace["rejected"]) == 1
    before, after = trace["rejected"][0]["binding"], trace["accepted"][0]["binding"]
    for key in ("robot_sha256", "initial_state_sha256", "scene_sha256", "protocol_sha256"):
        assert before[key] == after[key]
    assert before["candidate_sha256"] != after["candidate_sha256"]


@pytest.mark.parametrize("path", ["/context/robot", "/context/scene", "/context/initial_state",
                                  "/context/protocol", "/schema", "/candidate/action"])
def test_locked_context_mutations_are_persisted_unknown(api, path):
    client, _ = api
    world, url, _ = load(api)
    result = client.post(url + "/act", json={"patches": [
        {"op": "replace", "path": path, "value": {"accepted": True}}]}).json()
    assert result["outcome"] == "not_evaluated_kept_evidence"
    assert result["state_hash"] == world["state_hash"]
    trace = client.get(url + "/trace").json()
    assert len(trace["not_evaluated"]) == 1 and not trace["accepted"]
    assert client.post(url + "/replay").json()["matches_live_state"]


def test_pending_conflict_setup_reset_and_unknown_action(api):
    client, _ = api
    world, url, _ = load(api)
    submit(client, url, world["candidates"][0])
    assert submit(client, url, world["candidates"][1]).status_code == 409
    result = client.post(url + "/verify", json={"mode": "setup"}).json()
    assert result["outcome"] == "not_evaluated_kept_evidence"
    submit(client, url, world["candidates"][0])
    assert client.post(url + "/reset").json()["dropped_pending"]
    bad = copy.deepcopy(world["candidates"][0])
    bad["context_sha256"] = "0" * 64
    assert submit(client, url, bad).json()["outcome"] == "not_evaluated_kept_evidence"
    assert len(client.get(url + "/trace").json()["not_evaluated"]) == 3
    assert client.get(url + "/observe").json()["state_hash"] == world["state_hash"]
    assert client.post(url + "/verify", json={"mode": "robot_trajectory"}).status_code == 409


@pytest.mark.parametrize("field", ["robot", "initial_state", "scene", "protocol"])
def test_rebound_modified_context_cannot_reuse_validator(scene_contexts, tmp_path, field):
    context, candidates = copy.deepcopy(scene_contexts[ROBOTS[0]])
    context[field]["unexpected"] = True
    candidate = bind_candidate(context, candidates[0]["action"])
    receipt = verify_candidate(context, candidate, tmp_path / field, "test-run")
    assert receipt["status"] == "not_evaluated"
    assert receipt["fresh_rollout"] is False
    assert "drift" in receipt["reason"]


@pytest.mark.parametrize("fault", ["candidate_swap", "stale_base", "hash_tamper", "worker_error"])
def test_pending_faults_never_commit(api, monkeypatch, fault):
    client, _ = api
    world, url, directory = load(api)
    submit(client, url, world["candidates"][0])
    if fault == "worker_error":
        def fail(*args):
            raise RuntimeError("injected worker failure")
        monkeypatch.setattr(RobotControl, "_run_organoid", fail)
    else:
        with transaction(directory) as con:
            pending = get(con, "pending")
            if fault == "candidate_swap":
                pending["_candidate_state"]["candidate"] = world["candidates"][1]
            elif fault == "stale_base":
                pending["base_state_sha256"] = "0" * 64
            else:
                pending["binding"]["action_sha256"] = "0" * 64
            if fault != "hash_tamper":
                pending["pending_sha256"] = digest({k: v for k, v in pending.items()
                                                    if k != "pending_sha256"})
            put(con, "pending", pending)
    result = client.post(url + "/verify", json={"mode": "robot_trajectory"}).json()
    assert result["outcome"] == "not_evaluated_kept_evidence", result
    assert result["state_hash"] == world["state_hash"]
    assert len(client.get(url + "/trace").json()["not_evaluated"]) == 1


def test_double_verify_serializes_single_commit(api, monkeypatch):
    client, _ = api
    world, url, directory = load(api, ROBOTS[0], family=0)
    submit(client, url, world["candidates"][0])
    original = RobotControl._run_organoid
    entered = threading.Event()
    release = threading.Event()
    def blocked(self, request, evidence):
        entered.set()
        assert release.wait(30)
        return original(self, request, evidence)
    monkeypatch.setattr(RobotControl, "_run_organoid", blocked)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(client.post, url + "/verify", json={"mode": "robot_trajectory"})
        assert entered.wait(30)
        second = pool.submit(client.post, url + "/verify", json={"mode": "robot_trajectory"})
        release.set()
        responses = [first.result(timeout=120), second.result(timeout=120)]
    assert sorted(r.status_code for r in responses) == [200, 409]
    assert RobotControl(directory, ROOT).observe()["iterations_committed"] == 1


def test_missing_critic_and_custom_path_do_not_create_physical_verdict(scene_contexts):
    context, candidates = scene_contexts[ROBOTS[0]]
    custom = copy.deepcopy(candidates[0])
    custom["action"]["name"] = "custom"
    result = rank_candidates(context, candidates + [custom], checkpoint=None)
    assert all(r["score"] is None for r in result["ranking"])
    assert result["ranking"][-1]["critic"]["status"] == "not_evaluated_unsupported_path"
    assert all(r["critic"]["execution_authorized"] is False for r in result["ranking"])


def test_nonfinite_and_unsupported_scene_rejected_at_input(api):
    client, _ = api
    scene = asdict(sample_scenes(1, 20260911)[0])
    scene["robot"] = "unknown_robot"
    assert client.post("/worlds/load_robot", json={"scene_spec": scene}).status_code == 422
    scene["robot"] = ROBOTS[0]
    scene["task"] = "grasp"
    assert client.post("/worlds/load_robot", json={"scene_spec": scene}).status_code == 422
    scene["task"] = "reach_target"
    scene["control_noise"] = float("nan")
    with pytest.raises(ValueError):
        prepare_context(scene)


@pytest.mark.parametrize("fault", ["missing_check", "truthy_integer", "replay_failed",
                                   "goal_threshold", "numerical_failure"])
def test_explicit_gates_reject_inconsistent_worker_receipts(api, monkeypatch, fault):
    client, _ = api
    world, url, _ = load(api, ROBOTS[0], family=0)
    original = RobotControl._run_organoid
    def changed(self, request, evidence):
        original(self, request, evidence)
        path = evidence / "validation.json"
        receipt = json.loads(path.read_text())
        physics = receipt["physics"]
        if fault == "missing_check":
            physics["checks"].pop("no_external_forces")
        elif fault == "truthy_integer":
            physics["checks"]["no_external_forces"] = 1
        elif fault == "replay_failed":
            physics["replay"]["verified"] = False
        elif fault == "goal_threshold":
            physics["metrics"]["final_target_error_m"] = 1.
        else:
            physics["checks"]["numerically_valid"] = False
        physics_path = evidence / "episode/receipt.json"
        physics_path.write_text(json.dumps(physics))
        receipt["artifact_sha256"]["episode/receipt.json"] = sha256_file(physics_path)
        path.write_text(json.dumps(receipt))
    monkeypatch.setattr(RobotControl, "_run_organoid", changed)
    submit(client, url, world["candidates"][0])
    result = client.post(url + "/verify", json={"mode": "robot_trajectory"}).json()
    assert result["outcome"] == "not_evaluated_kept_evidence", result
    assert result["state_hash"] == world["state_hash"]


def test_evidence_tamper_between_validation_and_commit_is_blocked(api, monkeypatch):
    from kabuki import robot_control
    client, _ = api
    world, url, _ = load(api, ROBOTS[0], family=0)
    original = robot_control.seal_files
    def change(directory, evidence, run_id):
        with (evidence / "episode/trajectory.npz").open("ab") as handle:
            handle.write(b"changed-before-commit")
        return original(directory, evidence, run_id)
    monkeypatch.setattr(robot_control, "seal_files", change)
    submit(client, url, world["candidates"][0])
    result = client.post(url + "/verify", json={"mode": "robot_trajectory"}).json()
    assert result["outcome"] == "not_evaluated_kept_evidence"
    assert result["state_hash"] == world["state_hash"]
    assert client.post(url + "/replay").json()["matches_live_state"]


def test_rejected_evidence_also_has_integrity_protection(api):
    client, _ = api
    world, url, directory = load(api)
    submit(client, url, world["candidates"][0])
    result = client.post(url + "/verify", json={"mode": "robot_trajectory"}).json()
    assert result["outcome"] == "rejected_kept_evidence"
    receipt = directory / result["verification"]["evidence_directory"] / "validation.json"
    receipt.write_text("{}")
    assert client.post(url + "/replay").status_code == 409


def test_legacy_mode_is_preserved_but_robot_schema_cannot_bypass(api):
    client, _ = api
    assert client.post("/worlds/load", json={"world_spec": {
        "schema": "kabuki.robot-world.v1"}}).status_code == 400
    world = client.post("/worlds/load", json={"label": "legacy"}).json()
    url = "/worlds/" + world["world_id"]
    result = client.post(url + "/act", json={"patches": [
        {"op": "replace", "path": "/status", "value": "configured"}]})
    assert result.status_code == 200
    assert client.post(url + "/verify", json={"mode": "robot_trajectory"}).status_code == 400
    assert client.post(url + "/verify", json={"mode": "setup"}).json()["outcome"] == "committed"
    assert client.post(url + "/replay").json()["matches_live_state"]
