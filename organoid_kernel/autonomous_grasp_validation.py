"""Locked source-motion identity and fresh verification for autonomous transfer."""
from dataclasses import asdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .autonomous_grasp import CHECKS, SCHEMA, TransferRunner, TransferScene, initial_action
from .hashing import sha256_file, sha256_json
from .kuavo_sorting_validation import prepare_sorting_context
from .profile import PKG_ROOT
from .trajectory import read_trajectory

PROTOCOL = "organoid.autonomous-grasp-protocol.v1"


def load_prior(path):
    path = Path(path).resolve()
    prior = json.loads(path.read_text())
    if prior["schema"] != "organoid.captured-grasp-prior.v1" or prior["status"] != "accepted":
        raise ValueError("Source motion was not admitted")
    if not prior["source"]["checks"] or not all(v is True for v in prior["source"]["checks"].values()):
        raise ValueError("Source motion admission checks incomplete")
    for name, expected in (
        ("captured.parquet", prior["source"]["local_sha256"]),
        ("captured-motion.npz", prior["captured_motion_sha256"]),
    ):
        if sha256_file(path.parent / name) != expected:
            raise ValueError("Captured source evidence changed")
    if json.loads((path.parent / "audit/claim-ledger.json").read_text())["grade"] != "accepted":
        raise ValueError("Captured source audit is not accepted")
    return prior


def prepare_transfer_context(raw):
    from .kabuki_validation import bind_candidate
    if set(raw) != {"robot", "task", "parameters", "source_prior_path"} or raw["robot"] != "biped_s200049":
        raise ValueError("Malformed transfer scene")
    spec = TransferScene(**raw["parameters"])
    prior_path = Path(raw["source_prior_path"]).resolve()
    prior = load_prior(prior_path)
    context, _, _ = prepare_sorting_context({
        "robot": raw["robot"], "task": "visual_sorting", "parameters": asdict(spec.sorting_spec())})
    runner = TransferRunner(spec, prior, cameras=False)
    try:
        context["scene"] = {
            "spec": {"robot": raw["robot"], "task": "autonomous_grasp",
                     "parameters": asdict(spec), "source_prior_path": str(prior_path)},
            "xml": runner.xml, "calibration": runner.calibration,
            "source_prior": prior, "source_prior_sha256": sha256_file(prior_path),
            "source_audit_sha256": sha256_file(prior_path.parent / "audit/claim-ledger.json"),
        }
        context["initial_state"] = {"state_spec": int(runner.state_spec),
                                    "values": runner.initial_state.tolist()}
        context["protocol"].update(schema=PROTOCOL, required_checks=list(CHECKS),
                                    candidate_contract="absolute_grasp_goal_gap_and_transport_route")
        for name in ("source_grasp.py", "autonomous_grasp.py", "autonomous_grasp_validation.py"):
            context["protocol"]["code_sha256"][name] = sha256_file(PKG_ROOT / "organoid_kernel" / name)
        context = json.loads(json.dumps(context, allow_nan=False))
        return context, [bind_candidate(context, initial_action())], None
    finally:
        runner.close()


def verify_transfer(context, candidate, directory, run_id):
    from .kabuki_validation import binding, reconstruct
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "request.json").write_text(json.dumps(
        {"context": context, "candidate": candidate, "run_id": run_id}, indent=2))
    result = {"schema": "organoid.bound-validation.v1", "run_id": run_id,
              "status": "not_evaluated", "binding": None, "physics": None, "fresh_rollout": False}
    runner = None
    try:
        result["binding"] = binding(context, candidate)
        reconstruct(context)
        runner = TransferRunner(TransferScene(**context["scene"]["spec"]["parameters"]),
                                context["scene"]["source_prior"])
        if not np.array_equal(runner.initial_state, context["initial_state"]["values"]):
            raise ValueError("Initial state drift")
        physics = runner.run_transfer(candidate["action"], directory / "episode")
        result.update(fresh_rollout=True, physics=physics)
        reconstruct(context)
        if physics["replay"]["verified"] is True:
            result["status"] = physics["status"]
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if runner:
            runner.close()
    result["artifact_sha256"] = {str(p.relative_to(directory)): sha256_file(p)
                                 for p in sorted(directory.rglob("*")) if p.is_file()}
    (directory / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def check_transfer_episode(directory, request, physics):
    from .kabuki_validation import digest
    episode = Path(directory) / "episode"
    context = request["context"]
    manifest, arrays = read_trajectory(episode)
    metadata = manifest["metadata"]
    if physics["schema"] != SCHEMA:
        raise ValueError("Wrong transfer receipt")
    extension = metadata["extension"]
    if digest(extension["action"]) != digest(request["candidate"]["action"]):
        raise ValueError("Saved candidate differs from accepted request")
    if extension["source_prior_sha256"] != sha256_json(context["scene"]["source_prior"]):
        raise ValueError("Wrong captured motion source")
    if digest(extension["scene"]) != digest(context["scene"]["spec"]["parameters"]):
        raise ValueError("Wrong transfer scene")
    if not np.array_equal(arrays["initial_state"], context["initial_state"]["values"]):
        raise ValueError("Wrong initial integration state")
    tree = ET.fromstring(context["scene"]["xml"])
    for mesh in tree.findall("./asset/mesh"):
        mesh.set("file", "robot_assets/" + Path(mesh.get("file")).name)
    if (episode / "scene.xml").read_text() != ET.tostring(tree, encoding="unicode"):
        raise ValueError("Wrong portable scene")
    if sha256_file(episode / "sensors.h5") != metadata["sensor_sha256"]:
        raise ValueError("Wrong sensor content")
    with h5py.File(episode / "sensors.h5") as sensors:
        indices = sensors["control_index"][:]
        if not np.array_equal(indices, arrays["sensor_control_index"]) or indices[0] != 0 or indices[-1] != len(arrays["control"]):
            raise ValueError("Sensor clock incomplete")
        for camera in ("head", "wrist_right", "wrist_left"):
            if any(len(sensors[camera][key]) != len(indices) for key in ("rgb", "depth", "T_world_camera")):
                raise ValueError("Incomplete sensor stream")
    feedback = json.loads((episode / "feedback.json").read_text())
    if feedback["schema"] != "organoid.grasp-feedback.v1" or feedback["sensor_index"] != 0:
        raise ValueError("Unknown feedback contract")
