"""Shared Cartesian task candidates executed on pinned real-robot MJCF models.

No hardware execution. Full-model collision, force-limited servos, and saved
control replay are mandatory; feasibility labels describe only observed scope.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .hashing import sha256_file, sha256_json
from .profile import PKG_ROOT, load_profile
from .trajectory import read_trajectory, write_trajectory

ROBOTS = ("franka_emika_panda", "universal_robots_ur5e")
TASKS = ("reach_target", "obstacle_transfer", "inspection_sweep")
DT = .002
IK_TOL = .002
GOAL_TOL = .025
FEATURE_NAMES = (
    ["arm_dof"] + [f"initial_joint_{i}_rad" for i in range(7)]
    + [f"initial_joint_mask_{i}" for i in range(7)]
    + [f"tcp_start_{axis}_m" for axis in "xyz"]
    + [f"goal_delta_{axis}_m" for axis in "xyz"]
    + [f"obstacle_delta_{axis}_m" for axis in "xyz"]
    + [f"obstacle_half_{axis}_m" for axis in "xyz"]
    + ["obstacle_present", "waypoint_count", "path_length_m", "lift_m",
       "segment_duration_s", "control_noise_std_rad"]
    + [f"task_{name}" for name in TASKS]
)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


@lru_cache(maxsize=2)
def verified_profile(name):
    if name not in ROBOTS:
        raise ValueError(f"Unsupported simulation profile: {name}")
    profile = load_profile(name)
    manifest_path = PKG_ROOT / profile.raw["source_manifest"]
    manifest = json.loads(manifest_path.read_text())
    for relative, digest in manifest["sha256"].items():
        if sha256_file(manifest_path.parent / relative) != digest:
            raise ValueError(f"Robot source asset changed: {relative}")
    return profile


def model_tree(name):
    profile = verified_profile(name)
    tree = ET.parse(profile.mjcf_path()).getroot()
    compiler = tree.find("compiler")
    compiler.set("meshdir", str(profile.mjcf_path().parent / "assets"))
    tree.find("option").set("timestep", str(DT))
    if not profile.raw.get("tcp_site"):
        body = tree.find(f".//body[@name='{profile.raw['tcp_body']}']")
        ET.SubElement(body, "site", name="organoid_tcp",
                      pos=" ".join(map(str, profile.raw["tcp_position"])), size=".004")
    return tree


def tcp_name(profile):
    return profile.raw.get("tcp_site", "organoid_tcp")


@dataclass(frozen=True)
class SceneSpec:
    family: int
    robot: str
    task: str
    seed: int
    initial_jitter: float
    goal_offset: tuple
    obstacle_offset: tuple
    obstacle_half: tuple
    control_noise: float


def sample_scenes(families, seed):
    rng = np.random.default_rng(seed)
    result = []
    for family in range(families):
        task = TASKS[family % len(TASKS)]
        # Shared family split across robots; world placement is explicit per model.
        params = {
            "family": family, "task": task, "seed": seed + family,
            "initial_jitter": float(rng.uniform(0, .035)),
            "goal_offset": tuple(rng.uniform([-.08, .08, -.04], [.08, .20, .07])),
            "obstacle_offset": tuple(rng.uniform([-.012, -.012, -.025], [.012, .012, .025])),
            "obstacle_half": tuple(rng.uniform([.012, .012, .018], [.027, .027, .050])),
            "control_noise": float(rng.uniform(0, .0005)),
        }
        for robot in ROBOTS:
            result.append(SceneSpec(robot=robot, **params))
    return result


def scene_model(spec):
    profile = verified_profile(spec.robot)
    tree = model_tree(spec.robot)
    base = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    data = mujoco.MjData(base)
    mujoco.mj_resetDataKeyframe(base, data, base.key("home").id)
    jids = [base.joint(n).id for n in profile.joint_names]
    qadr = base.jnt_qposadr[jids]
    rng = np.random.default_rng(spec.seed)
    data.qpos[qadr] += rng.normal(0., spec.initial_jitter, len(jids))
    data.qpos[qadr] = np.clip(data.qpos[qadr], base.jnt_range[jids, 0] + .01,
                             base.jnt_range[jids, 1] - .01)
    data.ctrl[[base.actuator(n).id for n in profile.actuator_names]] = data.qpos[qadr]
    mujoco.mj_forward(base, data)
    start = data.site_xpos[base.site(tcp_name(profile)).id].copy()
    goal = start + spec.goal_offset
    obstacle = (start + goal) / 2 + spec.obstacle_offset
    if spec.task == "obstacle_transfer":
        ET.SubElement(tree.find("worldbody"), "geom", name="task_obstacle", type="box",
                      pos=" ".join(map(str, obstacle)), size=" ".join(map(str, spec.obstacle_half)),
                      rgba=".75 .2 .2 1")
    ET.SubElement(tree.find("worldbody"), "site", name="task_goal", type="sphere",
                  pos=" ".join(map(str, goal)), size=".015", rgba=".2 .7 .3 .7")
    xml = ET.tostring(tree, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    initial = mujoco.MjData(model)
    initial.qpos[:] = data.qpos
    initial.ctrl[:] = data.ctrl
    mujoco.mj_forward(model, initial)
    return model, initial, xml, start, goal, obstacle


def task_targets(task, start, goal):
    if task == "inspection_sweep":
        return np.asarray([goal + [-.035, 0, .025], goal + [.035, 0, .025], goal])
    return np.asarray([goal])


def candidate_paths(targets, start):
    def detour(height):
        path = []
        previous = start
        for target in targets:
            if height:
                path.extend([previous + [0, 0, height], target + [0, 0, height]])
            path.append(target.copy())
            previous = target
        return np.asarray(path)
    return [
        {"name": "direct", "lift": 0., "duration": 1.2, "waypoints": detour(0)},
        {"name": "raised", "lift": .13, "duration": 1.2, "waypoints": detour(.13)},
        {"name": "fast", "lift": 0., "duration": .18, "waypoints": detour(0)},
        {"name": "high_clearance", "lift": .65, "duration": 1.2, "waypoints": detour(.65)},
    ]


def initial_features(spec, model, initial, start, goal, obstacle, candidate):
    profile = verified_profile(spec.robot)
    qadr = [model.joint(name).qposadr[0] for name in profile.joint_names]
    joints = np.zeros(7)
    mask = np.zeros(7)
    joints[:len(qadr)] = initial.qpos[qadr]
    mask[:len(qadr)] = 1
    path = np.vstack([start, candidate["waypoints"]])
    obstacle_present = spec.task == "obstacle_transfer"
    vector = np.r_[
        len(qadr), joints, mask, start, goal - start,
        obstacle - start if obstacle_present else np.zeros(3),
        spec.obstacle_half if obstacle_present else np.zeros(3),
        float(obstacle_present), len(candidate["waypoints"]),
        np.linalg.norm(np.diff(path, axis=0), axis=1).sum(),
        candidate["lift"], candidate["duration"], spec.control_noise,
        [float(spec.task == task) for task in TASKS],
    ].astype(np.float32)
    assert len(vector) == len(FEATURE_NAMES)
    return vector


class ArmRunner:
    def __init__(self, spec, model, initial):
        self.spec, self.model = spec, model
        self.profile = verified_profile(spec.robot)
        self.data = mujoco.MjData(model)
        self.state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.initial_state = np.zeros(mujoco.mj_stateSize(model, self.state_spec))
        mujoco.mj_getState(model, initial, self.initial_state, self.state_spec)
        mujoco.mj_setState(model, self.data, self.initial_state, self.state_spec)
        mujoco.mj_forward(model, self.data)
        self.jids = np.asarray([model.joint(n).id for n in self.profile.joint_names])
        self.qadr = model.jnt_qposadr[self.jids]
        self.dadr = model.jnt_dofadr[self.jids]
        self.aids = np.asarray([model.actuator(n).id for n in self.profile.actuator_names])
        self.sid = model.site(tcp_name(self.profile)).id
        self.ik_data = mujoco.MjData(model)
        self.rng = np.random.default_rng(spec.seed + 100000)
        self.qpos, self.qvel = [self.data.qpos.copy()], [self.data.qvel.copy()]
        self.tcp = [self.data.site_xpos[self.sid].copy()]
        self.controls, self.references, self.forces, self.contacts = [], [], [], []
        self.ik_errors, self.contact_pairs = [], {}
        self.max_limit_excess = 0.
        self.invalid = False
        self.initial_warnings = np.asarray([w.number for w in self.data.warning])
        self.initial_contacts = self.contact_stats()

    def contact_stats(self):
        model, data = self.model, self.data
        obstacle_contact = False
        self_collision = False
        max_depth = 0.
        for contact in data.contact[:data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            names = [model.geom(g).name or f"geom_{g}" for g in (a, b)]
            bodies = model.geom_bodyid[[a, b]]
            # Distance is available even before the solver has produced force.
            if contact.dist >= -.0001:
                continue
            max_depth = max(max_depth, -float(contact.dist))
            obstacle_contact |= "task_obstacle" in names
            self_collision |= bool(bodies[0] != 0 and bodies[1] != 0 and bodies[0] != bodies[1])
            key = " | ".join(names)
            self.contact_pairs[key] = max(self.contact_pairs.get(key, 0.), -float(contact.dist))
        return [int(obstacle_contact), int(self_collision), max_depth]

    def ik(self, target, q0):
        model, data = self.model, self.ik_data
        data.qpos[:] = self.data.qpos
        def residual(q):
            data.qpos[self.qadr] = q
            mujoco.mj_kinematics(model, data)
            return data.site_xpos[self.sid] - target
        def jac(q):
            residual(q)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_comPos(model, data)
            mujoco.mj_jacSite(model, data, jp, jr, self.sid)
            return jp[:, self.dadr]
        lo = model.jnt_range[self.jids, 0] + .001
        hi = model.jnt_range[self.jids, 1] - .001
        result = least_squares(residual, np.clip(q0, lo, hi), jac=jac,
                               bounds=(lo, hi), max_nfev=75, gtol=1e-6)
        error = float(np.linalg.norm(residual(result.x)))
        self.ik_errors.append(error)
        return result.x, error

    def advance(self, reference):
        model, data = self.model, self.data
        data.ctrl[self.aids] = reference + self.rng.normal(0, self.spec.control_noise, len(reference))
        limited = model.actuator_ctrllimited[self.aids].astype(bool)
        bounded = self.aids[limited]
        data.ctrl[bounded] = np.clip(data.ctrl[bounded], model.actuator_ctrlrange[bounded, 0],
                                    model.actuator_ctrlrange[bounded, 1])
        self.controls.append(data.ctrl.copy())
        self.references.append(reference.copy())
        mujoco.mj_step(model, data)
        mujoco.mj_kinematics(model, data)
        self.qpos.append(data.qpos.copy())
        self.qvel.append(data.qvel.copy())
        self.tcp.append(data.site_xpos[self.sid].copy())
        self.forces.append(data.actuator_force.copy())
        stats = self.contact_stats()
        self.contacts.append(stats)
        q = data.qpos[self.qadr]
        bounds = model.jnt_range[self.jids]
        self.max_limit_excess = max(self.max_limit_excess, float(np.maximum(
            bounds[:, 0] - q, q - bounds[:, 1]).max()))
        self.invalid |= not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        self.invalid |= bool(np.any(np.asarray([w.number for w in data.warning]) != self.initial_warnings))
        return not (stats[0] or stats[1] or self.invalid)

    def run(self, candidate, targets):
        stop = None
        ik_failure = False
        reference = self.data.qpos[self.qadr].copy()
        if self.initial_contacts[0] or self.initial_contacts[1]:
            stop = "initial_collision"
        for waypoint in candidate["waypoints"]:
            if stop:
                break
            start = self.data.site_xpos[self.sid].copy()
            segments = max(2, int(np.ceil(np.linalg.norm(waypoint - start) / .025)))
            duration = candidate["duration"] / segments
            for part in range(segments):
                target = start + (waypoint - start) * (part + 1) / segments
                next_ref, error = self.ik(target, reference)
                if error > IK_TOL:
                    ik_failure, stop = True, "ik_tolerance_exceeded"
                    break
                steps = max(1, round(duration / DT))
                previous = reference.copy()
                for index in range(steps):
                    alpha = (index + 1) / steps
                    if not self.advance(previous + (next_ref - previous) * alpha):
                        stop = "numerical_error" if self.invalid else "collision"
                        break
                reference = next_ref
                if stop:
                    break
            if not stop:
                # Observe each requested waypoint before proceeding to the next.
                for _ in range(100):
                    if not self.advance(reference):
                        stop = "numerical_error" if self.invalid else "collision"
                        break
        if not stop:
            for _ in range(250):
                if not self.advance(reference):
                    stop = "numerical_error" if self.invalid else "collision"
                    break
        positions = np.asarray(self.tcp)
        visits, cursor = [], 0
        for target in targets:
            errors = np.linalg.norm(positions[cursor:] - target, axis=1)
            hits = np.flatnonzero(errors <= GOAL_TOL)
            visits.append(bool(len(hits)))
            if len(hits):
                cursor += int(hits[0])
        contact_arr = np.asarray([self.initial_contacts, *self.contacts])
        final_error = float(np.linalg.norm(positions[-1] - targets[-1]))
        forces = np.asarray(self.forces).reshape(-1, self.model.nu)
        effort_ok = bool(np.all(np.abs(forces) <= np.max(
            np.abs(self.model.actuator_forcerange), axis=1) + 1e-6))
        checks = {
            "completed_path": stop is None,
            "ordered_targets_reached": all(visits),
            "final_target_error": final_error <= GOAL_TOL,
            "no_obstacle_collision": not bool(contact_arr[:, 0].any()),
            "no_self_collision": not bool(contact_arr[:, 1].any()),
            "joint_limits": self.max_limit_excess <= .01,
            "effort_limits": effort_ok, "numerically_valid": not self.invalid,
            "no_external_forces": not bool(np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied)),
        }
        path_complete = stop is None
        labels = {
            "success": None if self.invalid else bool(all(checks.values())),
            "collision_free": False if contact_arr[:, :2].any() else (True if path_complete else None),
            "ik_feasible": False if ik_failure else (True if path_complete else None),
            "stable_grasp": None,
        }
        return {
            "schema": "organoid.multi-robot-receipt.v1",
            "status": "success" if all(checks.values()) else (
                "not_evaluated" if self.invalid else "rejected"),
            "stop_reason": stop, "checks": checks, "labels": labels,
            "metrics": {"final_target_error_m": final_error, "ordered_targets_hit": visits,
                        "max_ik_residual_m": max(self.ik_errors, default=None),
                        "max_joint_limit_excess_rad": self.max_limit_excess,
                        "contact_pairs_depth_m": self.contact_pairs},
            "scope": {
                "synthetic": True, "base": "fixed", "hardware": "not_evaluated",
                "orientation_task": "not_applicable_position_only",
                "grasp_stability": "not_applicable_no_grasp",
                "joint_speed_safety": "not_evaluated_no_velocity_limits_in_mjcf",
                "ik_failure": "bounded_solver_tolerance_failure_not_global_impossibility",
                "collision_threshold_m": .0001, "target_tolerance_m": GOAL_TOL,
            },
        }

    def save(self, directory, xml, candidate, targets, features, receipt):
        directory = Path(directory)
        directory.mkdir(parents=True)
        (directory / "scene.xml").write_text(xml)
        controls = np.asarray(self.controls, dtype=np.float64).reshape(-1, self.model.nu)
        arrays = {
            "state_time_s": np.arange(len(self.qpos)) * DT,
            "action_time_s": np.arange(len(controls)) * DT,
            "qpos": np.asarray(self.qpos), "qvel": np.asarray(self.qvel),
            "joint_position": np.asarray(self.qpos)[:, self.qadr],
            "joint_velocity": np.asarray(self.qvel)[:, self.dadr],
            "tcp_position": np.asarray(self.tcp), "control": controls,
            "joint_reference": np.asarray(self.references).reshape(-1, len(self.jids)),
            "actuator_force": np.asarray(self.forces).reshape(-1, self.model.nu),
            "contact": np.asarray(self.contacts).reshape(-1, 3),
            "initial_state": self.initial_state,
            "candidate_waypoints": candidate["waypoints"], "task_targets": targets,
            "initial_features": features,
        }
        clocks = {key: "state_time_s" for key in (
            "qpos", "qvel", "joint_position", "joint_velocity", "tcp_position", "state_time_s")}
        clocks.update({key: "action_time_s" for key in (
            "control", "joint_reference", "actuator_force", "contact", "action_time_s")})
        units = {"qpos": "rad_and_m", "qvel": "rad/s_and_m/s", "tcp_position": "m",
                 "joint_position": "rad", "joint_velocity": "rad/s",
                 "control": "model_native_see_actuator_names", "joint_reference": "rad",
                 "actuator_force": "N_or_N*m", "candidate_waypoints": "m", "task_targets": "m",
                 "state_time_s": "s", "action_time_s": "s"}
        streams = {key: {"unit": units.get(key, "mixed_see_metadata"),
                         "origin": "simulated" if key not in (
                             "candidate_waypoints", "task_targets", "initial_features") else "procedural",
                         "clock": clocks.get(key)} for key in arrays}
        streams["joint_reference"]["names"] = self.profile.joint_names
        streams["joint_position"]["names"] = self.profile.joint_names
        streams["joint_velocity"]["names"] = self.profile.joint_names
        manifest = write_trajectory(directory, arrays, streams, {
            "robot_profile": self.profile.name, "scene": asdict(self.spec),
            "state_spec": self.state_spec, "dt_s": DT, "synthetic": True,
            "joint_names": [self.model.joint(i).name for i in range(self.model.njnt)],
            "actuator_names": [self.model.actuator(i).name for i in range(self.model.nu)],
            "action_semantics": self.profile.raw["action_semantics"],
            "cartesian_frame": "model_world", "task_frame": "translation_at_initial_tcp_world_axes",
            "feature_names": FEATURE_NAMES, "hardware_test": "not_evaluated",
            "scene_sha256": sha256_file(directory / "scene.xml"),
            "source_manifest_sha256": sha256_file(PKG_ROOT / self.profile.raw["source_manifest"]),
            "mujoco_version": mujoco.__version__,
        })
        replay = replay_episode(directory)
        receipt["replay"] = replay
        if not replay["verified"]:
            receipt["labels"] = {key: None for key in receipt["labels"]}
            receipt["status"] = "not_evaluated"
        receipt["trajectory_sha256"] = manifest["sha256"]
        write_json(directory / "receipt.json", receipt)


def replay_episode(directory):
    directory = Path(directory)
    manifest, arrays = read_trajectory(directory)
    metadata = manifest["metadata"]
    profile = verified_profile(metadata["robot_profile"])
    if sha256_file(PKG_ROOT / profile.raw["source_manifest"]) != metadata["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    if sha256_file(directory / "scene.xml") != metadata["scene_sha256"]:
        raise ValueError("Scene hash mismatch")
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, arrays["initial_state"], metadata["state_spec"])
    mujoco.mj_forward(model, data)
    error = float(max(np.max(np.abs(data.qpos - arrays["qpos"][0])),
                      np.max(np.abs(data.qvel - arrays["qvel"][0]))))
    force_error = 0.
    tcp_error = float(np.max(np.abs(
        data.site_xpos[model.site(tcp_name(profile)).id] - arrays["tcp_position"][0])))
    warnings_before = np.asarray([w.number for w in data.warning])
    for index, control in enumerate(arrays["control"]):
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        mujoco.mj_kinematics(model, data)
        error = max(error, float(np.max(np.abs(data.qpos - arrays["qpos"][index + 1]))),
                    float(np.max(np.abs(data.qvel - arrays["qvel"][index + 1]))))
        force_error = max(force_error, float(np.max(
            np.abs(data.actuator_force - arrays["actuator_force"][index]))))
        tcp_error = max(tcp_error, float(np.max(np.abs(
            data.site_xpos[model.site(tcp_name(profile)).id] - arrays["tcp_position"][index + 1]))))
    warning_delta = np.asarray([w.number for w in data.warning]) - warnings_before
    finite = all(np.isfinite(value).all() for value in (arrays["qpos"], arrays["qvel"], arrays["control"]))
    return {"verified": bool(finite and max(error, force_error, tcp_error) <= 1e-10
                             and not warning_delta.any()),
            "max_state_error": error, "max_actuator_force_error": force_error,
            "max_tcp_error": tcp_error, "solver_warning_delta": warning_delta.tolist(),
            "control_steps": len(arrays["control"])}


def evaluate_scene(spec, output):
    directory = Path(output) / "episodes" / f"family_{spec.family:04d}" / spec.robot
    records = []
    try:
        model, initial, xml, start, goal, obstacle = scene_model(spec)
        targets = task_targets(spec.task, start, goal)
        for candidate in candidate_paths(targets, start):
            runner = ArmRunner(spec, model, initial)
            vector = initial_features(spec, model, initial, start, goal, obstacle, candidate)
            receipt = runner.run(candidate, targets)
            path = directory / candidate["name"]
            runner.save(path, xml, candidate, targets, vector, receipt)
            record = {
                "family": spec.family, "robot": spec.robot, "task": spec.task,
                "candidate": candidate["name"], "status": receipt["status"],
                "labels": receipt["labels"], "initial_features": vector.tolist(),
                "trajectory": str(path.relative_to(output)), "replay": receipt["replay"],
                "stop_reason": receipt["stop_reason"],
                "scene_hash": sha256_json(asdict(spec)),
                "initial_state_hash": sha256_json(runner.initial_state.tolist()),
            }
            write_json(path / "record.json", record)
            records.append(record)
    except Exception as exc:
        directory.mkdir(parents=True, exist_ok=True)
        error = {"family": spec.family, "robot": spec.robot, "task": spec.task,
                 "status": "error", "labels": {key: None for key in (
                     "success", "collision_free", "ik_feasible", "stable_grasp")},
                 "error": f"{type(exc).__name__}: {exc}"}
        write_json(directory / "error.json", error)
        records.append(error)
    return records
