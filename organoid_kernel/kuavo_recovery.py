"""Single-robot recovery experiments without editing the frozen grasp demo."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .autonomous_grasp import TransferRunner, TransferScene, observe_grasp, validate_action, transfer_scene_xml
from .hashing import sha256_array, sha256_file, sha256_json
from .humanoid_sorting import PARTS, SortingScene, SortingRunner
from .humanoid_tasks import HumanoidKinematics
from .physics_audit import FrameAuditor


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


@dataclass(frozen=True)
class RecoveryScene(TransferScene):
    kind: str = "box"
    object_size: tuple = (.022, .025, .040)

    def sorting_spec(self):
        if self.kind != "box":
            raise ValueError("This experiment covers cuboids only")
        size = np.asarray(self.object_size, dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size < .018) or np.any(size > .05):
            raise ValueError("Cuboid size outside experiment support")
        return SortingScene(
            seed=self.seed, orange_xy=self.source_xy, green_xy=self.goal_xy,
            size=tuple(size), mass=self.mass, friction=self.friction,
            barrier_height=self.barrier_height, depth_noise_m=self.depth_noise_m,
            head_blackout=self.head_blackout)


class KinematicOnlyIK(HumanoidKinematics):
    """The existing bounded SciPy solver, using FK instead of full dynamics."""

    def pose(self, q):
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.tcp].copy(), self.data.xmat[self.wrist_bid].reshape(3, 3).copy()

    def solve(self, position, rotation, initial=None, attempts=1):
        result = super().solve(position, rotation, initial, attempts)
        if result[2] > .002 or result[3] > .03:
            result = super().solve(position, rotation, initial, max(4, attempts))
        return result


class RecoveryRunner(TransferRunner):
    """Shared dynamics for all methods; consecutive attempts keep physical state."""

    def __init__(self, scene, prior):
        self.transfer_spec, self.prior = scene, prior
        def factory(_):
            config, xml, calibration = transfer_scene_xml(scene)
            tree = ET.fromstring(xml)
            # MuJoCo's default max-friction mixing otherwise masks all object
            # coefficients below the fixed 1.5 gripper coefficient. The object
            # material controls pair friction in this declared experiment.
            tree.find(".//geom[@name='object::target']").set("priority", "1")
            return config, ET.tostring(tree, encoding="unicode"), calibration
        SortingRunner.__init__(self, scene.sorting_spec(), scene_factory=factory)
        self.initial_observation, self.transfer_action = None, None
        self.iks = {side: KinematicOnlyIK(self.model, side) for side in ("r", "l")}

    def snapshot(self):
        state = np.empty_like(self.initial_state)
        mujoco.mj_getState(self.model, self.data, state, self.state_spec)
        return {
            "state": state.tolist(), "scene_sha256": sha256_json(self.xml),
            "reference": self.reference.tolist(), "angles": dict(self.angles),
            "positions": {s: p.tolist() for s, p in self.positions.items()},
            "camera_rng": deepcopy(self.observer.rng.bit_generator.state),
        }

    def restore_branch(self, snapshot):
        if snapshot["scene_sha256"] != sha256_json(self.xml):
            raise ValueError("Snapshot belongs to another frozen world")
        mujoco.mj_setState(self.model, self.data, np.asarray(snapshot["state"], dtype=float), self.state_spec)
        mujoco.mj_forward(self.model, self.data)
        self.reference = np.asarray(snapshot["reference"])
        self.angles = dict(snapshot["angles"])
        self.positions = {s: np.asarray(p) for s, p in snapshot["positions"].items()}
        self.observer.rng.bit_generator.state = deepcopy(snapshot["camera_rng"])
        # Restoring a branch must not leak another branch's tactile history.
        self.begin_segment()

    def begin_segment(self):
        """Reset evidence accumulators only, never qpos, qvel, objects or time."""
        mujoco.mj_getState(self.model, self.data, self.initial_state, self.state_spec)
        self.initial_warnings = np.array([w.number for w in self.data.warning])
        self.initial_objects = {p: self.data.qpos[a:a + 7].copy()
                                for p, a in self.object_addresses.items()}
        self.controls, self.references, self.phases, self.active_history = [], [], [], []
        self.positions_history = [self.data.qpos.copy()]
        self.velocities_history = [self.data.qvel.copy()]
        self.tcp_history = [np.array([self.data.site_xpos[self.tcp_ids[s]] for s in ("l", "r")])]
        self.forces_history, self.contacts_history = [], []
        self.sensor_indices, self.decisions, self.ik_errors = [], [], []
        self.max_speed[:] = 0
        self.max_force[:] = 0
        self.max_joint_limit_excess = self.max_penetration = self.max_self_penetration = 0.
        self.forbidden_contacts, self.contact_window = {}, []
        self.external_force_seen = False
        self.completed, self.stopped, self.active_part = [], None, None
        self.auditors = {name: FrameAuditor(self.model, self.data, joint)
                         for name, joint in (("orange", "objfree::target"), ("blue", "objfree::blue"))}
        self.carry = {p: [] for p in PARTS}
        self.max_lift_by_part = {p: 0. for p in PARTS}
        self.started, self.active_policy = False, None
        # A previous stop may have occurred partway through an interpolated move.
        self.reference = self.data.qpos[self.body_qadr].copy()
        self.positions = {s: self.data.site_xpos[self.tcp_ids[s]].copy() for s in ("r", "l")}
        self.angles = {s: float(self.data.qpos[self.model.joint(f"{s}_f_bar-1_joint").qposadr[0]])
                       for s in ("r", "l")}

    def policy_feedback(self, previous_action, previous=None):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        frame = self.observer.capture(self.data, "head")
        observation = observe_grasp(frame)
        columns = self.arm_columns["r"]
        tactile = np.asarray(self.contacts_history[-100:])
        normal = tactile[:, 1].mean(axis=0).tolist() if len(tactile) else [0., 0.]
        feedback = {
            "schema": "organoid.kuavo-recovery-feedback.v1",
            "initial_observation": observation,
            "gripper_calibration": self.gripper_calibration(),
            "previous_action": deepcopy(previous_action),
            "stop_reason": (previous or {}).get("reason", ""),
            "last_contact_fraction": (float(np.mean(np.all(tactile[:, 1] > .02, axis=1)))
                                      if len(tactile) else None),
            "finger_normal_force_N": normal,
            "joint_position_rad": self.data.qpos[self.body_qadr[columns]].tolist(),
            "joint_velocity_rad_s": self.data.qvel[self.body_dadr[columns]].tolist(),
            "tcp_position_m": self.data.site_xpos[self.tcp_ids["r"]].tolist(),
            "observed_claw_angle_rad": float(self.data.qpos[
                self.model.joint("r_f_bar-1_joint").qposadr[0]]),
            "reset_semantics": "live_post_failure_state",
        }
        return feedback, frame

    def execute(self, action, directory, *, continuation=True):
        directory = Path(directory)
        if continuation:
            self.begin_segment()
        start = time.monotonic()
        receipt = self.run_transfer(validate_action(action), directory)
        status = "accepted" if receipt["status"] == "success" else receipt["status"]
        if receipt.get("replay", {}).get("verified") is not True:
            status = "not_evaluated"
        receipt["scope"]["object_material_priority"] = 1
        return {
            "status": status, "reason": (receipt.get("stop") or {}).get("reason", ""),
            "action": action, "episode": str(directory), "receipt": receipt,
            "initial_state_sha256": sha256_array(self.initial_state),
            "initial_integration_sha256": sha256_json(self.initial_state.tolist()),
            "final_state_sha256": sha256_json(self.snapshot()["state"]),
            "scene_sha256": sha256_json(self.xml),
            "trajectory_sha256": sha256_file(directory / "trajectory.npz"),
            "seconds": time.monotonic() - start,
        }

    def sorting_receipt(self):
        receipt = super().sorting_receipt()
        receipt["scope"]["retry"] = "continuous_state_within_method_paired_branch_only_at_first_failure"
        receipt["scope"]["ik"] = "shared_multiseed_existing_bounded_scipy_solver"
        return receipt


def save_decision(directory, feedback, frame):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "feedback.json", feedback)
    np.savez_compressed(directory / "observation.npz", **frame)
