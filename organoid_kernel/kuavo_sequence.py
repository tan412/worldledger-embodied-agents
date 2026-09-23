"""Causal, phase-free action-sequence interface on the original Kuavo model.

This is fixed-base right-arm manipulation. Observations contain RGB-D-derived
geometry, proprioception and modeled contact sensing, never object state arrays.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import h5py
import mujoco
import numpy as np

from .autonomous_grasp import observe_grasp, initial_action, propose_candidates
from .hashing import sha256_file, sha256_json
from .humanoid_sorting import (
    MotionRejected, RGBDObserver, VisionUnavailable, default_policy, export_sorting,
)
from .humanoid_tasks import DT, SUBSTEPS, TABLE_Z
from .kuavo_recovery import RecoveryRunner, RecoveryScene, write_json

ACTION_DT = DT * SUBSTEPS
ACTION_DIM = 8
MAX_SECONDS = 24.
OBS_NAMES = [
    *[f"joint_position_{i}" for i in range(7)],
    *[f"joint_velocity_{i}" for i in range(7)],
    "claw_angle", "tcp_x", "tcp_y", "tcp_z",
    *[f"previous_command_{i}" for i in range(8)],
    "object_relative_x", "object_relative_y", "object_relative_z",
    "goal_relative_x", "goal_relative_y", "goal_relative_z",
    "observed_width", "object_visible", "observation_age_s",
    "finger_front_N", "finger_back_N",
    *[f"obstacle_relative_{i}" for i in range(6)], "obstacle_visible",
]


class SequenceObserver(RGBDObserver):
    def __init__(self, model, calibration, spec):
        self.model, self.spec = model, spec
        self.calibration = json.loads(json.dumps(calibration))
        for value in self.calibration.values():
            value["width"], value["height"] = 320, 240
            k = np.asarray(value["K"])
            k[:2] *= 2 / 3
            # Pixel centers must scale about the outside pixel boundary.
            k[0, 2] = (320 - 1) / 2
            k[1, 2] = (240 - 1) / 2
            value["K"] = k.tolist()
        self.renderer = mujoco.Renderer(model, height=240, width=320)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        self.rng = np.random.default_rng(spec.seed + 9100)


class SequenceRunner(RecoveryRunner):
    def __init__(self, scene: RecoveryScene):
        # Analytic minimum-jerk teacher; no captured trajectory or video prior.
        super().__init__(scene, {"curves": {}, "retargeting": "none"})
        self.observer.close()
        self.observer = SequenceObserver(self.model, self.calibration, self.sorting_scene)
        self.calibration = self.observer.calibration
        self.capture_stride = 100  # 5 Hz camera, 25 Hz action endpoints, 500 Hz physics.
        self.seq_obs, self.seq_actions, self.seq_indices = [], [], []
        self.last_command = np.r_[self.reference[self.arm_columns["r"]], self.angles["r"]]
        self.cached_geometry = None
        self.geometry_at = 0
        self.geometry_visible = False
        self.elevated_contacts, self.elevated_relative = [], []
        self.command_clamps = 0
        self.inference_calls = 0

    def _sample_sensors(self):
        index = len(self.controls)
        if self.sensor_indices and self.sensor_indices[-1] == index:
            return
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        frame = self.observer.capture(self.data, "head")
        geometry = observe_grasp(frame)
        self.geometry_visible = geometry["status"] == "observed"
        if self.geometry_visible:
            self.cached_geometry, self.geometry_at = geometry, index
        self.sensor_indices.append(index)
        self.latest_frames = {"head": frame}
        if self.sensor_file:
            group = self.sensor_file.require_group("head")
            group.attrs["K"] = frame["K"]
            for key in ("rgb", "depth", "T_world_camera"):
                value = frame[key]
                if key not in group:
                    group.create_dataset(key, shape=(0, *value.shape),
                                         maxshape=(None, *value.shape), dtype=value.dtype,
                                         chunks=(1, *value.shape), compression="gzip",
                                         compression_opts=1)
                ds = group[key]
                ds.resize(len(self.sensor_indices), axis=0)
                ds[-1] = value

    def observation(self):
        if self.cached_geometry is None:
            raise VisionUnavailable("no_initial_RGBD_object_and_goal")
        o = self.cached_geometry
        columns = self.arm_columns["r"]
        tcp = self.data.site_xpos[self.tcp_ids["r"]].copy()
        tactile = np.asarray(self.contact_window)
        force = tactile[:, 1].mean(axis=0) if len(tactile) else np.zeros(2)
        obstacle = o["obstacle_bounds_m"]
        bounds = (np.asarray(obstacle) - tcp).ravel() if obstacle else np.zeros(6)
        obs = np.r_[
            self.data.qpos[self.body_qadr[columns]],
            self.data.qvel[self.body_dadr[columns]],
            self.data.qpos[self.model.joint("r_f_bar-1_joint").qposadr[0]],
            tcp, self.last_command, np.asarray(o["center_m"]) - tcp,
            np.asarray(o["goal_m"]) - tcp, o["width_m"], float(self.geometry_visible),
            (len(self.controls) - self.geometry_at) * DT, force, bounds,
            float(obstacle is not None),
        ].astype(np.float32)
        if obs.shape != (len(OBS_NAMES),) or not np.isfinite(obs).all():
            raise VisionUnavailable("invalid_causal_observation")
        return obs

    def tick(self, reference, angles, phase, saved_control=None):
        index = len(self.controls)
        if index % SUBSTEPS == 0:
            self.seq_obs.append(self.observation())
            self.seq_indices.append(index)
        super().tick(reference, angles, phase, saved_control)
        if len(self.controls) % SUBSTEPS == 0:
            self.last_command = np.r_[reference[self.arm_columns["r"]], angles["r"]]
            self.seq_actions.append(self.last_command.astype(np.float32))
        # Acceptance uses a shared physical window, not a policy-supplied phase.
        adr = self.object_addresses["orange"]
        if self.data.qpos[adr + 2] - self.initial_objects["orange"][2] > .025:
            pads = [self.model.geom(f"claw::r_{f}_fingers").id for f in ("f", "b")]
            center = self.data.geom_xpos[pads].mean(axis=0)
            self.elevated_contacts.append(bool(np.all(self.contacts_history[-1][1] > .02)))
            self.elevated_relative.append(self.data.qpos[adr:adr + 3].copy() - center)

    def apply_endpoint(self, endpoint, phase="policy"):
        endpoint = np.asarray(endpoint, dtype=float)
        if endpoint.shape != (8,) or not np.isfinite(endpoint).all():
            raise MotionRejected("nonfinite_or_malformed_policy_action")
        columns = self.arm_columns["r"]
        bounds = self.model.jnt_range[self.body_jids[columns]]
        low, high = np.r_[bounds[:, 0], -.60], np.r_[bounds[:, 1], -.05]
        bounded = np.clip(endpoint, low, high)
        self.command_clamps += int(np.any(np.abs(bounded - endpoint) > 1e-7))
        # The same endpoint slew limit is applied to collected and learned actions.
        current = np.r_[self.reference[columns], self.angles["r"]]
        bounded = current + np.clip(bounded - current, -ACTION_DT, ACTION_DT)
        target = self.reference.copy()
        target[columns] = bounded[:7]
        old, angles0 = self.reference.copy(), dict(self.angles)
        for sub in range(SUBSTEPS):
            u = (sub + 1) / SUBSTEPS
            angles = dict(angles0)
            angles["r"] += (bounded[-1] - angles0["r"]) * u
            self.tick(old + (target - old) * u, angles, phase)
            if self.forbidden_contacts or self.max_self_penetration > .003:
                raise MotionRejected("forbidden_contact")
            if self.external_force_seen or not np.isfinite(self.data.qpos).all():
                raise MotionRejected("invalid_dynamics")
        self.reference, self.angles["r"] = target, float(bounded[-1])
        self.positions["r"] = self.data.site_xpos[self.tcp_ids["r"]].copy()

    def move_arm(self, side, destination, angle, duration, phase, *, path_shape=None):
        if side != "r":
            raise ValueError("Sequence experiment only controls the right arm")
        from .humanoid_tasks import minimum_jerk
        start = self.positions[side].copy()
        start_angle = self.angles[side]
        count = max(1, round(duration / ACTION_DT))
        for frame in range(count):
            u = minimum_jerk((frame + 1) / count)
            target = start + (np.asarray(destination) - start) * u
            columns = self.arm_columns[side]
            result = self.iks[side].solve(target, np.eye(3), initial=self.reference[columns])
            self.ik_errors.append((result[2], result[3]))
            if result[2] > .002 or result[3] > .03:
                raise MotionRejected(f"ik_tolerance_exceeded:{phase}")
            self.apply_endpoint(np.r_[result[1], start_angle + (angle - start_angle) * u], phase)

    def wait(self, duration, phase):
        for _ in range(max(1, round(duration / ACTION_DT))):
            self.apply_endpoint(self.last_command, phase)

    def teacher_action(self, variant=0):
        feedback = {
            "initial_observation": self.cached_geometry,
            "gripper_calibration": self.gripper_calibration(),
            "stop_reason": "",
        }
        proposals = propose_candidates(feedback, initial_action(), set(), batch_size=4)
        if not proposals["candidates"]:
            raise VisionUnavailable(proposals.get("reason", "no_teacher_proposal"))
        action = proposals["candidates"][-1]["action"]
        # The fixed-orientation Kuavo IK has a higher floor than the finger mesh.
        # This teacher workspace constraint is frozen before evaluation.
        action["grasp_m"][2] = max(action["grasp_m"][2],
                                    self.cached_geometry["table_z_m"] + .043)
        action["goal_m"][2] = action["grasp_m"][2]
        rng = np.random.default_rng(self.sorting_scene.seed * 19 + variant)
        if variant:
            offset = rng.uniform(-.002, .002, 2)
            action["grasp_m"][:2] = (np.asarray(action["grasp_m"][:2]) + offset).tolist()
            action["grasp_m"][2] += float(rng.uniform(0., .003))
            action["lift_m"] = float(rng.uniform(.045, .053))
        action["origin"] = "analytic_RGBD_teacher_sequence"
        return action

    def sorting_receipt(self):
        receipt = super().sorting_receipt()
        obj = receipt["objects"]["orange"]
        relative = np.asarray(self.elevated_relative)
        obj["bilateral_transport_fraction"] = (float(np.mean(self.elevated_contacts))
                                                if self.elevated_contacts else None)
        obj["transport_slip_m"] = (float(np.max(np.linalg.norm(relative - relative[0], axis=1)))
                                  if len(relative) else None)
        checks = receipt["checks"]
        checks["task_complete"] = bool(obj["fully_inside"] and obj["settled"]
                                       and obj["lift_m"] >= .04)
        checks["bilateral_transport"] = bool(len(relative) >= 50 and
                                             obj["bilateral_transport_fraction"] >= .8)
        checks["slip_below_20mm"] = bool(len(relative) and obj["transport_slip_m"] <= .02)
        checks["contact_conservation"] = bool(obj["contact_audit"]["ok"])
        checks["no_execution_stop"] = self.stopped is None
        receipt["status"] = (
            "not_evaluated" if not checks["numerically_valid"] or
            (self.stopped and self.stopped["status"] == "not_evaluated")
            else "success" if all(checks.values()) else "rejected")
        receipt["schema"] = "organoid.kuavo-sequence-acceptance.v1"
        receipt["scope"] = {
            "robot": "original_Kuavo_biped_s200049", "base": "fixed",
            "controlled_joints": "right_arm_7_and_right_claw",
            "observation": "44D_causal_RGBD_geometry_proprioception_and_modeled_contact",
            "camera_hz": 5, "action_hz": 1 / ACTION_DT, "physics_hz": 1 / DT,
            "phase_in_actor": False, "simulator_object_truth_in_actor": False,
            "contact_window": "all_physics_steps_object_lift_above_25mm",
            "online_learning": False, "teacher_fallback": False,
            "kabuki_commit": False, "hardware": False,
        }
        receipt["metrics"].update(
            command_clamps=self.command_clamps, inference_calls=self.inference_calls,
            elevated_contact_steps=len(relative))
        return receipt

    def run_sequence(self, directory, *, policy=None, variant=0):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.sensor_file = h5py.File(directory / "sensors.h5", "w")
        self.sensor_file.attrs["origin"] = "simulated_head_RGBD"
        self.sensor_file.attrs["calibration"] = json.dumps(self.calibration)
        self.active_policy = default_policy(max_retries=0)
        self.active_part, self.started = "orange", True
        start = time.monotonic()
        self.trajectory_extension = {
            "schema": "organoid.kuavo-sequence.v1", "obs_names": OBS_NAMES,
            "action_dt_s": ACTION_DT, "actor": "teacher" if policy is None else policy.name,
            "scene": asdict(self.transfer_spec),
        }
        try:
            self._sample_sensors()
            if self.cached_geometry is None:
                raise VisionUnavailable("no_initial_RGBD_object_and_goal")
            if policy is None:
                self.transfer_action = self.teacher_action(variant)
                self.trajectory_extension["teacher_action"] = self.transfer_action
                self._sort_one("orange", self.active_policy)
            else:
                history = [self.observation()]
                for _ in range(round(MAX_SECONDS / ACTION_DT / policy.execute_steps)):
                    actions = policy.predict(history[-2:])
                    self.inference_calls += 1
                    for action in actions[:policy.execute_steps]:
                        self.apply_endpoint(action)
                        history.append(self.observation())
        except VisionUnavailable as exc:
            self.stopped = {"status": "not_evaluated", "reason": str(exc)}
        except MotionRejected as exc:
            self.stopped = {"status": "rejected", "reason": str(exc)}
        except Exception as exc:
            self.stopped = {"status": "not_evaluated", "reason": f"{type(exc).__name__}: {exc}"}
        self._sample_sensors()
        self.sensor_file.create_dataset("control_index", data=self.sensor_indices)
        self.sensor_file.create_dataset("timestamp_s", data=np.asarray(self.sensor_indices) * DT)
        self.sensor_file.close()
        self.sensor_file = None
        n = len(self.seq_actions)
        np.savez_compressed(
            directory / "sequence.npz",
            obs=np.asarray(self.seq_obs[:n], dtype=np.float32).reshape(-1, len(OBS_NAMES)),
            action=np.asarray(self.seq_actions, dtype=np.float32).reshape(-1, ACTION_DIM),
            control_index=np.asarray(self.seq_indices[:n], dtype=np.int64),
        )
        receipt = self.sorting_receipt()
        export_sorting(self, directory, receipt)
        record = {
            "status": "accepted" if receipt["status"] == "success" else receipt["status"],
            "reason": (receipt.get("stop") or {}).get("reason"),
            "receipt": receipt, "seconds": time.monotonic() - start,
            "sequence_sha256": sha256_file(directory / "sequence.npz"),
            "scene_sha256": sha256_file(directory / "scene.xml"),
            "initial_state_sha256": sha256_json(self.initial_state.tolist()),
            "samples": n,
        }
        write_json(directory / "record.json", record)
        return record
