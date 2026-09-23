"""Observation-only candidate generation for captured-grasp transfer experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from .hashing import sha256_json
from .humanoid_tasks import TABLE_Z, claw_pose
from .humanoid_sorting import (
    SortingRunner, SortingScene, MotionRejected, VisionUnavailable, default_policy,
    detect_color, make_scene,
)

SCHEMA = "organoid.autonomous-grasp.v1"
CHECKS = (
    "task_complete", "object_in_bin", "object_settled", "object_lifted_40mm",
    "bilateral_transport", "slip_below_20mm", "no_forbidden_contact",
    "penetration_below_3mm", "self_penetration_below_3mm", "joint_limits",
    "joint_speed_limits", "effort_limits", "numerically_valid", "no_external_forces",
    "geometry_fidelity", "contact_conservation",
)


@dataclass(frozen=True)
class TransferScene:
    kind: str = "sphere"
    diameter: float = .032
    source_xy: tuple = (.30, -.415)
    goal_xy: tuple = (.30, -.22)
    mass: float = .025
    friction: float = .8
    barrier_height: float = .025
    depth_noise_m: float = .0001
    head_blackout: bool = False
    seed: int = 41

    def __post_init__(self):
        if self.kind not in ("sphere", "box") or not .02 <= self.diameter <= .06:
            raise ValueError("Transfer object outside declared model support")
        self.sorting_spec()

    def sorting_spec(self):
        return SortingScene(
            seed=self.seed, orange_xy=self.source_xy, green_xy=self.goal_xy,
            size=(self.diameter,) * 3 if self.kind == "sphere" else (.022, .025, .040),
            mass=self.mass, friction=self.friction, barrier_height=self.barrier_height,
            depth_noise_m=self.depth_noise_m, head_blackout=self.head_blackout)


def transfer_scene_xml(spec):
    config, xml, calibration = make_scene(spec.sorting_spec())
    root = ET.fromstring(xml)
    if spec.kind == "sphere":
        body = root.find(".//body[@name='object::target']")
        geom = body.find("geom")
        geom.set("type", "sphere")
        geom.set("size", str(spec.diameter / 2))
    return config, ET.tostring(root, encoding="unicode"), calibration


def color_points(frame, hue_low, hue_high, *, largest=True):
    hsv = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([hue_low, 100, 65]), np.array([hue_high, 255, 255]))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return np.empty((0, 3))
    v, u = np.nonzero(labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) if largest else labels > 0)
    depth = frame["depth"][v, u]
    valid = np.isfinite(depth) & (depth > .03) & (depth < 3.)
    v, u, depth = v[valid], u[valid], depth[valid]
    K = frame["K"]
    p = np.column_stack(((u - K[0, 2]) * depth / K[0, 0],
                         (v - K[1, 2]) * depth / K[1, 1], depth))
    T = frame["T_world_camera"]
    return p @ T[:3, :3].T + T[:3, 3]


def observe_grasp(frame):
    """No scene specification, simulator object pose or true dimensions are inputs."""
    points = color_points(frame, 3, 24)
    goal = detect_color(frame, "green")
    if len(points) < 16 or goal["status"] != "observed":
        return {"status": "not_evaluated", "reason": "object_or_goal_not_visible"}
    origin = points.mean(axis=0)
    p = points - origin
    solution, _, rank, _ = np.linalg.lstsq(
        np.column_stack((2 * p, np.ones(len(p)))), np.sum(p * p, axis=1), rcond=None)
    center = solution[:3] + origin
    radius = float(np.sqrt(max(0., solution[3] + solution[:3] @ solution[:3])))
    error = float(np.sqrt(np.mean((np.linalg.norm(points - center, axis=1) - radius) ** 2)))
    spherical = rank == 4 and .008 <= radius <= .04 and error < .0008
    low, high = np.percentile(points, [2, 98], axis=0)
    if not spherical:
        center = (low + high) / 2
    obstacles = color_points(frame, 0, 3, largest=False)
    obstacles = obstacles[obstacles[:, 1] * center[1] > 0]
    return {
        "status": "observed", "center_m": center.tolist(),
        "shape_estimate": "sphere" if spherical else "unclassified",
        "radius_m": radius if spherical else None, "fit_rmse_m": error,
        "top_m": float(center[2] + radius) if spherical else float(high[2]),
        "width_m": 2 * radius if spherical else float(max(high[0] - low[0], high[1] - low[1])),
        "table_z_m": float(goal["position_m"][2]),
        "goal_m": goal["position_m"],
        "obstacle_bounds_m": np.percentile(obstacles, [1, 99], axis=0).tolist() if len(obstacles) > 8 else None,
        "input_channels": ["head_rgb", "head_depth", "camera_calibration"],
    }


def validate_action(raw):
    if not isinstance(raw, dict) or set(raw) != {"grasp_m", "goal_m", "close_angle",
                                                "lift_m", "via_xy", "origin"}:
        raise ValueError("Malformed transfer candidate")
    for name in ("grasp_m", "goal_m"):
        p = np.asarray(raw[name], dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or not (
                .19 <= p[0] <= .41 and -.46 <= p[1] <= -.14 and .65 <= p[2] <= .75):
            raise ValueError("Candidate exceeds bounded workspace")
    if type(raw["close_angle"]) not in (int, float) or not -.60 <= raw["close_angle"] <= -.05:
        raise ValueError("Invalid jaw command")
    if type(raw["lift_m"]) not in (int, float) or not .041 <= raw["lift_m"] <= .075:
        raise ValueError("Invalid lift")
    via = np.asarray(raw["via_xy"], dtype=float).reshape(-1, 2)
    if len(via) > 2 or not np.isfinite(via).all():
        raise ValueError("Invalid route")
    if len(via) and (np.any(via[:, 0] < .19) or np.any(via[:, 0] > .41)
                     or np.any(via[:, 1] < -.46) or np.any(via[:, 1] > -.14)):
        raise ValueError("Route exceeds workspace")
    if not isinstance(raw["origin"], str) or len(raw["origin"]) > 100:
        raise ValueError("Invalid candidate provenance")
    return json.loads(json.dumps(raw, allow_nan=False))


def initial_action():
    return validate_action({"grasp_m": [.30, -.38, TABLE_Z + .043],
                            "goal_m": [.30, -.22, TABLE_Z + .043], "close_angle": -.25,
                            "lift_m": .049, "via_xy": [], "origin": "captured_block_prior_retarget"})


def propose_candidates(feedback, previous, seen, *, batch_size=4):
    """Pure, serializable boundary: evidence and model gripper calibration only."""
    observation = feedback["initial_observation"]
    if observation["status"] != "observed":
        return {"status": "not_evaluated", "reason": observation["reason"], "candidates": []}
    calibration = feedback["gripper_calibration"]
    width = observation["width_m"]
    if width > max(row["gap_m"] for row in calibration) - .001:
        return {"status": "unsupported", "reason": "observed_width_exceeds_gripper_opening", "candidates": []}
    previous_reason = feedback.get("stop_reason", "")
    candidates = []
    for compression in (.003, .007):
        sample = min(calibration, key=lambda row: abs(row["gap_m"] - (width - compression)))
        # Keep the modeled finger underside clear of the visually measured table.
        minimum_z = observation["table_z_m"] + .002 - sample["bottom_offset_m"]
        grasp_z = max(minimum_z, observation["center_m"][2] - sample["center_offset_m"])
        for dz in (0., .004, .008):
            grasp = list(observation["center_m"])
            grasp[2] = grasp_z + dz
            goal = list(observation["goal_m"])
            goal[2] += grasp[2] - observation["table_z_m"]
            routes = [[]]
            obstacle = observation["obstacle_bounds_m"]
            if obstacle and previous_reason == "forbidden_contact":
                low, high = np.asarray(obstacle)
                margin = width / 2 + .018
                routes = [[[float(high[0] + margin), grasp[1]],
                           [float(high[0] + margin), goal[1]]],
                          [[float(low[0] - margin), grasp[1]],
                           [float(low[0] - margin), goal[1]]]]
            for via in routes:
                raw = {"grasp_m": grasp, "goal_m": goal, "close_angle": sample["angle"],
                       "lift_m": previous["lift_m"], "via_xy": via,
                       "origin": "RGBD_fit_and_contact_failure_model_gap_search"}
                try:
                    action = validate_action(raw)
                except ValueError:
                    continue
                key = sha256_json({k: v for k, v in action.items() if k != "origin"})
                if key in seen:
                    continue
                candidates.append({"action": action, "action_geometry_sha256": key,
                                   "basis": {"observation": observation, "gripper_model": sample,
                                             "previous_stop_reason": previous_reason}})
                if len(candidates) == batch_size:
                    return {"status": "proposed", "candidates": candidates}
    return {"status": "proposed" if candidates else "exhausted", "candidates": candidates}


class TransferRunner(SortingRunner):
    def __init__(self, spec, prior, *, cameras=True):
        self.transfer_spec, self.prior = spec, prior
        super().__init__(spec.sorting_spec(), cameras=cameras,
                         scene_factory=lambda _: transfer_scene_xml(spec))
        self.initial_observation = None
        self.transfer_action = None

    def gripper_calibration(self):
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.data.qpos
        tcp = self.tcp_ids["r"]
        ids = [self.model.geom(f"claw::r_{s}_fingers").id for s in ("f", "b")]
        rows = []
        for angle in np.linspace(-.6, -.05, 112):
            for name, value in claw_pose(angle).items():
                data.qpos[self.model.joint(name).qposadr[0]] = value
            mujoco.mj_forward(self.model, data)
            points = data.geom_xpos[ids]
            axis = (points[0] - points[1]) / np.linalg.norm(points[0] - points[1])
            half = [np.abs(data.geom_xmat[i].reshape(3, 3).T @ axis) @ self.model.geom_size[i] for i in ids]
            bottom = min(data.geom_xpos[i, 2] - np.abs(data.geom_xmat[i].reshape(3, 3)[2]) @
                         self.model.geom_size[i] for i in ids) - data.site_xpos[tcp, 2]
            rows.append({"angle": float(angle), "gap_m": float(np.linalg.norm(points[0] - points[1]) - sum(half)),
                         "bottom_offset_m": float(bottom),
                         "center_offset_m": float(points[:, 2].mean() - data.site_xpos[tcp, 2])})
        return rows

    def _sort_one(self, part, policy):
        if part == "blue":
            self.completed.append("blue")
            return
        self.active_part = part
        self._sample_sensors()
        self.initial_observation = observe_grasp(self.latest_frames["head"])
        self.decisions.append({"phase": "initial_visual_observation", "control_index": 0, "sensor_index": 0,
                               "observation": self.initial_observation})
        if self.initial_observation["status"] != "observed":
            raise VisionUnavailable(self.initial_observation["reason"])
        a = self.transfer_action
        grasp, goal = np.array(a["grasp_m"]), np.array(a["goal_m"])
        close = a["close_angle"]
        self.move_arm("r", grasp + [0, 0, .042], -.6, 1.6, "approach")
        def move(destination, angle, duration, phase):
            self.move_arm("r", destination, angle, duration, phase,
                          path_shape=self.prior["curves"].get(phase))
        move(grasp, -.6, 2.2, "descend")
        move(grasp, close, 1.3, "close")
        self.wait(.5, "close")
        move(grasp + [0, 0, a["lift_m"]], close, 2.4, "lift")
        if not self._bilateral_feedback("r", "check_grasp"):
            raise MotionRejected("no_bilateral_grasp")
        targets = [np.r_[xy, grasp[2] + a["lift_m"]] for xy in a["via_xy"]]
        targets.append(np.r_[goal[:2], grasp[2] + a["lift_m"]])
        for target in targets:
            move(target, close, 3. / len(targets), "transport")
        if not self._bilateral_feedback("r", "check_transport"):
            raise MotionRejected("grasp_lost")
        move(goal, close, 2.4, "lower")
        move(goal, -.6, 1.3, "release")
        move(self.home, -.6, 2., "retract")
        self.wait(.8, "settle")
        observed, target = self.observe_task("orange", "green", "check_placement")
        if np.linalg.norm(observed[:2] - target[:2]) > .025:
            raise MotionRejected("visual_placement_error")
        self.completed.append("orange")

    def run_transfer(self, action, directory):
        self.transfer_action = validate_action(action)
        calibration = self.gripper_calibration()
        self.trajectory_extension = {"schema": SCHEMA, "action": action,
                                     "source_prior_sha256": sha256_json(self.prior),
                                     "retargeting": self.prior["retargeting"],
                                     "scene": asdict(self.transfer_spec)}
        receipt = self.run_sorting(default_policy(max_retries=0), directory)
        feedback = {
            "schema": "organoid.grasp-feedback.v1",
            "initial_observation": self.initial_observation,
            "gripper_calibration": calibration,
            "stop_reason": (self.stopped or {}).get("reason", ""),
            "last_contact_fraction": next((d["bilateral_contact_fraction"] for d in reversed(self.decisions)
                                           if "bilateral_contact_fraction" in d), None),
            "sensor_index": 0, "reset_semantics": "candidates_branch_from_locked_initial_state",
        }
        (directory / "feedback.json").write_text(json.dumps(feedback, indent=2, allow_nan=False))
        return receipt

    def sorting_receipt(self):
        receipt = super().sorting_receipt()
        obj = receipt["objects"]["orange"]
        if self.transfer_spec.kind == "sphere":
            pose = self.data.qpos[self.object_addresses["orange"]:self.object_addresses["orange"] + 3]
            radius = self.transfer_spec.diameter / 2
            error = pose[:2] - self.transfer_spec.goal_xy
            obj["fully_inside"] = bool(np.all(np.abs(error) + radius <= [.052, .046]))
            obj["settled"] = bool(obj["settled"] and abs(pose[2] - TABLE_Z - radius) < .005)
        checks = receipt["checks"]
        for key in ("task_sequence_complete", "both_objects_in_assigned_bins",
                    "both_objects_settled", "both_objects_lifted_40mm"):
            checks.pop(key)
        checks.update(task_complete=self.completed == ["orange", "blue"],
                      object_in_bin=obj["fully_inside"], object_settled=obj["settled"],
                      object_lifted_40mm=obj["lift_m"] >= .04,
                      bilateral_transport=obj["bilateral_transport_fraction"] is not None
                          and obj["bilateral_transport_fraction"] >= .8,
                      slip_below_20mm=obj["transport_slip_m"] is not None and obj["transport_slip_m"] <= .02)
        receipt["schema"] = SCHEMA
        receipt["objects"] = {"orange": obj}
        receipt["checks"] = {key: bool(checks[key]) for key in CHECKS}
        receipt["status"] = ("not_evaluated" if self.stopped and self.stopped["status"] == "not_evaluated"
                             or not checks["numerically_valid"] else
                             "success" if all(checks.values()) else "rejected")
        receipt["scope"] = {
            "robot": "original_biped_s200049", "base": "fixed",
            "source": "captured_action_FK_retargeted_phase_prior",
            "task": "single_object_pick_transport_place_blue_object_is_distractor",
            "planner_input": "RGBD_model_gripper_geometry_and_contact_feedback",
            "true_object_pose": "acceptance_labels_only",
            "retry": "new_candidate_from_locked_initial_state_not_continuous_physical_recovery",
        }
        return receipt
