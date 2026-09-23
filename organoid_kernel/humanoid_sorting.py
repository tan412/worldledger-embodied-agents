"""RGB-D-conditioned sorting on the original Kuavo model, with explicit feedback.

The teacher uses color/depth observations and modeled fingertip contact sensing.
Simulator object poses are reserved for labels, never passed to its planner.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import h5py
import mujoco
import numpy as np

from .humanoid_dataset import look_rotation
from .humanoid_tasks import (
    DT, FPS, SUBSTEPS, TABLE_Z, HumanoidConfig, HumanoidKinematics,
    HumanoidRunner, HumanoidStop, build_humanoid_scene, claw_pose, minimum_jerk,
)
from .hashing import sha256_file
from .physics_audit import FrameAuditor
from .trajectory import write_trajectory, read_trajectory

CAMERAS = ("head", "wrist_right", "wrist_left")
WIDTH, HEIGHT = 480, 360
COLORS = {
    "orange": [1., .40, .035, 1.], "blue": [.025, .25, .95, 1.],
    "green": [.06, .65, .22, 1.], "yellow": [.94, .86, .025, 1.],
}
HSV_BANDS = {"orange": (3, 24), "blue": (95, 130), "green": (38, 88), "yellow": (24, 37)}
PARTS = ("orange", "blue")
ASSIGNMENTS = {"orange": ("r", "green"), "blue": ("l", "yellow")}


@dataclass(frozen=True)
class SortingScene:
    seed: int = 11
    orange_xy: tuple = (.30, -.38)
    blue_xy: tuple = (.30, .38)
    green_xy: tuple = (.30, -.22)
    yellow_xy: tuple = (.30, .22)
    size: tuple = (.022, .025, .040)
    mass: float = .035
    friction: float = .65
    barrier_height: float = .025
    depth_noise_m: float = 0.
    head_blackout: bool = False

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("Invalid scene seed")
        for name in ("orange_xy", "blue_xy", "green_xy", "yellow_xy"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name}")
            if not .25 <= value[0] <= .35 or not .17 <= abs(value[1]) <= .43:
                raise ValueError(f"{name} exceeds the declared workspace")
            if (name in ("orange_xy", "green_xy")) != (value[1] < 0):
                raise ValueError("Objects must remain in their declared arm workspace")
        if not np.isfinite([self.barrier_height, self.depth_noise_m]).all():
            raise ValueError("Nonfinite scene parameters")
        if not .005 <= self.barrier_height <= .12 or not 0 <= self.depth_noise_m <= .003:
            raise ValueError("Scene uncertainty outside bounds")
        if type(self.head_blackout) is not bool:
            raise ValueError("head_blackout must be boolean")
        HumanoidConfig(size=self.size, mass=self.mass, friction=self.friction)


def make_scene(spec):
    config = HumanoidConfig(seed=spec.seed, size=spec.size, mass=spec.mass,
                            friction=spec.friction, grasp_offset_y=0.,
                            transport_clearance=.049, source_xy=spec.orange_xy,
                            goal_xy=spec.green_xy, barrier_height=spec.barrier_height)
    root = ET.fromstring(build_humanoid_scene(config))
    world = root.find("worldbody")
    table = world.find("geom[@name='object::table']")
    table.set("pos", f".36 0 {TABLE_Z - .0175}")
    table.set("size", ".24 .55 .0175")
    target = world.find("body[@name='object::target']")
    target.find("geom").set("rgba", " ".join(map(str, COLORS["orange"])))
    blue = ET.fromstring(ET.tostring(target, encoding="unicode"))
    blue.set("name", "object::blue")
    blue.set("pos", f"{spec.blue_xy[0]} {spec.blue_xy[1]} {TABLE_Z + spec.size[2] / 2}")
    blue.find("freejoint").set("name", "objfree::blue")
    blue.find("geom").set("name", "object::blue")
    blue.find("geom").set("rgba", " ".join(map(str, COLORS["blue"])))
    world.append(blue)
    marker = world.find("geom[@name='goal_marker']")
    marker.set("name", "bin_marker_green")
    marker.set("rgba", " ".join(map(str, COLORS["green"])))
    yellow = ET.fromstring(ET.tostring(marker, encoding="unicode"))
    yellow.set("name", "bin_marker_yellow")
    yellow.set("pos", f"{spec.yellow_xy[0]} {spec.yellow_xy[1]} {TABLE_Z + .0005}")
    yellow.set("rgba", " ".join(map(str, COLORS["yellow"])))
    world.append(yellow)
    for geom in list(world.findall("geom")):
        name = geom.get("name", "")
        if name.startswith("object::tray_") or name == "object::barrier":
            clone = ET.fromstring(ET.tostring(geom, encoding="unicode"))
            clone.set("name", name + "_left")
            pos = np.fromstring(geom.get("pos"), sep=" ")
            if name.startswith("object::tray_"):
                pos[:2] += np.asarray(spec.yellow_xy) - np.asarray(spec.green_xy)
            else:
                pos[1] *= -1
            clone.set("pos", " ".join(map(str, pos)))
            world.append(clone)
    wrist = root.find(".//body[@name='zarm_l7_link']")
    ET.SubElement(wrist, "site", name="grasp_tcp_left", pos="0 -.0008 -.197",
                  size=".002", rgba="0 0 0 0")
    # Virtual cameras use existing head and wrist links, not external truth poses.
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mounts = [
        ("head", "zhead_2_link", [.1074, -.0475, .1262], [.30, 0., .65], False, 64.),
        ("wrist_right", "zarm_r7_link", [0., -.055, -.1066], [0., .005, -.235], True, 78.),
        ("wrist_left", "zarm_l7_link", [0., .055, -.1066], [0., -.005, -.235], True, 78.),
    ]
    calibration = {}
    for name, parent, position, target_point, local_target, fovy in mounts:
        bid = model.body(parent).id
        rotation = data.xmat[bid].reshape(3, 3)
        center = data.xpos[bid]
        eye = center + rotation @ position
        target_point = center + rotation @ target_point if local_target else np.asarray(target_point)
        local_rotation = rotation.T @ look_rotation(eye, target_point)
        if name == "head":
            back = (eye - target_point) / np.linalg.norm(eye - target_point)
            right = np.array([0., -1., 0.])
            right -= back * (right @ back)
            right /= np.linalg.norm(right)
            local_rotation = rotation.T @ np.column_stack((right, np.cross(back, right), back))
        ET.SubElement(root.find(f".//body[@name='{parent}']"), "camera", name=name,
                      pos=" ".join(map(str, position)), fovy=str(fovy),
                      xyaxes=" ".join(map(str, np.r_[local_rotation[:, 0], local_rotation[:, 1]])))
        focal = HEIGHT / (2 * np.tan(np.radians(fovy) / 2))
        calibration[name] = {
            "parent": parent, "width": WIDTH, "height": HEIGHT,
            "K": [[focal, 0, (WIDTH - 1) / 2], [0, focal, (HEIGHT - 1) / 2], [0, 0, 1]],
            "local_position_m": position, "local_rotation_opengl": local_rotation.tolist(),
            "origin": "declared_virtual_mount_and_ideal_pinhole",
            "depth": "optical_axis_m", "camera_axes": "OpenCV_right_down_forward",
        }
    return config, ET.tostring(root, encoding="unicode"), calibration


class RGBDObserver:
    def __init__(self, model, calibration, spec):
        self.model, self.calibration, self.spec = model, calibration, spec
        self.renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        self.rng = np.random.default_rng(spec.seed + 9100)

    def capture(self, data, camera="head"):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=camera, scene_option=self.option)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        if self.spec.depth_noise_m:
            depth += self.rng.normal(0, self.spec.depth_noise_m, depth.shape).astype(np.float32)
        if camera == "head" and self.spec.head_blackout:
            rgb[:] = 0
        cid = self.model.camera(camera).id
        transform = np.eye(4)
        transform[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
        transform[:3, 3] = data.cam_xpos[cid]
        return {"rgb": rgb, "depth": depth, "T_world_camera": transform,
                "K": np.asarray(self.calibration[camera]["K"])}

    def close(self):
        self.renderer.close()


def detect_color(frame, color, *, top_surface=False):
    """Pure image/depth estimator: no model, qpos, object IDs or true poses."""
    hsv = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2HSV)
    low, high = HSV_BANDS[color]
    mask = cv2.inRange(hsv, np.array([low, 100, 70]), np.array([high, 255, 255]))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return {"status": "not_evaluated", "reason": "color_not_visible", "color": color}
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    v, u = np.nonzero(labels == component)
    depth = frame["depth"][v, u].astype(float)
    valid = np.isfinite(depth) & (depth > .02) & (depth < 3)
    u, v, depth = u[valid], v[valid], depth[valid]
    if len(u) < 10:
        return {"status": "not_evaluated", "reason": "insufficient_depth_pixels", "color": color}
    K = frame["K"]
    points = np.column_stack(((u - K[0, 2]) * depth / K[0, 0],
                              (v - K[1, 2]) * depth / K[1, 1], depth))
    transform = frame["T_world_camera"]
    points = points @ transform[:3, :3].T + transform[:3, 3]
    if top_surface:
        top = np.percentile(points[:, 2], 95)
        points = points[points[:, 2] >= top - .003]
        if len(points) < 8:
            return {"status": "not_evaluated", "reason": "top_surface_occluded", "color": color}
    center = (np.percentile(points, 5, axis=0) + np.percentile(points, 95, axis=0)) / 2
    return {"status": "observed", "color": color, "position_m": center.tolist(),
            "pixels": int(len(u)), "bbox_uvwh": stats[component, :4].tolist(),
            "origin": "RGB_color_components_and_depth_unprojection"}


class SortingRunner(HumanoidRunner):
    def __init__(self, spec: SortingScene, *, cameras=True, scene_factory=make_scene):
        self.sorting_scene = spec
        config, xml, self.calibration = scene_factory(spec)
        super().__init__(config, scene_xml=xml)
        self.iks = {"r": self.ik, "l": HumanoidKinematics(self.model, "l")}
        self.arm_columns = {s: np.array([self.joint_names.index(f"zarm_{s}{i}_joint")
                                        for i in range(1, 8)]) for s in ("r", "l")}
        self.tcp_ids = {"r": self.model.site("grasp_tcp").id,
                        "l": self.model.site("grasp_tcp_left").id}
        self.claw_ids = {s: self.model.actuator(f"act::{s}_f_bar-1_joint").id for s in ("r", "l")}
        self.angles = {"r": -.60, "l": -.60}
        self.positions = {"r": self.home.copy(), "l": np.array([.30, .40, .735])}
        left = self.iks["l"].solve(self.positions["l"], np.eye(3), attempts=3)
        if left[2] > .002 or left[3] > .03:
            raise ValueError(f"Left home unreachable: {left[2:]}")
        self.reference[self.arm_columns["l"]] = left[1]
        self.data.qpos[self.body_qadr] = self.reference
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(500):
            self._both_servo(self.reference, self.angles)
        mujoco.mj_getState(self.model, self.data, self.initial_state, self.state_spec)
        self.initial_warnings = np.array([w.number for w in self.data.warning])
        self.object_addresses = {
            "orange": int(self.model.joint("objfree::target").qposadr[0]),
            "blue": int(self.model.joint("objfree::blue").qposadr[0]),
        }
        self.object_dofs = {"orange": int(self.model.joint("objfree::target").dofadr[0]),
                            "blue": int(self.model.joint("objfree::blue").dofadr[0])}
        self.initial_objects = {name: self.data.qpos[adr:adr + 7].copy()
                                for name, adr in self.object_addresses.items()}
        self.observer = RGBDObserver(self.model, self.calibration, spec) if cameras else None
        self.controls, self.references, self.phases = [], [], []
        self.positions_history = [self.data.qpos.copy()]
        self.velocities_history = [self.data.qvel.copy()]
        self.tcp_history = [np.array([self.data.site_xpos[self.tcp_ids[s]] for s in ("l", "r")])]
        self.forces_history, self.contacts_history, self.active_history = [], [], []
        self.sensor_indices, self.decisions = [], []
        self.max_speed[:] = 0
        self.max_force[:] = 0
        self.max_joint_limit_excess = 0.
        self.max_penetration = self.max_self_penetration = 0.
        self.forbidden_contacts = {}
        self.external_force_seen = False
        self.contact_window = []
        self.stopped = None
        self.completed = []
        self.active_part = None
        self.auditors = {name: FrameAuditor(self.model, self.data, joint)
                         for name, joint in (("orange", "objfree::target"), ("blue", "objfree::blue"))}
        self.carry = {p: [] for p in PARTS}
        self.max_lift_by_part = {p: 0. for p in PARTS}
        self.sensor_file = None
        self.active_policy = None
        self.vision_status = "not_evaluated"
        self.capture_stride = 50
        self.started = False

    def _both_servo(self, reference, angles):
        kp = self.model.actuator_gainprm[self.body_aids, 0]
        self.data.ctrl[self.body_aids] = reference + self.data.qfrc_bias[self.body_dadr] / kp
        for side in ("r", "l"):
            self.data.ctrl[self.claw_ids[side]] = angles[side]
        control = self.data.ctrl.copy()
        mujoco.mj_step(self.model, self.data)
        return control

    def close(self):
        if self.observer:
            self.observer.close()
        if self.sensor_file:
            self.sensor_file.close()
            self.sensor_file = None

    def _sample_sensors(self):
        if self.observer is None:
            return
        index = len(self.controls)
        if self.sensor_indices and self.sensor_indices[-1] == index:
            return
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        frames = {name: self.observer.capture(self.data, name) for name in CAMERAS}
        self.sensor_indices.append(index)
        if self.sensor_file:
            for name, frame in frames.items():
                group = self.sensor_file.require_group(name)
                for key in ("rgb", "depth", "T_world_camera"):
                    value = frame[key]
                    if key not in group:
                        group.create_dataset(key, shape=(0, *value.shape),
                                             maxshape=(None, *value.shape), dtype=value.dtype,
                                             chunks=(1, *value.shape), compression="gzip", compression_opts=2)
                    dataset = group[key]
                    dataset.resize(len(self.sensor_indices), axis=0)
                    dataset[-1] = value
                group.attrs["K"] = frame["K"]
            self.sensor_file.flush()
        self.latest_frames = frames

    def observe_task(self, part, goal, phase):
        self._sample_sensors()
        frame = self.latest_frames["head"]
        observations = {
            "part": detect_color(frame, part, top_surface=True),
            "goal": detect_color(frame, goal),
        }
        self.decisions.append({
            "control_index": len(self.controls), "sensor_index": len(self.sensor_indices) - 1,
            "phase": phase, "part": part, "observation": observations,
            "input_channels": ["head_rgb", "head_depth", "camera_calibration"],
        })
        if any(o["status"] != "observed" for o in observations.values()):
            self.vision_status = "not_evaluated"
            raise VisionUnavailable("Part or destination is not observable")
        self.vision_status = "observed"
        return (np.asarray(observations["part"]["position_m"]),
                np.asarray(observations["goal"]["position_m"]))

    def tick(self, reference, angles, phase, saved_control=None):
        reference = np.asarray(reference, dtype=float)
        bounds = self.model.jnt_range[self.body_jids]
        if (reference.shape != self.reference.shape or not np.isfinite(reference).all()
                or np.any(reference < bounds[:, 0] - .01) or np.any(reference > bounds[:, 1] + .01)
                or any(not np.isfinite(a) or not -.698 <= a <= .698 for a in angles.values())):
            raise ValueError("Invalid original-robot command")
        self.external_force_seen |= bool(np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied))
        if saved_control is None:
            control = self._both_servo(reference, angles)
        else:
            control = np.asarray(saved_control)
            if control.shape != (self.model.nu,) or not np.isfinite(control).all():
                raise ValueError("Invalid saved control")
            self.data.ctrl[:] = control
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_kinematics(self.model, self.data)
        self.controls.append(control.copy())
        self.references.append(np.r_[reference, angles["l"], angles["r"]])
        self.phases.append(phase)
        self.active_history.append(PARTS.index(self.active_part) if self.active_part else -1)
        self.positions_history.append(self.data.qpos.copy())
        self.velocities_history.append(self.data.qvel.copy())
        self.forces_history.append(self.data.actuator_force.copy())
        self.tcp_history.append(np.array([self.data.site_xpos[self.tcp_ids[s]] for s in ("l", "r")]))
        self.max_speed = np.maximum(self.max_speed, np.abs(self.data.qvel[self.body_dadr]))
        self.max_force = np.maximum(self.max_force, np.abs(self.data.actuator_force))
        excess = np.maximum(bounds[:, 0] - self.data.qpos[self.body_qadr],
                            self.data.qpos[self.body_qadr] - bounds[:, 1])
        self.max_joint_limit_excess = max(self.max_joint_limit_excess, float(excess.max()))
        tactile = np.zeros((2, 2))
        for ci, contact in enumerate(self.data.contact[:self.data.ncon]):
            names = [self.geom_names[int(g)] for g in (contact.geom1, contact.geom2)]
            mujoco.mj_contactForce(self.model, self.data, ci, self.buffer)
            force = max(0., float(self.buffer[0]))
            if force < .005:
                continue
            depth = max(0., -float(contact.dist))
            robot = [n.startswith(("claw::", "collision::")) for n in names]
            physical_object = any(n in ("object::target", "object::blue") for n in names)
            if physical_object or (any(robot) and any(n.startswith("object::") for n in names)):
                self.max_penetration = max(self.max_penetration, depth)
            if all(robot) and not all(n.startswith("claw::") for n in names):
                self.max_self_penetration = max(self.max_self_penetration, depth)
            for si, side in enumerate(("l", "r")):
                for fi, finger in enumerate(("f", "b")):
                    if any(n.startswith(f"claw::{side}_{finger}") for n in names) and physical_object:
                        tactile[si, fi] += force
            obstacle = any(n.startswith("object::barrier") for n in names)
            table = "object::table" in names
            forearm = physical_object and any(n.startswith("collision::") for n in names)
            if (obstacle and (any(robot) or physical_object)) or (table and any(robot)) or forearm:
                pair = " | ".join(sorted(names))
                self.forbidden_contacts[pair] = max(force, self.forbidden_contacts.get(pair, 0.))
        self.contacts_history.append(tactile)
        self.contact_window.append(tactile)
        self.contact_window = self.contact_window[-100:]
        for part, adr in self.object_addresses.items():
            self.max_lift_by_part[part] = max(
                self.max_lift_by_part[part],
                float(self.data.qpos[adr + 2] - self.initial_objects[part][2]))
        if self.active_part and phase == "transport":
            side = ASSIGNMENTS[self.active_part][0]
            pads = [self.model.geom(f"claw::{side}_{finger}_fingers").id for finger in ("f", "b")]
            center = np.mean(self.data.geom_xpos[pads], axis=0)
            adr = self.object_addresses[self.active_part]
            bilateral = bool(np.all(tactile[0 if side == "l" else 1] > .02))
            self.carry[self.active_part].append((bilateral, self.data.qpos[adr:adr + 3] - center))
        if len(self.controls) % self.capture_stride == 0:
            for auditor in self.auditors.values():
                auditor.frame()
            self._sample_sensors()

    def move_arm(self, side, destination, angle, duration, phase, *, path_shape=None):
        destination = np.asarray(destination, dtype=float)
        start = self.positions[side].copy()
        initial_angle = self.angles[side]
        count = max(1, round(duration * FPS))
        for frame in range(count):
            alpha = minimum_jerk((frame + 1) / count)
            residual = np.zeros(3)
            if path_shape is not None:
                grid = np.linspace(0, 1, len(path_shape["progress"]))
                alpha = np.interp((frame + 1) / count, grid, path_shape["progress"])
                residual = .001 * np.array([np.interp((frame + 1) / count, grid,
                    np.asarray(path_shape["residual"])[:, axis]) for axis in range(3)])
            target = start + (destination - start) * alpha
            target += residual
            columns = self.arm_columns[side]
            result = self.iks[side].solve(target, np.eye(3), initial=self.reference[columns])
            self.ik_errors.append((result[2], result[3]))
            if result[2] > .002 or result[3] > .03:
                raise MotionRejected(f"ik_tolerance_exceeded:{side}:{phase}")
            target_ref = self.reference.copy()
            target_ref[columns] = result[1]
            target_angle = initial_angle + (angle - initial_angle) * alpha
            old_angles = dict(self.angles)
            for substep in range(SUBSTEPS):
                beta = (substep + 1) / SUBSTEPS
                angles = dict(old_angles)
                angles[side] += (target_angle - old_angles[side]) * beta
                self.tick(self.reference + (target_ref - self.reference) * beta, angles, phase)
                if self.forbidden_contacts or self.max_self_penetration > .003:
                    raise MotionRejected("forbidden_contact")
                if self.external_force_seen or not np.isfinite(self.data.qpos).all():
                    raise MotionRejected("invalid_dynamics")
            self.reference = target_ref
            self.angles[side] = target_angle
            self.positions[side] = target

    def wait(self, duration, phase):
        for _ in range(round(duration / DT)):
            self.tick(self.reference, self.angles, phase)
            if self.forbidden_contacts or self.max_self_penetration > .003:
                raise MotionRejected("forbidden_contact")

    def _bilateral_feedback(self, side, phase):
        window = np.asarray(self.contact_window)
        value = float(np.mean(np.all(window[:, 0 if side == "l" else 1] > .02, axis=1)))
        self.decisions.append({
            "control_index": len(self.controls), "phase": phase, "side": side,
            "bilateral_contact_fraction": value, "input_channels": ["simulated_fingertip_normal_force"],
        })
        return value >= .6

    def _sort_one(self, part, policy):
        side, goal_color = ASSIGNMENTS[part]
        self.active_part = part
        for attempt in range(policy["max_retries"] + 1):
            source, goal = self.observe_task(part, goal_color, "locate" if not attempt else "reobserve")
            if policy["controller"] == "nominal_xy_ablation":
                source[:2] = [.30, -.38 if side == "r" else .38]
                goal[:2] = [.30, -.22 if side == "r" else .22]
            if attempt == 0:
                source[1] += policy["initial_grasp_bias_y"]
            source[2] += .003
            goal[2] += self.sorting_scene.size[2] + .003
            hover = source + [0, 0, .042]
            self.move_arm(side, hover, -.60, 1.6, "approach")
            self.move_arm(side, source, -.60, 2.2, "descend")
            self.move_arm(side, source, -.25, 1.3, "close")
            self.wait(.5, "close")
            lift = source + [0, 0, policy["clearance_m"]]
            self.move_arm(side, lift, -.25, 2.4, "lift")
            if not self._bilateral_feedback(side, "check_grasp"):
                if attempt < policy["max_retries"] and policy["controller"] == "rgbd_feedback":
                    self.decisions.append({"control_index": len(self.controls), "phase": "retry_grasp",
                                           "part": part, "attempt": attempt + 1})
                    self.move_arm(side, source, -.25, 2., "recovery_lower")
                    self.move_arm(side, source, -.60, 1., "recovery_release")
                    self.move_arm(side, self.home if side == "r" else [.30, .40, .735],
                                  -.60, 1.8, "recovery_retract")
                    self.wait(.4, "observe")
                    continue
                raise MotionRejected("no_bilateral_grasp")
            transit = goal.copy()
            transit[2] = lift[2]
            self.move_arm(side, transit, -.25, 3., "transport")
            if not self._bilateral_feedback(side, "check_transport"):
                raise MotionRejected("grasp_lost")
            self.move_arm(side, goal, -.25, 2.4, "lower")
            self.move_arm(side, goal, -.60, 1.3, "release")
            self.move_arm(side, self.home if side == "r" else [.30, .40, .735],
                          -.60, 2., "retract")
            self.wait(.8, "settle")
            observed, target = self.observe_task(part, goal_color, "check_placement")
            error = float(np.linalg.norm(observed[:2] - target[:2]))
            self.decisions[-1]["observed_placement_error_m"] = error
            if error > .025:
                raise MotionRejected("visual_placement_error")
            self.completed.append(part)
            return

    def run_sorting(self, policy, directory=None):
        policy = validate_policy(policy)
        if self.started:
            raise ValueError("Each runner accepts exactly one candidate")
        if self.observer is None:
            raise ValueError("The sorting teacher requires camera observations")
        self.started, self.active_policy = True, policy
        if directory is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=False)
            self.sensor_file = h5py.File(directory / "sensors.h5", "w")
            self.sensor_file.attrs["origin"] = "simulated_RGB_D_attached_robot_cameras"
            self.sensor_file.attrs["calibration"] = json.dumps(self.calibration)
        try:
            self._sample_sensors()
            for part in policy["order"]:
                self._sort_one(part, policy)
        except VisionUnavailable as exc:
            self.stopped = {"status": "not_evaluated", "reason": str(exc)}
        except MotionRejected as exc:
            self.stopped = {"status": "rejected", "reason": str(exc)}
        except Exception as exc:
            self.stopped = {"status": "not_evaluated", "reason": f"{type(exc).__name__}: {exc}"}
        self._sample_sensors()
        if self.sensor_file:
            self.sensor_file.create_dataset("control_index", data=self.sensor_indices)
            self.sensor_file.create_dataset("timestamp_s", data=np.asarray(self.sensor_indices) * DT)
            self.sensor_file.close()
            self.sensor_file = None
        receipt = self.sorting_receipt()
        if directory is not None:
            export_sorting(self, directory, receipt)
        return receipt

    def sorting_receipt(self):
        objects = {}
        for part, adr in self.object_addresses.items():
            goal_color = ASSIGNMENTS[part][1]
            goal = np.asarray(getattr(self.sorting_scene, goal_color + "_xy"))
            pose = self.data.qpos[adr:adr + 7]
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, pose[3:])
            extent = np.abs(rotation.reshape(3, 3)) @ (np.asarray(self.sorting_scene.size) / 2)
            error = pose[:2] - goal
            dof = self.object_dofs[part]
            tail = np.asarray(self.velocities_history[-251:])[:, dof:dof + 6]
            carry = self.carry[part]
            relative = np.asarray([x[1] for x in carry])
            objects[part] = {
                "fully_inside": bool(np.all(np.abs(error) + extent[:2] <= [.052, .046])),
                "xy_error_m": float(np.linalg.norm(error)),
                "settled": bool(np.max(np.linalg.norm(tail[:, :3], axis=1)) < .015
                                and np.max(np.linalg.norm(tail[:, 3:], axis=1)) < .3
                                and abs(pose[2] - TABLE_Z - extent[2]) < .012),
                "lift_m": self.max_lift_by_part[part],
                "bilateral_transport_fraction": float(np.mean([x[0] for x in carry])) if carry else None,
                "transport_slip_m": float(np.max(np.linalg.norm(relative - relative[0], axis=1))) if carry else None,
                "contact_audit": self.auditors[part].verdict(),
            }
        speed_limits = np.array([{j.name: j for j in self.urdf.movable}[n].velocity for n in self.joint_names])
        warnings = np.array([w.number for w in self.data.warning]) - self.initial_warnings
        checks = {
            "task_sequence_complete": self.completed == (self.active_policy or {}).get("order"),
            "both_objects_in_assigned_bins": all(v["fully_inside"] for v in objects.values()),
            "both_objects_settled": all(v["settled"] for v in objects.values()),
            "both_objects_lifted_40mm": all(v["lift_m"] >= .04 for v in objects.values()),
            "bilateral_transport": all(v["bilateral_transport_fraction"] is not None
                                       and v["bilateral_transport_fraction"] >= .8 for v in objects.values()),
            "slip_below_20mm": all(v["transport_slip_m"] is not None
                                  and v["transport_slip_m"] <= .020 for v in objects.values()),
            "no_forbidden_contact": not self.forbidden_contacts,
            "penetration_below_3mm": self.max_penetration <= .003,
            "self_penetration_below_3mm": self.max_self_penetration <= .003,
            "joint_limits": self.max_joint_limit_excess <= .01,
            "joint_speed_limits": bool(np.all(self.max_speed <= speed_limits)),
            "effort_limits": bool(np.all(self.max_force <= self.model.actuator_forcerange[:, 1] + 1e-6)),
            "numerically_valid": bool(all(np.isfinite(x).all() for x in self.positions_history)
                                      and all(np.isfinite(x).all() for x in self.velocities_history)
                                      and not warnings.any()),
            "no_external_forces": not self.external_force_seen,
            "geometry_fidelity": self.fidelity["ok"],
            "contact_conservation": all(v["contact_audit"]["ok"] for v in objects.values()),
        }
        checks = {k: bool(v) for k, v in checks.items()}
        status = ("not_evaluated" if not checks["numerically_valid"] or (
                      self.stopped and self.stopped["status"] == "not_evaluated")
                  else "success" if all(checks.values()) else "rejected")
        return {"schema": "organoid.kuavo-visual-sorting.v1", "status": status,
                "checks": checks, "stop": self.stopped, "objects": objects,
                "metrics": {"control_steps": len(self.controls), "sensor_frames": len(self.sensor_indices),
                            "forbidden_contacts": self.forbidden_contacts,
                            "max_penetration_m": self.max_penetration,
                            "max_self_penetration_m": self.max_self_penetration,
                            "max_joint_limit_excess_rad": self.max_joint_limit_excess,
                            "max_joint_speed_rad_s": self.max_speed.tolist(),
                            "max_actuator_force": self.max_force.tolist(),
                            "solver_warning_delta": warnings.tolist(),
                            "max_ik_position_error_m": max((e[0] for e in self.ik_errors), default=None),
                            "max_ik_orientation_error_rad": max((e[1] for e in self.ik_errors), default=None)},
                "scene": asdict(self.sorting_scene), "policy": self.active_policy,
                "scope": {"robot": "original_Kuavo_biped_s200049", "base": "fixed",
                          "bimanual": "sequential_partitioned_workspace_not_cooperative_handover",
                          "vision": "color_components_plus_RGBD_geometry_not_learned_VLA",
                          "instruction": "structured_color_order_not_freeform_language_understanding",
                          "teacher": "procedural_observation_conditioned_feedback",
                          "true_object_pose": "privileged_labels_only",
                          "force_observation": "modeled_fingertip_contact_not_calibrated_force_sensor",
                          "contact_audit_hz": 10, "sensor_noise": asdict(self.sorting_scene)["depth_noise_m"]}}


class VisionUnavailable(RuntimeError):
    pass


class MotionRejected(RuntimeError):
    pass


def validate_policy(raw):
    if not isinstance(raw, dict) or set(raw) != {
            "controller", "order", "clearance_m", "max_retries", "initial_grasp_bias_y"}:
        raise ValueError("Malformed sorting policy")
    if raw["controller"] not in ("rgbd_feedback", "nominal_xy_ablation"):
        raise ValueError("Unsupported sorting controller")
    if not isinstance(raw["order"], list) or sorted(raw["order"]) != sorted(PARTS):
        raise ValueError("Each colored part must appear exactly once")
    if type(raw["max_retries"]) is not int or not 0 <= raw["max_retries"] <= 2:
        raise ValueError("Retry budget is 0..2")
    for key, low, high in (("clearance_m", .041, .055), ("initial_grasp_bias_y", -.05, .05)):
        value = raw[key]
        if type(value) not in (int, float) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid {key}")
    return json.loads(json.dumps(raw))


def default_policy(**changes):
    return validate_policy({"controller": "rgbd_feedback", "order": list(PARTS),
                            "clearance_m": .049, "max_retries": 1,
                            "initial_grasp_bias_y": 0., **changes})


def export_sorting(runner, directory, receipt):
    directory = Path(directory)
    # Keep a relocatable copy of the actual meshes next to this episode.
    import shutil
    root = ET.fromstring(runner.xml)
    assets = directory / "robot_assets"
    assets.mkdir()
    hashes = {}
    for mesh in root.findall("./asset/mesh"):
        source = Path(mesh.get("file"))
        target = assets / source.name
        if target.exists() and sha256_file(target) != sha256_file(source):
            raise ValueError("Conflicting original mesh names")
        if not target.exists():
            shutil.copy2(source, target)
        mesh.set("file", "robot_assets/" + source.name)
        hashes[str(target.relative_to(directory))] = sha256_file(target)
    (directory / "scene.xml").write_text(ET.tostring(root, encoding="unicode"))
    n = len(runner.controls)
    arrays = {
        "state_time_s": np.arange(n + 1) * DT, "action_time_s": np.arange(n) * DT,
        "qpos": np.asarray(runner.positions_history), "qvel": np.asarray(runner.velocities_history),
        "joint_position": np.asarray(runner.positions_history)[:, runner.body_qadr],
        "joint_velocity": np.asarray(runner.velocities_history)[:, runner.body_dadr],
        "object_pose": np.stack([np.asarray(runner.positions_history)[:, adr:adr + 7]
                                 for adr in runner.object_addresses.values()], axis=1),
        "tcp_position": np.asarray(runner.tcp_history),
        "control": np.asarray(runner.controls).reshape(-1, runner.model.nu),
        "joint_reference": np.asarray(runner.references).reshape(-1, len(runner.joint_names) + 2),
        "actuator_force": np.asarray(runner.forces_history).reshape(-1, runner.model.nu),
        "finger_normal_force": np.asarray(runner.contacts_history).reshape(-1, 2, 2),
        "active_part": np.asarray(runner.active_history, dtype=np.int32),
        "phase": np.asarray(runner.phases, dtype="U32"),
        "initial_state": runner.initial_state,
        "sensor_control_index": np.asarray(runner.sensor_indices, dtype=np.int64),
        "sensor_time_s": np.asarray(runner.sensor_indices) * DT,
    }
    streams = {}
    for key in arrays:
        clock = ("state_time_s" if key in ("qpos", "qvel", "tcp_position", "state_time_s",
                                           "joint_position", "joint_velocity", "object_pose")
                 else "sensor_time_s" if key.startswith("sensor_")
                 else None if key == "initial_state" else "action_time_s")
        streams[key] = {"unit": {"qpos": "model_native_rad_m", "qvel": "model_native_rad_s_m_s",
                                "tcp_position": "m", "finger_normal_force": "N",
                                "joint_reference": "rad", "phase": "phase_name",
                                "joint_position": "rad", "joint_velocity": "rad/s",
                                "object_pose": "xyz_m_quaternion_wxyz",
                                "active_part": "orange_0_blue_1_none_minus1"}.get(key, "see_metadata"),
                        "clock": clock, "origin": "simulated" if key not in (
                            "joint_reference", "phase", "active_part") else "teacher_reference"}
    for key in ("joint_position", "joint_velocity"):
        streams[key]["names"] = runner.joint_names
    state_names = [runner.model.joint(i).name for i in range(runner.model.njnt)]
    manifest = write_trajectory(directory, arrays, streams, {
        "robot_profile": runner.profile.name, "synthetic": True, "dt_s": DT,
        "state_spec": int(runner.state_spec), "scene": asdict(runner.sorting_scene),
        "joint_names": state_names, "reference_names": runner.joint_names + ["left_claw", "right_claw"],
        "actuator_names": [runner.model.actuator(i).name for i in range(runner.model.nu)],
        "sensor_file": "sensors.h5", "sensor_sha256": sha256_file(directory / "sensors.h5"),
        "scene_sha256": sha256_file(directory / "scene.xml"), "robot_assets_sha256": hashes,
        "task": {"order": runner.active_policy["order"], "assignments": ASSIGNMENTS},
        "action_semantics": "500Hz_force_limited_native_position_servo_control",
        "observations": "causal_RGBD_10Hz_plus_proprioception_and_fingertip_force",
        "privileged": ["full_qpos_contains_object_poses", "full_qvel", "active_part", "acceptance_labels"],
        "policy": runner.active_policy, "calibration": runner.calibration,
        "extension": getattr(runner, "trajectory_extension", None),
    })
    (directory / "decisions.json").write_text(json.dumps(runner.decisions, indent=2, allow_nan=False))
    receipt["trajectory_sha256"] = manifest["sha256"]
    receipt["replay"] = replay_sorting(directory)
    if not receipt["replay"]["verified"]:
        receipt["status"] = "not_evaluated"
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False))


def replay_sorting(directory):
    directory = Path(directory)
    manifest, arrays = read_trajectory(directory)
    metadata = manifest["metadata"]
    for relative, expected in {**metadata["robot_assets_sha256"],
                               "scene.xml": metadata["scene_sha256"],
                               "sensors.h5": metadata["sensor_sha256"]}.items():
        if sha256_file(directory / relative) != expected:
            raise ValueError(f"Changed episode artifact: {relative}")
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, arrays["initial_state"], metadata["state_spec"])
    mujoco.mj_forward(model, data)
    error = float(max(np.max(np.abs(data.qpos - arrays["qpos"][0])),
                      np.max(np.abs(data.qvel - arrays["qvel"][0]))))
    force_error = 0.
    warnings = np.array([w.number for w in data.warning])
    for i, control in enumerate(arrays["control"]):
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        error = max(error, float(np.max(np.abs(data.qpos - arrays["qpos"][i + 1]))),
                    float(np.max(np.abs(data.qvel - arrays["qvel"][i + 1]))))
        force_error = max(force_error, float(np.max(np.abs(data.actuator_force - arrays["actuator_force"][i]))))
    delta = np.array([w.number for w in data.warning]) - warnings
    return {"verified": max(error, force_error) <= 1e-10 and not bool(delta.any()),
            "max_state_error": error, "max_actuator_force_error": force_error,
            "solver_warning_delta": delta.tolist(), "control_steps": len(arrays["control"]),
            "mode": "saved_scene_initial_state_and_controls_no_teacher_no_vision"}
