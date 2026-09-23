"""Trajectory-backed observations and schema for the original humanoid dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .hashing import sha256_array, sha256_file
from .humanoid_tasks import DT, FPS, SUBSTEPS


WIDTH, HEIGHT = 848, 480
CAMERAS = ("camera_top", "camera_wrist_right", "camera_wrist_left")
LEG_NAMES = [f"{side}_{part}" for side in ("l", "r") for part in
             ("leg_roll", "leg_yaw", "leg_pitch", "knee", "foot_pitch", "foot_roll")]
ARM_NAMES = [f"zarm_{side}{i}_link" for side in ("l", "r") for i in range(1, 8)]
HEAD_NAMES = ["head_yaw", "head_pitch"]
GRIPPER_NAMES = ["left_claw", "right_claw"]
STATE_NAMES = LEG_NAMES + ARM_NAMES[:7] + ["left_claw"] + ARM_NAMES[7:] + ["right_claw"] + HEAD_NAMES
GROUPS = {"leg": LEG_NAMES, "arm": ARM_NAMES, "head": HEAD_NAMES, "effector": GRIPPER_NAMES}
JOINT_MAP = {
    **{name: f"leg_{side}{i + 1}_joint" for side, names in
       (("l", LEG_NAMES[:6]), ("r", LEG_NAMES[6:])) for i, name in enumerate(names)},
    **{name: name.replace("_link", "_joint") for name in ARM_NAMES},
    "head_yaw": "zhead_1_joint", "head_pitch": "zhead_2_joint",
    "left_claw": "l_f_bar-1_joint", "right_claw": "r_f_bar-1_joint",
}
TASK_TEXT = {
    "bin_placement": "用右夹爪夹起橙色长方体零件，越过红色挡板，放入绿色目标格。",
    "push_routing": "用闭合的右夹爪从红色挡板侧面推动橙色零件，将其送入绿色目标区。",
}
PHASES = ["approach", "descend", "close", "lift", "transport", "lower", "release",
          "retract", "settle", "safety_stop", "reposition", "contact", "push", "observe"]
PHASE_TEXT = {
    "approach": "进近零件", "descend": "下降至操作高度", "close": "闭合夹爪",
    "lift": "抬升零件", "transport": "搬运越障", "lower": "下降至目标格",
    "release": "松开夹爪", "retract": "撤回夹爪", "settle": "等待稳定",
    "safety_stop": "候选失败后停止轨迹并保持", "reposition": "重新定位夹爪",
    "contact": "建立推动接触", "push": "沿目标方向推送", "observe": "等待并更新物体位置",
}


def xyzw(rotation):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation).reshape(9))
    return np.roll(quat, -1)


def look_rotation(position, target):
    forward = np.asarray(target) - position
    forward /= np.linalg.norm(forward)
    up = np.array([0., 0., 1.])
    if abs(forward @ up) > .97:
        up = np.array([0., 1., 0.])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(right, forward), -forward))


def observation_camera_scene(scene_path, initial_qpos, initial_qvel):
    """Add only virtual cameras; their extrinsics are declared assumptions."""
    scene_path = Path(scene_path).resolve()
    root = ET.parse(scene_path).getroot()
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", str((scene_path.parent / mesh.get("file")).resolve()))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    mujoco.mj_forward(model, data)
    specs = [
        ("camera_top", "zhead_2_link", [.1074, -.0475, .1262], [.30, -.30, .67], False, 60.),
        ("camera_wrist_right", "zarm_r7_link", [0., -.055, -.1066], [0., .005, -.235], True, 78.),
        ("camera_wrist_left", "zarm_l7_link", [0., .055, -.1066], [0., -.005, -.235], True, 78.),
    ]
    calibration = {}
    for name, parent, position, target, local_target, fovy in specs:
        bid = model.body(parent).id
        parent_rotation = data.xmat[bid].reshape(3, 3)
        parent_position = data.xpos[bid]
        position = np.asarray(position)
        world_position = parent_position + parent_rotation @ position
        world_target = (parent_position + parent_rotation @ target if local_target else np.asarray(target))
        local_rotation = parent_rotation.T @ look_rotation(world_position, world_target)
        body = root.find(f".//body[@name='{parent}']")
        ET.SubElement(body, "camera", name=name, pos=" ".join(map(str, position)),
                      xyaxes=" ".join(map(str, np.r_[local_rotation[:, 0], local_rotation[:, 1]])),
                      fovy=str(fovy), mode="fixed")
        focal = HEIGHT / (2 * np.tan(np.radians(fovy) / 2))
        calibration[name] = {
            "parent_body": parent, "local_position_m": position.tolist(),
            "local_opengl_rotation": local_rotation.tolist(),
            "width": WIDTH, "height": HEIGHT, "fovy_degrees": fovy,
            "intrinsic_matrix": [[focal, 0, (WIDTH - 1) / 2],
                                 [0, focal, (HEIGHT - 1) / 2], [0, 0, 1]],
            "distortion_coefficients": [0, 0, 0, 0, 0],
            "calibration_origin": "assumed virtual mount near original head/wrist camera, not hardware calibration",
            "extrinsics_convention": "T_world_camera; OpenCV optical axes x right, y down, z forward",
            "noise": "none; ideal pinhole renderer",
        }
    return ET.tostring(root, encoding="unicode"), calibration


def features():
    result = {}

    def vector(key, names, unit, frame="", origin="simulated"):
        result[key] = {"dtype": "float32", "shape": [len(names)], "names": names,
                       "unit": unit, "frame": frame, "origin": origin}

    for group, names in GROUPS.items():
        vector(f"observation.state.{group}.position", names, "rad")
        vector(f"action.{group}.position", names, "rad", origin="procedural_reference")
        if group != "effector":
            vector(f"observation.state.{group}.velocity", names, "rad/s")
            vector(f"observation.state.{group}.effort", names, "N*m",
                   origin="simulated_actuator_torque_not_motor_current")
    vector("observation.state", STATE_NAMES, "rad")
    vector("action", STATE_NAMES, "rad", origin="next_40ms_endpoint_joint_reference")
    vector("next.observation.state", STATE_NAMES, "rad")
    vector("observation.state.end.position", [f"{s}_{v}" for s in ("left", "right") for v in "xyz"],
           "m", "world; original wrist-link origins", "forward_kinematics")
    vector("observation.state.end.orientation",
           [f"{s}_{v}" for s in ("left", "right") for v in ("x", "y", "z", "w")],
           "unit quaternion xyzw", "wrist-to-world", "forward_kinematics")
    for side in ("left", "right"):
        vector(f"observation.ee_pose_{side}", ["x", "y", "z", "qx", "qy", "qz", "qw"],
               "m + unit quaternion", "world; wrist-link origin", "forward_kinematics")
        vector(f"observation.tcp_pose_{side}", ["x", "y", "z", "qx", "qy", "qz", "qw"],
               "m + unit quaternion", "world; actual finger-pad midpoint", "forward_kinematics")
    vector("observation.gripper_width", ["left", "right"], "m", origin="mesh_pad_geometry")
    for name in ("observation.environment.object_pose", "next.environment.object_pose"):
        vector(name, ["x", "y", "z", "qx", "qy", "qz", "qw"], "m + unit quaternion", "world")
    vector("observation.environment.object_velocity", ["vx", "vy", "vz", "wx", "wy", "wz"],
           "m/s + rad/s", "world")
    vector("simulation.object_contact_wrench", ["fx", "fy", "fz", "tx", "ty", "tz"],
           "N + N*m", "world; torque about object COM", "simulated_contacts_not_wrist_sensor")
    vector("simulation.right_finger_normal_force", ["front", "back"], "N", origin="simulated_contacts")
    vector("imu.quat_xyzw", ["quat_x", "quat_y", "quat_z", "quat_w"],
           "unit quaternion", "base-to-world", "ideal_fixed_base_model_not_hardware_IMU")
    for key, unit in (("imu.acc_xyz", "m/s^2"), ("imu.free_acc_xyz", "m/s^2"),
                      ("imu.gyro_xyz", "rad/s")):
        vector(key, ["x", "y", "z"], unit, "base",
               "ideal_fixed_base_model_not_hardware_IMU")
    for name in CAMERAS:
        result[f"observation.images.{name}"] = {
            "dtype": "video", "shape": [3, HEIGHT, WIDTH], "names": ["channels", "height", "width"],
            "info": {"video.height": HEIGHT, "video.width": WIDTH, "video.codec": "h264",
                     "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                     "video.fps": FPS, "video.channels": 3, "has_audio": False},
            "origin": "robot_attached_virtual_camera_not_hardware_video",
        }
        vector(f"observation.camera_params.rotation_matrix_flat.{name}",
               [f"r{i}{j}" for i in range(3) for j in range(3)], "dimensionless",
               "OpenCV camera-to-world", "virtual_camera_calibration")
        vector(f"observation.camera_params.translation_vector.{name}", ["x", "y", "z"],
               "m", "world", "virtual_camera_calibration")
    for name in ("frame_index", "episode_index", "index", "task_index", "task.phase_index",
                 "simulation.control_index_start", "simulation.control_index_end"):
        result[name] = {"dtype": "int64", "shape": [1], "names": None}
    result["timestamp"] = {"dtype": "float32", "shape": [1], "names": None, "unit": "s"}
    for name in ("next.done", "next.truncated", "episode.success", "simulation.obstacle_contact"):
        result[name] = {"dtype": "bool", "shape": [1], "names": None}
    return result


@dataclass
class ReplayTrace:
    model: object
    qpos: np.ndarray
    qvel: np.ndarray
    controls: np.ndarray
    references: np.ndarray
    phases: np.ndarray
    actuator_force: np.ndarray
    body_names: list
    initial_state: np.ndarray
    state_spec: int
    checks: dict

    @property
    def frames(self):
        return len(self.controls) // SUBSTEPS


def replay_trace(directory, receipt):
    directory = Path(directory).resolve()
    for name, digest in receipt["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Source artifact changed: {name}")
    for name, digest in receipt["robot_assets_sha256"].items():
        if sha256_file(directory.parents[2] / "robot_assets" / name) != digest:
            raise ValueError(f"Source robot mesh changed: {name}")
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    with np.load(directory / "rollout.npz", allow_pickle=False) as saved:
        controls, references, phases = saved["controls"], saved["references"], saved["phases"]
        initial, spec = saved["initial_state"], int(saved["state_spec"])
        if sha256_array(controls) != receipt["hashes"]["controls"]:
            raise ValueError("Control hash mismatch")
        if not (len(controls) == len(references) == len(phases)):
            raise ValueError("Control stream lengths disagree")
        mujoco.mj_setState(model, data, initial, spec)
        mujoco.mj_forward(model, data)
        qpos = np.empty((len(controls) + 1, model.nq))
        qvel = np.empty((len(controls) + 1, model.nv))
        efforts = np.empty((len(controls), model.nu))
        qpos[0], qvel[0] = data.qpos, data.qvel
        warnings = np.array([w.number for w in data.warning])
        for index, control in enumerate(controls):
            data.ctrl[:] = control
            mujoco.mj_step(model, data)
            qpos[index + 1], qvel[index + 1] = data.qpos, data.qvel
            efforts[index] = data.actuator_force
        errors = {
            "saved_frame_qpos_error": float(np.max(np.abs(qpos[SUBSTEPS::SUBSTEPS] - saved["qpos"]))),
            "saved_frame_qvel_error": float(np.max(np.abs(qvel[SUBSTEPS::SUBSTEPS] - saved["qvel"]))),
            "final_qpos_error": float(np.max(np.abs(qpos[-1] - saved["final_qpos"]))),
            "final_qvel_error": float(np.max(np.abs(qvel[-1] - saved["final_qvel"]))),
        }
        if max(errors.values()) > 1e-10:
            raise ValueError(f"Saved dynamics replay diverged: {errors}")
        if np.any(np.array([w.number for w in data.warning]) != warnings):
            raise ValueError("Replay produced solver warnings")
    return ReplayTrace(model, qpos, qvel, controls, references, phases, efforts,
                       receipt["model"]["body_joint_names"], initial, spec,
                       {"deterministic": True, **errors})


class ObservationExtractor:
    def __init__(self, trace, camera_model):
        self.trace, self.model = trace, trace.model
        self.data = mujoco.MjData(self.model)
        self.camera_model = camera_model
        self.camera_data = mujoco.MjData(camera_model)
        self.jids = np.array([self.model.joint(JOINT_MAP[name]).id for name in STATE_NAMES])
        self.qadr = self.model.jnt_qposadr[self.jids]
        self.dadr = self.model.jnt_dofadr[self.jids]
        self.indices = {group: [STATE_NAMES.index(name) for name in names]
                        for group, names in GROUPS.items()}
        self.base = self.model.body("base_link").id
        if self.model.body_jntnum[self.base] != 0:
            raise ValueError("Ideal fixed-base IMU requires a fixed base")
        self.obj = self.model.body("object::target").id
        self.obj_qadr = self.model.joint("objfree::target").qposadr[0]
        self.wrists = [self.model.body(f"zarm_{side}7_link").id for side in ("l", "r")]
        self.pads = [[self.model.geom(f"claw::{side}_{part}_fingers").id for part in ("f", "b")]
                     for side in ("l", "r")]
        self.geom_names = [self.model.geom(i).name or "" for i in range(self.model.ngeom)]
        self.ref_indices = [trace.body_names.index(JOINT_MAP[name]) if name not in GRIPPER_NAMES else -1
                            for name in STATE_NAMES]
        self.left_aid = self.model.actuator("act::l_f_bar-1_joint").id

    def object_pose(self, qpos):
        pose = qpos[self.obj_qadr:self.obj_qadr + 7]
        return np.r_[pose[:3], pose[4:], pose[3]]

    def action(self, end_index):
        reference = self.trace.references[end_index - 1]
        return np.array([
            self.trace.controls[end_index - 1, self.left_aid] if name == "left_claw"
            else reference[-1] if name == "right_claw"
            else reference[index] for name, index in zip(STATE_NAMES, self.ref_indices)])

    def frame(self, index):
        trace, model, data = self.trace, self.model, self.data
        start, end = index * SUBSTEPS, (index + 1) * SUBSTEPS
        data.qpos[:] = trace.qpos[start]
        data.qvel[:] = trace.qvel[start]
        if start:
            data.ctrl[:] = trace.controls[start - 1]
        else:
            mujoco.mj_setState(model, data, trace.initial_state, trace.state_spec)
        mujoco.mj_forward(model, data)
        self.camera_data.qpos[:] = data.qpos
        self.camera_data.qvel[:] = data.qvel
        mujoco.mj_forward(self.camera_model, self.camera_data)
        state = data.qpos[self.qadr].copy()
        action = self.action(end)
        result = {
            "observation.state": state, "action": action,
            "next.observation.state": trace.qpos[end, self.qadr].copy(),
            "timestamp": index / FPS, "frame_index": index,
            "simulation.control_index_start": start, "simulation.control_index_end": end,
            "task.phase_index": PHASES.index(str(trace.phases[start])),
        }
        for group, indices in self.indices.items():
            result[f"observation.state.{group}.position"] = state[indices]
            result[f"action.{group}.position"] = action[indices]
            if group != "effector":
                result[f"observation.state.{group}.velocity"] = data.qvel[self.dadr[indices]].copy()
                result[f"observation.state.{group}.effort"] = data.qfrc_actuator[self.dadr[indices]].copy()
        end_pos, end_rot, widths = [], [], []
        for side, wrist, pads in zip(("left", "right"), self.wrists, self.pads):
            pos, rot = data.xpos[wrist].copy(), data.xmat[wrist].reshape(3, 3)
            quat = xyzw(rot)
            end_pos.extend(pos)
            end_rot.extend(quat)
            result[f"observation.ee_pose_{side}"] = np.r_[pos, quat]
            center = np.mean(data.geom_xpos[pads], axis=0)
            result[f"observation.tcp_pose_{side}"] = np.r_[center, quat]
            normal = rot[:, 1]
            gap = abs((data.geom_xpos[pads[0]] - data.geom_xpos[pads[1]]) @ normal)
            for pad in pads:
                gap -= np.abs(data.geom_xmat[pad].reshape(3, 3).T @ normal) @ model.geom_size[pad]
            widths.append(max(0., float(gap)))
        result["observation.state.end.position"] = np.array(end_pos)
        result["observation.state.end.orientation"] = np.array(end_rot)
        result["observation.gripper_width"] = np.array(widths)
        result["observation.environment.object_pose"] = self.object_pose(data.qpos)
        result["next.environment.object_pose"] = self.object_pose(trace.qpos[end])
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, self.obj, velocity, 0)
        result["observation.environment.object_velocity"] = np.r_[velocity[3:], velocity[:3]]
        rotation = data.xmat[self.base].reshape(3, 3)
        result["imu.quat_xyzw"] = xyzw(rotation)
        result["imu.acc_xyz"] = rotation.T @ -model.opt.gravity
        result["imu.free_acc_xyz"] = np.zeros(3)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, self.base, velocity, 1)
        result["imu.gyro_xyz"] = velocity[:3].copy()
        wrench, finger, force = np.zeros(6), np.zeros(2), np.zeros(6)
        obstacle = False
        for ci in range(data.ncon):
            contact = data.contact[ci]
            names = {self.geom_names[contact.geom1], self.geom_names[contact.geom2]}
            if "object::target" not in names:
                continue
            mujoco.mj_contactForce(model, data, ci, force)
            sign = 1 if model.geom_bodyid[contact.geom2] == self.obj else -1
            frame = contact.frame.reshape(3, 3).T
            world_force = sign * frame @ force[:3]
            wrench[:3] += world_force
            wrench[3:] += sign * frame @ force[3:] + np.cross(contact.pos - data.xipos[self.obj], world_force)
            for i, side in enumerate(("f", "b")):
                if any(name.startswith(f"claw::r_{side}") for name in names):
                    finger[i] += max(0., float(force[0]))
            obstacle |= "object::barrier" in names and force[0] > .005
        result["simulation.object_contact_wrench"] = wrench
        result["simulation.right_finger_normal_force"] = finger
        result["simulation.obstacle_contact"] = bool(obstacle)
        for name in CAMERAS:
            cid = self.camera_model.camera(name).id
            optical = self.camera_data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
            result[f"observation.camera_params.rotation_matrix_flat.{name}"] = optical.ravel()
            result[f"observation.camera_params.translation_vector.{name}"] = self.camera_data.cam_xpos[cid].copy()
        return result
