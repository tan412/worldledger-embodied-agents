"""New procedural tasks on the repository's original Kuavo humanoid asset.

The base is fixed, as in grasp_transplant's manipulation protocol. Original
joint topology, inertias, collision geometry and effort limits are retained.
No captured motion, Cartesian rig, wrist weld or object attachment is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import mjworld, physics_audit
from .fk import load_urdf
from .hashing import sha256_array, sha256_bytes, sha256_file
from .profile import load_profile
from .worldrev import admit_object


DT = .002
FPS = 25
SUBSTEPS = 20
ROOT_Z = .89
TABLE_Z = .65
WRIST = "zarm_r7_link"
ARM_NAMES = tuple(f"zarm_r{i}_joint" for i in range(1, 8))


@lru_cache(maxsize=1)
def robot_assets():
    profile = load_profile("biped_s200049")
    urdf = load_urdf(profile.urdf_path(), profile.mesh_path())
    return profile, urdf


def claw_pose(angle, side="r"):
    return {
        f"{side}_f_bar-1_joint": angle, f"{side}_f_bar-3": angle,
        f"{side}_b_bar-1": -angle, f"{side}_b_bar-3": -angle,
        f"{side}_f_bar-2": -angle, f"{side}_b_bar-2": angle,
    }


@dataclass(frozen=True)
class HumanoidConfig:
    task: str = "bin_placement"
    seed: int = 7
    size: tuple[float, float, float] = (.022, .025, .040)
    mass: float = .035
    friction: float = .65
    grasp_offset_y: float = .055
    transport_clearance: float = .015
    route: str = "direct"
    source_xy: tuple[float, float] = (.30, -.38)
    goal_xy: tuple[float, float] | None = None
    barrier_xy: tuple[float, float] = (.30, -.30)
    barrier_height: float = .035
    initial_arm_noise: float = 0.0
    speed_scale: float = 1.0
    control_noise: float = 0.0

    def __post_init__(self):
        if self.task not in ("bin_placement", "push_routing"):
            raise ValueError("Unknown humanoid task")
        if self.route not in ("direct", "detour"):
            raise ValueError("Unknown route")
        if self.goal_xy is None:
            object.__setattr__(self, "goal_xy",
                               (.24 if self.task == "push_routing" else .30, -.22))
        if any(np.asarray(value).shape != (2,) for value in
               (self.source_xy, self.goal_xy, self.barrier_xy)):
            raise ValueError("Scene XY coordinates must have two components")
        admit_object("target", {"kind": "box", "size": self.size,
                                "mass": self.mass, "friction": self.friction})
        if not np.isfinite([self.grasp_offset_y, self.transport_clearance,
                            *self.source_xy, *self.goal_xy, *self.barrier_xy,
                            self.barrier_height, self.initial_arm_noise,
                            self.speed_scale, self.control_noise]).all():
            raise ValueError("Trajectory parameters must be finite")
        if abs(self.grasp_offset_y) > .07 or not 0 < self.transport_clearance <= .055:
            raise ValueError("Trajectory parameters exceed the bounded demonstration workspace")
        if not .5 <= self.speed_scale <= 1.8 or self.control_noise < 0 or self.control_noise > .04:
            raise ValueError("Randomized control parameters exceed bounds")
        if not 0 < self.barrier_height <= .15 or not 0 <= self.initial_arm_noise <= .15:
            raise ValueError("Scene height and initial perturbation exceed bounds")

    def to_dict(self):
        return asdict(self)

    @property
    def source(self):
        return np.array([*self.source_xy, TABLE_Z + self.size[2] / 2])

    @property
    def goal(self):
        return np.array([*self.goal_xy, TABLE_Z + self.size[2] / 2])


def build_humanoid_scene(config, *, with_scene=True):
    profile, urdf = robot_assets()
    objects = []
    if with_scene:
        objects = [
            {"name": "table", "kind": "static_box", "size": [.42, .50, .035],
             "pos": [.36, -.30, TABLE_Z - .0175], "rgba": [.65, .70, .70, 1]},
            {"name": "barrier", "kind": "static_box",
             "pos": [*config.barrier_xy, TABLE_Z + config.barrier_height / 2],
             "size": [.065, .012, config.barrier_height],
             "rgba": [.72, .23, .21, 1]},
            {"name": "target", "kind": "box", "size": list(config.size),
             "pos": config.source.tolist(), "mass": config.mass,
             "friction": config.friction, "rgba": [.96, .59, .13, 1]},
        ]
        if config.task == "bin_placement":
            for axis in (0, 1):
                for sign in (-1, 1):
                    half = [.052, .046]
                    pos = config.goal.copy()
                    pos[axis] += sign * (half[axis] + .002)
                    size = [.108, .096, .008]
                    size[axis] = .004
                    objects.append({
                        "name": f"tray_{axis}_{sign}", "kind": "static_box",
                        "size": size, "pos": [*pos[:2], TABLE_Z + .004],
                        "rgba": [.22, .35, .30, 1],
                    })
    info = mjworld.build_world(
        urdf, profile, ground_height=0., free_root=False, root_pos=(0, 0, ROOT_Z),
        objects=objects, with_visual_meshes=True, full_body_actuators=True,
        claw_collision=True, claw_closed_chain=True, weld_body=None,
        smooth_friction_links=("zarm_l7_link", "zarm_r7_link"),
        omit_wrist_marker_ball=False,
    )
    tree = ET.fromstring(info.xml)
    option = tree.find("option")
    option.set("timestep", str(DT))
    option.set("iterations", "80")
    option.set("cone", "elliptic")
    default = ET.SubElement(tree, "default")
    ET.SubElement(default, "geom", solref=".006 1", solimp=".95 .99 .001")
    visual = ET.SubElement(tree, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "headlight", ambient=".35 .35 .35", diffuse=".65 .65 .65")
    ET.SubElement(tree.find("asset"), "texture", name="sky", type="skybox",
                  builtin="flat", rgb1=".86 .89 .89", width="32", height="192")
    world = tree.find("worldbody")
    ET.SubElement(world, "light", pos="1 -2 3", dir="-.2 .4 -1", diffuse=".8 .8 .8")
    floor = world.find("geom[@name='floor']")
    floor.set("rgba", ".78 .82 .82 1")
    if with_scene:
        ET.SubElement(world, "geom", name="goal_marker", type="box",
                      size=".052 .046 .0004",
                      pos=f"{config.goal[0]} {config.goal[1]} {TABLE_Z + .0005}",
                      rgba=".24 .62 .42 1", contype="0", conaffinity="0")
    # All original mesh/collision parts remain visible and active. The TCP is
    # only a kinematic reference fitted from the original two finger pads.
    wrist = tree.find(f".//body[@name='{WRIST}']")
    ET.SubElement(wrist, "site", name="grasp_tcp", pos="0 .0008 -.197",
                  size=".002", rgba="0 0 0 0")
    return ET.tostring(tree, encoding="unicode")


class HumanoidKinematics:
    def __init__(self, model, side="r"):
        if side not in ("l", "r"):
            raise ValueError("Unknown arm side")
        self.side = side
        self.model = model
        self.data = mujoco.MjData(model)
        self.jids = np.array([model.joint(f"zarm_{side}{i}_joint").id for i in range(1, 8)])
        self.qadr = model.jnt_qposadr[self.jids]
        self.dadr = model.jnt_dofadr[self.jids]
        self.wrist_bid = model.body(f"zarm_{side}7_link").id
        self.tcp = model.site("grasp_tcp" if side == "r" else "grasp_tcp_left").id
        self.bounds = model.jnt_range[self.jids].T
        self.bounds[0] += .001
        self.bounds[1] -= .001

    def pose(self, q):
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.tcp].copy(), self.data.xmat[self.wrist_bid].reshape(3, 3).copy()

    def solve(self, position, rotation, initial=None, attempts=1):
        seeds = ([initial] if initial is not None else []) + [
            [-.65, -.45, .8, -1.5, -.5, .4, .4],
            [-.5, -.8, 1.2, -1.4, -.8, .5, .3],
            [-.4, -.4, -1.0, -1.2, .8, .4, -.2],
        ]
        if self.side == "l":
            seeds = ([initial] if initial is not None else []) + [
                np.asarray(q) * [-1, -1, 1, -1, 1, -1, 1] for q in seeds[-3:]]
        best = None
        for seed in seeds[:attempts]:
            q0 = np.clip(seed, self.bounds[0], self.bounds[1])

            def residual(q):
                pos, rot = self.pose(q)
                return np.r_[pos - position,
                             .16 * Rotation.from_matrix(rotation @ rot.T).as_rotvec()]

            result = least_squares(residual, q0, bounds=self.bounds,
                                   max_nfev=180, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            pos, rot = self.pose(result.x)
            error = float(np.linalg.norm(pos - position))
            angle = float(np.linalg.norm(Rotation.from_matrix(rotation @ rot.T).as_rotvec()))
            candidate = (error + .16 * angle, result.x.copy(), error, angle)
            if best is None or candidate[0] < best[0]:
                best = candidate
            if error < .0002 and angle < .002:
                break
        return best


def minimum_jerk(a):
    return a ** 3 * (10 + a * (-15 + 6 * a))


class HumanoidStop(RuntimeError):
    """A candidate cannot continue after a measured task or safety failure."""


class HumanoidRunner:
    def __init__(self, config: HumanoidConfig, *, scene_xml=None):
        self.config = config
        self.profile, self.urdf = robot_assets()
        self.xml = scene_xml if scene_xml is not None else build_humanoid_scene(config)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.ik = HumanoidKinematics(self.model)
        self.rotation = (Rotation.from_euler("y", -.25).as_matrix()
                         if config.task == "push_routing" else np.eye(3))
        self.joint_names = [
            j.name for j in self.urdf.movable
            if any(j.name.startswith(prefix) for prefix in self.profile.joint_prefixes)]
        self.body_aids = np.array([self.model.actuator(f"fb::{n}").id for n in self.joint_names])
        self.body_jids = np.array([self.model.joint(n).id for n in self.joint_names])
        self.body_qadr = self.model.jnt_qposadr[self.body_jids]
        self.body_dadr = self.model.jnt_dofadr[self.body_jids]
        self.arm_indices = np.array([self.joint_names.index(n) for n in ARM_NAMES])
        self.claw_aid = self.model.actuator("act::r_f_bar-1_joint").id
        self.left_aid = self.model.actuator("act::l_f_bar-1_joint").id
        self.object_qadr = int(self.model.joint("objfree::target").qposadr[0])
        self.object_dadr = int(self.model.joint("objfree::target").dofadr[0])
        self.object_bid = self.model.body("object::target").id
        self.tcp = self.model.site("grasp_tcp").id
        self.pad_ids = [self.model.geom(f"claw::r_{s}_fingers").id for s in ("f", "b")]
        self.geom_names = [self.model.geom(i).name or "" for i in range(self.model.ngeom)]
        self.home = np.array([.30, -.40, .735])
        result = self.ik.solve(self.home, self.rotation, attempts=3)
        if result[2] > .002 or result[3] > .025:
            raise ValueError(f"Unreachable home: {result[2:]}")
        self.reference = np.zeros(len(self.joint_names))
        self.reference[self.arm_indices] = result[1]
        if config.initial_arm_noise:
            rng = np.random.default_rng(config.seed)
            self.reference[self.arm_indices] += rng.normal(
                0., config.initial_arm_noise, len(self.arm_indices))
            self.reference[self.arm_indices] = np.clip(
                self.reference[self.arm_indices],
                self.model.jnt_range[self.body_jids[self.arm_indices], 0] + .001,
                self.model.jnt_range[self.body_jids[self.arm_indices], 1] - .001)
        # Left arm is in its URDF neutral pose, outside the right-arm work area.
        self.data.qpos[self.body_qadr] = self.reference
        for side in ("l", "r"):
            for name, value in claw_pose(-.60, side).items():
                self.data.qpos[self.model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(self.model, self.data)
        self.open_angle = -.60
        self.close_angle = -.25
        self.current_angle = self.open_angle
        self.position_ref = self.home.copy()
        self.ik_errors = []
        self.control_rng = np.random.default_rng(config.seed)
        self.controls, self.references, self.phases, self.frames = [], [], [], []
        for _ in range(500):
            self._servo_step(self.reference, self.current_angle)
        self.initial_time = float(self.data.time)
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.initial_state = np.zeros(mujoco.mj_stateSize(self.model, self.state_spec))
        mujoco.mj_getState(self.model, self.data, self.initial_state, self.state_spec)
        self.initial_object = self.object_position.copy()
        self.initial_warnings = np.array([w.number for w in self.data.warning])
        self.fidelity = physics_audit.geometry_fidelity(
            self.model, self.data, body_prefixes=self.profile.physics_audit_body_prefixes)
        self.auditor = physics_audit.FrameAuditor(
            self.model, self.data, "objfree::target", soft_object=False)
        self.controls, self.references, self.phases, self.frames = [], [], [], []
        self.max_speed = np.zeros(len(self.joint_names))
        self.max_force = np.zeros(self.model.nu)
        self.max_penetration = 0.
        self.obstacle_steps = self.tool_table_steps = 0
        self.other_robot_contact_steps = 0
        self.max_self_penetration = 0.
        self.max_joint_limit_excess = 0.
        self.collision_pairs = {}
        self.bucket = self._bucket()
        self.buffer = np.zeros(6)
        self.close_alignment = None
        self.stopped = None
        self.directed_push_contact_steps = 0
        self.external_force_seen = False

    @property
    def object_position(self):
        return self.data.qpos[self.object_qadr:self.object_qadr + 3]

    def _servo_step(self, reference, angle):
        # Model-based gravity/Coriolis compensation is submitted through the
        # existing force-limited actuators, never through qfrc_applied.
        kp = self.model.actuator_gainprm[self.body_aids, 0]
        self.data.ctrl[self.body_aids] = (
            reference + self.data.qfrc_bias[self.body_dadr] / kp)
        self.data.ctrl[self.claw_aid] = angle
        self.data.ctrl[self.left_aid] = self.open_angle if hasattr(self, "open_angle") else -.60
        if self.config.control_noise:
            self.data.ctrl[self.body_aids] += self.control_rng.normal(
                0., self.config.control_noise, len(self.body_aids))
            # Position actuator controls are radians, not actuator force limits.
            bounds = self.model.jnt_range[self.body_jids]
            self.data.ctrl[self.body_aids] = np.clip(self.data.ctrl[self.body_aids],
                                                    bounds[:, 0], bounds[:, 1])
        control = self.data.ctrl.copy()
        mujoco.mj_step(self.model, self.data)
        return control

    @staticmethod
    def _bucket():
        return {"front_N": 0., "back_N": 0., "table_contact": False,
                "bilateral_substeps": 0, "obstacle_contact": False,
                "tcp_tracking_error_m": 0.}

    def step(self, reference, angle, phase, control=None):
        reference = np.asarray(reference, dtype=float)
        if reference.shape != self.reference.shape or not np.isfinite(reference).all():
            raise ValueError("Expected finite original-body joint references")
        bounds = self.model.jnt_range[self.body_jids]
        if np.any(reference < bounds[:, 0] - .01) or np.any(reference > bounds[:, 1] + .01):
            raise ValueError("Joint reference exceeds original joint range")
        if not np.isfinite(angle) or not -.698 <= angle <= .698:
            raise ValueError("Gripper command exceeds calibrated model range")
        if control is not None:
            control = np.asarray(control, dtype=float)
            if control.shape != (self.model.nu,) or not np.isfinite(control).all():
                raise ValueError("Expected a finite saved actuator control vector")
        self.external_force_seen |= bool(
            np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied))
        if control is None:
            control = self._servo_step(reference, angle)
        else:
            self.data.ctrl[:] = control
            mujoco.mj_step(self.model, self.data)
        self.controls.append(np.asarray(control).copy())
        self.references.append(np.r_[reference, angle])
        self.phases.append(phase)
        self.max_speed = np.maximum(self.max_speed, np.abs(self.data.qvel[self.body_dadr]))
        self.max_force = np.maximum(self.max_force, np.abs(self.data.actuator_force))
        q = self.data.qpos[self.body_qadr]
        bounds = self.model.jnt_range[self.body_jids]
        self.max_joint_limit_excess = max(
            self.max_joint_limit_excess,
            float(np.max(np.maximum(bounds[:, 0] - q, q - bounds[:, 1]))))
        fn = {"f": 0., "b": 0.}
        obstacle = tool_table = other_robot = False
        for ci in range(self.data.ncon):
            contact = self.data.contact[ci]
            a, b = self.geom_names[contact.geom1], self.geom_names[contact.geom2]
            names = {a, b}
            mujoco.mj_contactForce(self.model, self.data, ci, self.buffer)
            force = max(0., float(self.buffer[0]))
            if force < .005:
                continue
            robot = any(n.startswith(("claw::", "collision::")) for n in names)
            target = "object::target" in names
            depth = max(0., -float(contact.dist))
            if target or (robot and any(n.startswith("object::") for n in names)):
                self.max_penetration = max(self.max_penetration, depth)
            if target:
                for side in ("f", "b"):
                    if any(n.startswith(f"claw::r_{side}") for n in names):
                        fn[side] += force
                self.bucket["table_contact"] |= "object::table" in names
                other_robot |= any(n.startswith("collision::") for n in names)
            obstacle |= "object::barrier" in names and (robot or target)
            tool_table |= "object::table" in names and robot
            if (all(n.startswith(("claw::", "collision::")) for n in names)
                    and not all(n.startswith("claw::") for n in names)):
                self.max_self_penetration = max(self.max_self_penetration, depth)
            if (target or (robot and any(n.startswith("object::") for n in names))):
                pair = " | ".join(sorted(names))
                self.collision_pairs[pair] = max(force, self.collision_pairs.get(pair, 0.))
        self.obstacle_steps += int(obstacle)
        self.tool_table_steps += int(tool_table)
        self.other_robot_contact_steps += int(other_robot)
        self.directed_push_contact_steps += int(phase == "push" and sum(fn.values()) > .02)
        self.bucket["obstacle_contact"] |= obstacle
        self.bucket["bilateral_substeps"] += int(fn["f"] > .02 and fn["b"] > .02)
        self.bucket["front_N"] = max(self.bucket["front_N"], fn["f"])
        self.bucket["back_N"] = max(self.bucket["back_N"], fn["b"])
        self.bucket["tcp_tracking_error_m"] = max(
            self.bucket["tcp_tracking_error_m"],
            float(np.linalg.norm(self.data.site_xpos[self.tcp] - self.position_ref)))
        if len(self.controls) % SUBSTEPS == 0:
            self.auditor.frame()
            self.frames.append({
                "time_s": len(self.controls) * DT, "phase": phase,
                "qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(),
                "joint_position": self.data.qpos[self.body_qadr].copy(),
                "object_pose": self.data.qpos[self.object_qadr:self.object_qadr + 7].copy(),
                "object_velocity": self.data.qvel[self.object_dadr:self.object_dadr + 6].copy(),
                "tcp": self.data.site_xpos[self.tcp].copy(),
                "pad_center": np.mean(self.data.geom_xpos[self.pad_ids], axis=0),
                "lift_m": float(self.object_position[2] - self.initial_object[2]),
                **self.bucket,
            })
            self.bucket = self._bucket()

    def move(self, destination, angle, duration, phase):
        destination = np.asarray(destination, dtype=float)
        start = self.position_ref.copy()
        start_angle = self.current_angle
        nframes = max(1, int(round(duration * FPS / self.config.speed_scale)))
        for frame in range(nframes):
            alpha = minimum_jerk((frame + 1) / nframes)
            pos = start + (destination - start) * alpha
            result = self.ik.solve(pos, self.rotation,
                                   initial=self.reference[self.arm_indices])
            self.ik_errors.append((result[2], result[3]))
            if result[2] > .002 or result[3] > .03:
                raise ValueError(f"IK infeasible in {phase}: {result[2:]}, target={pos}")
            target_ref = self.reference.copy()
            target_ref[self.arm_indices] = result[1]
            target_angle = start_angle + (angle - start_angle) * alpha
            prev_angle = self.current_angle
            prev_ref = self.reference.copy()
            prev_pos = self.position_ref.copy()
            for sub in range(SUBSTEPS):
                u = (sub + 1) / SUBSTEPS
                ref = prev_ref + (target_ref - prev_ref) * u
                self.position_ref = prev_pos + (pos - prev_pos) * u
                self.step(ref, prev_angle + (target_angle - prev_angle) * u, phase)
            self.reference = target_ref
            self.current_angle = target_angle
            self.position_ref = pos
            if self.obstacle_steps or self.tool_table_steps:
                self.stopped = {"phase": phase, "time_s": len(self.controls) * DT,
                                "reason": "obstacle_contact" if self.obstacle_steps else "tool_table_contact"}
                raise HumanoidStop(self.stopped["reason"])

    def hold(self, duration, phase):
        for _ in range(int(duration / DT)):
            self.step(self.reference, self.current_angle, phase)

    def run(self):
        try:
            if self.config.task == "bin_placement":
                self._pick()
            else:
                self._push()
        except HumanoidStop:
            self.reference = self.data.qpos[self.body_qadr].copy()
            self.current_angle = float(
                self.data.qpos[self.model.joint("r_f_bar-1_joint").qposadr[0]])
            self.position_ref = self.data.site_xpos[self.tcp].copy()
            self.hold(1., "safety_stop")
        return self.receipt()

    def _pick(self):
        cfg = self.config
        source = cfg.source.copy()
        source[2] = TABLE_Z + .043
        source[1] += cfg.grasp_offset_y
        goal = cfg.goal.copy()
        goal[2] = source[2]
        hover = source.copy()
        hover[2] = .735
        self.move(hover, self.open_angle, 1.6, "approach")
        self.move(source, self.open_angle, 2.2, "descend")
        self.close_alignment = (self.object_position
                                - np.mean(self.data.geom_xpos[self.pad_ids], axis=0)).copy()
        self.move(source, self.close_angle, 1.3, "close")
        self.hold(.5, "close")
        lift = source.copy()
        lift[2] += cfg.transport_clearance
        self.move(lift, self.close_angle, 2.4, "lift")
        if not any(f["bilateral_substeps"] >= 10 for f in self.frames[-10:]):
            self.stopped = {"phase": "lift", "time_s": len(self.controls) * DT,
                            "reason": "no_bilateral_grasp"}
            raise HumanoidStop("No bilateral grasp during the initial lift")
        transit = goal.copy()
        transit[2] = lift[2]
        self.move(transit, self.close_angle, 3.0, "transport")
        self.move(goal, self.close_angle, 2.4, "lower")
        self.move(goal, self.open_angle, 1.3, "release")
        goal[2] = .735
        self.move(goal, self.open_angle, 2.0, "retract")
        self.hold(1.0, "settle")

    def _push(self):
        cfg = self.config
        self.close_angle = -.015
        destinations = ([cfg.goal[:2]] if cfg.route == "direct" else
                        [np.array([cfg.goal[0], cfg.source[1]]), cfg.goal[:2]])
        for destination in destinations:
            source = self.object_position[:2].copy()
            direction = destination - source
            distance = float(np.linalg.norm(direction))
            direction /= distance
            rotation = self.data.xmat[self.object_bid].reshape(3, 3)
            support = float(np.abs(rotation.T @ np.r_[direction, 0]) @ (np.array(cfg.size) / 2))
            finger_support = float(np.abs(direction) @ [.014, .011])
            offset = support + finger_support + .001
            behind = source - direction * (offset + .005)
            start = source - direction * offset
            finish = destination - direction * offset
            self.move([*behind, .718], self.close_angle, 1.8, "reposition")
            self.move([*behind, .695], self.close_angle, 1.6, "descend")
            self.move([*start, .695], self.close_angle, .8, "contact")
            self.move([*finish, .695], self.close_angle, max(2.0, distance / .035), "push")
            self.hold(.5, "push")
            self.move([*finish, .718], self.close_angle, 1.8, "retract")
            self.hold(.5, "observe")
        self.hold(1.0, "settle")

    def receipt(self):
        tail = self.frames[-FPS:]
        transit = [f for f in self.frames if f["phase"] == "transport"]
        bilateral = float(np.mean([f["bilateral_substeps"] / SUBSTEPS for f in transit])) if transit else None
        slip = None
        if transit:
            relative = np.array([f["object_pose"][:3] - f["pad_center"] for f in transit])
            slip = float(np.max(np.linalg.norm(relative - relative[0], axis=1)))
        rotation = self.data.xmat[self.object_bid].reshape(3, 3)
        extent = np.abs(rotation) @ (np.array(self.config.size) / 2)
        xy_error = self.object_position[:2] - self.config.goal[:2]
        lin_speed = max(float(np.linalg.norm(f["object_velocity"][:3])) for f in tail)
        angular_speed = max(float(np.linalg.norm(f["object_velocity"][3:])) for f in tail)
        warning_delta = np.array([w.number for w in self.data.warning]) - self.initial_warnings
        max_lift = max(f["lift_m"] for f in self.frames)
        by_name = {j.name: j for j in self.urdf.movable}
        speed_limits = np.array([by_name[n].velocity for n in self.joint_names])
        metrics = {
            "final_xy_error_m": xy_error.tolist(),
            "final_xy_error_norm_m": float(np.linalg.norm(xy_error)),
            "final_object_position_m": self.object_position.tolist(),
            "transport_bilateral_fraction": bilateral, "transport_slip_m": slip,
            "max_lift_m": max_lift,
            "obstacle_contact_duration_s": self.obstacle_steps * DT,
            "tool_table_contact_duration_s": self.tool_table_steps * DT,
            "max_penetration_m": self.max_penetration,
            "max_self_penetration_m": self.max_self_penetration,
            "max_joint_limit_excess_rad": self.max_joint_limit_excess,
            "other_robot_contact_duration_s": self.other_robot_contact_steps * DT,
            "max_joint_speed_rad_s": self.max_speed.tolist(),
            "max_actuator_force_Nm": self.max_force.tolist(),
            "max_ik_position_error_m": max((e[0] for e in self.ik_errors), default=None),
            "max_ik_orientation_error_rad": max((e[1] for e in self.ik_errors), default=None),
            "settle_linear_speed_m_s": lin_speed, "settle_angular_speed_rad_s": angular_speed,
            "collision_peak_force_N": self.collision_pairs,
            "close_alignment_m": self.close_alignment.tolist() if self.close_alignment is not None else None,
            "solver_warning_delta": warning_delta.tolist(),
            "push_contact_duration_s": self.directed_push_contact_steps * DT,
        }
        checks = {
            "object_fully_in_target": bool(np.all(np.abs(xy_error) + extent[:2] <= [.052, .046])),
            "placed_settled": (lin_speed < .015 and angular_speed < .3
                               and abs(self.object_position[2] - self.config.goal[2]) < .012),
            "no_obstacle_contact": self.obstacle_steps == 0,
            "no_tool_table_collision": self.tool_table_steps == 0,
            "no_forearm_object_contact": self.other_robot_contact_steps == 0,
            "penetration_below_3mm": self.max_penetration <= .003,
            "self_penetration_below_3mm": self.max_self_penetration <= .003,
            "joint_position_within_urdf_tolerance": self.max_joint_limit_excess <= .01,
            "joint_speed_within_urdf": bool(np.all(self.max_speed <= speed_limits)),
            "effort_within_limits": bool(np.all(self.max_force <= self.model.actuator_forcerange[:, 1] + 1e-6)),
            "numerically_valid": bool(np.isfinite(self.data.qpos).all() and not np.any(warning_delta > 0)),
            "no_external_force_injection": not self.external_force_seen,
            "geometry_fidelity": self.fidelity["ok"],
            "contact_conservation_and_mesh_overlap": self.auditor.verdict()["ok"],
        }
        if self.config.task == "bin_placement":
            checks.update({
                "bilateral_transport": bilateral is not None and bilateral >= .80,
                "lift_above_40mm": max_lift >= .04,
                "slip_below_20mm": slip is not None and slip <= .020,
            })
        else:
            checks.update({"finger_push_contact": self.directed_push_contact_steps > 20,
                           "object_not_lifted": max_lift < .020})
        checks = {key: bool(value) for key, value in checks.items()}
        return {
            "schema": "organoid-kernel.humanoid-native-task.v1",
            "config": self.config.to_dict(), "success": all(checks.values()),
            "checks": checks, "metrics": metrics,
            "safety_stop": self.stopped,
            "model": {"profile": self.profile.name,
                      "urdf": self.profile.urdf,
                      "urdf_sha256": sha256_file(self.profile.urdf_path()),
                      "body_joint_names": self.joint_names,
                      "actuator_names": [self.model.actuator(i).name for i in range(self.model.nu)],
                      "force_limits_Nm": self.model.actuator_forcerange.tolist()},
            "provenance": {
                "synthetic": True, "source_episodes": [], "captured_video": False,
                "robot": "original biped_s200049 / Kuavo 4 Pro assets",
                "trajectory_source": "new Cartesian waypoints + bounded original-arm IK",
                "base": "fixed; matches grasp_transplant manipulation protocol",
                "whole_body_balance": "not_evaluated",
                "wrist_weld": False, "object_attachment": False,
                "control": "original position servos + model bias feedforward through ctrl",
                "original_effort_limits": "enforced",
                "hardware_test": "not_evaluated; no robot connection available",
                "gripper_geometry": "existing mesh-derived collisions and parallel-link approximation",
                "gripper_effort_limit": "repository profile approximation, not hardware calibration",
                "object_orientation_goal": "unconstrained; push may tip the rectangular part",
                "generalization": "not_evaluated; deterministic demonstration fixture",
            },
            "simulation": {"engine": "MuJoCo", "version": mujoco.__version__,
                           "dt": DT, "fps": FPS, "frames": len(self.frames),
                           "control_steps": len(self.controls)},
            "hashes": {"xml": sha256_bytes(self.xml.encode()),
                       "controls": sha256_array(np.asarray(self.controls))},
            "geometry_fidelity": self.fidelity,
            "contact_audit": self.auditor.verdict(),
        }


def propose_humanoid_repair(config, receipt):
    if receipt["success"]:
        return None, {"rule": "complete", "reason": "All original-robot task and dynamics gates passed."}
    metrics = receipt["metrics"]
    if config.task == "bin_placement":
        if receipt["safety_stop"] and receipt["safety_stop"]["reason"] == "no_bilateral_grasp":
            error = metrics["close_alignment_m"]
            if error is not None and abs(error[1]) > .02:
                value = float(np.clip(config.grasp_offset_y + error[1], -.07, .07))
                return replace(config, grasp_offset_y=value), {
                    "rule": "align_original_gripper",
                    "measured_y_error_m": error[1],
                    "change": {"grasp_offset_y": [config.grasp_offset_y, value]},
                }
        if metrics["obstacle_contact_duration_s"] > 0 and config.transport_clearance < .049:
            return replace(config, transport_clearance=.049), {
                "rule": "raise_original_arm_clearance",
                "change": {"transport_clearance": [config.transport_clearance, .049]},
                "obstacle_contact_duration_s": metrics["obstacle_contact_duration_s"],
            }
    elif config.route == "direct" and metrics["obstacle_contact_duration_s"] > 0:
        return replace(config, route="detour"), {
            "rule": "route_original_fingers_around_obstacle",
            "change": {"route": ["direct", "detour"]},
        }
    return None, {"rule": "stop_honestly",
                  "failed_checks": [name for name, passed in receipt["checks"].items() if not passed]}
