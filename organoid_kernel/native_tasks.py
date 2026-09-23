"""Procedural tabletop tasks with force-limited MuJoCo control.

This is a generic Cartesian test rig, not a vendor robot or a learned policy.
No recorded episode, mocap body, object attachment, or rollout state overwrite
is used. Simulation object state is privileged feedback for the push planner.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .hashing import sha256_array, sha256_bytes, sha256_json
from .worldrev import admit_object


DT = 0.002
FPS = 25
SUBSTEPS = int(round(1 / FPS / DT))
CONTROL_NAMES = ("axis_x_m", "axis_y_m", "axis_z_m",
                 "left_finger_m", "right_finger_m")
JOINT_NAMES = ("axis_x", "axis_y", "axis_z", "finger_left", "finger_right")
LIMITS = np.array([[-.43, .43], [-.32, .32], [.032, .26],
                   [0, .052], [0, .052]])
VELOCITY_LIMITS = np.array([.8, .8, .8, .3, .3])
GOAL_HALF_SIZE = np.array([.058, .062])
PROVENANCE = {
    "synthetic": True,
    "motion_source": "procedural_minimum_jerk_and_state_feedback",
    "raw_episodes": [],
    "original_videos": [],
    "robot": "generic_cartesian_parallel_jaw_rig_v1",
    "control": "force_limited_joint_position_servos",
    "object_attachment": False,
    "mocap_or_weld": False,
    "privileged_feedback": "MuJoCo object pose used between push segments",
    "scope": "task-level dynamics of this explicit generic rig only",
    "real_robot_executability": "not_proven",
    "material_calibration": "assumed_not_measured",
}


@dataclass(frozen=True)
class TaskConfig:
    task: str
    seed: int = 7
    object_size: tuple[float, float, float] = (.04, .04, .04)
    mass: float = .055
    friction: float = .65
    grip_offset_x: float = .06
    lift_height: float = .082
    route: str = "direct"

    def __post_init__(self):
        if self.task not in ("bin_placement", "push_routing"):
            raise ValueError(f"Unknown task: {self.task}")
        if self.route not in ("direct", "detour"):
            raise ValueError(f"Unknown route: {self.route}")
        if len(self.object_size) != 3:
            raise ValueError("object_size must contain three full dimensions")
        values = [*self.object_size, self.mass, self.friction,
                  self.grip_offset_x, self.lift_height]
        if not np.isfinite(values).all():
            raise ValueError("Task parameters must be finite")
        if abs(self.grip_offset_x) > .08 or not .06 <= self.lift_height <= .23:
            raise ValueError("Repair parameters exceed the rig workspace")
        admit_object("part", {"kind": "box", "size": self.object_size,
                              "mass": self.mass, "friction": self.friction})

    @property
    def start_xy(self):
        jitter = np.random.default_rng(self.seed).uniform(-.006, .006, 2)
        base = (-.25, -.13) if self.task == "bin_placement" else (-.25, 0)
        return np.asarray(base) + jitter

    @property
    def goal_xy(self):
        return np.array([.24, .12 if self.task == "bin_placement" else 0.0])

    @property
    def barrier_size(self):
        return (.035, .24, .07) if self.task == "bin_placement" else (.045, .18, .065)

    def to_dict(self):
        return asdict(self)


def _numbers(values):
    return " ".join(f"{float(v):.8g}" for v in values)


def build_scene(config: TaskConfig) -> str:
    root = ET.Element("mujoco", model="organoid_native_tabletop")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=str(DT), gravity="0 0 -9.81",
                  integrator="implicitfast", iterations="80",
                  cone="elliptic", impratio="5")
    ET.SubElement(root, "size", njmax="500", nconmax="150")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "quality", shadowsize="2048")
    ET.SubElement(visual, "headlight", ambient=".35 .35 .35",
                  diffuse=".5 .5 .5", specular=".1 .1 .1")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", friction=".7 .005 .0001",
                  solref=".006 1", solimp=".95 .99 .001", condim="4")
    ET.SubElement(default, "joint", damping="1", armature=".01")
    assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "texture", type="skybox", builtin="flat",
                  rgb1=".82 .85 .85", width="32", height="192")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="-.4 -.5 1.5", dir=".2 .2 -1",
                  diffuse=".8 .8 .8", castshadow="true")
    ET.SubElement(world, "light", pos=".5 .3 1.2", dir="-.2 -.1 -1",
                  diffuse=".45 .45 .45", castshadow="false")

    def geom(parent, name, size, pos=(0, 0, 0), rgba=".7 .7 .7 1",
             kind="box", collision=True, **attrs):
        return ET.SubElement(parent, "geom", name=name, type=kind,
                             size=_numbers(size), pos=_numbers(pos), rgba=rgba,
                             **({} if collision else {"contype": "0", "conaffinity": "0"}),
                             **attrs)

    geom(world, "floor", (3, 3, .05), (0, 0, -.69), ".86 .88 .88 1")
    geom(world, "table", (.55, .43, .025), (0, 0, -.025), ".73 .76 .77 1")
    geom(world, "table_edge", (.558, .438, .008), (0, 0, -.052), ".22 .26 .28 1",
         collision=False)
    for x in (-.48, .48):
        for y in (-.36, .36):
            geom(world, f"table_leg_{x}_{y}", (.022, .022, .30),
                 (x, y, -.36), ".32 .36 .38 1", collision=False)
        geom(world, f"upright_{x}", (.022, .023, .27),
             (x, .35, .27), ".30 .34 .36 1")
    geom(world, "cross_rail", (.50, .022, .022), (0, .35, .52),
         ".22 .27 .29 1")
    # Rail and spindle visuals do not pretend to be a full hardware model.
    xbody = ET.SubElement(world, "body", name="x_stage")
    ET.SubElement(xbody, "inertial", pos="0 0 .5", mass=".7",
                  diaginertia=".02 .02 .02")
    ET.SubElement(xbody, "joint", name="axis_x", type="slide", axis="1 0 0",
                  range=_numbers(LIMITS[0]))
    geom(xbody, "y_rail", (.015, .41, .018), (0, 0, .50),
         ".38 .44 .46 1", collision=False)
    geom(xbody, "x_carriage", (.04, .037, .033), (0, .35, .52),
         ".12 .38 .40 1", collision=False)
    ybody = ET.SubElement(xbody, "body", name="y_stage")
    ET.SubElement(ybody, "inertial", pos="0 0 .48", mass=".4",
                  diaginertia=".005 .005 .005")
    ET.SubElement(ybody, "joint", name="axis_y", type="slide", axis="0 1 0",
                  range=_numbers(LIMITS[1]))
    geom(ybody, "y_carriage", (.032, .035, .03), (0, 0, .48),
         ".12 .38 .40 1", collision=False)
    tool = ET.SubElement(ybody, "body", name="tool")
    ET.SubElement(tool, "inertial", pos="0 0 .08", mass=".35",
                  diaginertia=".002 .002 .001")
    ET.SubElement(tool, "joint", name="axis_z", type="slide", axis="0 0 1",
                  range=_numbers(LIMITS[2]))
    geom(tool, "spindle", (.013, .20), (0, 0, .24),
         ".61 .67 .69 1", kind="cylinder", collision=False)
    geom(tool, "palm", (.034, .066, .018), (0, 0, .07), ".16 .22 .24 1")
    ET.SubElement(tool, "site", name="tcp", pos="0 0 0", size=".003",
                  rgba="0 0 0 0")
    for side, sign in (("left", 1), ("right", -1)):
        finger = ET.SubElement(tool, "body", name=f"finger_{side}",
                               pos=f"0 {sign * .008} 0")
        ET.SubElement(finger, "joint", name=f"finger_{side}", type="slide",
                      axis=f"0 {sign} 0", range="0 .052", damping="2")
        geom(finger, f"pad_{side}", (.015, .008, .032), (0, 0, .007),
             ".10 .13 .14 1", mass=".06", friction="1.1 .01 .001",
             solref=".004 1")
        geom(finger, f"finger_trim_{side}", (.014, .006, .007), (0, 0, .041),
             ".72 .75 .75 1", collision=False, mass=".005")
    if config.task == "push_routing":
        geom(tool, "pusher_stem", (.009, .036), (0, 0, .014),
             ".44 .48 .49 1", kind="cylinder", mass=".025")
        geom(tool, "pusher_tip", (.026, .026, .020), (0, 0, -.038),
             ".12 .18 .20 1", mass=".03",
             friction=".25 .003 .0001")
    bsize = np.array(config.barrier_size)
    geom(world, "barrier", bsize / 2, (0, 0, bsize[2] / 2),
         ".68 .24 .23 1")
    for x in (-.007, .007):
        geom(world, f"barrier_mark_{x}", (.002, bsize[1] / 2, .0007),
             (x, 0, bsize[2] + .0008), ".96 .78 .35 1", collision=False)
    goal = config.goal_xy
    geom(world, "goal_region", (*GOAL_HALF_SIZE, .0004), (*goal, .0005),
         ".27 .57 .43 1", collision=False)
    for axis in range(2):
        for sign in (-1, 1):
            half = GOAL_HALF_SIZE.copy()
            half[axis] = .003
            pos = goal.copy()
            pos[axis] += sign * (GOAL_HALF_SIZE[axis] + .003)
            height = .008 if config.task == "bin_placement" else .0006
            geom(world, f"goal_edge_{axis}_{sign}", (*half, height),
                 (*pos, height), ".35 .40 .40 1",
                 collision=config.task == "bin_placement")
    obj = ET.SubElement(world, "body", name="part",
                        pos=_numbers((*config.start_xy, config.object_size[2] / 2 + .001)))
    ET.SubElement(obj, "freejoint", name="part_free")
    geom(obj, "part_geom", np.array(config.object_size) / 2,
         rgba=".95 .61 .16 1", mass=str(config.mass),
         friction=f"{config.friction} .005 .0001")
    geom(obj, "part_mark", (config.object_size[0] * .30, .002, .0004),
         (0, 0, config.object_size[2] / 2 + .0004),
         ".98 .96 .91 1", collision=False, mass="0")
    actuator = ET.SubElement(root, "actuator")
    for index, name in enumerate(JOINT_NAMES):
        ET.SubElement(actuator, "position", name=CONTROL_NAMES[index], joint=name,
                      kp="6000" if index < 3 else "800",
                      kv="180" if index < 3 else "12",
                      forcerange="-120 120" if index < 3 else "-6 6",
                      ctrlrange=_numbers(LIMITS[index]))
    return ET.tostring(root, encoding="unicode")


def minimum_jerk(alpha):
    alpha = np.clip(alpha, 0, 1)
    return alpha ** 3 * (10 + alpha * (-15 + 6 * alpha))


def admission(config: TaskConfig) -> dict:
    reasons = []
    if config.task == "bin_placement":
        if config.object_size[1] + .008 > 2 * LIMITS[3, 1]:
            reasons.append("object_width_exceeds_open_jaw_with_clearance")
        if np.any(np.asarray(config.object_size[:2]) > 2 * GOAL_HALF_SIZE - .008):
            reasons.append("object_does_not_fit_target_bin")
    return {"admitted": not reasons, "reasons": reasons}


class SafetyStop(RuntimeError):
    """Measured forbidden contact stops a candidate without changing state."""


class NativeTaskRunner:
    """All motion after reset enters MuJoCo through actuator controls."""

    def __init__(self, config: TaskConfig, *, xml: str | None = None):
        self.config = config
        self.xml = build_scene(config) if xml is None else xml
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.jids = [self.model.joint(n).id for n in JOINT_NAMES]
        self.qadr = self.model.jnt_qposadr[self.jids]
        self.dadr = self.model.jnt_dofadr[self.jids]
        self.object_qadr = self.model.joint("part_free").qposadr[0]
        self.object_dadr = self.model.joint("part_free").dofadr[0]
        self.object_bid = self.model.body("part").id
        self.part_gid = self.model.geom("part_geom").id
        self.tool_bid = self.model.body("tool").id
        self.geom_names = [self.model.geom(i).name for i in range(self.model.ngeom)]
        self.home = np.array([-.35, -.27, .23, .047, .047])
        self.data.qpos[self.qadr] = self.home
        self.data.ctrl[:] = self.home
        mujoco.mj_forward(self.model, self.data)
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.initial_state = np.zeros(mujoco.mj_stateSize(self.model, self.state_spec))
        mujoco.mj_getState(self.model, self.data, self.initial_state, self.state_spec)
        self.initial_time = float(self.data.time)
        self.initial_object = self.object_position.copy()
        self.warning_counts = np.array([w.number for w in self.data.warning])
        self.controls = []
        self.phases = []
        self.frames = []
        self.max_joint_speed = np.zeros(5)
        self.max_penetration = 0.0
        self.obstacle_steps = self.table_collision_steps = 0
        self.peak_contact_force = 0.0
        self.nonfinite = False
        self.contact_buffer = np.zeros(6)
        self.close_error = None
        self.safety_stop = None
        self._bucket = self._new_bucket()

    @property
    def object_position(self):
        return self.data.qpos[self.object_qadr:self.object_qadr + 3]

    @staticmethod
    def _new_bucket():
        return {"left_N": 0., "right_N": 0., "pusher_N": 0.,
                "obstacle_contact": False, "table_contact": False,
                "bilateral_substeps": 0, "penetration_m": 0.}

    def step(self, control, phase):
        control = np.asarray(control, dtype=float)
        if control.shape != (5,) or not np.isfinite(control).all():
            raise ValueError("Control must contain five finite positions")
        if np.any(control < LIMITS[:, 0] - 1e-9) or np.any(control > LIMITS[:, 1] + 1e-9):
            raise ValueError(f"Out-of-range control: {control}")
        self.data.ctrl[:] = control
        mujoco.mj_step(self.model, self.data)
        self.controls.append(control.copy())
        self.phases.append(phase)
        self.nonfinite |= not (np.isfinite(self.data.qpos).all()
                               and np.isfinite(self.data.qvel).all())
        self.max_joint_speed = np.maximum(
            self.max_joint_speed, np.abs(self.data.qvel[self.dadr]))
        contacts = {"pad_left": 0., "pad_right": 0., "pusher_tip": 0.}
        obstacle = table_collision = False
        for ci in range(self.data.ncon):
            contact = self.data.contact[ci]
            names = {self.geom_names[contact.geom1], self.geom_names[contact.geom2]}
            mujoco.mj_contactForce(self.model, self.data, ci, self.contact_buffer)
            force = max(0., float(self.contact_buffer[0]))
            active = force > .01
            if "part_geom" in names:
                self.peak_contact_force = max(self.peak_contact_force, force)
                self._bucket["table_contact"] |= "table" in names and active
                for key in contacts:
                    if key in names:
                        contacts[key] += force
            relevant = "part_geom" in names or bool(
                names & {"pad_left", "pad_right", "palm", "pusher_tip", "pusher_stem"})
            if relevant and names != {"pad_left", "pad_right"}:
                depth = max(0., -float(contact.dist))
                self.max_penetration = max(self.max_penetration, depth)
                self._bucket["penetration_m"] = max(self._bucket["penetration_m"], depth)
            obstacle |= "barrier" in names and relevant and active
            table_collision |= ("table" in names and "part_geom" not in names
                                and relevant and active)
        self.obstacle_steps += int(obstacle)
        self.table_collision_steps += int(table_collision)
        self._bucket["obstacle_contact"] |= obstacle
        self._bucket["bilateral_substeps"] += int(
            contacts["pad_left"] > .02 and contacts["pad_right"] > .02)
        for key, name in (("left_N", "pad_left"), ("right_N", "pad_right"),
                          ("pusher_N", "pusher_tip")):
            self._bucket[key] = max(self._bucket[key], contacts[name])
        if len(self.controls) % SUBSTEPS == 0:
            self._sample(phase)

    def _sample(self, phase):
        q = self.object_qadr
        v = self.object_dadr
        self.frames.append({
            "time_s": round(float(self.data.time - self.initial_time), 6),
            "phase": phase,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "joint_position": self.data.qpos[self.qadr].copy(),
            "control": self.data.ctrl.copy(),
            "object_pose": self.data.qpos[q:q + 7].copy(),
            "object_velocity": self.data.qvel[v:v + 6].copy(),
            "tool_position": self.data.xpos[self.tool_bid].copy(),
            "lift_m": float(self.object_position[2] - self.initial_object[2]),
            **self._bucket,
        })
        self._bucket = self._new_bucket()

    def move(self, target, duration, phase):
        start = self.data.ctrl.copy()
        target = np.asarray(target, dtype=float)
        count = max(SUBSTEPS, int(round(duration / (DT * SUBSTEPS))) * SUBSTEPS)
        for i in range(count):
            alpha = minimum_jerk((i + 1) / count)
            self.step(start + (target - start) * alpha, phase)
            if self.safety_stop is None and (self.obstacle_steps or self.table_collision_steps):
                self.safety_stop = {
                    "phase": phase, "time_s": len(self.controls) * DT,
                    "reason": ("obstacle_contact" if self.obstacle_steps else
                               "tool_table_collision"),
                }
                raise SafetyStop(self.safety_stop["reason"])

    def hold(self, duration, phase):
        self.move(self.data.ctrl.copy(), duration, phase)

    def run(self):
        check = admission(self.config)
        if not check["admitted"]:
            raise ValueError("Scene rejected before simulation: " + ", ".join(check["reasons"]))
        try:
            if self.config.task == "bin_placement":
                self._pick_and_place()
            else:
                self._push_route()
            self.hold(1.0, "settle")
        except SafetyStop:
            brake = np.clip(self.data.qpos[self.qadr], LIMITS[:, 0], LIMITS[:, 1])
            for _ in range(int(1 / DT)):
                self.step(brake, "safety_stop")
        return self.receipt()

    def _pick_and_place(self):
        source = self.initial_object[:2].copy()
        source[0] += self.config.grip_offset_x
        goal = self.config.goal_xy
        grasp_z = self.config.object_size[2] / 2 + .018
        close = max(.003, self.config.object_size[1] / 2 - .007)
        opened = min(.05, self.config.object_size[1] / 2 + .021)

        def target(xy, z, jaw):
            return [*xy, z, jaw, jaw]

        self.move(target(source, .22, opened), 1.2, "approach")
        self.move(target(source, grasp_z, opened), 1.5, "descend")
        self.close_error = (self.object_position[:2]
                            - self.data.xpos[self.tool_bid, :2]).copy()
        self.move(target(source, grasp_z, close), .8, "close")
        self.hold(.3, "close")
        self.move(target(source, self.config.lift_height, close), 1.5, "lift")
        self.move(target(goal, self.config.lift_height, close), 3.4, "transport")
        self.move(target(goal, grasp_z + .001, close), 1.5, "lower")
        self.move(target(goal, grasp_z + .001, opened), .8, "release")
        self.move(target(goal, .22, opened), 1.4, "retract")

    def _push_route(self):
        cfg = self.config
        goal = cfg.goal_xy
        lane = -(cfg.barrier_size[1] / 2 + cfg.object_size[1] / 2 + .07)
        destinations = [goal] if cfg.route == "direct" else [
            np.array([self.initial_object[0], lane]),
            np.array([goal[0], lane]), goal]
        for destination in destinations:
            self.hold(.24, "observe")
            source = self.object_position[:2].copy()
            delta = destination - source
            distance = float(np.linalg.norm(delta))
            if distance < .003:
                continue
            direction = delta / distance
            # Support radius of the current oriented box, not a copied path.
            rotation = self.data.xmat[self.object_bid].reshape(3, 3)
            local_direction = rotation.T @ np.array([*direction, 0.])
            support = float(np.dot(np.abs(local_direction),
                                   np.asarray(cfg.object_size) / 2))
            pusher_support = .026 * float(np.abs(direction).sum())
            offset = support + pusher_support + .001
            behind = source - direction * (offset + .014)
            contact = source - direction * offset
            finish = destination - direction * offset
            jaw = .001
            self.move([*behind, .22, jaw, jaw], 1.4, "reposition")
            self.move([*behind, .064, jaw, jaw], 1.2, "descend")
            self.move([*contact, .064, jaw, jaw], .6, "contact")
            self.move([*finish, .064, jaw, jaw], max(1.6, distance / .065), "push")
            self.hold(.25, "push")
            self.move([*finish, .22, jaw, jaw], 1.2, "retract")

    def receipt(self):
        cfg = self.config
        tail = self.frames[-max(1, int(FPS * .8)):]
        rotation = self.data.xmat[self.object_bid].reshape(3, 3)
        half_extent = np.abs(rotation) @ (np.asarray(cfg.object_size) / 2)
        error = self.object_position[:2] - cfg.goal_xy
        final_inside = bool(np.all(np.abs(error) + half_extent[:2]
                                   <= GOAL_HALF_SIZE))
        linear_speed = max(float(np.linalg.norm(f["object_velocity"][:3])) for f in tail)
        angular_speed = max(float(np.linalg.norm(f["object_velocity"][3:])) for f in tail)
        height_error = abs(float(self.object_position[2]) - cfg.object_size[2] / 2)
        final_stable = linear_speed < .015 and angular_speed < .3 and height_error < .008
        transport = [f for f in self.frames if f["phase"] == "transport"]
        bilateral = (float(np.mean([f["bilateral_substeps"] / SUBSTEPS for f in transport]))
                     if transport else None)
        slip = None
        if transport:
            relative = np.array([f["object_pose"][:3] - f["tool_position"] for f in transport])
            slip = float(np.linalg.norm(relative - relative[0], axis=1).max())
        max_lift = max(f["lift_m"] for f in self.frames)
        warning_delta = (np.array([w.number for w in self.data.warning])
                         - self.warning_counts)
        numerics_ok = not self.nonfinite and not np.any(warning_delta > 0)
        checks = {
            "object_fully_in_target": final_inside,
            "object_settled_on_table": final_stable,
            "no_obstacle_contact": self.obstacle_steps == 0,
            "no_tool_table_collision": self.table_collision_steps == 0,
            "penetration_within_3mm": self.max_penetration <= .003,
            "finite_state_without_solver_warnings": numerics_ok,
            "joint_velocity_within_protocol": bool(
                np.all(self.max_joint_speed <= VELOCITY_LIMITS)),
        }
        if cfg.task == "bin_placement":
            checks.update({
                "bilateral_grasp_during_transport": bilateral is not None and bilateral >= .85,
                "lift_above_65mm": max_lift >= .065,
                "gripper_slip_below_15mm": slip is not None and slip <= .015,
            })
        else:
            checks["push_contact_observed"] = any(f["pusher_N"] > .05 for f in self.frames)
            checks["object_not_lifted"] = max_lift < .02
        metrics = {
            "final_xy_error_m": error.tolist(),
            "final_xy_error_norm_m": float(np.linalg.norm(error)),
            "final_aabb_half_extent_m": half_extent.tolist(),
            "final_height_error_m": height_error,
            "settle_linear_speed_max_m_s": linear_speed,
            "settle_angular_speed_max_rad_s": angular_speed,
            "max_lift_m": max_lift,
            "transport_bilateral_fraction": bilateral,
            "transport_slip_m": slip,
            "obstacle_contact_duration_s": self.obstacle_steps * DT,
            "tool_table_contact_duration_s": self.table_collision_steps * DT,
            "max_penetration_m": self.max_penetration,
            "peak_object_contact_force_N": self.peak_contact_force,
            "max_joint_speed": self.max_joint_speed.tolist(),
            "solver_warning_delta": warning_delta.tolist(),
            "grasp_alignment_error_xy_m": (
                self.close_error.tolist() if self.close_error is not None else None),
        }
        return {
            "schema": "organoid-kernel.native-task-receipt.v1",
            "config": cfg.to_dict(), "provenance": dict(PROVENANCE),
            "simulation": {"engine": "MuJoCo", "version": mujoco.__version__,
                           "timestep_s": DT, "observation_fps": FPS,
                           "control_hz": round(1 / DT),
                           "duration_s": len(self.controls) * DT,
                           "frames": len(self.frames), "control_steps": len(self.controls),
                           "actuator_force_limits_N": [120, 120, 120, 6, 6],
                           "velocity_limits_m_s": VELOCITY_LIMITS.tolist()},
            "scene": {"start_xy_m": self.initial_object[:2].tolist(),
                      "goal_xy_m": cfg.goal_xy.tolist(),
                      "goal_half_size_m": GOAL_HALF_SIZE.tolist(),
                      "barrier_size_m": list(cfg.barrier_size)},
            "success": all(checks.values()),
            "checks": checks, "metrics": metrics,
            "safety_stop": self.safety_stop,
            "hashes": {"mjcf_sha256": sha256_bytes(self.xml.encode()),
                       "config_sha256": sha256_json(cfg.to_dict()),
                       "controls_sha256": sha256_array(np.asarray(self.controls)),
                       "initial_state_sha256": sha256_array(self.initial_state)},
        }

    def replay_check(self):
        replay = NativeTaskRunner(self.config)
        for control, phase in zip(self.controls, self.phases):
            replay.step(control, phase)
        receipt = replay.receipt()
        pose_error = float(np.max(np.abs(
            np.array([f["qpos"] for f in replay.frames])
            - np.array([f["qpos"] for f in self.frames]))))
        return {"mode": "recorded_controls_only_no_planner",
                "success": receipt["success"],
                "checks_equal": receipt["checks"] == self.receipt()["checks"],
                "max_qpos_absolute_error": pose_error,
                "deterministic": pose_error <= 1e-10,
                "steps": len(replay.controls)}


def propose_repair(config: TaskConfig, receipt: dict):
    if receipt["success"]:
        return None, {"rule": "complete", "reason": "All task and physics checks passed."}
    metrics = receipt["metrics"]
    if config.task == "bin_placement":
        error = metrics["grasp_alignment_error_xy_m"]
        if (metrics["transport_bilateral_fraction"] is not None
                and metrics["transport_bilateral_fraction"] < .2 and error
                and abs(error[0]) > .02):
            corrected = float(np.clip(config.grip_offset_x + error[0], -.08, .08))
            return replace(config, grip_offset_x=corrected), {
                "rule": "align_grasp", "measured_error_xy_m": error,
                "change": {"grip_offset_x": [config.grip_offset_x, corrected]},
                "reason": "No sustained bilateral contact and measured grasp misalignment.",
            }
        if metrics["obstacle_contact_duration_s"] > 0:
            lift = config.barrier_size[2] + config.object_size[2] + .055
            if config.lift_height < lift <= .23:
                return replace(config, lift_height=lift), {
                    "rule": "raise_clearance",
                    "obstacle_contact_duration_s": metrics["obstacle_contact_duration_s"],
                    "change": {"lift_height": [config.lift_height, lift]},
                    "reason": "Transport hit the barrier; clearance includes part and jaw geometry.",
                }
    elif config.route == "direct" and metrics["obstacle_contact_duration_s"] > 0:
        return replace(config, route="detour"), {
            "rule": "route_around_barrier",
            "obstacle_contact_duration_s": metrics["obstacle_contact_duration_s"],
            "change": {"route": ["direct", "detour"]},
            "reason": "Direct push contacted the barrier; reapproach from three sides.",
        }
    return None, {"rule": "stop_honestly", "reason": "No bounded repair rule applies.",
                  "failed_checks": [k for k, v in receipt["checks"].items() if not v]}
