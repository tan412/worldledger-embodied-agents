"""OpenLET/Kuavo ROS bag adapter.

This adapter is deliberately a fact extractor.  In particular, it does not
collapse command-side and feedback-side signals into one trajectory:

* ``/sensors_data_raw`` is the measured joint/IMU stream;
* ``/joint_cmd`` is the controller command stream, including torque limits,
  ratios, gains and control modes;
* auxiliary arm, claw, Dex force and tactile topics are preserved as separate
  streams instead of being silently discarded or overwritten.

All stream timestamps come from the message ``header.stamp``.  The sensor
position stream remains the primary time axis and gets the historical nominal
frequency frame grid used by the kinematic validators.  Other streams retain
their native timestamps so delay and rate mismatch remain measurable.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ..evidence import EvidencePackage, Stream
from .base import file_record

JOINT_TOPIC = "/sensors_data_raw"
COMMAND_TOPIC = "/joint_cmd"
ARM_TOPIC = "/kuavo_arm_traj"
CLAW_COMMAND_TOPIC = "/leju_claw_command"
CLAW_STATE_TOPIC = "/leju_claw_state"
TF_TOPICS = ("/tf", "/tf_static")
DEG_DETECT_THRESHOLD = 3.2


def _stamp(msg, fallback_ns: int) -> float:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    if stamp is None:
        return float(fallback_ns) / 1e9
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


def _joint_names(dim: int, explicit: list | None, pkg: EvidencePackage) -> list:
    """Return the Kuavo whole-body order without silently shifting DoFs.

    Kuavo 4 Pro samples have 28 DoF.  Kuavo 5 samples in LET-Body have one
    additional ``waist_yaw_joint`` between the legs and the arms.  Unknown
    widths remain explicitly anonymous rather than pretending to be a 28-DoF
    model.
    """
    if explicit is not None:
        if len(explicit) != dim:
            raise ValueError(f"joint_names 长度 {len(explicit)} 与数据宽度 {dim} 不符")
        return list(explicit)
    names = (
        [f"leg_l{i}_joint" for i in range(1, 7)]
        + [f"leg_r{i}_joint" for i in range(1, 7)]
    )
    if dim == 28:
        names += [f"zarm_l{i}_joint" for i in range(1, 8)]
        names += [f"zarm_r{i}_joint" for i in range(1, 8)]
        names += ["zhead_1_joint", "zhead_2_joint"]
    elif dim == 29:
        names += ["waist_yaw_joint"]
        names += [f"zarm_l{i}_joint" for i in range(1, 8)]
        names += [f"zarm_r{i}_joint" for i in range(1, 8)]
        names += ["zhead_1_joint", "zhead_2_joint"]
    else:
        names = [f"joint_{i}" for i in range(dim)]
        pkg.adapter_notes.append(
            f"未知 Kuavo 关节宽度 {dim}: 不套用 28/29 DoF 顺序，使用匿名 joint_i")
    return names


def _sort_rows(rows: list[tuple[float, object]]) -> list[tuple[float, object]]:
    return sorted(rows, key=lambda row: row[0])


def _stack(rows: list[tuple[float, object]], field: str, pkg: EvidencePackage):
    """Stack fixed-width numeric rows, dropping only malformed rows."""
    good = []
    widths = set()
    for t, value in rows:
        arr = np.asarray(value)
        if arr.ndim != 1:
            pkg.adapter_notes.append(f"{field}: 跳过非一维消息")
            continue
        widths.add(int(arr.size))
        if arr.size:
            good.append((float(t), arr.astype(float, copy=False)))
    if not good:
        return None, None
    if len(widths) != 1:
        pkg.adapter_notes.append(f"{field}: 消息宽度变化 {sorted(widths)}，仅保留主宽度")
        width = max(widths, key=lambda x: sum(a.size == x for _, a in good))
        good = [(t, a) for t, a in good if a.size == width]
    good.sort(key=lambda row: row[0])
    return np.stack([a for _, a in good]), np.asarray([t for t, _ in good])


def _add_numeric(pkg, name, rows, columns, unit, source_field, source_file,
                 note=""):
    data, timestamps = _stack(rows, source_field, pkg)
    if data is None:
        return False
    pkg.add(Stream(name, "observed", data=data, columns=list(columns), unit=unit,
                   timestamps=timestamps, source_file=source_file,
                   source_field=source_field, note=note))
    return True


def _nearest_indices(reference: np.ndarray, target: np.ndarray):
    idx = np.searchsorted(target, reference)
    idx = np.clip(idx, 0, len(target) - 1)
    left = np.maximum(idx - 1, 0)
    use_left = np.abs(target[left] - reference) <= np.abs(target[idx] - reference)
    return np.where(use_left, left, idx)


def _topic_rows(reader, connections, topic, callback):
    rows = []
    for conn, bag_time, raw in reader.messages(connections=connections):
        msg = reader.deserialize(raw, conn.msgtype)
        value = callback(msg)
        if value is not None:
            rows.append((_stamp(msg, bag_time), value))
    return _sort_rows(rows)


def _prefixed_columns(prefix, columns):
    return [f"{prefix}_{c}" for c in columns]


def _wrench_value(msg):
    w = getattr(msg, "wrench", None)
    if w is None:
        return None
    return [w.force.x, w.force.y, w.force.z,
            w.torque.x, w.torque.y, w.torque.z]


def _joint_state_value(msg):
    pos = np.asarray(getattr(msg, "position", []), dtype=float)
    if pos.size == 0:
        return None
    return pos


def _read_auxiliary(bag_path: Path) -> dict:
    """Read non-primary topics in a fresh reader context.

    Keeping this separate is intentional: the primary reader is closed before
    streams are assembled, while every auxiliary row is still decoded with
    the same header-clock rule.
    """
    from rosbags.highlevel import AnyReader

    result = {
        "arm": [],
        "arm_velocity": [],
        "arm_effort": [],
        "arm_names": None,
        "claw_command": [],
        "claw_command_names": None,
        "claw_state": [],
        "wrench": {},
        "tactile": {},
        "calibration": {},
    }
    with AnyReader([bag_path]) as reader:
        by_topic = {}
        for conn in reader.connections:
            by_topic.setdefault(conn.topic, []).append(conn)

        if ARM_TOPIC in by_topic:
            converted_any = False
            for conn, bag_time, raw in reader.messages(
                    connections=by_topic[ARM_TOPIC]):
                msg = reader.deserialize(raw, conn.msgtype)
                stamp = _stamp(msg, bag_time)
                pos = np.asarray(msg.position, dtype=float)
                result["arm_names"] = result["arm_names"] or list(msg.name)
                converted = bool(pos.size and
                                np.max(np.abs(pos)) > DEG_DETECT_THRESHOLD)
                converted_any = converted_any or converted
                if pos.size:
                    result["arm"].append((stamp, np.deg2rad(pos)
                                          if converted else pos))
                vel = np.asarray(getattr(msg, "velocity", []), dtype=float)
                if vel.size:
                    result["arm_velocity"].append((stamp, np.deg2rad(vel)
                                                   if converted else vel))
                effort = np.asarray(getattr(msg, "effort", []), dtype=float)
                if effort.size:
                    result["arm_effort"].append((stamp, effort))
            result["arm_degrees_converted"] = converted_any

        if CLAW_COMMAND_TOPIC in by_topic:
            for conn, bag_time, raw in reader.messages(
                    connections=by_topic[CLAW_COMMAND_TOPIC]):
                msg = reader.deserialize(raw, conn.msgtype)
                data = getattr(msg, "data", msg)
                result["claw_command_names"] = (
                    result["claw_command_names"] or list(
                        getattr(data, "name", [])))
                pos = np.asarray(getattr(data, "position", []), dtype=float)
                vel = np.asarray(getattr(data, "velocity", []), dtype=float)
                eff = np.asarray(getattr(data, "effort", []), dtype=float)
                if pos.size:
                    result["claw_command"].append(
                        (_stamp(msg, bag_time), np.concatenate([pos, vel, eff])))

        if CLAW_STATE_TOPIC in by_topic:
            for conn, bag_time, raw in reader.messages(
                    connections=by_topic[CLAW_STATE_TOPIC]):
                msg = reader.deserialize(raw, conn.msgtype)
                data = msg.data
                state = np.asarray(msg.state, dtype=float)
                pos = np.asarray(data.position, dtype=float)
                vel = np.asarray(data.velocity, dtype=float)
                eff = np.asarray(data.effort, dtype=float)
                if pos.size:
                    result["claw_state"].append(
                        (_stamp(msg, bag_time),
                         np.concatenate([state, pos, vel, eff])))

        for topic, conns in by_topic.items():
            if topic.startswith("/force6d_") and topic.endswith(
                    "_force_torque"):
                result["wrench"][topic] = _topic_rows(
                    reader, conns, topic, _wrench_value)
            elif topic.startswith("/cb_") and topic.endswith(
                    "_matrix_touch_pc2"):
                result["tactile"][topic] = _topic_rows(
                    reader, conns, topic,
                    lambda msg: np.asarray(msg.data, dtype=float))
            elif topic in ("/kuavo/arm_zeros", "/kuavo/offset"):
                result["calibration"][topic] = _topic_rows(
                    reader, conns, topic,
                    lambda msg: np.asarray(getattr(msg, "data", []),
                                           dtype=float))
    return result


def load(bag_path: Path, marks_path: Path = None, joint_names: list = None,
         nominal_hz: float = 500.0) -> EvidencePackage:
    from rosbags.highlevel import AnyReader
    bag_path = Path(bag_path)
    marks_path = marks_path or bag_path.with_suffix(".json")

    pkg = EvidencePackage(
        episode_id=bag_path.stem, dataset_format="rosbag",
        fps=nominal_hz, raw_files=[file_record(bag_path, with_hash=False)])

    times, qs, vs, vds, efforts, sensor_times = [], [], [], [], [], []
    sensor_imu = []
    command_rows = {key: [] for key in (
        "q", "v", "tau", "tau_max", "tau_ratio", "kp", "kd", "mode")}
    tf_offsets: dict = {}
    odom = []                       # (t, xyz, quat)
    with AnyReader([bag_path]) as reader:
        by_topic = {}
        for c in reader.connections:
            by_topic.setdefault(c.topic, []).append(c)
        pkg.meta["topics"] = sorted(by_topic)
        pkg.meta["camera_topics"] = sorted(
            topic for topic in by_topic
            if topic.startswith(("/cam_", "/camera"))
        )

        if JOINT_TOPIC in by_topic:
            for conn, _t, raw in reader.messages(connections=by_topic[JOINT_TOPIC]):
                msg = reader.deserialize(raw, conn.msgtype)
                stamp = _stamp(msg, _t)
                q = np.asarray(msg.joint_data.joint_q, dtype=float)
                times.append(stamp)
                qs.append(q)
                vs.append(np.asarray(msg.joint_data.joint_v, dtype=float))
                vds.append(np.asarray(msg.joint_data.joint_vd, dtype=float))
                efforts.append(np.asarray(msg.joint_data.joint_torque, dtype=float))
                sensor_clock = getattr(msg, "sensor_time", None)
                sensor_times.append(
                    float(sensor_clock.sec) + float(sensor_clock.nanosec) / 1e9
                    if sensor_clock is not None else np.nan)
                imu = msg.imu_data
                sensor_imu.append((
                    [imu.quat.x, imu.quat.y, imu.quat.z, imu.quat.w],
                    [imu.acc.x, imu.acc.y, imu.acc.z],
                    [imu.free_acc.x, imu.free_acc.y, imu.free_acc.z],
                    [imu.gyro.x, imu.gyro.y, imu.gyro.z],
                ))
        if COMMAND_TOPIC in by_topic:
            for conn, _t, raw in reader.messages(connections=by_topic[COMMAND_TOPIC]):
                msg = reader.deserialize(raw, conn.msgtype)
                stamp = _stamp(msg, _t)
                for key, attr in (
                    ("q", "joint_q"), ("v", "joint_v"), ("tau", "tau"),
                    ("tau_max", "tau_max"), ("tau_ratio", "tau_ratio"),
                    ("kp", "joint_kp"), ("kd", "joint_kd"),
                    ("mode", "control_modes")):
                    command_rows[key].append((stamp, np.asarray(getattr(msg, attr),
                                                                 dtype=float)))
        for topic in TF_TOPICS:
            connections = by_topic.get(topic)
            if not connections:
                continue
            for conn, _t, raw in reader.messages(connections=connections):
                msg = reader.deserialize(raw, conn.msgtype)
                for tr in msg.transforms:
                    key = (tr.header.frame_id, tr.child_frame_id)
                    t = tr.transform.translation
                    tf_offsets.setdefault(key, [t.x, t.y, t.z])
                    if key == ("odom", "base_link"):
                        r = tr.transform.rotation
                        odom.append((tr.header.stamp.sec + tr.header.stamp.nanosec / 1e9,
                                     [t.x, t.y, t.z], [r.x, r.y, r.z, r.w]))

    if not qs:
        pkg.adapter_notes.append(f"{JOINT_TOPIC} 不存在或无消息")
        return pkg
    times = np.asarray(times)
    order = np.argsort(times, kind="stable")
    times = times[order]
    qs = [qs[i] for i in order]
    vs = [vs[i] for i in order]
    vds = [vds[i] for i in order]
    efforts = [efforts[i] for i in order]
    sensor_times = [sensor_times[i] for i in order]
    sensor_imu = [sensor_imu[i] for i in order]
    dim = len(qs[0])
    names = _joint_names(dim, joint_names, pkg)

    # 帧栅格:名义采样率,撞格顺延(实测最大顺延 2 格,不影响速度换算)
    t0 = float(times[0])
    frames, used = [], set()
    for t in times:
        f = int(round((t - t0) * nominal_hz))
        while f in used:
            f += 1
        used.add(f)
        frames.append(f)

    pkg.add(Stream("robot.joint_position", "observed",
                   data=np.stack(qs), columns=list(names), unit="rad",
                   timestamps=times, source_file=str(bag_path),
                   source_field=f"{JOINT_TOPIC}.joint_data.joint_q",
                   provenance={"frame_index": frames,
                               "frame_grid_hz": nominal_hz,
                               "time_source": "header.stamp"}))
    for values, stream_name, field, unit in (
        (vs, "robot.joint_velocity", "joint_v", "rad/s"),
        (vds, "robot.joint_acceleration", "joint_vd", "vendor_raw"),
        (efforts, "robot.joint_effort", "joint_torque", "Nm"),
    ):
        pkg.add(Stream(stream_name, "observed", data=np.stack(values),
                       columns=list(names), unit=unit, timestamps=times,
                       source_file=str(bag_path),
                       source_field=f"{JOINT_TOPIC}.joint_data.{field}",
                       provenance={"time_source": "header.stamp",
                                    "semantic": "vendor field preserved",
                                    "unit_requires_vendor_confirmation": field == "joint_vd",
                                    "not_used_as_derived_acceleration": field == "joint_vd"}))
    if sensor_times:
        sensor_clock = np.asarray(sensor_times, dtype=float)
        pkg.add(Stream(
            "extra.sensors_data_raw.sensor_time", "observed",
            data=sensor_clock[:, None], columns=["sensor_time_s"], unit="s",
            timestamps=times, source_file=str(bag_path),
            source_field=f"{JOINT_TOPIC}.sensor_time",
            provenance={"time_source": "header.stamp",
                        "clock": "embedded_sensor_time",
                        "header_minus_sensor_time_s": round(
                            float(np.nanmedian(times - sensor_clock)), 6)}))
    if sensor_imu:
        imu_data = np.asarray([
            [*quat, *acc, *free_acc, *gyro]
            for quat, acc, free_acc, gyro in sensor_imu
        ], dtype=float)
        pkg.add(Stream(
            "robot.imu", "observed", data=imu_data,
            columns=["quat_x", "quat_y", "quat_z", "quat_w",
                     "acc_x", "acc_y", "acc_z",
                     "free_acc_x", "free_acc_y", "free_acc_z",
                     "gyro_x", "gyro_y", "gyro_z"],
            unit="quat,m/s^2,rad/s", timestamps=times,
            source_file=str(bag_path), source_field=f"{JOINT_TOPIC}.imu_data",
            provenance={"time_source": "header.stamp"}))

    # Command side: every channel is kept on the command topic's own clock.
    for key, stream_name, field, unit in (
        ("q", "robot.action", "joint_q", "rad"),
        ("v", "robot.action_velocity", "joint_v", "rad/s"),
        ("tau", "robot.command_effort", "tau", "Nm"),
        ("tau_max", "robot.torque_limit", "tau_max", "Nm"),
        ("tau_ratio", "robot.torque_ratio", "tau_ratio", "ratio"),
        ("kp", "robot.controller_kp", "joint_kp", "raw"),
        ("kd", "robot.controller_kd", "joint_kd", "raw"),
        ("mode", "robot.control_mode", "control_modes", "enum"),
    ):
        rows = command_rows[key]
        data, cmd_ts = _stack(rows, f"{COMMAND_TOPIC}.{field}", pkg)
        if data is None:
            continue
        columns = list(names) if data.shape[1] == len(names) else [
            f"command[{i}]" for i in range(data.shape[1])]
        pkg.add(Stream(stream_name, "observed", data=data, columns=columns,
                       unit=unit, timestamps=cmd_ts, source_file=str(bag_path),
                       source_field=f"{COMMAND_TOPIC}.{field}",
                       provenance={"time_source": "header.stamp",
                                   "clock": "command_topic",
                                   "semantic": "controller-side field",
                                   "torque_semantics_requires_vendor_confirmation":
                                       field in ("tau", "tau_max")}))
    if COMMAND_TOPIC not in by_topic:
        pkg.adapter_notes.append(f"{COMMAND_TOPIC} 不存在: 无法审计命令-执行延迟、限矩和控制模式")

    aux = _read_auxiliary(bag_path)

    # The arm trajectory is a separate high-level command.  Its position is
    # degrees in the observed samples; do not overwrite the whole-body action.
    if aux["arm"]:
        arm_rows = aux["arm"]
        arm_vel_rows = aux["arm_velocity"]
        arm_eff_rows = aux["arm_effort"]
        arm_names = aux["arm_names"] or [f"arm_joint_{i+1}" for i in range(14)]
        arm_names = [str(n) for n in arm_names]
        arm_names = [n if n.startswith("zarm_") else
                     f"zarm_{'l' if i < len(arm_names) // 2 else 'r'}{i % 7 + 1}_joint"
                     for i, n in enumerate(arm_names)]
        _add_numeric(pkg, "extra.kuavo_arm_traj.position", arm_rows, arm_names,
                     "rad", f"{ARM_TOPIC}.position", str(bag_path),
                     note="JointState 高层手臂目标；原始样例按角度制转为 rad")
        if arm_vel_rows:
            _add_numeric(pkg, "extra.kuavo_arm_traj.velocity", arm_vel_rows, arm_names,
                         "rad/s", f"{ARM_TOPIC}.velocity", str(bag_path))
        if arm_eff_rows:
            _add_numeric(pkg, "extra.kuavo_arm_traj.effort", arm_eff_rows, arm_names,
                         "Nm", f"{ARM_TOPIC}.effort", str(bag_path))

    if aux["claw_command"]:
        rows = aux["claw_command"]
        names_ = aux["claw_command_names"] or []
        width = len(rows[0][1])
        n = len(names_)
        cols = (list(names_) + [f"{x}_velocity" for x in names_]
                + [f"{x}_effort" for x in names_])
        if len(cols) != width:
            cols = [f"extra.leju_claw.command[{i}]" for i in range(width)]
        _add_numeric(pkg, "extra.leju_claw.command", rows, cols, "raw",
                     f"{CLAW_COMMAND_TOPIC}.data", str(bag_path))
    if aux["claw_state"]:
        rows = aux["claw_state"]
        _add_numeric(pkg, "extra.leju_claw.state", rows,
                     [f"state_data_{i}" for i in range(rows[0][1].size)],
                     "raw", f"{CLAW_STATE_TOPIC}.(state,data)", str(bag_path))

    # Dex hand telemetry.  Per-hand streams prevent the second hand from
    # replacing the first one in EvidencePackage.streams.
    wrench_rows = aux["wrench"]
    tactile_rows = aux["tactile"]
    for topic, rows in wrench_rows.items():
        side = "left" if "left" in topic else "right"
        _add_numeric(pkg, f"extra.sensor.force_torque.{side}", rows,
                     _prefixed_columns(side, ["fx", "fy", "fz", "tx", "ty", "tz"]),
                     "N,Nm", topic, str(bag_path))
    if wrench_rows:
        ordered = sorted(wrench_rows)
        ref_topic = ordered[0]
        ref_rows = wrench_rows[ref_topic]
        ref_data, ref_ts = _stack(ref_rows, ref_topic, pkg)
        pieces = [ref_data]
        deltas = []
        for topic in ordered[1:]:
            other_data, other_ts = _stack(wrench_rows[topic], topic, pkg)
            idx = _nearest_indices(ref_ts, other_ts)
            pieces.append(other_data[idx])
            deltas.append(float(np.max(np.abs(other_ts[idx] - ref_ts))))
        if len(pieces) == 1:
            data, cols = pieces[0], _prefixed_columns(
                "left" if "left" in ref_topic else "right",
                ["fx", "fy", "fz", "tx", "ty", "tz"])
        else:
            data = np.concatenate(pieces, axis=1)
            cols = (["left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                     "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz"])
        pkg.add(Stream("sensor.force_torque", "observed", data=data, columns=cols,
                       unit="N,Nm", timestamps=ref_ts, source_file=str(bag_path),
                       source_field="+".join(ordered),
                       provenance={"pairing": "nearest_to_first_force_topic",
                                   "max_pair_delta_s": max(deltas, default=0.0)}))
    for topic, rows in tactile_rows.items():
        side = "left" if "left" in topic else "right"
        _add_numeric(pkg, f"extra.sensor.tactile.{side}", rows,
                     [f"{side}_cell_{i}" for i in range(len(rows[0][1]))],
                     "raw", topic, str(bag_path))
    if tactile_rows:
        ordered = sorted(tactile_rows)
        ref_data, ref_ts = _stack(tactile_rows[ordered[0]], ordered[0], pkg)
        pieces = [ref_data]
        deltas = []
        for topic in ordered[1:]:
            other_data, other_ts = _stack(tactile_rows[topic], topic, pkg)
            idx = _nearest_indices(ref_ts, other_ts)
            pieces.append(other_data[idx])
            deltas.append(float(np.max(np.abs(other_ts[idx] - ref_ts))))
        pkg.add(Stream("sensor.tactile", "observed",
                       data=np.concatenate(pieces, axis=1), timestamps=ref_ts,
                       source_file=str(bag_path),
                       source_field="+".join(ordered),
                       provenance={"pairing": "nearest_to_first_tactile_topic",
                                   "max_pair_delta_s": max(deltas, default=0.0)}))

    # Keep additional calibration topics visible for Body bags.
    for topic, rows in aux["calibration"].items():
            _add_numeric(pkg, f"extra{topic.replace('/', '.')}", rows,
                         [f"value_{i}" for i in range(len(rows[0][1]))],
                         "raw", topic, str(bag_path))

    if odom:
        ot = np.asarray([o[0] for o in odom])
        # 最近邻查表到关节时刻,不插值(插值会掩盖缺样)
        idx = np.clip(np.searchsorted(ot, times), 0, len(odom) - 1)
        pose = np.array([[*odom[i][1], *odom[i][2]] for i in idx])
        pkg.add(Stream("robot.base_pose", "observed", data=pose,
                       columns=["x", "y", "z", "qx", "qy", "qz", "qw"],
                       timestamps=times, source_file=str(bag_path),
                       source_field="/tf odom:base_link",
                       provenance={"lookup": "nearest, no interpolation",
                                   "max_lookup_delta_s": float(np.max(
                                       np.abs(ot[idx] - times))) }))
    if tf_offsets:
        pkg.add(Stream("robot.link_offsets", "observed",
                       data={(p, c): v for (p, c), v in tf_offsets.items()},
                       source_field="/tf(+static)", source_file=str(bag_path),
                       note="parent→child 平移观测,几何指纹身份核定的输入"))

    # 标注(OpenLET marks JSON)
    if Path(marks_path).exists():
        data = json.loads(Path(marks_path).read_text())
        segs = []
        for i, mark in enumerate(data.get("marks", []), 1):
            def epoch(key):
                return datetime.strptime(mark[key].strip(),
                                         "%Y-%m-%d %H:%M:%S.%f").timestamp()
            try:
                segs.append({"id": i, "start_s": epoch("markStart") - t0,
                             "end_s": epoch("markEnd") - t0,
                             "text": (mark.get("skillDetail") or "").strip(),
                             "text_en": (mark.get("enSkillDetail") or "").strip()})
            except (KeyError, ValueError):
                pkg.adapter_notes.append(f"mark {i} 字段异常,原样跳过")
        pkg.add(Stream("annotation.language_segments",
                       "observed" if segs else "missing", data=segs,
                       source_file=str(marks_path), source_field="marks"))
    return pkg
