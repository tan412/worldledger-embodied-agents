"""LeRobot v2.1 / v3.0 Adapter(v2.1 按乐聚商业交付实测,v3 按门户 limx 实测)。

无损原则的落实:
  * 关节流优先按厂商分组字段(乐聚)组装;没有分组字段时回退到通用
    `observation.state` 的 names 数组 —— 列名即事实,不写死;
  * v3 布局(data/chunk-*/file-*.parquet 多 episode 合装)按 episode_index 切片;
  * 末端执行器/手指通道不并进本体轨迹,单独成流(本体 URDF 无对应关节);
  * IMU、力/触觉、相机内外参、视频路径全部成流;未识别字段挂 extra.*;
  * timestamp 原样保留 —— 是不是合成栅格由能力清单裁定,Adapter 不遮丑。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..evidence import EvidencePackage, Stream
from .base import file_record

# 分组字段 → 流内列名(names 数组直接用;去掉 _link 后缀得关节名之外,保留原名)
JOINT_GROUPS = ("observation.state.leg.position", "observation.state.arm.position",
                "observation.state.head.position", "observation.state.waist.position")
EFFECTOR_COLS = ("observation.state.effector.position",
                 "observation.state.hand_left.position",
                 "observation.state.hand_right.position")


def _names(info: dict, col: str) -> list:
    return list(info.get("features", {}).get(col, {}).get("names") or [])


def _joint_name(raw: str) -> str:
    """厂商列名 → URDF 关节名:zarm_l1_link → zarm_l1_joint;leg 组是缩写,查表。"""
    if raw.endswith("_link"):
        return raw[:-5] + "_joint"
    leg_map = {"l_leg_roll": "leg_l1_joint", "l_leg_yaw": "leg_l2_joint",
               "l_leg_pitch": "leg_l3_joint", "l_knee": "leg_l4_joint",
               "l_foot_pitch": "leg_l5_joint", "l_foot_roll": "leg_l6_joint",
               "r_leg_roll": "leg_r1_joint", "r_leg_yaw": "leg_r2_joint",
               "r_leg_pitch": "leg_r3_joint", "r_knee": "leg_r4_joint",
               "r_foot_pitch": "leg_r5_joint", "r_foot_roll": "leg_r6_joint",
               "head_yaw": "zhead_1_joint", "head_pitch": "zhead_2_joint"}
    return leg_map.get(raw, raw)


def _read_episode_df(task_dir: Path, info: dict, episode: int):
    """v2.1: 每 episode 一个 parquet;v3: file-*.parquet 合装,按 episode_index 切片。"""
    import pandas as pd
    v21 = task_dir / f"data/chunk-000/episode_{episode:06d}.parquet"
    if v21.exists():
        return pd.read_parquet(v21), v21
    frames = []
    src = None
    for pq in sorted(task_dir.glob("data/chunk-*/file-*.parquet")):
        df = pd.read_parquet(pq)
        if "episode_index" not in df.columns:
            continue
        hit = df[df["episode_index"] == episode]
        if len(hit):
            frames.append(hit)
            src = src or pq
        elif frames:
            break                       # episode 连续存放,已越过
    if not frames:
        raise FileNotFoundError(f"episode {episode} 不在 {task_dir}/data 下")
    return pd.concat(frames, ignore_index=True), src


def load(task_dir: Path, episode: int) -> EvidencePackage:
    task_dir = Path(task_dir)
    info = json.loads((task_dir / "meta/info.json").read_text())
    df, parquet = _read_episode_df(task_dir, info, episode)
    fps = float(info.get("fps", 30))
    n = len(df)

    pkg = EvidencePackage(
        episode_id=f"{task_dir.name}-ep{episode}",
        dataset_format=f"lerobot_{info.get('codebase_version', '?')}",
        fps=fps,
        raw_files=[file_record(parquet)],
        meta={"robot_type": info.get("robot_type"), "task_dir": task_dir.name,
              "total_episodes": info.get("total_episodes")})
    if "synthetic" in info:
        pkg.meta["synthetic"] = bool(info["synthetic"])
        pkg.meta["hardware_test"] = info.get("hardware_test", "not_evaluated")
        pkg.meta["action_semantics"] = info.get("action_semantics")
        pkg.meta["gripper_units"] = info.get("gripper_units")
        pkg.meta["organoid_schema"] = info.get("organoid_schema")

    ts = df["timestamp"].to_numpy().astype(float) if "timestamp" in df.columns else None
    frame_idx = (df["frame_index"].to_numpy().astype(int).tolist()
                 if "frame_index" in df.columns else list(range(n)))

    # ---- 本体关节:分组拼接,列名走事实映射
    cols, mats = [], []
    for group in JOINT_GROUPS:
        if group not in df.columns:
            continue
        names = _names(info, group)
        arr = np.stack(df[group].to_numpy())
        if names and arr.shape[1] != len(names):
            pkg.adapter_notes.append(f"{group}: names {len(names)} 与数据宽度 {arr.shape[1]} 不符,按宽度截断")
        for i in range(arr.shape[1]):
            raw = names[i] if i < len(names) else f"{group}[{i}]"
            cols.append(_joint_name(raw))
        mats.append(arr)
    if mats:
        joint = np.concatenate(mats, axis=1)
        pkg.add(Stream("robot.joint_position", "observed", data=joint, columns=cols,
                       unit="rad", timestamps=ts, source_file=str(parquet),
                       source_field="+".join(g for g in JOINT_GROUPS if g in df.columns),
                       provenance={"frame_index": frame_idx,
                                   "name_mapping": "vendor names → URDF joint names"}))
    elif "observation.state" in df.columns:
        # 通用 LeRobot:没有厂商分组字段时,observation.state 的 names 即关节名
        arr = np.stack(df["observation.state"].to_numpy())
        names = _names(info, "observation.state") or \
            [f"state[{i}]" for i in range(arr.shape[1])]
        pkg.add(Stream("robot.joint_position", "observed", data=arr,
                       columns=[str(x) for x in names[:arr.shape[1]]], unit="rad",
                       timestamps=ts, source_file=str(parquet),
                       source_field="observation.state",
                       provenance={"frame_index": frame_idx,
                                   "name_mapping": "generic observation.state names"}))
        pkg.adapter_notes.append("通用 LeRobot 布局:关节流取 observation.state 全量,"
                                 "本体/末端拆分需 Profile 提供关节前缀")

    # velocity/effort 与 position 使用同一事实列顺序，不能只保留 position。
    # 这些通道是动力学校准和控制器辨识的观测，不应由校准脚本绕过 adapter 读 parquet。
    for suffix, stream_name, unit in (
            ("velocity", "robot.joint_velocity", "rad/s"),
            ("effort", "robot.joint_effort", "Nm")):
        vcols, vmats = [], []
        for group in JOINT_GROUPS:
            key = f"{group.rsplit('.', 1)[0]}.{suffix}"
            if key not in df.columns:
                continue
            arr = np.stack(df[key].to_numpy())
            names = _names(info, key) or _names(info, group)
            for i in range(arr.shape[1]):
                raw = names[i] if i < len(names) else f"{key}[{i}]"
                vcols.append(_joint_name(raw))
            vmats.append(arr)
        if vmats:
            pkg.add(Stream(stream_name, "observed",
                           data=np.concatenate(vmats, axis=1),
                           columns=vcols, unit=unit, timestamps=ts,
                           source_file=str(parquet),
                           source_field="+".join(
                               f"{g.rsplit('.', 1)[0]}.{suffix}"
                               for g in JOINT_GROUPS
                               if f"{g.rsplit('.', 1)[0]}.{suffix}" in df.columns),
                           provenance={"frame_index": frame_idx,
                                       "name_mapping": "vendor names → URDF joint names"}))

    # ---- 通用 LeRobot 的末端位姿与夹爪(limx: observation.ee_pose_*;极智 UMI 交付:
    #      observation.state.*_ee_pose / *_gripper_width —— 无关节流时走 UMI 源数据口径)
    ee_cols = [c for c in df.columns
               if ("ee_pose" in c and "cmd" not in c) and c.startswith("observation")]
    if ee_cols:
        arr = np.concatenate([np.stack(df[c].to_numpy()) for c in ee_cols], axis=1)
        target = "robot.end_effector_pose" if pkg.has("robot.joint_position") else "umi.gripper_pose"
        pkg.add(Stream(target, "observed", data=arr, timestamps=ts,
                       source_field="+".join(ee_cols), source_file=str(parquet),
                       columns=[f"{c.split('.')[-1]}[{i}]" for c in ee_cols
                                for i in range(np.stack(df[c].to_numpy()).shape[1])]))
    grip_cols = [c for c in df.columns if "gripper_width" in c]
    if grip_cols:
        arr = np.concatenate([np.stack(df[c].to_numpy()).reshape(n, -1)
                              for c in grip_cols], axis=1)
        pkg.add(Stream("umi.gripper_width", "observed", data=arr, timestamps=ts,
                       source_field="+".join(grip_cols), source_file=str(parquet)))

    # ---- 末端执行器(不并入本体)
    for col in EFFECTOR_COLS:
        if col in df.columns:
            arr = np.stack(df[col].to_numpy())
            pkg.add(Stream("robot.end_effector_pose" if "end" in col else f"extra.{col}",
                           "observed", data=arr, columns=_names(info, col) or
                           [f"{col}[{i}]" for i in range(arr.shape[1])],
                           unit="raw", timestamps=ts, source_file=str(parquet),
                           source_field=col,
                           note="末端执行器通道,本体 URDF 无对应关节,单独成流"))

    # ---- IMU
    imu_cols, imu_mats = [], []
    for col, names in (("imu.quat_xyzw", ["quat_x", "quat_y", "quat_z", "quat_w"]),
                       ("imu.acc_xyz", ["acc_x", "acc_y", "acc_z"]),
                       ("imu.gyro_xyz", ["gyro_x", "gyro_y", "gyro_z"])):
        if col in df.columns:
            imu_mats.append(np.stack(df[col].to_numpy()))
            imu_cols += names
    if imu_mats:
        pkg.add(Stream("robot.imu", "observed", data=np.concatenate(imu_mats, axis=1),
                       columns=imu_cols, timestamps=ts, source_file=str(parquet),
                       source_field="imu.*"))

    # ---- action / 力 / 相机参数 / 其余字段
    if "action" in df.columns:
        arr = np.stack(df["action"].to_numpy())
        pkg.add(Stream("robot.action", "observed", data=arr,
                       columns=[_joint_name(c) for c in _names(info, "action")] or
                       [f"action[{i}]" for i in range(arr.shape[1])],
                       unit="rad", timestamps=ts, source_field="action",
                       source_file=str(parquet)))
    extrinsic_cols, extrinsic_arrays, extrinsic_names = [], [], []
    for col in df.columns:
        if "force_torque" in col:
            pkg.add(Stream("sensor.force_torque", "observed",
                           data=np.stack(df[col].to_numpy()), timestamps=ts,
                           source_field=col, source_file=str(parquet)))
        elif "touch_matrix" in col:
            pkg.add(Stream("sensor.tactile", "observed",
                           data=np.stack(df[col].to_numpy()), timestamps=ts,
                           source_field=col, source_file=str(parquet)))
        elif col.startswith("observation.camera_params"):
            array = np.stack(df[col].to_numpy())
            extrinsic_cols.append(col)
            extrinsic_arrays.append(array)
            extrinsic_names.extend([f"{col}[{i}]" for i in range(array.shape[1])])
    if extrinsic_arrays:
        pkg.add(Stream("camera.extrinsics", "observed",
                       data=np.concatenate(extrinsic_arrays, axis=1),
                       columns=extrinsic_names, timestamps=ts,
                       source_field="+".join(extrinsic_cols), source_file=str(parquet),
                       provenance={"fields": extrinsic_cols},
                       note="全部相机外参数组按列保留，不以最后一台相机覆盖前面的通道"))

    # ---- 视频文件(不解码,只登记路径与哈希;抽帧由语义层按需做)
    vids = sorted((task_dir / "videos/chunk-000").glob(f"*/episode_{episode:06d}.mp4"))
    if vids:
        pkg.add(Stream("camera.rgb", "observed", files=[str(v) for v in vids],
                       source_file=str(vids[0]),
                       provenance={"cameras": [v.parent.name for v in vids]}))
    params = sorted((task_dir / "parameters").glob("*.json")) if (task_dir / "parameters").exists() else []
    if params:
        pkg.add(Stream("camera.intrinsics", "observed",
                       files=[str(p) for p in params], note="内外参 JSON 原样引用"))

    # ---- 标注(乐聚变体:metadata.json 的 label_info.action_config;帧号 → 秒)
    meta_path = task_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        ep_meta = meta.get(str(episode), {})
        pkg.meta.update({k: ep_meta.get(k) for k in
                         ("task_name", "scene_name", "sub_scene_name", "sn_code",
                          "episode_status", "data_gen_mode") if k in ep_meta})
        segs = []
        for k, seg in enumerate(ep_meta.get("label_info", {}).get("action_config", []), 1):
            segs.append({"id": k, "start_s": seg["start_frame"] / fps,
                         "end_s": seg["end_frame"] / fps,
                         "start_frame": seg["start_frame"], "end_frame": seg["end_frame"],
                         "skill": seg.get("skill"),
                         "text": (seg.get("action_text") or "").strip(),
                         "text_en": (seg.get("english_action_text") or "").strip(),
                         "is_mistake": bool(seg.get("is_mistake"))})
        pkg.add(Stream("annotation.language_segments",
                       "observed" if segs else "missing", data=segs,
                       source_file=str(meta_path), source_field="label_info.action_config",
                       provenance={"time_base": "frame/fps(与轨迹同源)"},
                       note="" if segs else "标注数组为空"))
    if info.get("organoid_schema") == "humanoid-trajectories.v1":
        for field, name, unit, frame in (
            ("observation.environment.object_pose", "scene.object_pose",
             "m + quaternion xyzw", "world"),
            ("episode.success", "annotation.success", "bool", ""),
        ):
            array = np.stack(df[field].to_numpy()).reshape(n, -1)
            pkg.add(Stream(name, "derived", data=array, timestamps=ts, unit=unit, frame=frame,
                           source_field=field, source_file=str(parquet),
                           columns=_names(info, field)))
        represented = {field for stream in pkg.streams.values()
                       for field in stream.source_field.split("+")}
        for field in df.columns:
            if field in represented or field in ("timestamp", "index", "frame_index", "episode_index", "task_index"):
                continue
            array = np.stack(df[field].to_numpy()).reshape(n, -1)
            feature = info["features"].get(field, {})
            pkg.add(Stream(f"extra.{field}", "derived", data=array,
                           columns=_names(info, field), timestamps=ts,
                           unit=feature.get("unit", ""), frame=feature.get("frame", ""),
                           source_field=field, source_file=str(parquet),
                           provenance={"generator": feature.get("origin")}))
    if info.get("synthetic"):
        for stream in pkg.streams.values():
            if stream.origin == "observed":
                stream.origin = "derived"
            stream.provenance.update({
                "synthetic": True, "hardware_measured": False,
                "dataset_origin": info.get("data_gen_mode", "simulation"),
            })
        pkg.adapter_notes.append("合成数据保留 synthetic 来源，derived 不表示真实硬件测量；"
                                 "电机电流、传感器噪声和硬件标定不由此补造")
    return pkg
