"""Pico VR 第一人称采集 Adapter(场景数据集产品口径)。

一个场景目录 = 一条 episode:
  data/trackingData_*.txt        逐帧 JSONL:头显/双手柄位姿、全身骨架、26 点手部
                                 关键点、TrackerState、纳秒时间戳(首行是设备头)
  data/*_segments_description.json  动作分段标注(OpenLET 同款字段)
  data/camera_params.json        双目内参与畸变
  data/CameraRecord_*.mp4 + video/*  第一视角视频
  data/points_aligned.ply        场景点云

映射口径(按各通道实测活性定,不按名字想当然):
  真实的手 = Hand.HandJointLocations(26 点手部追踪,实测更新占比 89~99%),
  腕点(joint[0])→ umi.gripper_pose(双手位置轨迹),全部关键点 → human.hand_landmarks;
  Controller 通道实测在多数场景整场静止(1~2 个取值 —— 闲置手柄/占位),
  按活性分级:活跃 → extra.controller_poses,死通道 → missing 并注明;
  头显位姿 → camera.extrinsics(第一视角相机轨迹);全身骨架挂 extra;
  TrackerState 占比入 provenance。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..evidence import EvidencePackage, Stream
from .base import file_record


def _hms_to_s(text: str) -> float:
    parts = [float(p) for p in text.strip().split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def _pose7(text: str):
    v = [float(x) for x in str(text).split(",")]
    return v if len(v) == 7 else None


def load(scene_dir: Path) -> EvidencePackage:
    scene_dir = Path(scene_dir)
    data = scene_dir / "data"
    track = sorted(data.glob("trackingData_*.txt"))
    if not track:
        raise SystemExit(f"{scene_dir} 下没有 trackingData_*.txt")
    track = track[0]

    pkg = EvidencePackage(episode_id=scene_dir.name, dataset_format="pico_vr_ego",
                          raw_files=[file_record(track)],
                          meta={"collection": "VR 第一人称(头显+手柄+全身+手部追踪)"})

    ts, head, ctrl, hands, states = [], [], [], [], []
    n_body = 0
    with track.open() as fh:
        header = json.loads(fh.readline())
        pkg.meta.update({k: header.get(k) for k in ("SN", "Version") if k in header})
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = row.get("timeStampNs")
            if t is None:
                continue
            hp = _pose7(row.get("Head", {}).get("pose", ""))
            lp = _pose7(row.get("Controller", {}).get("left", {}).get("pose", ""))
            rp = _pose7(row.get("Controller", {}).get("right", {}).get("pose", ""))
            if hp is None or lp is None or rp is None:
                continue
            ts.append(t / 1e9)
            head.append(hp)
            ctrl.append(lp + rp)
            states.append(row.get("TrackerState") == "accurate")
            hand = row.get("Hand") or {}
            pts = []
            for side in ("leftHand", "rightHand"):
                joints = (hand.get(side) or {}).get("HandJointLocations") or []
                for j in joints:
                    p = j.get("p") or j.get("pose")
                    if p:
                        pts.extend([float(x) for x in str(p).split(",")][:3])
            hands.append(pts)
            n_body = max(n_body, len((row.get("Body") or {}).get("joints") or []))

    ts = np.asarray(ts)
    ts_rel = ts - ts[0]
    accurate_ratio = float(np.mean(states)) if states else 0.0

    ctrl_arr = np.asarray(ctrl)
    ctrl_live = (float((np.linalg.norm(np.diff(ctrl_arr[:, 0:3], axis=0), axis=1)
                        > 1e-4).mean()) if len(ctrl_arr) > 1 else 0.0)
    pkg.add(Stream("extra.controller_poses",
                   "observed" if ctrl_live >= 0.05 else "missing",
                   data=ctrl_arr,
                   columns=[f"{s}_{c}" for s in ("left", "right")
                            for c in ("x", "y", "z", "qx", "qy", "qz", "qw")],
                   unit="m/quat", frame="VR 世界系", timestamps=ts_rel,
                   source_file=str(track), source_field="Controller.left/right.pose",
                   provenance={"update_ratio": round(ctrl_live, 4),
                               "tracker_accurate_ratio": round(accurate_ratio, 4)},
                   note="" if ctrl_live >= 0.05 else
                   f"死通道:整场更新占比仅 {ctrl_live:.1%}(闲置手柄/占位),不作手部数据"))
    pkg.add(Stream("camera.extrinsics", "observed", data=np.asarray(head),
                   columns=["x", "y", "z", "qx", "qy", "qz", "qw"],
                   timestamps=ts_rel, source_file=str(track), source_field="Head.pose",
                   note="头显位姿 = 第一视角相机轨迹"))
    widths = {len(h) for h in hands}
    if widths and max(widths) > 0:
        w = max(widths)
        arr = np.full((len(hands), w), np.nan)
        for i, h in enumerate(hands):
            arr[i, :len(h)] = h
        pkg.add(Stream("human.hand_landmarks", "observed", data=arr,
                       timestamps=ts_rel, source_file=str(track),
                       source_field="Hand.leftHand/rightHand.HandJointLocations",
                       note=f"每手最多 26 关键点 xyz;缺失帧为 NaN"))
        # 真实的手 = 手部追踪的腕点(每手 joint[0]);Controller 通道是闲置手柄,不用
        if w >= 26 * 3 * 2:
            wrist = np.concatenate([arr[:, 0:3], arr[:, 26 * 3:26 * 3 + 3]], axis=1)
            live = float((np.linalg.norm(np.diff(wrist[:, 0:3], axis=0), axis=1)
                          > 1e-4).mean())
            pose = np.zeros((len(wrist), 14))
            pose[:, 0:3] = wrist[:, 0:3]
            pose[:, 6] = 1.0                      # 姿态占位:单位四元数
            pose[:, 7:10] = wrist[:, 3:6]
            pose[:, 13] = 1.0
            pkg.add(Stream("umi.gripper_pose", "observed", data=pose,
                           columns=[f"{s}_{c}" for s in ("left", "right")
                                    for c in ("x", "y", "z", "qx", "qy", "qz", "qw")],
                           unit="m(姿态为占位单位四元数)", frame="VR 世界系",
                           timestamps=ts_rel, source_file=str(track),
                           source_field="Hand.*.HandJointLocations[0](腕点)",
                           provenance={"update_ratio": round(live, 4),
                                       "tracker_accurate_ratio": round(accurate_ratio, 4)},
                           note="双手腕点位置轨迹 = 遥操双手"))
    if n_body:
        pkg.add(Stream("extra.body_joints", "observed", source_file=str(track),
                       source_field="Body.joints", note=f"全身骨架 {n_body} 关节,原样引用"))

    fps = round(1.0 / float(np.median(np.diff(ts_rel))), 2) if len(ts_rel) > 2 else 0.0
    pkg.fps = fps

    # 标注(OpenLET 同款字段,HH:MM:SS 时基)
    seg_files = sorted(data.glob("*_segments_description.json"))
    if seg_files:
        raw = json.loads(seg_files[0].read_text())
        segs = [{"id": s.get("id", i + 1),
                 "start_s": _hms_to_s(s["start_timestamp"]),
                 "end_s": _hms_to_s(s["end_timestamp"]),
                 "text": (s.get("action_description") or "").strip(),
                 "text_en": (s.get("action_description_en") or "").strip(),
                 "skill": s.get("atomic_action"),
                 "interacting_hand": s.get("interacting_hand"),
                 "is_noise": bool(s.get("is_noise"))}
                for i, s in enumerate(raw)]
        pkg.add(Stream("annotation.language_segments", "observed" if segs else "missing",
                       data=segs, source_file=str(seg_files[0])))

    cam = data / "camera_params.json"
    if cam.exists():
        pkg.add(Stream("camera.intrinsics", "observed", files=[str(cam)],
                       note="双目内参+equiDis62 畸变模型"))
    vids = sorted(data.glob("CameraRecord_*.mp4")) + \
        sorted((scene_dir / "video").glob("*.m*")) if (scene_dir / "video").exists() \
        else sorted(data.glob("CameraRecord_*.mp4"))
    if vids:
        import cv2
        cap = cv2.VideoCapture(str(vids[0]))
        seq = sum(bool(cap.read()[0]) for _ in range(3))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n_frames // 2))
        seekable = bool(cap.read()[0])
        vfps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        pkg.add(Stream("camera.rgb", "observed", files=[str(v) for v in vids],
                       provenance={"fps": vfps, "frames": n_frames,
                                   "sequential_decodable_of_3": seq, "seekable": seekable}))
    ply = data / "points_aligned.ply"
    if ply.exists():
        pkg.add(Stream("scene.environment", "observed", files=[str(ply)],
                       note="对齐点云,原样引用"))
    return pkg
