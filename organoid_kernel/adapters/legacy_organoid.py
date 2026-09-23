"""legacy organoid 文件契约 Adapter:episode 产物目录直接进内核。

接受如下目录布局:
  trajectory.csv   Frame,X,Y,Z,QX,QY,QZ,QW,<关节列…>
  segments.json    [{id,start_timestamp,end_timestamp,action_description,…}]
  episode.json     元信息(可选)
  camera.json      相机参数(可选,原样引用)
  hands.jsonl      21 点人手关键点(可选)→ human.hand_landmarks
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..evidence import EvidencePackage, Stream
from .base import file_record


def _hms_to_s(text: str) -> float:
    parts = [float(p) for p in str(text).strip().split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def load(run_dir: Path, fps: float = None) -> EvidencePackage:
    run_dir = Path(run_dir)
    traj = run_dir / "trajectory.csv"
    pkg = EvidencePackage(episode_id=run_dir.name, dataset_format="legacy_organoid",
                          raw_files=[file_record(traj, with_hash=False)])

    import csv
    with traj.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [[float(v) for v in r] for r in reader if r]
    arr = np.asarray(rows)
    cols = {c: i for i, c in enumerate(header)}
    joint_cols = [c for c in header if c not in
                  ("Frame", "X", "Y", "Z", "QX", "QY", "QZ", "QW")]

    meta_path = run_dir / "episode.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rep_path = run_dir / "adapter_report.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    fps = fps or rep.get("frame_grid_hz") or meta.get("fps") or 30.0
    pkg.fps = float(fps)
    pkg.meta.update({k: meta[k] for k in ("task_name", "robot_type") if k in meta})

    frames = arr[:, cols["Frame"]].astype(int).tolist() if "Frame" in cols else None
    ts = (np.asarray(frames, dtype=float) / pkg.fps) if frames is not None else None
    pkg.add(Stream("robot.joint_position", "observed",
                   data=arr[:, [cols[c] for c in joint_cols]], columns=joint_cols,
                   unit="rad", timestamps=ts, source_file=str(traj),
                   source_field=",".join(joint_cols[:3]) + ",…",
                   provenance={"frame_index": frames}))
    if all(c in cols for c in ("X", "Y", "Z", "QX", "QY", "QZ", "QW")):
        pose = arr[:, [cols[c] for c in ("X", "Y", "Z", "QX", "QY", "QZ", "QW")]]
        origin = "observed" if np.isfinite(pose[:, 0]).all() else "missing"
        pkg.add(Stream("robot.base_pose", origin, data=pose,
                       columns=["x", "y", "z", "qx", "qy", "qz", "qw"],
                       timestamps=ts, source_file=str(traj),
                       note="" if origin == "observed" else "根平移为 nan(老契约的不可评估口径)"))

    seg_path = run_dir / "segments.json"
    if seg_path.exists():
        raw = json.loads(seg_path.read_text())
        raw = raw if isinstance(raw, list) else raw.get("segments", [])
        segs = [{"id": s.get("id", i + 1),
                 "start_s": _hms_to_s(s["start_timestamp"]),
                 "end_s": _hms_to_s(s["end_timestamp"]),
                 "text": (s.get("action_description") or "").strip(),
                 "text_en": (s.get("action_description_en") or "").strip(),
                 "is_noise": bool(s.get("is_noise"))} for i, s in enumerate(raw)]
        pkg.add(Stream("annotation.language_segments",
                       "observed" if segs else "missing", data=segs,
                       source_file=str(seg_path)))

    hands = run_dir / "hands.jsonl"
    if hands.exists():
        pts = []
        with hands.open() as fh:
            for line in fh:
                row = json.loads(line)
                flat = []
                for hand in ("left", "right"):
                    for p in (row.get(hand) or []):
                        flat.extend(p[:3])
                pts.append(flat)
        if pts:
            w = max(len(p) for p in pts)
            mat = np.full((len(pts), w), np.nan)
            for i, p in enumerate(pts):
                mat[i, :len(p)] = p
            pkg.add(Stream("human.hand_landmarks", "observed", data=mat,
                           source_file=str(hands), note="21 点人手关键点(老契约)"))
    cam = run_dir / "camera.json"
    if cam.exists():
        pkg.add(Stream("camera.intrinsics", "observed", files=[str(cam)]))
    return pkg
