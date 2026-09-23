"""Ego / MP4 Adapter(阶段 7):视频本体数据 —— 没有机器人轨迹,不假装有。

走 visual_only_v1 策略:视频完整性 + 标注覆盖;MuJoCo/运动学 not_applicable。
"""
from __future__ import annotations

import json
from pathlib import Path

from ..evidence import EvidencePackage, Stream
from .base import file_record


def load(mp4_path: Path, annotation_json: Path = None) -> EvidencePackage:
    mp4_path = Path(mp4_path)
    pkg = EvidencePackage(episode_id=mp4_path.stem, dataset_format="ego_mp4",
                          raw_files=[file_record(mp4_path)])
    import cv2
    cap = cv2.VideoCapture(str(mp4_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 顺序解码前 3 帧(硬指标)+ 随机寻址中点(索引质量,单独记录)
    sequential = sum(bool(cap.read()[0]) for _ in range(3))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames // 2))
    seekable = bool(cap.read()[0])
    cap.release()
    pkg.fps = float(fps)
    pkg.add(Stream("camera.rgb", "observed", files=[str(mp4_path)],
                   source_file=str(mp4_path),
                   provenance={"fps": fps, "frames": frames, "resolution": [w, h],
                               "sequential_decodable_of_3": sequential,
                               "seekable": seekable}))
    ann = annotation_json or mp4_path.with_suffix(".json")
    if Path(ann).exists():
        data = json.loads(Path(ann).read_text())
        segs = [{"id": i + 1, "start_s": s.get("start_s", 0), "end_s": s.get("end_s", 0),
                 "text": s.get("text", "")} for i, s in enumerate(data.get("segments", []))]
        pkg.add(Stream("annotation.language_segments",
                       "observed" if segs else "missing", data=segs,
                       source_file=str(ann)))
    return pkg
