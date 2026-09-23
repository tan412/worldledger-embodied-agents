"""简智 GenDAS(decode_output)Adapter:VIO 末端位姿 CSV + 多路相机 mp4。

原始 .mcap(专有 protobuf schema,400+ MB/条)不解;厂商自带的 decode_output
已含 robot0_vio_eef_pose.csv 与压缩视频 —— 按 UMI 源数据口径成流。
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..evidence import EvidencePackage, Stream
from .base import file_record


def load(decode_dir: Path) -> EvidencePackage:
    decode_dir = Path(decode_dir)
    pkg = EvidencePackage(episode_id=decode_dir.parent.name or decode_dir.name,
                          dataset_format="gendas_decoded",
                          meta={"vendor": "jianzhi", "container": "mcap(未解,用厂商 decode_output)"})
    eef = decode_dir / "robot0_vio_eef_pose.csv"
    if eef.exists():
        with eef.open() as fh:
            reader = csv.reader(fh)
            header = next(reader)
            rows = [[float(v) for v in r] for r in reader if r]
        arr = np.asarray(rows)
        pkg.raw_files.append(file_record(eef))
        # 首列通常是时间戳(秒或纳秒)
        ts = arr[:, 0]
        if ts.max() > 1e12:
            ts = ts / 1e9
        ts = ts - ts[0]
        pkg.fps = round(1.0 / float(np.median(np.diff(ts))), 2) if len(ts) > 2 else 0.0
        pkg.add(Stream("umi.gripper_pose", "observed", data=arr[:, 1:],
                       columns=header[1:], timestamps=ts, source_file=str(eef),
                       source_field="robot0_vio_eef_pose", unit="m/quat(VIO)"))
    finger = decode_dir / "robot1_finger_eef_pose.csv"
    if finger.exists():
        pkg.add(Stream("extra.finger_eef_pose", "observed", files=[str(finger)],
                       source_file=str(finger)))
    vids = sorted(decode_dir.glob("robot0_sensor_camera*_compressed.mp4"))
    if vids:
        pkg.add(Stream("camera.rgb", "observed", files=[str(v) for v in vids],
                       provenance={"cameras": [v.stem for v in vids]}))
    return pkg
