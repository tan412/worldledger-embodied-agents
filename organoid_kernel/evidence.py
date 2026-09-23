"""Canonical Evidence Package(阶段 2):类型化流 + provenance,原数据只引用不改写。

一个流(Stream)= 一段带来源与等级的证据:
  origin: observed(源字段原样)| derived(按显式规则派生,带条件与上界)|
          assumed(外部假设)  | missing(缺失或占位)
Adapter 只做事实提取与无损映射;所有裁决在 Validator/Policy。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .hashing import sha256_array, sha256_file

# 标准流名(introduction.md §4);未识别字段以 extra.* 原样挂载,不丢弃
KNOWN_STREAMS = {
    "robot.joint_position", "robot.joint_velocity", "robot.joint_effort",
    "robot.joint_acceleration", "robot.action_velocity", "robot.command_effort",
    "robot.torque_limit", "robot.torque_ratio", "robot.controller_kp",
    "robot.controller_kd", "robot.control_mode",
    "robot.base_pose", "robot.base_velocity", "robot.imu",
    "robot.end_effector_pose", "robot.action", "robot.link_offsets",
    "camera.rgb", "camera.depth", "camera.intrinsics", "camera.extrinsics",
    "sensor.force_torque", "sensor.tactile", "sensor.audio",
    "annotation.language_segments", "annotation.success", "annotation.intervention",
    "scene.object_pose", "scene.environment",
    "human.hand_landmarks", "umi.gripper_pose", "umi.gripper_width",
    "uav.pose", "uav.velocity", "uav.action", "uav.gimbal_state",
    "uav.camera_state", "scene.target_truth", "scene.keepout",
    "mission.events", "mission.result", "agent.plan", "agent.trace",
}


@dataclass
class Stream:
    name: str                       # 标准流名或 extra.<原字段>
    origin: str                     # observed / derived / assumed / missing
    data: object = None             # numpy 数组、dict、或 None(文件型证据)
    columns: list = field(default_factory=list)   # 数组列名(如关节名)
    unit: str = ""                  # rad / m / s / raw ...
    frame: str = ""                 # 坐标系说明
    timestamps: object = None       # 每行时刻(秒)或 None
    source_file: str = ""           # 来源文件
    source_field: str = ""          # 来源字段路径
    files: list = field(default_factory=list)     # 文件型证据(视频等)的路径列表
    provenance: dict = field(default_factory=dict)  # 变换、插值、派生规则、条件实测值与上界
    note: str = ""

    def content_hash(self) -> str:
        if self.data is not None and hasattr(self.data, "tobytes"):
            return sha256_array(self.data)
        if self.files:
            return sha256_file(Path(self.files[0]))
        return ""

    def summary(self) -> dict:
        shape = None
        if self.data is not None and hasattr(self.data, "shape"):
            shape = list(self.data.shape)
        return {
            "name": self.name, "origin": self.origin, "shape": shape,
            "columns": self.columns[:64] or None, "unit": self.unit or None,
            "frame": self.frame or None,
            "source_file": self.source_file or None,
            "source_field": self.source_field or None,
            "files": [str(f) for f in self.files] or None,
            "provenance": self.provenance or None, "note": self.note or None,
        }


@dataclass
class EvidencePackage:
    """一条 episode 的全部证据:流 + 原始文件清单 + 元信息。原始文件不可变。"""
    episode_id: str
    dataset_format: str                       # lerobot_v2.1 / rosbag / hdf5 / umi_zarr / ego_mp4
    fps: float = 0.0
    raw_files: list = field(default_factory=list)     # [{path,bytes,sha256}]
    streams: dict = field(default_factory=dict)       # name -> Stream
    meta: dict = field(default_factory=dict)          # 任务名/场景/设备等事实字段
    adapter_notes: list = field(default_factory=list)  # 映射决策、单位换算、字段丢失记录

    def add(self, stream: Stream) -> None:
        self.streams[stream.name] = stream

    def has(self, name: str, min_origin: str = "derived") -> bool:
        """流存在且证据等级达标(observed 恒达标;derived 在 min_origin=observed 时不算)。"""
        s = self.streams.get(name)
        if s is None or s.origin == "missing":
            return False
        if min_origin == "observed":
            return s.origin == "observed"
        return s.origin in ("observed", "derived")

    def get(self, name: str):
        return self.streams.get(name)

    def summary(self) -> dict:
        return {
            "schema": "organoid-kernel.evidence.v1",
            "episode_id": self.episode_id,
            "dataset_format": self.dataset_format,
            "fps": self.fps,
            "raw_files": self.raw_files,
            "meta": self.meta,
            "streams": {k: v.summary() for k, v in sorted(self.streams.items())},
            "adapter_notes": self.adapter_notes,
        }
