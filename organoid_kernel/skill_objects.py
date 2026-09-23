"""Object models used by the reusable skill layer.

The skill layer does not require a full deformable-body simulator.  It keeps
the smallest state representation that is useful for planning and evidence
gating, while preserving the distinction between rigid and deformable objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


OBJECT_KINDS = (
    "rigid",
    "deformable_surface",
    "cloth",
    "bag",
    "rope",
    "soft_tool",
    "articulated",
)


def _tuple_of_floats(value: Any, width: int | None = None) -> tuple[float, ...]:
    if value is None:
        return ()
    result = tuple(float(v) for v in value)
    if width is not None and result and len(result) != width:
        raise ValueError(f"数值状态宽度应为 {width}，收到 {len(result)}")
    return result


@dataclass
class EndEffectorTrajectory:
    """Robot-agnostic end-effector trajectory in a declared coordinate frame."""

    arm: str
    timestamps: tuple[float, ...]
    poses: tuple[tuple[float, ...], ...]
    gripper: tuple[float, ...] = ()
    frame: str = "world"
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.arm not in {"left", "right", "both"}:
            raise ValueError("arm 必须是 left/right/both")
        self.timestamps = tuple(float(t) for t in self.timestamps)
        self.poses = tuple(_tuple_of_floats(p, 7) for p in self.poses)
        self.gripper = tuple(float(v) for v in self.gripper)
        if len(self.timestamps) != len(self.poses):
            raise ValueError("timestamps 与 poses 长度不一致")
        if self.gripper and len(self.gripper) != len(self.poses):
            raise ValueError("gripper 与 poses 长度不一致")
        if any(not math.isfinite(t) for t in self.timestamps):
            raise ValueError("timestamps 必须是有限数")
        if any(b < a for a, b in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("timestamps 必须单调不减")
        if not self.frame:
            raise ValueError("frame 不能为空")

    def validate(self) -> dict:
        quaternion_norms = [
            math.sqrt(sum(v * v for v in pose[3:])) for pose in self.poses
        ]
        bad_quaternions = [
            i for i, norm in enumerate(quaternion_norms)
            if not math.isfinite(norm) or norm < 1e-8
        ]
        dt = [b - a for a, b in zip(self.timestamps, self.timestamps[1:])]
        return {
            "valid": not bad_quaternions,
            "frames": len(self.poses),
            "bad_quaternion_frames": bad_quaternions,
            "duration_s": (self.timestamps[-1] - self.timestamps[0]
                           if self.timestamps else 0.0),
            "dt_min_s": min(dt) if dt else None,
            "dt_max_s": max(dt) if dt else None,
        }

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "timestamps": list(self.timestamps),
            "poses": [list(pose) for pose in self.poses],
            "gripper": list(self.gripper),
            "frame": self.frame,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "EndEffectorTrajectory":
        return cls(
            arm=str(value["arm"]),
            timestamps=tuple(value.get("timestamps") or ()),
            poses=tuple(tuple(pose) for pose in value.get("poses") or ()),
            gripper=tuple(value.get("gripper") or ()),
            frame=str(value.get("frame", "world")),
            source=str(value.get("source", "")),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ObjectSpec:
    """Static object description used for skill compatibility checks."""

    object_id: str
    kind: str = "rigid"
    material: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id 不能为空")
        if self.kind not in OBJECT_KINDS:
            raise ValueError(f"未知物体类型: {self.kind}")

    @property
    def deformable(self) -> bool:
        return self.kind in {
            "deformable_surface", "cloth", "bag", "rope", "soft_tool"
        }

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "kind": self.kind,
            "material": self.material,
            "properties": dict(self.properties),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "ObjectSpec":
        return cls(
            object_id=str(value["object_id"]),
            kind=str(value.get("kind", "rigid")),
            material=value.get("material"),
            properties=dict(value.get("properties") or {}),
        )


@dataclass
class ObjectState:
    """Runtime object state for planning and post-condition checks.

    ``edge_points`` and ``fold_lines`` are deliberately geometric summaries,
    not a replacement for cloth simulation.  They are enough to represent
    flatten/fold/hang/wipe preconditions without inventing hidden dynamics.
    """

    object_id: str
    kind: str = "rigid"
    center: tuple[float, ...] = ()
    pose: tuple[float, ...] = ()
    edge_points: tuple[tuple[float, ...], ...] = ()
    grasp_points: tuple[tuple[float, ...], ...] = ()
    fold_lines: tuple[tuple[float, ...], ...] = ()
    flatten_ratio: float | None = None
    tension: str | None = None
    contact_regions: tuple[str, ...] = ()
    occluded: bool = False
    facts: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in OBJECT_KINDS:
            raise ValueError(f"未知物体类型: {self.kind}")
        self.center = _tuple_of_floats(self.center, 3)
        self.pose = _tuple_of_floats(self.pose)
        self.edge_points = tuple(_tuple_of_floats(p, 3) for p in self.edge_points)
        self.grasp_points = tuple(_tuple_of_floats(p, 3) for p in self.grasp_points)
        self.fold_lines = tuple(_tuple_of_floats(p) for p in self.fold_lines)
        if self.flatten_ratio is not None and not 0.0 <= self.flatten_ratio <= 1.0:
            raise ValueError("flatten_ratio 必须位于 [0, 1]")
        if self.tension is not None and self.tension not in {"slack", "taut", "unknown"}:
            raise ValueError("tension 必须是 slack/taut/unknown")

    @property
    def deformable(self) -> bool:
        return self.kind in {
            "deformable_surface", "cloth", "bag", "rope", "soft_tool"
        }

    def has_fact(self, fact: str) -> bool:
        return fact in self.facts

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "kind": self.kind,
            "center": list(self.center),
            "pose": list(self.pose),
            "edge_points": [list(p) for p in self.edge_points],
            "grasp_points": [list(p) for p in self.grasp_points],
            "fold_lines": [list(p) for p in self.fold_lines],
            "flatten_ratio": self.flatten_ratio,
            "tension": self.tension,
            "contact_regions": list(self.contact_regions),
            "occluded": self.occluded,
            "facts": sorted(self.facts),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "ObjectState":
        return cls(
            object_id=str(value["object_id"]),
            kind=str(value.get("kind", "rigid")),
            center=tuple(value.get("center") or ()),
            pose=tuple(value.get("pose") or ()),
            edge_points=tuple(tuple(p) for p in value.get("edge_points") or ()),
            grasp_points=tuple(tuple(p) for p in value.get("grasp_points") or ()),
            fold_lines=tuple(tuple(p) for p in value.get("fold_lines") or ()),
            flatten_ratio=value.get("flatten_ratio"),
            tension=value.get("tension"),
            contact_regions=tuple(value.get("contact_regions") or ()),
            occluded=bool(value.get("occluded", False)),
            facts=set(value.get("facts") or ()),
            metadata=dict(value.get("metadata") or {}),
        )


def object_skill_compatible(object_kind: str, supported_kinds: tuple[str, ...]) -> bool:
    """Return whether a skill explicitly supports an object kind."""
    if object_kind not in OBJECT_KINDS:
        raise ValueError(f"未知物体类型: {object_kind}")
    return "*" in supported_kinds or object_kind in supported_kinds
