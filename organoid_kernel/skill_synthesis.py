"""Bridge reusable skill graphs to the current synthetic-data backends.

The skill layer describes a task independently of a robot or simulator.  This
module is the small boundary where that description is checked against the
backends that actually exist today.  A graph that has no executable backend is
rejected instead of being turned into a synthetic episode with misleading
metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any

from .hashing import sha256_json
from .skill_catalog import default_skill_library
from .skill_objects import OBJECT_KINDS, ObjectSpec
from .skills import SkillGraph, SkillStep, SkillValidation


GEOMETRY_TO_OBJECT_KIND = {
    "box": "rigid",
    "cylinder": "rigid",
    "sphere": "rigid",
}

GRASP_GRAPH_SEQUENCE = ("grasp", "lift", "transport", "place", "release")
GRASP_BACKEND_ID = "grasp_transplant"


class UnsupportedSkillBackend(ValueError):
    """Raised when a valid skill graph has no physical synthesis backend."""


def normalize_object_kind(kind: str) -> str:
    """Map simulator geometry names to the abstract skill object vocabulary."""
    kind = str(kind)
    if kind in OBJECT_KINDS:
        return kind
    if kind in GEOMETRY_TO_OBJECT_KIND:
        return GEOMETRY_TO_OBJECT_KIND[kind]
    raise ValueError(f"无法把物体类型/几何 {kind!r} 归一化为技能物体类型")


def graph_definition(graph: SkillGraph) -> dict:
    """Return graph content without mutable validation results."""
    return {
        "schema": "organoid-kernel.skill-graph.v1",
        "steps": [step.to_dict() for step in graph.steps],
        "initial_state": sorted(graph.initial_state),
        "metadata": dict(graph.metadata),
    }


def graph_hash(graph: SkillGraph) -> str:
    return sha256_json(graph_definition(graph))


@dataclass
class SkillSynthesisTask:
    graph: SkillGraph
    object_specs: dict[str, ObjectSpec] = field(default_factory=dict)
    backend_id: str = "auto"
    source_episode: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema": "organoid-kernel.skill-synthesis-task.v1",
            "graph": graph_definition(self.graph),
            "object_specs": {
                object_id: spec.to_dict()
                for object_id, spec in sorted(self.object_specs.items())
            },
            "backend_id": self.backend_id,
            "source_episode": self.source_episode,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "SkillSynthesisTask":
        graph_value = value.get("graph", value)
        graph = SkillGraph.from_dict(graph_value)
        normalized_steps = []
        inferred_specs = {}
        for step in graph.steps:
            kind = step.object_kind
            if kind:
                kind = normalize_object_kind(kind)
            normalized_steps.append(replace(step, object_kind=kind))
            if step.object_id and step.object_id not in inferred_specs:
                inferred_specs[step.object_id] = ObjectSpec(
                    step.object_id, kind or "rigid")
        graph = replace(graph, steps=normalized_steps)
        specs = {}
        for object_id, spec in (value.get("object_specs") or {}).items():
            spec_value = dict(spec)
            spec_value["object_id"] = str(object_id)
            spec_value["kind"] = normalize_object_kind(
                spec_value.get("kind", "rigid"))
            specs[str(object_id)] = ObjectSpec.from_dict(spec_value)
        if not specs:
            specs = inferred_specs
        return cls(
            graph=graph,
            object_specs=specs,
            backend_id=str(value.get("backend_id", "auto")),
            source_episode=str(value.get("source_episode", "")),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass
class SynthesisValidation:
    valid: bool
    backend_id: str | None
    graph_hash: str
    skill_sequence: tuple[str, ...]
    graph_validation: SkillValidation
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "backend_id": self.backend_id,
            "graph_hash": self.graph_hash,
            "skill_sequence": list(self.skill_sequence),
            "graph_validation": self.graph_validation.to_dict(),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def default_grasp_task(*, object_id: str = "target",
                       object_kind: str = "rigid",
                       source_episode: str = "") -> SkillSynthesisTask:
    """Build the currently executable pick-and-place task graph."""
    object_kind = normalize_object_kind(object_kind)
    graph = SkillGraph(
        steps=[
            SkillStep("grasp", object_id=object_id,
                      object_kind=object_kind, arm="any"),
            SkillStep("lift", object_id=object_id,
                      object_kind=object_kind, arm="any"),
            SkillStep("transport", object_id=object_id,
                      object_kind=object_kind, arm="any"),
            SkillStep("place", object_id=object_id,
                      object_kind=object_kind, arm="any"),
            SkillStep("release", object_id=object_id,
                      object_kind=object_kind, arm="any"),
        ],
        initial_state={"visible", "aligned", "hand_open",
                       "object_free", "target_reachable"},
        metadata={
            "name": "single-object-pick-and-place",
            "execution_note": (
                "当前 grasp_transplant 后端把整张图映射到一次完整抓取重放；"
                "图中阶段用于约束、分段和收据，不把阶段误报为独立真机验证"
            ),
        },
    )
    return SkillSynthesisTask(
        graph=graph,
        object_specs={object_id: ObjectSpec(object_id, object_kind)},
        backend_id=GRASP_BACKEND_ID,
        source_episode=source_episode,
    )


def bind_object_geometry(task: SkillSynthesisTask, geometry: str, *,
                         object_id: str = "target",
                         material: str | None = None,
                         properties: dict[str, Any] | None = None
                         ) -> SkillSynthesisTask:
    """Bind one simulator geometry to the graph's abstract object."""
    kind = normalize_object_kind(geometry)
    specs = dict(task.object_specs)
    previous = specs.get(object_id)
    merged_properties = dict(previous.properties if previous else {})
    merged_properties["geometry"] = geometry
    if properties:
        merged_properties.update(properties)
    specs[object_id] = ObjectSpec(
        object_id=object_id,
        kind=kind,
        material=material if material is not None else (
            previous.material if previous else None),
        properties=merged_properties,
    )
    steps = [
        replace(step, object_kind=kind)
        if step.object_id == object_id or step.object_id is None else step
        for step in task.graph.steps
    ]
    graph = replace(task.graph, steps=steps)
    return replace(task, graph=graph, object_specs=specs)


def backend_for_graph(graph: SkillGraph) -> str:
    """Resolve a graph to a real backend, fail-closed for unsupported tasks."""
    sequence = tuple(step.skill for step in graph.steps)
    if sequence == GRASP_GRAPH_SEQUENCE:
        return GRASP_BACKEND_ID
    if not sequence:
        raise UnsupportedSkillBackend("技能图为空，不能生成合成 episode")
    unsupported = sorted(set(sequence) - set(GRASP_GRAPH_SEQUENCE))
    if unsupported:
        raise UnsupportedSkillBackend(
            "当前物理合成器未实现技能后端: " + ", ".join(unsupported))
    raise UnsupportedSkillBackend(
        "grasp_transplant 只接受完整序列 "
        + " → ".join(GRASP_GRAPH_SEQUENCE)
        + "，收到 " + " → ".join(sequence))


def validate_synthesis_task(
    task: SkillSynthesisTask,
    *,
    library=None,
) -> SynthesisValidation:
    """Validate graph semantics and backend support before any simulation."""
    library = library or default_skill_library()
    sequence = tuple(step.skill for step in task.graph.steps)
    errors: list[str] = []
    warnings: list[str] = []

    if not task.graph.steps:
        errors.append("技能图为空")

    for step in task.graph.steps:
        if step.object_id and step.object_id not in task.object_specs:
            errors.append(
                f"步骤 {step.skill} 引用了未声明物体: {step.object_id}")

    try:
        backend = backend_for_graph(task.graph)
    except UnsupportedSkillBackend as exc:
        backend = None
        errors.append(str(exc))

    if backend and task.backend_id not in {"", "auto", backend}:
        errors.append(
            f"技能图解析出的后端为 {backend}，但任务声明为 {task.backend_id}")

    graph_validation = task.graph.validate(
        library,
        object_specs=task.object_specs,
    )
    errors.extend(error for error in graph_validation.errors
                  if error not in errors)
    warnings.extend(graph_validation.warnings)

    return SynthesisValidation(
        valid=not errors and graph_validation.valid,
        backend_id=backend,
        graph_hash=graph_hash(task.graph),
        skill_sequence=sequence,
        graph_validation=graph_validation,
        errors=errors,
        warnings=warnings,
    )


def graph_to_segments(
    graph: SkillGraph,
    frame_count: int,
    close_frame: int,
    release_frame: int,
    fps: float,
    *,
    library=None,
) -> list[dict]:
    """Partition one rollout into graph-derived semantic segments.

    The current motion source exposes reliable close/release anchors but no
    per-skill timestamps.  The interior graph steps therefore share the
    close-to-release interval deterministically.  This changes labels and
    metadata only; it does not invent a new trajectory.
    """
    if frame_count <= 0:
        return []
    if not graph.steps:
        raise ValueError("技能图为空，不能生成动作分段")
    library = library or default_skill_library()
    last_frame = frame_count - 1
    close = max(0, min(int(close_frame), last_frame))
    release = max(close, min(int(release_frame), last_frame))
    count = len(graph.steps)

    if count == 1:
        ranges = [(0, last_frame)]
    elif count == 2:
        ranges = [(0, close), (release, last_frame)]
    else:
        interior = count - 2
        boundaries = [
            int(round(close + (release - close) * i / interior))
            for i in range(interior + 1)
        ]
        ranges = [(0, close)]
        ranges.extend((boundaries[i], boundaries[i + 1])
                      for i in range(interior))
        ranges.append((release, last_frame))

    def hms(frame: int) -> str:
        total = max(0.0, float(frame) / float(fps))
        seconds = int(total)
        return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"

    segments = []
    for index, (step, (start, end)) in enumerate(zip(graph.steps, ranges), 1):
        skill = library.get(step.skill)
        description = skill.description if skill else step.skill
        object_suffix = f" ({step.object_id})" if step.object_id else ""
        segments.append({
            "id": index,
            "skill": step.skill,
            "skill_index": index - 1,
            "object_id": step.object_id,
            "object_kind": step.object_kind,
            "arm": step.arm,
            "start_frame": int(start),
            "end_frame": int(end),
            "start_timestamp": hms(start),
            "end_timestamp": hms(end),
            "action_description": description + object_suffix,
            "action_description_en": step.skill,
            "source_segment_id": step.source_segment_id,
            "source_text": step.source_text,
            "source_text_en": step.source_text_en,
            "confidence": step.confidence,
            "semantic_scope": "graph_partition_of_one_rollout",
        })
    return segments


def task_from_json(path: str | Path) -> SkillSynthesisTask:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SkillSynthesisTask.from_dict(payload)


def task_to_json(task: SkillSynthesisTask, path: str | Path,
                 validation: SynthesisValidation | None = None) -> None:
    payload = task.to_dict()
    if validation is not None:
        payload["validation"] = validation.to_dict()
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
