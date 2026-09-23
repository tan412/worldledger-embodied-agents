"""Reusable, state-checked manipulation skills.

This module describes what an atomic skill means and how several skills may be
composed.  It intentionally stops before robot-specific control: a valid graph
is a kinematically and semantically coherent plan, not proof that a particular
robot can execute it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .skill_objects import ObjectSpec, object_skill_compatible


def _facts(value: Iterable[str] | None) -> frozenset[str]:
    return frozenset(str(v) for v in (value or ()))


def _options(value: Iterable[Iterable[str]] | None) -> tuple[frozenset[str], ...]:
    raw = tuple(_facts(option) for option in (value or ()))
    return raw or (frozenset(),)


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    input_states: tuple[str, ...] = ()
    output_states: tuple[str, ...] = ()
    preconditions: tuple[frozenset[str], ...] = (frozenset(),)
    postconditions: frozenset[str] = frozenset()
    clears: frozenset[str] = frozenset()
    sensors: tuple[str, ...] = ()
    control_mode: str = "unspecified"
    object_kinds: tuple[str, ...] = ("*",)
    robot_types: tuple[str, ...] = ("*",)
    evidence_sources: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("技能名称不能为空")
        if not self.preconditions:
            object.__setattr__(self, "preconditions", (frozenset(),))

    def missing_preconditions(self, state: Iterable[str]) -> set[str]:
        current = set(state)
        missing_options = [set(option) - current for option in self.preconditions]
        return min(missing_options, key=lambda missing: (len(missing), sorted(missing)))

    def can_start(self, state: Iterable[str]) -> bool:
        return not self.missing_preconditions(state)

    def supports_object(self, obj: ObjectSpec | str | None) -> bool:
        if obj is None:
            return True
        kind = obj if isinstance(obj, str) else obj.kind
        return object_skill_compatible(kind, self.object_kinds)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "input_states": list(self.input_states),
            "output_states": list(self.output_states),
            "preconditions": [sorted(option) for option in self.preconditions],
            "postconditions": sorted(self.postconditions),
            "clears": sorted(self.clears),
            "sensors": list(self.sensors),
            "control_mode": self.control_mode,
            "object_kinds": list(self.object_kinds),
            "robot_types": list(self.robot_types),
            "evidence_sources": list(self.evidence_sources),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "Skill":
        return cls(
            name=str(value["name"]),
            category=str(value.get("category", "manipulation")),
            input_states=tuple(value.get("input_states") or ()),
            output_states=tuple(value.get("output_states") or ()),
            preconditions=_options(value.get("preconditions")),
            postconditions=_facts(value.get("postconditions")),
            clears=_facts(value.get("clears")),
            sensors=tuple(value.get("sensors") or ()),
            control_mode=str(value.get("control_mode", "unspecified")),
            object_kinds=tuple(value.get("object_kinds") or ("*",)),
            robot_types=tuple(value.get("robot_types") or ("*",)),
            evidence_sources=tuple(value.get("evidence_sources") or ()),
            description=str(value.get("description", "")),
        )


@dataclass
class SkillStep:
    skill: str
    object_id: str | None = None
    object_kind: str | None = None
    arm: str = "any"
    parameters: dict[str, Any] = field(default_factory=dict)
    source_segment_id: int | None = None
    source_text: str = ""
    source_text_en: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "object_id": self.object_id,
            "object_kind": self.object_kind,
            "arm": self.arm,
            "parameters": dict(self.parameters),
            "source_segment_id": self.source_segment_id,
            "source_text": self.source_text,
            "source_text_en": self.source_text_en,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "SkillStep":
        return cls(
            skill=str(value["skill"]),
            object_id=value.get("object_id"),
            object_kind=value.get("object_kind"),
            arm=str(value.get("arm", "any")),
            parameters=dict(value.get("parameters") or {}),
            source_segment_id=value.get("source_segment_id"),
            source_text=str(value.get("source_text", "")),
            source_text_en=str(value.get("source_text_en", "")),
            confidence=float(value.get("confidence", 1.0)),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass
class SkillTransition:
    source_index: int
    target_index: int
    source_skill: str
    target_skill: str
    passed: bool
    missing_preconditions: tuple[str, ...] = ()
    continuity_violations: tuple[str, ...] = ()
    object_compatibility: bool = True
    state_before: tuple[str, ...] = ()
    state_after: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "source_skill": self.source_skill,
            "target_skill": self.target_skill,
            "passed": self.passed,
            "missing_preconditions": list(self.missing_preconditions),
            "continuity_violations": list(self.continuity_violations),
            "object_compatibility": self.object_compatibility,
            "state_before": list(self.state_before),
            "state_after": list(self.state_after),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class SkillValidation:
    valid: bool
    final_state: tuple[str, ...]
    transitions: list[SkillTransition] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "final_state": list(self.final_state),
            "transitions": [t.to_dict() for t in self.transitions],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _continuity_diagnostics(previous: SkillStep, current: SkillStep,
                            position_tol_m: float, velocity_tol: float,
                            acceleration_tol: float) -> tuple[list[str], dict]:
    """Check boundary metadata when both neighboring steps provide it.

    Missing boundary measurements are warnings at this layer.  If both sides
    are present, a discontinuity is an error because blindly concatenating
    such trajectories can create an impulse.
    """
    errors = []
    diagnostics: dict[str, Any] = {}

    pairs = (
        ("end_pose", "start_pose", position_tol_m, "pose_gap_m"),
        ("end_velocity", "start_velocity", velocity_tol, "velocity_gap"),
        ("end_acceleration", "start_acceleration", acceleration_tol,
         "acceleration_gap"),
    )
    for left_key, right_key, tolerance, output_key in pairs:
        left = previous.metadata.get(left_key)
        right = current.metadata.get(right_key)
        if left is None or right is None:
            continue
        try:
            gap = sum((float(a) - float(b)) ** 2
                      for a, b in zip(left, right)) ** 0.5
        except (TypeError, ValueError):
            errors.append(f"{left_key}/{right_key} 格式无效")
            continue
        diagnostics[output_key] = round(gap, 8)
        if gap > tolerance:
            errors.append(f"{output_key}={gap:.6g}>{tolerance:.6g}")

    left_gripper = previous.metadata.get("end_gripper")
    right_gripper = current.metadata.get("start_gripper")
    if left_gripper is not None and right_gripper is not None:
        diagnostics["gripper_gap"] = abs(float(left_gripper) - float(right_gripper))
        if diagnostics["gripper_gap"] > 1e-6:
            errors.append(f"gripper_gap={diagnostics['gripper_gap']:.6g}")

    if previous.arm not in {"any", current.arm} and current.arm != "any":
        # A handover may intentionally use different arms.
        if current.skill != "handover" and previous.skill != "handover":
            errors.append(f"arm_conflict:{previous.arm}->{current.arm}")
    return errors, diagnostics


@dataclass
class SkillGraph:
    steps: list[SkillStep] = field(default_factory=list)
    initial_state: set[str] = field(default_factory=set)
    transitions: list[SkillTransition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: SkillStep) -> "SkillGraph":
        self.steps.append(step)
        return self

    def extend(self, steps: Iterable[SkillStep]) -> "SkillGraph":
        self.steps.extend(steps)
        return self

    def validate(self, library: "SkillLibrary", *,
                 initial_state: Iterable[str] | None = None,
                 object_specs: dict[str, ObjectSpec] | None = None,
                 position_tol_m: float = 0.05,
                 velocity_tol: float = 0.5,
                 acceleration_tol: float = 2.0) -> SkillValidation:
        state = set(self.initial_state if initial_state is None else initial_state)
        specs = object_specs or {}
        transitions: list[SkillTransition] = []
        errors: list[str] = []
        warnings: list[str] = []
        previous: SkillStep | None = None
        previous_state = set(state)

        for index, step in enumerate(self.steps):
            skill = library.get(step.skill)
            if skill is None:
                errors.append(f"未知技能: {step.skill}")
                transitions.append(SkillTransition(
                    index - 1, index, previous.skill if previous else "",
                    step.skill, False, state_before=tuple(sorted(state)),
                    state_after=tuple(sorted(state)),
                    diagnostics={"unknown_skill": True},
                ))
                previous = step
                continue

            obj = (specs.get(step.object_id) if step.object_id else None)
            obj = obj if obj is not None else step.object_kind
            try:
                object_ok = skill.supports_object(obj)
            except ValueError:
                object_ok = False
                errors.append(
                    f"步骤 {index} 使用了未知物体类型: {step.object_kind}")
            arm_ok = (
                skill.control_mode != "dual_arm"
                or step.arm in {"both", "dual", "left+right"}
            )
            if not object_ok:
                errors.append(
                    f"技能 {step.skill} 不支持物体类型 "
                    f"{step.object_kind or getattr(obj, 'kind', None)}")
            if not arm_ok:
                errors.append(
                    f"技能 {step.skill} 需要双末端，当前 arm={step.arm}")

            missing = skill.missing_preconditions(state)
            transition_errors = []
            if missing:
                transition_errors.append(
                    f"缺少前置条件: {sorted(missing)}")
                errors.append(
                    f"步骤 {index}({step.skill}) 缺少前置条件: {sorted(missing)}")

            diagnostics = {}
            if previous is not None:
                continuity, diagnostics = _continuity_diagnostics(
                    previous, step, position_tol_m, velocity_tol, acceleration_tol)
                transition_errors.extend(continuity)
                errors.extend(
                    f"步骤 {index - 1}->{index}: {item}" for item in continuity)

            if not missing and object_ok:
                state.difference_update(skill.clears)
                state.update(skill.postconditions)
            else:
                warnings.append(
                    f"步骤 {index} 未应用后置状态，后续状态可能继续失败")

            if not arm_ok:
                transition_errors.append("dual_arm_requires_two_end_effectors")
            passed = not transition_errors and object_ok and arm_ok
            transitions.append(SkillTransition(
                index - 1, index, previous.skill if previous else "",
                step.skill, passed,
                missing_preconditions=tuple(sorted(missing)),
                continuity_violations=tuple(transition_errors),
                object_compatibility=object_ok,
                state_before=tuple(sorted(previous_state)),
                state_after=tuple(sorted(state)),
                diagnostics=diagnostics,
            ))
            previous_state = set(state)
            previous = step

        self.transitions = transitions
        return SkillValidation(
            valid=not errors,
            final_state=tuple(sorted(state)),
            transitions=transitions,
            errors=errors,
            warnings=warnings,
        )

    def to_dict(self) -> dict:
        return {
            "schema": "organoid-kernel.skill-graph.v1",
            "steps": [step.to_dict() for step in self.steps],
            "initial_state": sorted(self.initial_state),
            "transitions": [transition.to_dict() for transition in self.transitions],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "SkillGraph":
        graph = cls(
            steps=[SkillStep.from_dict(step) for step in value.get("steps") or ()],
            initial_state=set(value.get("initial_state") or ()),
            metadata=dict(value.get("metadata") or {}),
        )
        return graph


class SkillLibrary:
    def __init__(self, skills: Iterable[Skill] | None = None):
        self._skills: dict[str, Skill] = {}
        for skill in skills or ():
            self.register(skill)

    def register(self, skill: Skill) -> Skill:
        if skill.name in self._skills:
            raise ValueError(f"技能重复注册: {skill.name}")
        self._skills[skill.name] = skill
        return skill

    def upsert(self, skill: Skill) -> Skill:
        self._skills[skill.name] = skill
        return skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def require(self, name: str) -> Skill:
        skill = self.get(name)
        if skill is None:
            raise KeyError(f"未知技能: {name}")
        return skill

    def names(self) -> tuple[str, ...]:
        return tuple(self._skills)

    def compose(self, steps: Iterable[SkillStep], *,
                initial_state: Iterable[str] = (),
                object_specs: dict[str, ObjectSpec] | None = None,
                validate: bool = True) -> SkillGraph:
        graph = SkillGraph(list(steps), set(initial_state))
        if validate:
            result = graph.validate(self, object_specs=object_specs)
            if not result.valid:
                raise ValueError("技能组合不合法: " + "; ".join(result.errors))
        return graph

    def to_dict(self) -> dict:
        return {
            "schema": "organoid-kernel.skill-library.v1",
            "skills": [self._skills[name].to_dict() for name in self._skills],
        }

    @classmethod
    def from_dict(cls, value: dict) -> "SkillLibrary":
        return cls(Skill.from_dict(item) for item in value.get("skills") or ())
