"""Execution contracts for reusable skills.

An execution contract names the backend that can test a skill and the evidence
it consumes.  It keeps simulation-specific details out of the skill graph while
making the boundary explicit in every produced receipt.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillExecutionContract:
    backend_id: str
    skill: str
    mode: str
    required_streams: tuple[str, ...] = ()
    optional_streams: tuple[str, ...] = ()
    output_artifacts: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def inspect(self, pkg) -> dict:
        missing = [name for name in self.required_streams
                   if not pkg.has(name)]
        optional_missing = [name for name in self.optional_streams
                            if not pkg.has(name)]
        return {
            "backend_id": self.backend_id,
            "skill": self.skill,
            "mode": self.mode,
            "runnable": not missing,
            "missing_required_streams": missing,
            "missing_optional_streams": optional_missing,
            "output_artifacts": list(self.output_artifacts),
            "limitations": list(self.limitations),
        }

    def to_dict(self) -> dict:
        return {
            "backend_id": self.backend_id,
            "skill": self.skill,
            "mode": self.mode,
            "required_streams": list(self.required_streams),
            "optional_streams": list(self.optional_streams),
            "output_artifacts": list(self.output_artifacts),
            "limitations": list(self.limitations),
        }


_CONTRACTS = (
    SkillExecutionContract(
        backend_id="grasp_transplant",
        skill="grasp",
        mode="hybrid_dynamics",
        required_streams=("robot.joint_position", "robot.action"),
        optional_streams=("camera.rgb", "annotation.language_segments",
                          "sensor.force_torque", "sensor.tactile"),
        output_artifacts=("grasp-transplant.json", "action-labels/*.csv",
                          "truth/contact_forces.csv"),
        limitations=(
            "当前后端以显式物体和台面假设在 MuJoCo 中检验接触可行性",
            "weld 版本是仿真诊断；真机等价性必须看无 weld action_closed_loop",
            "视觉-only 或末端-only 数据不能直接进入该后端",
        ),
    ),
)


def execution_contract(backend_id: str) -> SkillExecutionContract:
    for contract in _CONTRACTS:
        if contract.backend_id == backend_id:
            return contract
    raise KeyError(f"未知技能执行后端: {backend_id}")


def execution_contracts() -> tuple[SkillExecutionContract, ...]:
    return _CONTRACTS
