"""Validation Policy(阶段 3):按平台与用途决定必需 claim 与晋级规则。

Policy 不做验证,只回答两个问题:哪些 claim 是这类数据的必需项;
必需项之外的判负/警示如何折算成最终分级(含修复分级)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import ledger as L


@dataclass
class Policy:
    name: str
    required: list = field(default_factory=list)     # 必需 claim:rejected→整体 rejected,
                                                     # not_evaluated→整体 not_evaluable
    optional: list = field(default_factory=list)     # 参考 claim:不影响分级,仅入账
    description: str = ""

    def grade(self, ledger: L.Ledger) -> None:
        """把逐 claim 状态折算成最终分级。修复/警示信息由 planner 事先写进 ledger。"""
        rejected, blocked, errors = [], [], []
        unassessed = []
        for name in self.required:
            st = ledger.status_of(name)
            if st == L.REJECTED:
                rejected.append(name)
            elif st in (L.NOT_EVALUATED, L.INCONCLUSIVE):
                blocked.append(name)
            elif st == L.ERROR:
                errors.append(name)
        if self.name == "source_quality_v1":
            for claim in ledger.claims:
                if claim.name in self.required or claim.name in self.optional:
                    continue
                if claim.status in (L.NOT_EVALUATED, L.INCONCLUSIVE):
                    unassessed.append(claim.name)
        if errors:
            ledger.grade = "error"
            ledger.grade_reason = f"必需检查执行出错: {errors}"
        elif rejected:
            ledger.grade = "rejected"
            ledger.grade_reason = f"必需检查判负: {rejected}"
        elif blocked:
            ledger.grade = "not_evaluable"
            ledger.grade_reason = f"必需检查缺少可信输入: {blocked}"
        elif ledger.repairs:
            ledger.grade = "accepted_repaired"
            ledger.grade_reason = f"修复后通过(修复项: {sorted(ledger.repairs)})"
        elif ledger.warnings:
            ledger.grade = "accepted_with_warnings"
            ledger.grade_reason = f"通过但带警示(警示项: {sorted(ledger.warnings)})"
        elif unassessed:
            ledger.grade = "accepted_with_warnings"
            ledger.warnings["unassessed"] = sorted(set(unassessed))
            ledger.grade_reason = (
                f"{self.name} 范围内通过；未评估模型相关检查: "
                f"{sorted(set(unassessed))}")
        else:
            ledger.grade = "accepted"
            ledger.grade_reason = "必需检查全部通过"


POLICIES = {
    # 自由根双足:保留原 G1/Kuavo 全部门禁口径
    "free_root_biped_v1": Policy(
        name="free_root_biped_v1",
        required=["timebase_integrity", "kinematic_precheck", "world_binding",
                  "foot_ground_contact", "balance", "self_collision",
                  "ground_penetration"],
        optional=["pairing", "motion_language", "annotation_coverage",
                  "visual_mesh_integrity", "sensor_health"],
        description="真机双足,准静态口径;世界绑定审计+物理五项+预检联判"),
    # 固定基机械臂/上身:不要求自由根、脚接触或平衡
    "fixed_base_manipulator_v1": Policy(
        name="fixed_base_manipulator_v1",
        required=["timebase_integrity", "kinematic_precheck", "self_collision"],
        optional=["pairing", "motion_language", "annotation_coverage",
                  "visual_mesh_integrity", "sensor_health"],
        description="固定基座操作;关节+自碰撞,无双足门禁"),
    "mobile_manipulator_v1": Policy(
        name="mobile_manipulator_v1",
        required=["timebase_integrity", "kinematic_precheck", "self_collision"],
        optional=["pairing", "annotation_coverage"],
        description="轮式/移动平台;双足门禁不适用"),
    # UMI:只声明源数据质量,不声明机器人可执行
    "umi_source_v1": Policy(
        name="umi_source_v1",
        required=["timebase_integrity", "end_effector_kinematics"],
        optional=["gripper_channel", "annotation_coverage", "video_integrity"],
        description="UMI 手持源数据;机器人可执行 claim 须经重定向 derived revision"),
    # 未知机器人但有关节流:只声明源数据质量,模型相关检查等 Profile 补齐后解锁
    "source_quality_v1": Policy(
        name="source_quality_v1",
        required=["timebase_integrity"],
        optional=["pairing", "annotation_coverage", "stream_liveness",
                  "sensor_health", "video_integrity"],
        description="身份未核定的机器人数据;跑与模型无关的门禁,不冒充完整验证"),
    # 纯视觉(Ego 等):MuJoCo 缺输入不是数据失败
    "visual_only_v1": Policy(
        name="visual_only_v1",
        required=["video_integrity"],
        optional=["annotation_coverage", "motion_language", "hand_quality",
                  "object_tracking", "hand_object_contact"],
        description="视频/标注数据;物理与运动学 not_applicable;人手视频加手-物-接触深检"),
    # 任务级仿真:机器人+物体+环境联合验证
    "simulation_task_v1": Policy(
        name="simulation_task_v1",
        required=["timebase_integrity", "kinematic_precheck", "self_collision",
                  "object_motion_continuity"],
        optional=["task_contact_plausibility", "annotation_coverage"],
        description="场景状态完整时的任务级验证"),
}


def pick_policy(profile, pkg) -> str:
    """按 Profile 平台类型与可用流选默认 Policy;调用方可显式覆盖。"""
    if profile is None:
        if pkg.has("umi.gripper_pose"):
            return "umi_source_v1"
        if pkg.has("robot.joint_position"):
            return "source_quality_v1"
        return "visual_only_v1"
    if pkg.has("scene.object_pose"):
        return "simulation_task_v1"
    return {"free_root": "free_root_biped_v1",
            "fixed_base": "fixed_base_manipulator_v1",
            "mobile": "mobile_manipulator_v1"}.get(profile.base_type,
                                                   "fixed_base_manipulator_v1")
