"""物理底座契约回归:解析字段、世界纯函数和 MuJoCo 参数不得丢失。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from organoid_kernel import mjworld  # noqa: E402
from organoid_kernel.fk import load_urdf  # noqa: E402
from organoid_kernel.physics_rules import (DynamicMonitor, dynamic_rule_failures,
                                           dynamic_rules)  # noqa: E402
from organoid_kernel.profile import load_profile  # noqa: E402


URDF = ROOT / "assets/robots/biped_s200049/biped_s200049.prepped.urdf"


def test_urdf_parser_keeps_dynamics_and_inertial_frame(tmp_path):
    path = tmp_path / "tiny.urdf"
    path.write_text(
        """<robot name="tiny">
        <link name="base"/>
        <link name="tip">
          <inertial>
            <origin xyz="1 2 3" rpy="0.1 0.2 0.3"/>
            <mass value="2"/>
            <inertia ixx="1" iyy="2" izz="3" ixy="0.1" ixz="0.2" iyz="0.3"/>
          </inertial>
        </link>
        <joint name="j" type="revolute">
          <parent link="base"/><child link="tip"/>
          <limit lower="-1" upper="2" effort="7" velocity="8"/>
          <dynamics damping="0.4" friction="0.05"/>
        </joint>
        </robot>""",
        encoding="utf-8",
    )
    model = load_urdf(path)
    joint = model.joints[0]
    assert (joint.effort, joint.damping, joint.friction) == (7.0, 0.4, 0.05)
    assert model.links["tip"].inertia_rpy == (0.1, 0.2, 0.3)


def test_dynamic_contract_fails_closed_on_nonfinite_and_speed():
    rules = {"root_drop_m": 0.25, "tilt_deg": 25.0,
             "root_drift_m": 0.5, "velocity_limit_ratio": 1.0}
    velocity = {"j": {"violated": True}}
    assert dynamic_rule_failures(0.0, 0.0, 0.0, velocity, rules,
                                 finite_state=False) == [
                                     "nonfinite_state", "joint_velocity"]


@pytest.mark.skipif(not URDF.exists(), reason="缺机器人资产")
def test_fixed_root_monitor_does_not_apply_free_root_thresholds():
    mujoco = pytest.importorskip("mujoco")
    profile = load_profile("biped_s200049")
    model_u = load_urdf(profile.urdf_path(), profile.mesh_path())
    info = mjworld.build_world(model_u, profile, -3.0, free_root=False,
                               full_body_actuators=True)
    model = mjworld.compile_model(info)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    monitor = DynamicMonitor(
        model_u, model, data, ["leg_l1_joint"], 999.0, [999.0, 999.0],
        dynamic_rules(profile), root_mode="fixed_root")
    sample = monitor.observe(0.0)

    assert sample["root_drop_m"] == 0.0
    assert sample["tilt_deg"] == 0.0
    assert sample["root_drift_m"] == 0.0
    assert monitor.invalid_at_s is None


@pytest.mark.skipif(not URDF.exists(), reason="缺机器人资产")
def test_claw_variant_does_not_mutate_parsed_model():
    profile = load_profile("biped_s200049")
    model = load_urdf(profile.urdf_path(), profile.mesh_path())
    original_types = {j.name: j.type for j in model.joints}

    plain = mjworld.build_world(model, profile, -3.0, free_root=False,
                                 claw_collision=False)
    claw = mjworld.build_world(model, profile, -3.0, free_root=False,
                               claw_collision=True)
    plain_again = mjworld.build_world(model, profile, -3.0, free_root=False,
                                      claw_collision=False)

    assert len(plain.joint_order) == 36
    assert len(claw.joint_order) == 40
    assert len(plain_again.joint_order) == 36
    assert {j.name: j.type for j in model.joints} == original_types


@pytest.mark.skipif(not URDF.exists(), reason="缺机器人资产")
def test_mujoco_keeps_urdf_dynamics_and_caps_actuator_effort():
    mujoco = pytest.importorskip("mujoco")
    profile = load_profile("biped_s200049")
    model_u = load_urdf(profile.urdf_path(), profile.mesh_path())
    info = mjworld.build_world(model_u, profile, -3.0, free_root=False,
                                full_body_actuators=True)
    model = mjworld.compile_model(info)

    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "leg_l5_joint")
    dof = model.jnt_dofadr[joint_id]
    assert model.dof_damping[dof] == pytest.approx(0.2)
    assert model.dof_frictionloss[dof] == pytest.approx(0.0)

    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fb::leg_l5_joint")
    assert model.actuator_forcerange[actuator_id].tolist() == pytest.approx(
        [-36.0, 36.0])
