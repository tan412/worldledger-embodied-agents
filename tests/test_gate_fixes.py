"""阶段 0 回归:三处上游门禁缺陷在新内核中的原生修复,各一个用例钉死。

  #1 预检收据缺失时,物理联判必须 not_evaluated 且写明原因 —— 不许静默移除该检查;
  #2 配对时长必须取时间戳墙钟跨度 —— 丢帧不得被误判成时长偏差;
  #3 重复帧冲突量必须对四元数做 q/−q 半球归一 —— 同一姿态不得虚报冲突。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from organoid_kernel.evidence import EvidencePackage, Stream
from organoid_kernel.validators import pairing as pairing_v
from organoid_kernel.validators import kinematics as kin_v
from organoid_kernel.profile import load_profile


def make_pkg(n=100, fps=50.0, drop_ratio=0.0, dup_quat_flip=False):
    """合成 episode:可注入丢帧与 q/−q 重复帧。"""
    keep = np.arange(n)
    if drop_ratio:
        rng = np.random.RandomState(7)
        keep = np.sort(rng.choice(n, int(n * (1 - drop_ratio)), replace=False))
    ts = keep / fps
    m = len(keep)
    joints = np.zeros((m, 2))
    joints[:, 0] = np.linspace(0, 0.3, m)
    pkg = EvidencePackage(episode_id="synthetic", dataset_format="test", fps=fps)
    pkg.add(Stream("robot.joint_position", "observed", data=joints,
                   columns=["zarm_l1_joint", "zarm_r1_joint"], timestamps=ts,
                   provenance={"frame_index": keep.tolist()}))
    base = np.zeros((m, 7))
    base[:, 6] = 1.0
    if dup_quat_flip:
        # 末帧与首帧同帧号,姿态为 −q(同一姿态的另一半球表示)
        pkg.streams["robot.joint_position"].provenance["frame_index"][-1] = int(keep[0])
        joints[-1] = joints[0]
        base[-1, 3:7] = -base[0, 3:7]
    pkg.add(Stream("robot.base_pose", "observed", data=base,
                   columns=["x", "y", "z", "qx", "qy", "qz", "qw"], timestamps=ts))
    segs = [{"id": 1, "start_s": 0.0, "end_s": (n - 1) / fps, "text": "动作"}]
    pkg.add(Stream("annotation.language_segments", "observed", data=segs))
    return pkg


@unittest.skipUnless((Path(__file__).resolve().parents[1] / "assets/robots/biped_s200049/biped_s200049.prepped.urdf").exists(), "robot mesh bundle is downloaded separately")
class GateFixes(unittest.TestCase):
    def test_fix1_missing_precheck_not_silently_dropped(self):
        """物理联判缺预检收据 → kinematic_precheck_gate 必须 not_evaluated 且给原因。"""
        from organoid_kernel.validators import physics_mujoco
        pkg = make_pkg()
        ctx = {"profile": load_profile("biped_s200049"), "receipts": {}}   # 无预检收据
        receipt, claims = physics_mujoco.run(pkg, {}, ctx)
        gate = {c.name: c for c in claims}["kinematic_precheck_gate"]
        self.assertEqual(gate.status, "not_evaluated")
        self.assertIn("拒绝联判", gate.reason)

    def test_fix2_pairing_uses_wall_clock(self):
        """丢 20% 帧:墙钟跨度不变,配对必须仍然通过(帧数÷fps 的旧算法会假判负)。"""
        pkg = make_pkg(n=200, fps=50.0, drop_ratio=0.2)
        receipt, claims = pairing_v.run(pkg, {"timebase": {"synthetic": False}}, {})
        self.assertEqual(claims[0].status, "accepted")
        self.assertEqual(receipt["trajectory_duration_source"], "wall_clock")
        # 反证:帧数÷fps 会短 20%,超 0.12 容差
        frames_based = len(pkg.get("robot.joint_position").data) / 50.0
        wall = receipt["durations_sec"]["trajectory"]
        self.assertGreater(abs(frames_based - wall) / wall, pairing_v.TOLERANCE)

    def test_fix3_quaternion_sign_normalized_in_duplicates(self):
        """q 与 −q 是同一姿态:重复帧冲突量必须 ≈ 0,不得虚报 ~2.0。"""
        pkg = make_pkg(dup_quat_flip=True)
        ctx = {"profile": load_profile("biped_s200049")}
        receipt, claims = kin_v.run(pkg, {}, ctx)
        self.assertFalse(receipt["checks"]["unique_source_frames"])   # 重复帧如实上报
        self.assertLess(receipt["duplicate_frame_conflict_rad"], 1e-6)  # 但冲突量为零


if __name__ == "__main__":
    unittest.main(verbosity=2)
