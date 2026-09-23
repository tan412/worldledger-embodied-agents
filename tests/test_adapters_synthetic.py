"""阶段 7/8 合成夹具测试:四种格式的 Adapter + 传感器/任务级 Validator。

真实 bag/H5/Zarr 不在盘上 —— 用各自官方库合成最小数据集,验证适配与验证链。
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from organoid_kernel.adapters import base as ad_base
from organoid_kernel.evidence import EvidencePackage, Stream
from organoid_kernel import planner

TMP = Path(tempfile.mkdtemp(prefix="kernel-test-"))
SAMPLE = Path(__file__).resolve().parents[1] / "samples/WL_01_04(整理货架)"


class HDF5Adapter(unittest.TestCase):
    def test_roundtrip(self):
        import h5py
        p = TMP / "arm.h5"
        with h5py.File(p, "w") as f:
            f["qpos"] = np.linspace(0, 0.5, 300).reshape(100, 3)
            f["timestamp"] = np.arange(100) / 30.0 + np.random.RandomState(0).rand(100) * 1e-4
            f["action"] = np.zeros((100, 3))
            f["tactile_raw"] = np.ones((100, 8))
        from organoid_kernel.adapters import hdf5_generic
        pkg = hdf5_generic.load(p, mapping={"joint_names": ["j1", "j2", "j3"]})
        self.assertEqual(ad_base.detect_format(p), "hdf5")
        self.assertTrue(pkg.has("robot.joint_position"))
        self.assertAlmostEqual(pkg.fps, 30.0, delta=0.5)
        self.assertTrue(any(k.startswith("extra.") for k in pkg.streams))  # 未识别字段不丢


@unittest.skipUnless(SAMPLE.exists(), "public release omits the large sample dataset")
class LeRobotStandardStreams(unittest.TestCase):
    def test_joint_velocity_and_effort_are_preserved(self):
        """动力学校准使用标准流，不应再次绕过 adapter 读取 parquet。"""
        from organoid_kernel.adapters import lerobot_v21
        pkg = lerobot_v21.load(SAMPLE, 0)
        self.assertTrue(pkg.has("robot.joint_position"))
        self.assertTrue(pkg.has("robot.joint_velocity"))
        self.assertTrue(pkg.has("robot.joint_effort"))
        q = pkg.get("robot.joint_position")
        for name in ("robot.joint_velocity", "robot.joint_effort"):
            stream = pkg.get(name)
            self.assertEqual(stream.columns, q.columns)
            self.assertEqual(stream.data.shape, q.data.shape)


class UMIZarrAdapter(unittest.TestCase):
    def test_umi_policy_chain(self):
        import zarr
        p = TMP / "umi.zarr"
        root = zarr.open(str(p), mode="w")
        n = 120
        t = np.linspace(0, 4, n)
        pos = np.stack([0.3 * np.sin(t), 0.3 * np.cos(t), 0.2 + 0.05 * t], axis=1)
        root["data/robot0_eef_pos"] = pos
        root["data/robot0_gripper_width"] = 0.04 + 0.02 * np.sin(t)[:, None]
        root["meta/episode_ends"] = np.array([n])
        from organoid_kernel.adapters import umi_zarr
        pkg = umi_zarr.load(p, 0)
        pkg.fps = 30.0
        self.assertTrue(pkg.has("umi.gripper_pose"))
        ledger = planner.run_episode(pkg, TMP / "umi-run", policy_name="umi_source_v1",
                                     validators=["source_media"])
        st = {c.name: c.status for c in ledger.claims}
        self.assertEqual(st["end_effector_kinematics"], "accepted")
        self.assertEqual(st["gripper_channel"], "accepted")
        # UMI 策略绝不产生机器人可执行 claim
        self.assertNotIn("kinematic_precheck", st)


@unittest.skipUnless(SAMPLE.exists(), "public release omits the large sample dataset")
class EgoMP4Adapter(unittest.TestCase):
    def test_visual_only(self):
        mp4 = next((SAMPLE / "videos/chunk-000/observation.images.camera_top").glob("*.mp4"))
        from organoid_kernel.adapters import ego_mp4
        pkg = ego_mp4.load(mp4)
        ledger = planner.run_episode(pkg, TMP / "ego-run", policy_name="visual_only_v1",
                                     validators=["source_media"])
        st = {c.name: c.status for c in ledger.claims}
        self.assertEqual(st["video_integrity"], "accepted")
        self.assertEqual(ledger.grade, "accepted")


class RosbagAdapter(unittest.TestCase):
    def test_synthetic_bag(self):
        try:
            from rosbags.rosbag1 import Writer
            from rosbags.typesys import Stores, get_typestore
        except ImportError:
            self.skipTest("rosbags 写接口不可用")
        store = get_typestore(Stores.ROS1_NOETIC)
        bag = TMP / "mini.bag"
        JointState = store.types["sensor_msgs/msg/JointState"]
        Header = store.types["std_msgs/msg/Header"]
        TimeT = store.types["builtin_interfaces/msg/Time"]
        with Writer(bag) as w:
            conn = w.add_connection("/sensors_data_raw_js", JointState.__msgtype__,
                                    typestore=store)
            for i in range(50):
                t = i / 100.0
                msg = JointState(
                    header=Header(seq=i, stamp=TimeT(sec=int(t), nanosec=int((t % 1) * 1e9)),
                                  frame_id=""),
                    name=[f"j{k}" for k in range(3)],
                    position=np.array([0.01 * i, 0.0, -0.01 * i]),
                    velocity=np.zeros(3), effort=np.zeros(3))
                w.write(conn, int(t * 1e9), store.serialize_ros1(msg, JointState.__msgtype__))
        # 探测格式即可(专有 joint_data 消息类型合成成本高;kuavo 口径由实测 90 条背书)
        self.assertEqual(ad_base.detect_format(bag), "rosbag")


class OpenLETRawAdapter(unittest.TestCase):
    def test_local_raw_samples_keep_controller_and_dex_channels(self):
        """Regression test against the checked-in local OpenLET sample cache."""
        root = Path(__file__).resolve().parents[2] / "let_sample"
        bags = {
            "base": next(root.glob("base/*.bag"), None),
            "dex": next(root.glob("dex/*.bag"), None),
            "body": next(root.glob("body/*.bag"), None),
        }
        if not all(bags.values()):
            self.skipTest("local OpenLET sample cache is not available")
        from organoid_kernel.adapters import rosbag_let

        base = rosbag_let.load(bags["base"])
        self.assertEqual(base.get("robot.joint_position").data.shape[1], 28)
        for name in (
            "robot.joint_velocity", "robot.joint_acceleration",
            "robot.joint_effort", "robot.action", "robot.command_effort",
            "robot.torque_limit", "robot.controller_kp",
            "robot.controller_kd", "robot.control_mode",
        ):
            self.assertTrue(base.has(name), name)
        self.assertEqual(base.get("robot.action").data.shape[1], 28)

        dex = rosbag_let.load(bags["dex"])
        self.assertTrue(dex.has("sensor.force_torque"))
        self.assertEqual(dex.get("sensor.force_torque").data.shape[1], 12)
        self.assertTrue(dex.has("sensor.tactile"))
        self.assertGreaterEqual(dex.get("sensor.tactile").data.shape[1], 720)

        body = rosbag_let.load(bags["body"])
        self.assertEqual(body.get("robot.joint_position").data.shape[1], 29)
        self.assertIn("waist_yaw_joint", body.get("robot.joint_position").columns)
        self.assertTrue(body.has("extra.kuavo.arm_zeros"))
        self.assertTrue(body.has("extra.kuavo.offset"))


class SourceQualityScope(unittest.TestCase):
    def test_unknown_model_is_not_reported_as_full_acceptance(self):
        pkg = EvidencePackage(episode_id="unknown-model", dataset_format="test",
                              fps=30.0)
        pkg.add(Stream("robot.joint_position", "observed",
                       data=np.zeros((5, 1)), columns=["joint_0"],
                       timestamps=np.arange(5) / 30.0))
        ledger = planner.run_episode(pkg, TMP / "unknown-model-run",
                                     validators=["quality", "kinematics"])
        self.assertEqual(ledger.grade, "accepted_with_warnings")
        self.assertIn("kinematic_precheck", ledger.warnings["unassessed"])


@unittest.skipUnless((Path(__file__).resolve().parents[1] / "assets/robots/biped_s200049/biped_s200049.prepped.urdf").exists(), "robot mesh bundle is downloaded separately")
class TaskScene(unittest.TestCase):
    def _pkg(self, teleport=False):
        n, fps = 90, 30.0
        joints = np.zeros((n, 2))
        pkg = EvidencePackage(episode_id="scene-test", dataset_format="test", fps=fps)
        pkg.add(Stream("robot.joint_position", "observed", data=joints,
                       columns=["zarm_l1_joint", "zarm_r1_joint"],
                       timestamps=np.arange(n) / fps))
        obj = np.zeros((n, 7))
        obj[:, 0] = np.linspace(0.3, 0.5, n)     # 物体平滑移动
        obj[:, 6] = 1.0
        if teleport:
            obj[n // 2, 1] = 5.0                  # 瞬移一帧
        pkg.add(Stream("scene.object_pose", "observed", data=obj))
        pkg.add(Stream("annotation.language_segments", "observed",
                       data=[{"id": 1, "start_s": 0.2, "end_s": 2.0, "text": "抓取物体"}]))
        return pkg

    def test_continuity_pass_and_fail(self):
        from organoid_kernel.validators import task_scene
        from organoid_kernel.profile import load_profile
        ctx = {"profile": load_profile("biped_s200049"), "out_dir": TMP}
        _, claims = task_scene.run(self._pkg(), {}, ctx)
        st = {c.name: c.status for c in claims}
        self.assertEqual(st["object_motion_continuity"], "accepted")
        _, claims = task_scene.run(self._pkg(teleport=True), {}, ctx)
        st = {c.name: c.status for c in claims}
        self.assertEqual(st["object_motion_continuity"], "rejected")


class StaticBaseRule(unittest.TestCase):
    def test_refuses_when_walking(self):
        """腿在动 → 派生拒绝;这是派生与伪造的分界线。"""
        from organoid_kernel.inventory import derive_static_base
        n = 100
        pkg = EvidencePackage(episode_id="walk", dataset_format="test", fps=30)
        joints = np.zeros((n, 2))
        joints[:, 0] = np.linspace(0, 0.5, n)     # 腿关节大幅活动
        pkg.add(Stream("robot.joint_position", "observed", data=joints,
                       columns=["leg_l1_joint", "zarm_l1_joint"],
                       timestamps=np.arange(n) / 30))
        imu = np.zeros((n, 7))
        imu[:, 3] = 1.0   # quat_w? columns 顺序按 xyzw:置 w=1 于第 4 列
        imu = np.zeros((n, 4)); imu[:, 3] = 1.0
        imu[:, 0] = 0.001
        acc = np.tile([0.1, 0.0, 9.8], (n, 1))
        data = np.concatenate([imu, acc], axis=1)
        pkg.add(Stream("robot.imu", "observed", data=data,
                       columns=["quat_x", "quat_y", "quat_z", "quat_w",
                                "acc_x", "acc_y", "acc_z"]))
        r = derive_static_base(pkg, ["leg_l1_joint"])
        self.assertFalse(r["applied"])
        self.assertFalse(pkg.has("robot.base_pose"))


class LegacyOrganoidAdapter(unittest.TestCase):
    def test_old_contract_roundtrip(self):
        """老 organoid 的 trajectory.csv + segments.json 有入口,且能走全链分级。"""
        d = TMP / "legacy-ep"
        d.mkdir(exist_ok=True)
        n = 60
        with (d / "trajectory.csv").open("w") as fh:
            fh.write("Frame,X,Y,Z,QX,QY,QZ,QW,zarm_l1_joint,zarm_r1_joint\n")
            for i in range(n):
                fh.write(f"{i},0,0,0.8,0,0,0,1,{0.01*i:.4f},0\n")
        (d / "segments.json").write_text(json.dumps([
            {"id": 1, "start_timestamp": "00:00:00", "end_timestamp": "00:00:02",
             "action_description": "抬左臂"}]))
        from organoid_kernel.adapters import base as ab, legacy_organoid
        self.assertEqual(ab.detect_format(d), "legacy_organoid")
        pkg = legacy_organoid.load(d, fps=30.0)
        self.assertTrue(pkg.has("robot.joint_position"))
        self.assertTrue(pkg.has("robot.base_pose"))
        self.assertEqual(len(pkg.get("annotation.language_segments").data), 1)


class SynthesisVariants(unittest.TestCase):
    def test_deterministic_and_provenance(self):
        """同 seed 逐位相同;变体标记 derived、provenance 记源哈希;原始包不动。"""
        from organoid_kernel import synthesis
        from organoid_kernel.hashing import sha256_array
        pkg = EvidencePackage(episode_id="src", dataset_format="test", fps=30.0)
        arr = np.linspace(0, 1, 120).reshape(60, 2)
        pkg.add(Stream("robot.joint_position", "observed", data=arr.copy(),
                       columns=["a", "b"], timestamps=np.arange(60) / 30.0))
        h0 = sha256_array(arr)
        v1 = synthesis.make_variant(pkg, "noisy", seed=7, joint_noise_rad=0.01)
        v2 = synthesis.make_variant(pkg, "noisy", seed=7, joint_noise_rad=0.01)
        v3 = synthesis.make_variant(pkg, "noisy", seed=8, joint_noise_rad=0.01)
        a1 = np.asarray(v1.get("robot.joint_position").data)
        self.assertEqual(sha256_array(a1),
                         sha256_array(np.asarray(v2.get("robot.joint_position").data)))
        self.assertNotEqual(sha256_array(a1),
                            sha256_array(np.asarray(v3.get("robot.joint_position").data)))
        self.assertEqual(sha256_array(np.asarray(pkg.get("robot.joint_position").data)), h0)
        s = v1.get("robot.joint_position")
        self.assertEqual(s.origin, "derived")
        self.assertEqual(s.provenance["synthesis"]["source_joint_hash"], h0)
        self.assertTrue(v1.episode_id.endswith("@noisy"))

    def test_time_scale(self):
        from organoid_kernel import synthesis
        pkg = EvidencePackage(episode_id="src", dataset_format="test", fps=30.0)
        pkg.add(Stream("robot.joint_position", "observed",
                       data=np.zeros((60, 1)), columns=["a"],
                       timestamps=np.arange(60) / 30.0))
        pkg.add(Stream("annotation.language_segments", "observed",
                       data=[{"id": 1, "start_s": 0.0, "end_s": 1.0, "text": "x"}]))
        v = synthesis.make_variant(pkg, "slow", time_scale=2.0)
        ts = np.asarray(v.get("robot.joint_position").timestamps)
        self.assertAlmostEqual(float(ts[-1]), (59 / 30.0) * 2.0, places=6)
        self.assertAlmostEqual(v.get("annotation.language_segments").data[0]["end_s"], 2.0)
        self.assertAlmostEqual(v.fps, 15.0)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
