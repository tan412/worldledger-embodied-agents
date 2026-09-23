"""Derived videos and learning views; immutable Kabuki evidence is never edited."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import cv2
import h5py
import imageio_ffmpeg
import mujoco
import numpy as np

from .trajectory import read_trajectory
from .hashing import sha256_file, sha256_json

CAMERAS = ("head", "wrist_right", "wrist_left")


def causal_indices(sample_times, query_times):
    sample_times = np.asarray(sample_times)
    query_times = np.asarray(query_times)
    if (sample_times.ndim != 1 or not len(sample_times)
            or not np.isfinite(sample_times).all()
            or not np.isfinite(query_times).all()
            or np.any(np.diff(sample_times) <= 0)):
        raise ValueError("Sensor clock must be finite and strictly increasing")
    indices = np.searchsorted(sample_times, query_times, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("Observation unavailable before the first sensor sample")
    return indices


def transition_indices(control_steps, stride=50):
    if control_steps <= 0 or stride <= 0:
        raise ValueError("Training episodes require nonempty controls and a positive stride")
    start = np.arange(0, control_steps, stride)
    return start, np.minimum(start + stride, control_steps)


def encode_video(path, frames, size, fps=10):
    path = Path(path)
    process = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
    if code:
        raise RuntimeError(f"Video encoding failed: {path}")


def render_episode(episode, destination):
    episode, destination = Path(episode), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    _, arrays = read_trajectory(episode)
    times = np.arange(max(2, int(np.ceil(arrays["state_time_s"][-1] * 10)) + 1)) / 10
    indices = causal_indices(arrays["sensor_time_s"], times)
    with h5py.File(episode / "sensors.h5") as sensors:
        for camera in CAMERAS:
            images = sensors[camera]["rgb"]
            height, width = images.shape[1:3]
            encode_video(destination / (camera + ".mp4"),
                         (images[int(index)] for index in indices), (width, height))
            cv2.imwrite(str(destination / (camera + ".jpg")),
                        cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))
    model = mujoco.MjModel.from_xml_path(str(episode / "scene.xml"))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.14, 0., .88]
    camera.distance, camera.azimuth, camera.elevation = 2.6, 205, -22
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    renderer = mujoco.Renderer(model, height=720, width=960)
    pose_indices = causal_indices(arrays["state_time_s"], times)
    def frames():
        for frame_number, index in enumerate(pose_indices):
            data.qpos[:] = arrays["qpos"][index]
            data.qvel[:] = arrays["qvel"][index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            image = renderer.render().copy()
            if frame_number == 0:
                cv2.imwrite(str(destination / "robot.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            yield image
    try:
        encode_video(destination / "robot.mp4", frames(), (960, 720))
    finally:
        renderer.close()
    verification = {}
    for name in ("robot", *CAMERAS):
        capture = cv2.VideoCapture(str(destination / (name + ".mp4")))
        decoded, first, last, std = 0, None, None, []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            first = frame if first is None else first
            last = frame
            std.append(float(frame.std()))
            decoded += 1
        capture.release()
        if first is None or decoded != len(times):
            raise ValueError(f"Incomplete video: {name}")
        verification[name] = {"decoded_frames": decoded, "expected_frames": len(times),
                              "pixel_std_min": min(std),
                              "first_last_mean_pixel_change": float(np.abs(
                                  last.astype(float) - first.astype(float)).mean())}
        if name != "head" and min(std) < 5:
            raise ValueError(f"Blank camera/robot video: {name}")
    (destination / "media-check.json").write_text(json.dumps(verification, indent=2))
    return {"duration_s": len(times) / 10, "frames": len(times), "verification": verification}


def export_training(root):
    root = Path(root)
    records = json.loads((root / "results.json").read_text())
    output = root / "training.hdf5"
    if output.exists():
        raise FileExistsError("Refusing to overwrite training export")
    accepted = [r for r in records if r["outcome"]["verification"]["accepted"] is True]
    train, valid, index = [], [], []
    critic_rows = []
    for record in records:
        receipt = record["receipt"] or {}
        status = record["outcome"]["verification"]["status"]
        contacts = receipt.get("metrics", {}).get("forbidden_contacts", {})
        reason = (receipt.get("stop") or {}).get("reason", "")
        labels = {
            "success": None if status == "not_evaluated" else status == "accepted",
            "collision_free": False if contacts else (True if status == "accepted" else None),
            "ik_feasible": False if reason.startswith("ik_tolerance_exceeded") else (
                True if status == "accepted" else None),
            "stable_grasp": True if status == "accepted" else (
                False if reason in ("no_bilateral_grasp", "grasp_lost") else None),
        }
        critic_rows.append({
            "case": record["case"], "candidate": record["candidate"],
            "context_sha256": record["context_sha256"], "binding": record["binding"],
            "episode": record["episode"], "status": status,
            "labels": labels, "label_mask": {key: value is not None for key, value in labels.items()},
            "label_scope": "candidate_under_frozen_controller_not_robot_global_feasibility",
            "reason": reason,
        })
    (root / "critic-labels.json").write_text(json.dumps(critic_rows, indent=2))
    with h5py.File(output, "w") as h5:
        data = h5.create_group("data")
        data.attrs["schema"] = "organoid.kuavo-learning-view.v1"
        data.attrs["format"] = "robomimic-style_demonstrations_custom_environment_registration_required"
        data.attrs["action_semantics"] = "30_joint_and_gripper_reference_endpoints_up_to_100ms_see_action_dt_s"
        data.attrs["origin"] = "simulated_teacher_demonstrations"
        total = 0
        for number, record in enumerate(accepted):
            episode = root / record["episode"]
            manifest, arrays = read_trajectory(episode)
            model = mujoco.MjModel.from_xml_path(str(episode / "scene.xml"))
            start, end = transition_indices(len(arrays["control"]))
            n = len(start)
            image_indices = causal_indices(arrays["sensor_control_index"], start)
            next_indices = causal_indices(arrays["sensor_control_index"], end)
            name = f"demo_{number}"
            demo = data.create_group(name)
            demo.attrs["num_samples"] = n
            demo.attrs["context_sha256"] = record["context_sha256"]
            demo.attrs["trajectory_sha256"] = manifest["sha256"]
            demo.attrs["source_episode"] = record["episode"]
            demo.attrs["reference_names"] = json.dumps(manifest["metadata"]["reference_names"])
            scene = dict(manifest["metadata"]["scene"])
            scene.pop("seed")
            demo.attrs["scene_family_sha256"] = sha256_json(scene)
            demo.create_dataset("actions", data=arrays["joint_reference"][end - 1].astype(np.float32))
            demo.create_dataset("action_dt_s", data=arrays["state_time_s"][end] - arrays["state_time_s"][start])
            demo.create_dataset("state_time_s", data=arrays["state_time_s"][start])
            demo.create_dataset("next_state_time_s", data=arrays["state_time_s"][end])
            done = np.zeros(n, dtype=bool)
            done[-1] = True
            demo.create_dataset("dones", data=done)
            rewards = np.zeros(n, dtype=np.float32)
            rewards[-1] = 1.
            demo.create_dataset("rewards", data=rewards)
            claws = [int(model.joint(f"{side}_f_bar-1_joint").qposadr[0]) for side in ("l", "r")]
            proprio = np.column_stack((arrays["joint_position"], arrays["joint_velocity"],
                                      arrays["qpos"][:, claws])).astype(np.float32)
            order = manifest["metadata"]["policy"]["order"]
            task = np.tile([1., 0.] if order[0] == "orange" else [0., 1.], (n, 1)).astype(np.float32)
            with h5py.File(episode / "sensors.h5") as sensors:
                for group_name, state_indices, visual_indices in (
                    ("obs", start, image_indices), ("next_obs", end, next_indices)):
                    group = demo.create_group(group_name)
                    group.create_dataset("proprio", data=proprio[state_indices])
                    group.create_dataset("task_order", data=task)
                    group.create_dataset("sensor_time_s", data=arrays["sensor_time_s"][visual_indices])
                    group.create_dataset("sensor_age_s", data=arrays["state_time_s"][state_indices]
                                         - arrays["sensor_time_s"][visual_indices])
                    for camera in CAMERAS:
                        source = sensors[camera]["rgb"]
                        destination = group.create_dataset(camera + "_rgb", shape=(n, *source.shape[1:]),
                                                           dtype="uint8", chunks=(1, *source.shape[1:]),
                                                           compression="gzip", compression_opts=2)
                        for i, image_index in enumerate(visual_indices):
                            destination[i] = source[int(image_index)]
            (valid if record["case"] == "shifted" else train).append(name.encode())
            index.append({"demo": name, "case": record["case"], "candidate": record["candidate"],
                          "samples": n, "context_sha256": record["context_sha256"],
                          "scene_family_sha256": demo.attrs["scene_family_sha256"]})
            total += n
        train_families = {r["scene_family_sha256"] for r in index if r["case"] != "shifted"}
        valid_families = {r["scene_family_sha256"] for r in index if r["case"] == "shifted"}
        if train_families & valid_families:
            raise ValueError("Related scene families must not cross the training/validation split")
        h5.create_dataset("mask/train", data=np.asarray(train, dtype="S32"))
        h5.create_dataset("mask/valid", data=np.asarray(valid, dtype="S32"))
        data.attrs["total"] = total
    report = {"schema": "organoid.kuavo-training-export.v1", "episodes": len(accepted),
              "transitions": total, "action_dimensions": 30, "proprio_dimensions": 58,
              "actor_images": list(CAMERAS), "privileged_poses_in_actor_observations": False,
              "action_dt_s": "0.1_except_explicit_final_partial_transition",
              "rgbd_calibration_and_dense_controls": "retained_in_each_source_episode",
              "training_hdf5_sha256": sha256_file(output),
              "dataset": index, "critic_episodes": len(critic_rows),
              "model_trained": False, "learning_gain_evaluated": False,
              "split": "shifted_scene_held_out_all_related_recovery_candidates_in_training",
              "compatibility": "dataset_layout_ready_environment_wrapper_needed_for_closed_loop_training_eval"}
    (root / "training-export.json").write_text(json.dumps(report, indent=2))
    return report


def build_showcase(root):
    root = Path(root)
    records = json.loads((root / "results.json").read_text())
    display = []
    titles = {"nominal": "双臂顺序整理", "shifted": "位置变化 · 蓝色优先",
              "grasp_recovery": "抓取偏差", "blocked": "高挡板", "blackout": "头部相机失效"}
    for record in sorted(records, key=lambda r: (r["case"] != "nominal", r["case"], r["candidate"])):
        if not record["receipt"]:
            continue
        episode = root / record["episode"]
        destination = root / "media" / (record["case"] + "-" + record["candidate"])
        media = render_episode(episode, destination)
        receipt = record["receipt"]
        decisions = json.loads((episode / "decisions.json").read_text())
        title = titles[record["case"]]
        if record["case"] == "grasp_recovery":
            title += " · " + ("反馈重抓" if record["candidate"] == "feedback_retry" else "不重试")
        display.append({
            "title": title, "status": record["outcome"]["verification"]["status"],
            "media": str(destination.relative_to(root)), "episode": record["episode"],
            "duration_s": media["duration_s"], "policy": receipt["policy"],
            "metrics": receipt["metrics"], "objects": receipt["objects"],
            "failed_checks": ([k for k, v in receipt["checks"].items() if not v]
                              if receipt["status"] != "not_evaluated" else []),
            "reason": (receipt.get("stop") or {}).get("reason"),
            "retries": sum(d["phase"] == "retry_grasp" for d in decisions),
            "events": [{"time": d["control_index"] * .002, "phase": d["phase"],
                        "part": d.get("part", "")} for d in decisions],
        })
    html = (Path(__file__).resolve().parents[1] / "assets/humanoid_sorting/index.html").read_text()
    payload = json.dumps(display, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (root / "index.html").write_text(html.replace("__EPISODES__", payload), encoding="utf-8")
    return display
