#!/usr/bin/env python3
"""Render saved robot state trajectories, with original visual assets."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import imageio_ffmpeg
import mujoco
import numpy as np
from organoid_kernel.trajectory import read_trajectory
from organoid_kernel.multi_robot_tasks import replay_episode


def render(directory, output):
    directory, output = Path(directory), Path(output)
    if not replay_episode(directory)["verified"]:
        raise ValueError("Refusing to render divergent evidence")
    manifest, arrays = read_trajectory(directory)
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [.20, .15, .40]
    camera.distance = 1.75
    camera.azimuth = -55 if "ur5e" in manifest["metadata"]["robot_profile"] else 125
    camera.elevation = -25
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    fps = 25
    indices = np.arange(0, len(arrays["qpos"]), max(1, round(1/fps/manifest["metadata"]["dt_s"])))
    if indices[-1] != len(arrays["qpos"]) - 1:
        indices = np.r_[indices, len(arrays["qpos"]) - 1]
    process = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x720",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], stdin=subprocess.PIPE)
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        try:
            for i in indices:
                data.qpos[:] = arrays["qpos"][i]
                data.qvel[:] = arrays["qvel"][i]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                frame = renderer.render()
                cv2.putText(frame, f"{manifest['metadata']['robot_profile']} | {directory.name}",
                            (18, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (235, 235, 235), 2)
                process.stdin.write(frame.tobytes())
            cv2.imwrite(str(output.with_suffix(".png")), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            process.stdin.close()
            if process.wait():
                raise RuntimeError("Video encoding failed")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.episode, args.out)
