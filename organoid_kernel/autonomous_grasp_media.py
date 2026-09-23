"""Derived presentation and learning index, outside sealed transfer evidence."""
import json
from pathlib import Path
import shutil
import subprocess

import imageio_ffmpeg

from .hashing import sha256_file
from .humanoid_sorting_media import render_episode


def build_showcase(root):
    root = Path(root)
    records = json.loads((root / "results.json").read_text())
    assets = root / "media"
    assets.mkdir(exist_ok=True)
    template_root = Path(__file__).resolve().parents[1] / "assets"
    shutil.copy2(template_root / "humanoid_tasks/synchronized_video.js", assets / "synchronized_video.js")
    titles = {"source_baseline": "积木动作基线", "ball32": "32 mm 球", "ball36": "36 mm 球",
              "camera_blackout": "相机失效", "ball50": "50 mm 球"}
    display, learning = [], []
    for world in records:
        attempts = []
        for row in world["attempts"]:
            if row["episode"] is None or row["receipt"] is None:
                attempts.append({"index": row["index"], "status": "not_evaluated",
                                 "reason": row["outcome"]["verification"].get("reason", ""),
                                 "media": None, "origin": row["origin"], "action": row["action"]})
                continue
            path = assets / f"{world['case']}-{row['index']}"
            render_episode(root / row["episode"], path)
            receipt = row["receipt"]
            status = row["outcome"]["verification"]["status"]
            attempts.append({
                "index": row["index"], "status": status, "origin": row["origin"],
                "media": str(path.relative_to(root)), "episode": row["episode"],
                "action": row["action"], "reason": (receipt.get("stop") or {}).get("reason"),
                "steps": receipt["metrics"]["control_steps"], "object": receipt["objects"]["orange"],
                "replay": receipt["replay"]["verified"],
                "failed_checks": [k for k, v in receipt["checks"].items() if not v] if status != "not_evaluated" else [],
            })
            learning.append({
                "scene_family": world["case"], "attempt": row["index"], "episode": row["episode"],
                "origin": row["origin"], "status": status, "binding": row["binding"],
                "success_label": None if status == "not_evaluated" else status == "accepted",
                "success_label_mask": status != "not_evaluated",
                "behavior_cloning_eligible": status == "accepted" and row["origin"] != "independent_confirmation",
                "synthetic": True, "captured_source_prior": world["source_prior"],
                "privileged_channels": ["object_pose", "full_qpos_object_components", "acceptance_labels"],
            })
        display.append({
            "case": world["case"], "title": titles.get(world["case"], world["case"]),
            "status": world["status"], "scene": world["scene"],
            "attempts": attempts, "selected": world["selected_attempt"],
            "generations": world["generations"], "context_sha256": world["context_sha256"],
        })
    prior_path = Path(records[0]["source_prior"])
    prior = json.loads(prior_path.read_text())
    source = root / "source"
    source.mkdir(exist_ok=True)
    for name in ("prior.json", "captured.parquet", "captured-motion.npz", "info.json"):
        shutil.copy2(prior_path.parent / name, source / name)
    shutil.copytree(prior_path.parent / "audit", source / "audit", dirs_exist_ok=True)
    start = prior["frames"]["start"] / prior["source_fps"]
    duration = (prior["frames"]["end"] - prior["frames"]["start"] + 1) / prior["source_fps"]
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-i", prior["source"]["source_video"], "-t", str(duration),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(source / "captured-grasp.mp4"),
    ], check=True)
    payload = {"worlds": display, "source": {
        "task": "搭建简单积木", "episode": prior["source"]["episode"],
        "frames": prior["frames"], "fps": prior["source_fps"],
        "portal_url": prior["source"]["portal_url"],
        "parquet_sha256": prior["source"]["local_sha256"],
        "portal_bytes_match": prior["source"]["local_sha256"] == prior["source"]["portal_sha256"],
    }}
    html = (template_root / "autonomous_grasp/index.html").read_text()
    (root / "index.html").write_text(html.replace("__DATA__", json.dumps(
        payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")), encoding="utf-8")
    (root / "learning-index.json").write_text(json.dumps(learning, indent=2, ensure_ascii=False))
    (root / "presentation-manifest.json").write_text(json.dumps({
        "index_sha256": sha256_file(root / "index.html"),
        "learning_index_sha256": sha256_file(root / "learning-index.json"),
        "videos": len([r for w in display for r in w["attempts"] if r["media"]]) * 4,
        "browser_validation": "not_evaluated_local_file_navigation_policy",
    }, indent=2))
    return payload
