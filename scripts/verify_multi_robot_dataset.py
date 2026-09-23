#!/usr/bin/env python3
"""Validate exported candidate/label/split evidence; optionally replay every episode."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from organoid_kernel.hashing import sha256_file, sha256_json
from organoid_kernel.multi_robot_tasks import FEATURE_NAMES, replay_episode, write_json
from organoid_kernel.trajectory import read_trajectory


def verify_episode(item):
    root, record, replay = item
    directory = root / record["trajectory"]
    manifest, arrays = read_trajectory(directory)
    receipt = json.loads((directory / "receipt.json").read_text())
    meta = manifest["metadata"]
    assert receipt["trajectory_sha256"] == manifest["sha256"]
    assert record["labels"] == receipt["labels"]
    assert record["status"] == receipt["status"]
    assert record["scene_hash"] == sha256_json(meta["scene"])
    assert record["initial_state_hash"] == sha256_json(arrays["initial_state"].tolist())
    assert meta["feature_names"] == FEATURE_NAMES
    np.testing.assert_array_equal(arrays["initial_features"], record["initial_features"])
    assert len(arrays["qpos"]) == len(arrays["control"]) + 1
    for time in ("state_time_s", "action_time_s"):
        assert np.all(np.diff(arrays[time]) > 0)
    assert meta["scene_sha256"] == sha256_file(directory / "scene.xml")
    assert receipt["labels"]["stable_grasp"] is None
    assert receipt["labels"]["success"] == all(receipt["checks"].values())
    assert receipt["replay"]["verified"]
    if replay:
        assert replay_episode(directory)["verified"]
    return len(arrays["control"])


def verify(root, replay=False, workers=4):
    root = Path(root).resolve()
    records = json.loads((root / "records.json").read_text())
    split = json.loads((root / "split.json").read_text())
    predictions = json.loads((root / "critic-predictions.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert len(records) == summary["episode_records"]
    prediction_by_path = {p["trajectory"]: p for p in predictions}
    groups = defaultdict(list)
    for record in records:
        assert record["status"] != "error"
        assert prediction_by_path[record["trajectory"]]["split"] == split[str(record["family"])]
        groups[(record["family"], record["robot"])].append(record)
    for group in groups.values():
        assert len(group) == 4
        assert len({r["scene_hash"] for r in group}) == 1
        assert len({r["initial_state_hash"] for r in group}) == 1
    for path, digest in json.loads((root / "source-hashes.json").read_text()).items():
        assert sha256_file(root / "source_snapshot" / path) == digest
    with ProcessPoolExecutor(max_workers=workers) as pool:
        steps = list(pool.map(verify_episode, [(root, record, replay) for record in records]))
    report = {
        "schema": "organoid.multi-robot-artifact-verification.v1",
        "ok": True, "episodes": len(records), "control_steps": sum(steps),
        "same_initial_scene_groups": len(groups), "family_split_disjoint": True,
        "source_snapshot_hashes_verified": True,
        "independent_replays_this_verification": len(records) if replay else 0,
    }
    write_json(root / "artifact-verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--replay-all", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset, args.replay_all, args.workers), indent=2))
