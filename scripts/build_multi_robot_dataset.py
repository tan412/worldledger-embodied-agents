#!/usr/bin/env python3
"""Generate paired multi-robot task candidates, replay them, then train critic."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from organoid_kernel.multi_robot_tasks import (
    ROBOTS, TASKS, evaluate_scene, sample_scenes, write_json, replay_episode,
)
from organoid_kernel.hashing import sha256_file


def summarize(records):
    by_robot_task = {}
    for robot in ROBOTS:
        for task in TASKS:
            subset = [r for r in records if r["robot"] == robot and r["task"] == task]
            by_robot_task[f"{robot}/{task}"] = dict(Counter(r["status"] for r in subset))
    groups = defaultdict(list)
    for record in records:
        if "candidate" in record:
            groups[(record["family"], record["robot"])].append(record)
    corrections = []
    for key, group in groups.items():
        original = next(r for r in group if r["candidate"] == "direct")
        valid = [r for r in group if r["labels"]["success"] is True and r["replay"]["verified"]]
        if original["labels"]["success"] is False and valid:
            chosen = valid[0]
            assert original["initial_state_hash"] == chosen["initial_state_hash"]
            assert original["scene_hash"] == chosen["scene_hash"]
            corrections.append({"family": key[0], "robot": key[1], "task": chosen["task"],
                                "original": original["trajectory"], "accepted": chosen["trajectory"],
                                "scope": "same-world procedural candidate correction",
                                "selection": "first physically accepted candidate in fixed order",
                                "scene_hash": chosen["scene_hash"]})
    return {
        "schema": "organoid.multi-robot-dataset.v1", "episode_records": len(records),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "by_robot_task": by_robot_task,
        "replay_verified": sum(r.get("replay", {}).get("verified", False) for r in records),
        "label_counts": {
            head: {state: sum(
                (r["labels"][head] is None if state == "not_evaluated" else
                 r["labels"][head] is (state == "positive")) for r in records)
                   for state in ("positive", "negative", "not_evaluated")}
            for head in ("success", "collision_free", "ik_feasible", "stable_grasp")},
        "correction_pairs": corrections,
        "hardware_test": "not_evaluated",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=ROOT / "runs_multi_robot_v1")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    if args.replay:
        print(replay_episode(args.replay))
        return
    if args.families < 1 or args.workers < 1:
        parser.error("families and workers must be positive")
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    scenes = sample_scenes(args.families, args.seed)
    source_files = [
        "organoid_kernel/multi_robot_tasks.py", "organoid_kernel/trajectory.py",
        "organoid_kernel/feasibility.py", "organoid_kernel/profile.py",
        "scripts/build_multi_robot_dataset.py",
        *[f"profiles/{robot}.json" for robot in ROBOTS],
    ]
    for relative in source_files:
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    write_json(output / "source-hashes.json", {name: sha256_file(ROOT / name) for name in source_files})
    write_json(output / "protocol.json", {
        "seed": args.seed, "family_count": args.families, "profiles": list(ROBOTS),
        "tasks": list(TASKS), "candidate_count_per_scene": 4, "dt_s": .002,
        "all_candidates_retained": True,
        "randomized": ["initial_joints", "goal_xyz", "obstacle_xyz", "obstacle_size", "control_noise"],
        "candidate_variants": ["direct", "raised", "fast", "high_clearance"],
        "collision_geometry": "pinned source model collision geometry retained",
        "hardware": "not_evaluated",
    })
    write_json(output / "scene-specs.json", [asdict(s) for s in scenes])
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_scene, s, output) for s in scenes]
        for i, future in enumerate(as_completed(futures), 1):
            batch = future.result()
            records.extend(batch)
            write_json(output / "progress.json", {"scenes_finished": i, "scenes_total": len(scenes),
                                                 "status_counts": dict(Counter(r["status"] for r in records))})
            print(f"[{i}/{len(scenes)} scenes] {dict(Counter(r['status'] for r in records))}", flush=True)
    records.sort(key=lambda r: (r["family"], r["robot"], r.get("candidate", "")))
    write_json(output / "records.json", records)
    summary = summarize(records)
    write_json(output / "summary.json", summary)
    if not args.skip_training:
        from organoid_kernel.feasibility import train_critic
        evaluation = train_critic(records, output, args.seed)
        print("Test ranking:", evaluation["test_ranking"])
    print("Episodes:", len(records), "Statuses:", summary["status_counts"],
          "Verified replay:", summary["replay_verified"],
          "Correction pairs:", len(summary["correction_pairs"]))


if __name__ == "__main__":
    main()
