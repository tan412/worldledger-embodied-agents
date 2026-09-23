#!/usr/bin/env python3
"""Exercise the actual Kabuki HTTP service and retain all three outcomes."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from organoid_kernel.hashing import sha256_file
from organoid_kernel.multi_robot_tasks import sample_scenes, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kabuki", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "runs_multi_robot_480_v1/feasibility_critic.pt")
    parser.add_argument("--families", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    if not 1 <= args.families <= 60:
        parser.error("--families must be 1..60")
    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory must be new or empty")
    args.out.mkdir(parents=True, exist_ok=True)
    kabuki = args.kabuki.resolve()
    # Keep the socket open until uvicorn inherits it, avoiding a port-picking race.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    base = f"http://127.0.0.1:{listener.getsockname()[1]}"
    env = dict(os.environ, KABUKI_PATH=str(kabuki), ORGANOID_KERNEL=str(ROOT),
               ORGANOID_CRITIC=str(args.checkpoint.resolve()),
               KABUKI_WORLDS=str(args.out / "worlds"))
    outcomes, worlds, transcript = [], [], []

    def api(method, path, body=None):
        request = urllib.request.Request(
            base + path, method=method,
            data=json.dumps(body, allow_nan=False).encode() if body is not None else None,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.loads(response.read())
        transcript.append({"method": method, "path": path, "request": body, "response": result})
        write_json(args.out / "http-transcript.json", transcript)
        return result

    with (args.out / "server.log").open("w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--app-dir",
             str(kabuki / "mcp/service"), "--fd", str(listener.fileno())],
            env=env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(listener.fileno(),))
        listener.close()
        try:
            for _ in range(120):
                if server.poll() is not None:
                    raise RuntimeError("Kabuki service exited; inspect server.log")
                try:
                    urllib.request.urlopen(base + "/openapi.json", timeout=.5).close()
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(.1)
            else:
                raise RuntimeError("Kabuki service did not become ready")
            for spec in sample_scenes(args.families, args.seed):
                world = api("POST", "/worlds/load_robot",
                            {"scene_spec": asdict(spec), "label": f"{spec.robot}-{spec.family}"})
                prefix = "/worlds/" + world["world_id"]
                ranking = api("POST", prefix + "/rank", {"candidates": world["candidates"]})
                assert ranking["execution_authorized"] is False
                for row in ranking["ranking"]:
                    before = api("GET", prefix + "/observe")["state_hash"]
                    pending = api("POST", prefix + "/act", {"patches": [
                        {"op": "replace", "path": "/candidate", "value": row["candidate"]}]})
                    assert pending["status"] == "pending_verification"
                    assert api("GET", prefix + "/observe")["state_hash"] == before
                    result = api("POST", prefix + "/verify", {"mode": "robot_trajectory"})
                    if result["verification"]["accepted"] is not True:
                        assert result["state_hash"] == before
                    outcomes.append({"world_id": world["world_id"], "robot": spec.robot,
                                     "task": spec.task, "candidate": row["candidate"]["action"]["name"],
                                     "critic_score": row["score"], "critic_status": row["critic"]["status"],
                                     "binding": pending["binding"], **result})
                    print(spec.robot, spec.task, row["candidate"]["action"]["name"],
                          result["outcome"], flush=True)
                # Each world retains an unsupported-context attempt and a setup bypass attempt.
                unknown = api("POST", prefix + "/act", {"patches": [
                    {"op": "replace", "path": "/context/scene", "value": {}}]})
                assert unknown["outcome"] == "not_evaluated_kept_evidence"
                api("POST", prefix + "/act", {"patches": [
                    {"op": "replace", "path": "/candidate", "value": world["candidates"][0]}]})
                setup = api("POST", prefix + "/verify", {"mode": "setup"})
                assert setup["outcome"] == "not_evaluated_kept_evidence"
                trace = api("GET", prefix + "/trace")
                replay = api("POST", prefix + "/replay")
                assert replay["matches_live_state"] and replay["evidence_hashes_checked"]
                export = args.out / world["world_id"]
                export.mkdir()
                write_json(export / "trace.json", trace)
                write_json(export / "replay.json", replay)
                for status in ("accepted", "rejected", "not_evaluated"):
                    with (export / f"{status}.jsonl").open("w") as handle:
                        for record in trace[status]:
                            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                worlds.append({"world_id": world["world_id"], "robot": spec.robot, "task": spec.task,
                               "counts": {s: len(trace[s]) for s in (
                                   "accepted", "rejected", "not_evaluated")}, "replay": replay})
                write_json(args.out / "outcomes.json", outcomes)
                write_json(args.out / "worlds.json", worlds)
            counts = Counter()
            for world in worlds:
                counts.update(world["counts"])
            summary = {
                "schema": "kabuki.robot-integration-demo.v1", "world_count": len(worlds),
                "counts": dict(counts), "fresh_candidate_attempts": len(outcomes),
                "all_world_replays_match": all(w["replay"]["matches_live_state"] for w in worlds),
                "critic_authorizes_execution": False, "hardware_executed": False,
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "seed": args.seed, "purpose": "integration_test_not_heldout_generalization_evaluation",
                "sources": {str(p): sha256_file(p) for p in (
                    ROOT / "organoid_kernel/kabuki_validation.py",
                    ROOT / "scripts/verify_kabuki_candidate.py",
                    kabuki / "kabuki/robot_control.py",
                    kabuki / "mcp/service/app.py", kabuki / "mcp/shell/server.js")},
                "worlds": worlds,
            }
            write_json(args.out / "summary.json", summary)
            print(json.dumps({k: v for k, v in summary.items() if k not in ("sources", "worlds")}, indent=2))
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
