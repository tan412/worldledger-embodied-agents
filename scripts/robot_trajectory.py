#!/usr/bin/env python3
"""Import explicitly named external states or transfer a Cartesian task."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from organoid_kernel.robot_trajectory_io import import_joint_file, transfer_episode
from organoid_kernel.multi_robot_tasks import ROBOTS


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("import")
    ingest.add_argument("source", type=Path)
    ingest.add_argument("--sidecar", type=Path, required=True)
    ingest.add_argument("--out", type=Path, required=True)
    transfer = sub.add_parser("transfer")
    transfer.add_argument("source", type=Path)
    transfer.add_argument("--robot", choices=ROBOTS, required=True)
    transfer.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = (import_joint_file(args.source, args.sidecar, args.out) if args.command == "import"
              else transfer_episode(args.source, args.robot, args.out))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
