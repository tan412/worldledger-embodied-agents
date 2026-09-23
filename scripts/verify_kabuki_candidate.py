#!/usr/bin/env python3
"""Isolated Organoid worker for the Kabuki robot-trajectory verification gate."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from organoid_kernel.kabuki_validation import verify_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    result = verify_candidate(request["context"], request["candidate"],
                              args.out, request["run_id"])
    print(json.dumps({"status": result["status"], "run_id": result["run_id"]}))


if __name__ == "__main__":
    main()
