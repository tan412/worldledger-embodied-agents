#!/usr/bin/env python3
"""Score declared pre-execution features with OOD rejection; never authorize control."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from organoid_kernel.feasibility import score_candidate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    print(json.dumps(score_candidate(args.checkpoint, record["robot"],
                                     record["initial_features"]), indent=2))
