#!/usr/bin/env python3
"""Fetch a pinned, licensed subset of MuJoCo Menagerie, with file hashes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REVISION = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
MODELS = {"franka_emika_panda": "panda.xml", "universal_robots_ur5e": "ur5e.xml"}
BASE = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--max-time", "90",
                        "--silent", "--show-error", url, "-o", str(temporary)], check=True)
        if not temporary.stat().st_size:
            raise ValueError(f"Empty source: {url}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def install(name):
    directory = ROOT / "assets/robots" / name
    manifest = directory / "source-manifest.json"
    if manifest.exists():
        record = json.loads(manifest.read_text())
        if record["revision"] != REVISION:
            raise ValueError("Existing model revision differs")
        if all((directory / p).is_file() and digest(directory / p) == value
               for p, value in record["sha256"].items()):
            print(f"{name}: verified existing assets", flush=True)
            return
        raise ValueError("Existing assets differ from manifest; refusing overwrite")
    model = MODELS[name]
    prefix = f"{BASE}/{REVISION}/{name}"
    fetch(f"{prefix}/{model}", directory / model)
    tree = ET.parse(directory / model)
    resources = [model, "LICENSE", "README.md"]
    resources += ["assets/" + mesh.get("file") for mesh in tree.findall("./asset/mesh")]
    def download(resource):
        if resource != model:
            fetch(f"{prefix}/{resource}", directory / resource)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(download, resources))
    record = {
        "repository": "https://github.com/google-deepmind/mujoco_menagerie",
        "revision": REVISION, "model": model,
        "sha256": {p: digest(directory / p) for p in sorted(resources)},
        "license_file": "LICENSE", "hardware_calibration": "not_evaluated",
    }
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{name}: installed {len(resources)} pinned files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=[*MODELS, "all"], default="all")
    args = parser.parse_args()
    for robot in MODELS if args.robot == "all" else [args.robot]:
        install(robot)
