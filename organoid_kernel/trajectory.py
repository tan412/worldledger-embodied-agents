"""Versioned, multi-rate trajectory storage with explicit provenance and masks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .hashing import sha256_file

SCHEMA = "organoid.trajectory.v2"


def write_trajectory(directory, arrays, streams, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "trajectory.npz").exists():
        raise FileExistsError("Refusing to overwrite trajectory")
    if set(arrays) != set(streams):
        raise ValueError("Every array needs exactly one stream definition")
    normalized = {key: np.asarray(value) for key, value in arrays.items()}
    for key, value in normalized.items():
        if value.dtype.hasobject:
            raise ValueError(f"Object arrays are not portable: {key}")
        spec = streams[key]
        if not all(k in spec for k in ("unit", "origin", "clock")):
            raise ValueError(f"Incomplete stream metadata: {key}")
        clock = spec["clock"]
        if clock is not None:
            ts = normalized[clock]
            if ts.ndim != 1 or value.shape[0] != len(ts):
                raise ValueError(f"Stream clock mismatch: {key}")
            if not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
                raise ValueError(f"Nonmonotonic clock: {clock}")
        if spec.get("names") and value.shape[-1] != len(spec["names"]):
            raise ValueError(f"Named channel count mismatch: {key}")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            if not spec.get("allows_missing", False):
                raise ValueError(f"Undeclared missing values: {key}")
    np.savez_compressed(directory / "trajectory.npz", **normalized)
    manifest = {
        "schema": SCHEMA, "metadata": metadata,
        "streams": {key: {**streams[key], "shape": list(value.shape),
                           "dtype": str(value.dtype)} for key, value in normalized.items()},
        "sha256": sha256_file(directory / "trajectory.npz"),
    }
    (directory / "trajectory.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return manifest


def read_trajectory(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "trajectory.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unsupported trajectory schema")
    if sha256_file(directory / "trajectory.npz") != manifest["sha256"]:
        raise ValueError("Trajectory hash mismatch")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if set(arrays) != set(manifest["streams"]):
        raise ValueError("Trajectory stream set mismatch")
    for key, value in arrays.items():
        spec = manifest["streams"][key]
        if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
            raise ValueError(f"Trajectory schema mismatch: {key}")
    return manifest, arrays
