"""Load the versioned trajectory into the existing evidence/inventory CLI."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ..evidence import EvidencePackage, Stream
from ..trajectory import read_trajectory
from .base import file_record


def load(directory):
    directory = Path(directory)
    manifest, arrays = read_trajectory(directory)
    metadata = manifest["metadata"]
    names = {"joint_position": "robot.joint_position",
             "joint_velocity": "robot.joint_velocity",
             "joint_reference": "robot.action",
             "tcp_position": "extra.tcp_position"}
    package = EvidencePackage(
        episode_id=directory.name, dataset_format="organoid_trajectory_v2",
        raw_files=[file_record(directory / name) for name in ("trajectory.json", "trajectory.npz")],
        meta={**metadata, "robot_type": metadata.get("robot_profile", "")},
    )
    for key, array in arrays.items():
        spec = manifest["streams"][key]
        simulated = spec["origin"] in ("simulated", "procedural")
        package.add(Stream(
            names.get(key, "extra." + key), "derived" if simulated else spec["origin"],
            data=array, columns=spec.get("names", []), unit=spec["unit"],
            timestamps=arrays.get(spec["clock"]), source_file=str(directory / "trajectory.npz"),
            source_field=key, provenance={"origin": spec["origin"], "synthetic": simulated,
                                         "trajectory_schema": manifest["schema"]},
        ))
    time = arrays.get("state_time_s", arrays.get("timestamp_s"))
    if time is not None and len(time) > 1:
        package.fps = float(1 / np.median(np.diff(time)))
    return package
