"""Robot Profile 与身份核定(阶段 3 + §5.4)。

Profile 是版本化 JSON,描述机器人事实;身份核定决定"这条 episode 绑定哪个 Profile",
分级 fingerprint_exact / fingerprint_family_tie / inferred / unknown,逐条入收据。
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]      # new_organoid/
PROFILE_DIR = PKG_ROOT / "profiles"
ASSET_DIR = PKG_ROOT / "assets"


@dataclass
class RobotProfile:
    name: str
    version: str
    base_type: str               # free_root / fixed_base / mobile / none
    root_link: str = ""
    urdf: str = ""               # 相对 new_organoid 的路径
    mesh_dir: str = ""
    joint_prefixes: list = field(default_factory=list)   # 本体关节前缀(判关节归属)
    leg_columns: list = field(default_factory=list)
    foot_bodies: dict = field(default_factory=dict)      # link -> left/right
    foot_contact_margin: float = 0.0
    expected_foot_spheres: int = 0
    adjacent_filter_depth: int = 3
    vendor_names: list = field(default_factory=list)     # 数据集里可能声明的名字
    joint_count: int = 0                                  # 本体可动关节数(推断用)
    dynamic_rules: dict = field(default_factory=dict)
    fullbody_servo: dict = field(default_factory=dict)
    robot_friction: float = 0.8
    floor_friction: float = 0.8
    claw_friction: float = 1.5
    claw_joint_damping: float = 1.5
    claw_joint_armature: float = 0.005
    claw_actuator_kp: float = 60.0
    claw_actuator_force_limit: float = 2.0
    physics_audit_body_prefixes: list = field(default_factory=list)
    collision_mesh_body_prefixes: list = field(default_factory=list)
    mjcf: str = ""
    joint_names: list = field(default_factory=list)
    actuator_names: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def urdf_path(self) -> Path:
        return PKG_ROOT / self.urdf

    def mesh_path(self) -> Path:
        return PKG_ROOT / self.mesh_dir if self.mesh_dir else self.urdf_path().parent / "meshes"

    def mjcf_path(self) -> Path:
        if not self.mjcf:
            raise ValueError(f"{self.name} has no MJCF backend")
        return PKG_ROOT / self.mjcf

    def map_joint_columns(self, columns, unit):
        """Require named joints and explicit units; never match by width alone."""
        if unit != "rad" or not self.joint_names:
            raise ValueError("Explicit radian joint schema is required")
        if len(columns) != len(set(columns)) or set(columns) != set(self.joint_names):
            raise ValueError("Joint names differ from the selected robot")
        return [columns.index(name) for name in self.joint_names]


def load_profile(name: str) -> RobotProfile:
    data = json.loads((PROFILE_DIR / f"{name}.json").read_text())
    return RobotProfile(raw=data, **{k: v for k, v in data.items()
                                     if k in RobotProfile.__dataclass_fields__ and k != "raw"})


def list_profiles() -> list:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


# ---------------------------------------------------------------- 身份核定

def urdf_joint_origins(urdf: Path) -> dict:
    """URDF 里每对 parent→child 的平移,几何指纹的比对基准。"""
    root = ET.parse(urdf).getroot()
    out = {}
    for j in root.findall("joint"):
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        origin = j.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None
                                  else "0 0 0").split()]
        out[(parent, child)] = xyz
    return out


def fingerprint_match(observed_offsets: dict, urdf: Path,
                      skeleton_prefixes=("leg", "waist", "zarm", "zhead", "torso", "base"),
                      tol: float = 1e-6) -> dict:
    """观测 parent→child 平移 vs URDF 逐位比对。只认骨架;附件(相机/雷达/夹爪)按台安装,
    拿附件偏差否候选会误杀 —— OpenLET 实测口径。"""
    expect = urdf_joint_origins(urdf)
    matched = mismatched = 0
    worst = 0.0
    for (parent, child), xyz in observed_offsets.items():
        if not any(child.startswith(p) or parent.startswith(p) for p in skeleton_prefixes):
            continue
        if (parent, child) not in expect:
            continue
        err = max(abs(a - b) for a, b in zip(xyz, expect[(parent, child)]))
        if err < tol:
            matched += 1
        else:
            mismatched += 1
            worst = max(worst, err)
    return {"matched": matched, "mismatched": mismatched, "worst_error_m": worst}


def identify(pkg, profiles: list = None) -> dict:
    """身份核定:有 link_offsets 流走指纹,否则关节数+厂商声明推断。

    返回 {status, profile, tied_with, evidence};status ∈
    fingerprint_exact / fingerprint_family_tie / inferred / unknown。
    """
    profiles = profiles or list_profiles()
    offsets_stream = pkg.get("robot.link_offsets")
    if offsets_stream is not None and offsets_stream.data:
        scores = []
        for name in profiles:
            prof = load_profile(name)
            if not prof.urdf:
                continue
            r = fingerprint_match(offsets_stream.data, prof.urdf_path())
            scores.append((name, r))
        exact = [(n, r) for n, r in scores if r["matched"] > 0 and r["mismatched"] == 0]
        if len(exact) == 1:
            return {"status": "fingerprint_exact", "profile": exact[0][0],
                    "tied_with": [], "evidence": exact[0][1]}
        if len(exact) > 1:
            best = max(exact, key=lambda x: x[1]["matched"])
            return {"status": "fingerprint_family_tie", "profile": best[0],
                    "tied_with": [n for n, _ in exact if n != best[0]],
                    "evidence": best[1]}
    # 推断:关节数 + 厂商声明。关节数按 Profile 的本体前缀过滤后再比
    #(末端/夹爪列混在轨迹里时不误伤 —— joint_prefixes 的消费点)
    joint = pkg.get("robot.joint_position")
    all_cols = joint.columns if joint is not None else []
    declared = str(pkg.meta.get("robot_type", "")).lower()
    for name in profiles:
        prof = load_profile(name)
        if not prof.joint_count:
            continue
        if prof.joint_names and set(all_cols) != set(prof.joint_names):
            continue
        cols = ([c for c in all_cols
                 if any(c.startswith(p) for p in prof.joint_prefixes)]
                if prof.joint_prefixes else all_cols)
        if len(cols) == prof.joint_count and (
                not declared or any(v.lower() in declared or declared in v.lower()
                                    for v in prof.vendor_names) or not prof.vendor_names):
            return {"status": "inferred", "profile": name, "tied_with": [],
                    "evidence": {"joint_count_matched": len(cols),
                                 "total_columns": len(all_cols),
                                 "vendor_declared": declared}}
    return {"status": "unknown", "profile": None, "tied_with": [],
            "evidence": {"joint_count": len(all_cols), "vendor_declared": declared}}
