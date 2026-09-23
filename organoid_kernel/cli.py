"""CLI:python3 -m organoid_kernel.cli <inspect|run|batch|golden-compare> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import planner, report
from .adapters import base as adapters_base
from .skill_catalog import skill_catalog
from .skill_mining import mine_evidence_package


def _load_episode(path: Path, episode: int):
    fmt = adapters_base.detect_format(path)
    if fmt == "organoid_trajectory_v2":
        from .adapters import unified_trajectory
        return unified_trajectory.load(path)
    if fmt.startswith("lerobot"):
        from .adapters import lerobot_v21
        return lerobot_v21.load(path, episode)
    if fmt == "rosbag":
        from .adapters import rosbag_let
        bag = path if path.is_file() else sorted(path.glob("*.bag"))[episode]
        return rosbag_let.load(bag)
    if fmt == "legacy_organoid":
        from .adapters import legacy_organoid
        return legacy_organoid.load(path)
    if fmt == "fourier":
        from .adapters import fourier_hdf5
        return fourier_hdf5.load(path)
    if fmt == "pico_vr":
        from .adapters import vr_ego
        return vr_ego.load(path)
    if fmt == "gendas":
        from .adapters import gendas_decoded
        return gendas_decoded.load(path)
    if fmt == "hdf5":
        from .adapters import hdf5_generic
        h5 = path if path.is_file() else sorted(list(path.glob("*.h5")) + list(path.glob("*.hdf5")))[episode]
        return hdf5_generic.load(h5)
    if fmt == "umi_zarr":
        from .adapters import umi_zarr
        return umi_zarr.load(path, episode)
    if fmt == "ego_mp4":
        from .adapters import ego_mp4
        mp4 = path if path.is_file() else sorted(path.glob("*.mp4"))[episode]
        return ego_mp4.load(mp4)
    raise SystemExit(
        f"无法识别数据格式: {path}\n"
        "当前支持: LeRobot v2.1/v3.0(meta/info.json)、rosbag(*.bag)、通用 HDF5(*.h5/hdf5)、"
        "傅利叶 HDF5(proprio_stats/)、UMI-Zarr(.zgroup/zarr.json)、GenDAS(robot0_vio_eef_pose.csv)、"
        "纯视频(*.mp4)、Pico VR(data/trackingData_*.txt)、legacy organoid(trajectory.csv+segments.json)。"
        "括号内为探测特征;新格式需在 organoid_kernel/adapters/ 实现适配器并在 cli._load_episode 注册。")


def _run_one(path: Path, episode: int, out: Path, policy, profile, skip):
    pkg = _load_episode(path, episode)
    names = [n for n, _ in planner.REGISTRY if n not in set(skip)]
    ledger = planner.run_episode(pkg, out, policy_name=policy,
                                 profile_name=profile, validators=names)
    report.write_report(out)          # 每次 run 都落人读质检报告
    return pkg, ledger


def _episode_count(path: Path) -> int:
    fmt = adapters_base.detect_format(path)
    if fmt.startswith("lerobot"):
        v21 = sorted(path.glob("data/chunk-*/episode_*.parquet"))
        if v21:
            return len(v21)
        info = json.loads((path / "meta/info.json").read_text())
        return int(info.get("total_episodes") or 1)
    if fmt == "ego_mp4" and path.is_dir():
        return len(list(path.glob("*.mp4")))
    return 1


def cmd_golden_compare(new_dir: Path, golden_dir: Path) -> None:
    """新账本 vs 老 summary 收据:预检/物理/配对三口径逐条对账。"""
    golden = {}
    for gp in Path(golden_dir).glob("*.summary.json"):
        g = json.loads(gp.read_text())
        key = (g.get("task_dir", g.get("task", "")).split("/")[-1], g.get("episode"))
        golden[key] = g
    matched, diffs = 0, []
    for lp in sorted(Path(new_dir).glob("*/claim-ledger.json")):
        led = json.loads(lp.read_text())
        claims = {c["claim"]: c["status"] for c in led["claims"]}
        eid = led["episode_id"]
        hit = next((g for (task, ep), g in golden.items()
                    if task in eid and f"ep{ep}" in eid), None)
        if hit is None:
            continue
        new_pre = (claims.get("kinematic_precheck") == "accepted"
                   and not (led.get("repairs") or led.get("warnings")))
        phys = [claims.get(c) for c in ("balance", "foot_ground_contact",
                                        "ground_penetration", "self_collision")]
        new_phys = ("not_evaluated" if "not_evaluated" in phys else
                    "rejected" if "rejected" in phys
                    or claims.get("kinematic_precheck_gate") == "rejected" else "accepted")
        d = []
        if hit.get("precheck_passed") is not None and bool(hit["precheck_passed"]) != new_pre:
            d.append(f"预检 旧{hit['precheck_passed']}→新{new_pre}")
        if hit.get("physics_decision") and hit["physics_decision"] != new_phys:
            d.append(f"物理 旧{hit['physics_decision']}→新{new_phys}")
        if d:
            diffs.append({"episode": eid, "diffs": d})
        else:
            matched += 1
    print(f"golden 对账: {matched}/{matched + len(diffs)} 一致")
    for d in diffs:
        print("  差异:", d["episode"], d["diffs"])


def main() -> None:
    ap = argparse.ArgumentParser(prog="organoid_kernel")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="探测格式并打印证据流盘点")
    p.add_argument("path", type=Path)
    p.add_argument("--episode", type=int, default=0)

    p = sub.add_parser("run", help="单条 episode 走全链,产出收据与质检报告")
    p.add_argument("path", type=Path)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--policy", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--skip", nargs="*", default=[], help="跳过的 validator 名")

    p = sub.add_parser("batch", help="一个数据集目录逐 episode 全跑")
    p.add_argument("path", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--policy", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--skip", nargs="*", default=["visual"])

    p = sub.add_parser("golden-compare", help="新账本目录 vs 老 summary 收据目录对账")
    p.add_argument("runs", type=Path)
    p.add_argument("golden", type=Path)

    p = sub.add_parser("mine-skills", help="从 episode 的语言分段抽取原子技能")
    p.add_argument("path", type=Path)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--with-catalog", action="store_true",
                   help="同时写出标准技能和数据集来源目录")

    args = ap.parse_args()
    if args.cmd == "inspect":
        pkg = _load_episode(args.path, args.episode)
        print(json.dumps(pkg.summary(), ensure_ascii=False, indent=1)[:4000])
    elif args.cmd == "run":
        pkg, ledger = _run_one(args.path, args.episode, args.out,
                               args.policy, args.profile, args.skip)
        print(f"{pkg.episode_id}: {ledger.grade} ({ledger.grade_reason})")
        for c in ledger.claims:
            print(f"  {c.name:26s} {c.status:14s} {c.reason[:70]}")
        print(f"质检报告: {args.out / 'quality-report.md'}")
    elif args.cmd == "batch":
        n = _episode_count(args.path)
        if args.limit:
            n = min(n, args.limit)
        for ep in range(n):
            try:
                pkg, ledger = _run_one(args.path, ep, args.out / f"ep{ep:04d}",
                                       args.policy, args.profile, args.skip)
                print(f"[{ep+1}/{n}] {pkg.episode_id}: {ledger.grade}")
            except Exception as e:
                print(f"[{ep+1}/{n}] ep{ep} 失败: {str(e)[:120]}", file=sys.stderr)
    elif args.cmd == "golden-compare":
        cmd_golden_compare(args.runs, args.golden)
    elif args.cmd == "mine-skills":
        pkg = _load_episode(args.path, args.episode)
        results = mine_evidence_package(pkg)
        output = {
            "schema": "organoid-kernel.episode-skills.v1",
            "episode_id": pkg.episode_id,
            "dataset_format": pkg.dataset_format,
            "source_meta": pkg.meta,
            "segments": [result.to_dict() for result in results],
            "summary": {
                "segments": len(results),
                "steps": sum(len(result.steps) for result in results),
                "unrecognized_fragments": sorted({
                    fragment for result in results
                    for fragment in result.unrecognized
                }),
            },
        }
        if args.with_catalog:
            output["catalog"] = skill_catalog()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"{pkg.episode_id}: 抽取 {output['summary']['steps']} 个技能步骤"
              f" → {args.out}")


if __name__ == "__main__":
    main()
