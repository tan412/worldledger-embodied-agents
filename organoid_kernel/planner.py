"""Planner(阶段 4):按能力清单决定可运行的 Validator,执行并汇账。

Validator 注册项声明输入契约(requires);缺输入 → 该 Validator 的 claim 全部
not_evaluated 并写明缺什么。修复回路(repair.py)在预检判负后按病因介入。
"""
from __future__ import annotations

import importlib
import json
import time
import traceback
from pathlib import Path

from . import policy as policy_mod
from . import profile as profile_mod
from .inventory import build_inventory
from .ledger import Claim, Ledger, NOT_EVALUATED, ERROR, REJECTED

# 注册表:名字 → (模块, 额外条件)。REQUIRES 里 "robot.model" 表示需要身份核定成功。
REGISTRY = [
    ("quality", "organoid_kernel.validators.quality"),
    ("pairing", "organoid_kernel.validators.pairing"),
    ("kinematics", "organoid_kernel.validators.kinematics"),
    ("physics", "organoid_kernel.validators.physics_mujoco"),
    ("motion_language", "organoid_kernel.validators.semantics"),
    ("sensors", "organoid_kernel.validators.sensors"),
    ("source_media", "organoid_kernel.validators.source_media"),
    ("hand_video", "organoid_kernel.validators.hand_video"),
    ("visual", "organoid_kernel.validators.visual_blender"),
    ("task_scene", "organoid_kernel.validators.task_scene"),
]


def _requirement_met(req: str, pkg, ctx) -> bool:
    if req == "robot.model":
        profile = ctx.get("profile")
        # These legacy validators consume URDFModel, not arbitrary MJCF files.
        return profile is not None and bool(profile.urdf) and profile.urdf_path().is_file()
    return pkg.has(req)


def _compare_physics(before: dict, after: dict) -> dict:
    """修复前后物理收据非回归比较:平衡裕度/穿透/自碰撞不得因数据修复而变差。"""
    def worst(r):
        w = r.get("worst") or {}
        return (w.get("min_balance_margin_m"), w.get("max_ground_penetration_m"),
                w.get("max_self_penetration_m"))
    b, a = worst(before), worst(after)
    regressions = []
    if b[0] is not None and a[0] is not None and a[0] < b[0] - 1e-3:
        regressions.append(f"平衡裕度 {b[0]:.4f}→{a[0]:.4f} m")
    for i, name in ((1, "地面穿透"), (2, "自碰撞深度")):
        if b[i] is not None and a[i] is not None and a[i] > b[i] + 1e-4:
            regressions.append(f"{name} {b[i]:.4f}→{a[i]:.4f} m")
    return {"passed": not regressions,
            "before": {"balance_m": b[0], "penetration_m": b[1], "self_m": b[2]},
            "after": {"balance_m": a[0], "penetration_m": a[1], "self_m": a[2]},
            "regressions": regressions or None}


def run_episode(pkg, out_dir: Path, policy_name: str = None,
                profile_name: str = None, validators: list = None) -> Ledger:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evidence.json").write_text(
        json.dumps(pkg.summary(), ensure_ascii=False, indent=1), encoding="utf-8")

    # 身份核定 → Profile 绑定
    identity = profile_mod.identify(pkg)
    if profile_name:
        identity = {**identity, "profile": profile_name,
                    "status": identity.get("status", "inferred")
                    if identity.get("profile") == profile_name else "operator_pinned"}
    prof = profile_mod.load_profile(identity["profile"]) if identity.get("profile") else None

    inventory = build_inventory(pkg, prof.leg_columns if prof else ())
    (out / "capability-inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=1), encoding="utf-8")

    pol_name = policy_name or policy_mod.pick_policy(prof, pkg)
    pol = policy_mod.POLICIES[pol_name]
    ledger = Ledger(episode_id=pkg.episode_id, policy=pol_name)
    ledger.identity = identity

    ctx = {"profile": prof, "receipts": {}, "out_dir": out, "inventory": inventory}
    plan = []
    for name, module_path in REGISTRY:
        if validators is not None and name not in validators:
            continue
        mod = importlib.import_module(module_path)
        missing = [r for r in mod.REQUIRES if not _requirement_met(r, pkg, ctx)]
        plan.append({"validator": name, "runnable": not missing, "missing": missing})
        if missing:
            for claim in mod.CLAIMS:
                ledger.add(Claim(claim, NOT_EVALUATED, f"缺少输入: {missing}"))
            continue
        t0 = time.monotonic()
        try:
            receipt, claims = mod.run(pkg, inventory, ctx)
            receipt["elapsed_s"] = round(time.monotonic() - t0, 2)
            rp = out / f"receipt-{name}.json"
            rp.write_text(json.dumps(receipt, ensure_ascii=False, indent=1,
                                     default=str), encoding="utf-8")
            ctx["receipts"][name] = receipt
            for c in claims:
                c.receipt = rp.name
                ledger.add(c)
        except Exception:
            for claim in mod.CLAIMS:
                ledger.add(Claim(claim, ERROR, traceback.format_exc()[-400:]))
    (out / "plan.json").write_text(json.dumps(
        {"schema": "organoid-kernel.plan.v1", "policy": pol_name, "plan": plan},
        ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 修复回路:预检判负时按病因定向处置(测量侧修数据 / 稀疏台阶降警示)
    if ledger.status_of("kinematic_precheck") == REJECTED:
        from . import repair
        outcome = repair.attempt(pkg, ctx, ledger, out)
        if outcome.get("changed"):
            # 修复后重验预检(同一 Validator、同一口径)
            mod = importlib.import_module("organoid_kernel.validators.kinematics")
            receipt, claims = mod.run(pkg, inventory, ctx)
            (out / "receipt-kinematics.repaired.json").write_text(
                json.dumps(receipt, ensure_ascii=False, indent=1, default=str),
                encoding="utf-8")
            ctx["receipts"]["kinematics"] = receipt
            ledger.claims = [c for c in ledger.claims
                             if c.name not in ("kinematic_precheck",
                                               "kinematic_precheck_gate")]
            for c in claims:
                c.receipt = "receipt-kinematics.repaired.json"
                ledger.add(c)
            # 数据变了 → 物理重跑 + 前后非回归比较(compare_physical_receipts 口径):
            # 修复只许把预检修好,不许改变物理结论
            before = ctx["receipts"].get("physics")
            if before is not None and (validators is None or "physics" in validators):
                pmod = importlib.import_module(
                    "organoid_kernel.validators.physics_mujoco")
                p_receipt, p_claims = pmod.run(pkg, inventory, ctx)
                (out / "receipt-physics.repaired.json").write_text(
                    json.dumps(p_receipt, ensure_ascii=False, indent=1,
                               default=str), encoding="utf-8")
                phys_names = set(pmod.CLAIMS)
                ledger.claims = [c for c in ledger.claims if c.name not in phys_names]
                for c in p_claims:
                    c.receipt = "receipt-physics.repaired.json"
                    ledger.add(c)
                ledger.repairs["physics_non_regression"] = _compare_physics(
                    before, p_receipt)
                ctx["receipts"]["physics"] = p_receipt

    pol.grade(ledger)
    ledger.save(out / "claim-ledger.json")
    return ledger
