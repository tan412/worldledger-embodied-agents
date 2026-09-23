"""人读质检报告生成器:把一次 run 的全部收据渲染成一份 markdown 质检报告。

原则:报告里每个数字都能在同目录的 JSON 收据里找到出处;
"没验的"与"验过的"同等显眼 —— not_evaluated 不藏在附录里。
"""
from __future__ import annotations

import json
from pathlib import Path

STATE_ZH = {"accepted": "✅ 通过", "rejected": "❌ 判负", "inconclusive": "⚠️ 存疑",
            "not_evaluated": "◻️ 未评估", "not_applicable": "— 不适用", "error": "💥 出错"}
GRADE_ZH = {"accepted": "通过", "accepted_repaired": "修复后通过",
            "accepted_with_warnings": "带警示通过", "rejected": "判负",
            "not_evaluable": "不可评估", "error": "执行出错"}


def render(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    def j(name):
        p = run_dir / name
        return json.loads(p.read_text()) if p.exists() else {}
    ledger = j("claim-ledger.json")
    evidence = j("evidence.json")
    inv = j("capability-inventory.json")
    plan = j("plan.json")

    lines = [f"# 质检报告:{ledger.get('episode_id', run_dir.name)}", ""]
    grade = ledger.get("grade", "?")
    lines += [f"**最终分级:{GRADE_ZH.get(grade, grade)}** — {ledger.get('grade_reason','')}",
              f"策略:`{ledger.get('policy')}` · 身份核定:"
              f"`{(ledger.get('identity') or {}).get('status')}` → "
              f"{(ledger.get('identity') or {}).get('profile') or '未绑定模型'}", ""]

    # 数据里有什么
    lines += ["## 数据里有什么(证据流盘点)", "",
              "| 流 | 等级 | 规模 | 来源 |", "|---|---|---|---|"]
    for name, s in (evidence.get("streams") or {}).items():
        shape = "×".join(str(v) for v in (s.get("shape") or [])) or \
                (f"{len(s.get('files') or [])} 文件" if s.get("files") else "—")
        src = s.get("source_field") or (Path(s["files"][0]).name if s.get("files") else "")
        note = f";{s['note']}" if s.get("note") else ""
        lines.append(f"| `{name}` | {s.get('origin')} | {shape} | {src}{note} |")
    tb = inv.get("timebase") or {}
    if tb.get("applicable"):
        lines += ["", f"时间基:{'**合成栅格**(帧号/fps,真实采样时刻已丢)' if tb.get('synthetic') else '真实时间戳'}"
                  f",dt 中位 {tb.get('dt_median_s')} s"]
    der = ((inv.get("derivations") or {}).get("static_base") or {})
    if der.get("applied"):
        lines += [f"派生:静止基座推导生效(腿摆幅 {der['leg_range_rad']:.1e} rad、"
                  f"IMU 摆幅 {der['quat_range']:.1e};忽略平移上界 "
                  f"{der.get('neglected_translation_bound_m', 0)*100:.1f} cm)"]

    # 验了什么
    lines += ["", "## 验了什么(逐 claim 裁决)", "",
              "| 检查 | 结果 | 说明 |", "|---|---|---|"]
    for c in ledger.get("claims") or []:
        reason = (c.get("reason") or "").replace("|", "/")[:90]
        lines.append(f"| {c['claim']} | {STATE_ZH.get(c['status'], c['status'])} | {reason} |")

    if ledger.get("repairs"):
        lines += ["", f"**修复记录**:{json.dumps(ledger['repairs'], ensure_ascii=False)}"
                  "(逐格补丁见 repair-patch.json)"]
    if ledger.get("warnings"):
        lines += ["", f"**警示单**:{json.dumps(ledger['warnings'], ensure_ascii=False)}"]

    # 关键数字摘录
    kin = j("receipt-kinematics.json") or j("receipt-kinematics.repaired.json")
    phys = j("receipt-physics.json")
    vis = j("receipt-visual.json")
    extras = []
    if kin:
        pv, vv = kin.get("position_violations") or {}, kin.get("velocity_violations") or {}
        if pv:
            extras.append("限位越界:" + "; ".join(
                f"{k} {v['count']} 帧(最大超 {v['max_excess_rad']:.4f} rad)"
                for k, v in list(pv.items())[:5]))
        if vv:
            extras.append("限速违规:" + "; ".join(
                f"{k} {v['count']} 样本(峰值 {v['max_observed_rad_s']} rad/s)"
                for k, v in list(vv.items())[:5]))
    if phys and phys.get("coverage"):
        cov = phys["coverage"]
        w = phys.get("worst") or {}
        extras.append(f"物理({phys.get('scope')}):平衡覆盖 {cov.get('balanced')},"
                      f"最差裕度 {w.get('min_balance_margin_m')} m @frame "
                      f"{w.get('min_balance_margin_frame')};支撑覆盖 {cov.get('support')};"
                      f"最大穿透 {w.get('max_ground_penetration_m')} m")
        extras.append(f"地面标定:{(phys.get('ground') or {}).get('estimated_height_m')} m"
                      f"({(phys.get('ground') or {}).get('method')})")
    if vis and vis.get("shots"):
        for s in vis["shots"]:
            extras.append(f"视觉 frame {s['frame']}:导入 {s.get('imported')}/"
                          f"{s.get('requested_meshes')} mesh,画幅覆盖 {s.get('in_frame_ratio')}")
    if extras:
        lines += ["", "## 关键数字", ""] + [f"- {e}" for e in extras]

    # 没验的以及为什么
    blocked = [p for p in (plan.get("plan") or []) if not p.get("runnable")]
    if blocked:
        lines += ["", "## 没验的以及为什么", ""]
        for p in blocked:
            lines.append(f"- `{p['validator']}`:缺少 {p['missing']}")
    lines += ["", "---", "*本报告由 organoid_kernel 生成;每个数字的出处见同目录 JSON 收据。*", ""]
    return "\n".join(lines)


def write_report(run_dir: Path) -> Path:
    out = Path(run_dir) / "quality-report.md"
    out.write_text(render(run_dir), encoding="utf-8")
    return out
