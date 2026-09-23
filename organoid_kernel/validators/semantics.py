"""语义侧证据层(§7.6):动作-文字一致性(纯运动学)+ 关键帧证据包(纯解码)。

VLM 抽查刻意不内置:advisory-only 工具依赖本地权重环境,保持外挂;
本层产出的关键帧包就是它的标准输入。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..ledger import Claim, ACCEPTED, REJECTED, NOT_EVALUATED

REQUIRES = ["robot.joint_position", "annotation.language_segments"]
CLAIMS = ["motion_language"]

RIGHT = re.compile(r"右手|右臂|right\s*(hand|arm)", re.I)
LEFT = re.compile(r"左手|左臂|left\s*(hand|arm)", re.I)
BOTH = re.compile(r"双手|双臂|两只手|both\s*(hands|arms)", re.I)
IDLE_RATE = 0.03          # rad/s:双臂路径速率低于此视为没动
DOMINANCE = 0.25          # 点名的手活动量须不低于另一只的该比例


def _arm_cols(columns, side):
    return [i for i, c in enumerate(columns) if c.startswith(f"zarm_{side}")]


def run(pkg, inventory: dict, ctx: dict) -> tuple:
    joint = pkg.get("robot.joint_position")
    segs = pkg.get("annotation.language_segments")
    arr = np.asarray(joint.data, dtype=float)
    fps = pkg.fps or 30.0
    li, ri = _arm_cols(joint.columns, "l"), _arm_cols(joint.columns, "r")
    receipt = {"schema": "organoid-kernel.motion-language.v1",
               "episode_id": pkg.episode_id, "segments": []}
    if not li or not ri:
        return receipt, [Claim("motion_language", NOT_EVALUATED, "轨迹无双臂关节列")]

    rows, mismatches = [], []
    for seg in segs.data:
        a = max(0, int(seg["start_s"] * fps))
        b = min(len(arr), int(seg["end_s"] * fps) + 1)
        text = seg.get("text") or ""
        if b - a < 2:
            rows.append({"segment_id": seg["id"], "verdict": "not_applicable",
                         "reason": "段太短"})
            continue
        dl = np.abs(np.diff(arr[a:b][:, li], axis=0)).sum() / ((b - a - 1) / fps) / len(li)
        dr = np.abs(np.diff(arr[a:b][:, ri], axis=0)).sum() / ((b - a - 1) / fps) / len(ri)
        named = ("both" if BOTH.search(text) else
                 "right" if RIGHT.search(text) else
                 "left" if LEFT.search(text) else None)
        verdict, reason = "not_applicable", "标注未点名手"
        if named:
            if dl < IDLE_RATE and dr < IDLE_RATE:
                verdict, reason = "no_motion", "标注描述动作但双臂都几乎没动"
            elif named == "both":
                verdict, reason = "consistent", ""
            else:
                lead, other = (dr, dl) if named == "right" else (dl, dr)
                if lead < IDLE_RATE and other > IDLE_RATE and lead < DOMINANCE * other:
                    verdict = "mismatch"
                    reason = f"点名{named},实测另一只臂活动量是它的 {other/max(lead,1e-9):.1f} 倍"
                else:
                    verdict, reason = "consistent", ""
        row = {"segment_id": seg["id"], "text": text[:60], "named": named,
               "left_rate": round(float(dl), 4), "right_rate": round(float(dr), 4),
               "verdict": verdict, "reason": reason or None}
        rows.append(row)
        if verdict in ("mismatch", "no_motion"):
            mismatches.append(row)
    receipt["segments"] = rows
    receipt["counts"] = {v: sum(1 for r in rows if r["verdict"] == v)
                         for v in ("consistent", "mismatch", "no_motion", "not_applicable")}

    # ---- 关键帧证据包(有视频才抽;段首/中/末三帧,索引绑内容哈希)
    rgb = pkg.get("camera.rgb")
    if rgb is not None and rgb.files:
        try:
            _extract_keyframes(pkg, segs, rgb, ctx["out_dir"] / "keyframes")
            receipt["keyframes"] = "keyframes/keyframes-index.json"
        except Exception as e:
            receipt["keyframes_error"] = str(e)[:200]

    ok = not mismatches
    return receipt, [Claim("motion_language", ACCEPTED if ok else REJECTED,
                           "" if ok else f"{len(mismatches)} 段动作与文字不符(疑似,须人工复核)",
                           detail={"flagged": [m["segment_id"] for m in mismatches]})]


def _extract_keyframes(pkg, segs, rgb, out: Path, per_segment: int = 3):
    import cv2
    import hashlib, json
    out.mkdir(parents=True, exist_ok=True)
    video = Path(rgb.files[0])                     # 首路(顶部)相机
    targets = {}
    for seg in segs.data:
        s, e = int(seg.get("start_frame", seg["start_s"] * pkg.fps)), \
               int(seg.get("end_frame", seg["end_s"] * pkg.fps))
        for k in range(per_segment):
            f = s + (e - s) * (0.5 if per_segment == 1 else k / (per_segment - 1))
            targets.setdefault(int(round(f)), []).append((seg["id"], k))
    cap = cv2.VideoCapture(str(video))
    got, idx = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in targets:
            got[idx] = frame
        idx += 1
    cap.release()
    by_seg = {}
    for f, hits in sorted(targets.items()):
        if f not in got:
            continue
        for sid, k in hits:
            name = f"seg{sid}-{k}.jpg"
            cv2.imwrite(str(out / name), got[f], [cv2.IMWRITE_JPEG_QUALITY, 92])
            by_seg.setdefault(sid, []).append({"file": name, "frame_index": f})
    index = {"schema": "organoid-kernel.keyframes.v1", "video": video.name,
             "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
             "segments": [{"segment_id": s["id"], "description": s.get("text", ""),
                           "frames": by_seg.get(s["id"], [])} for s in segs.data]}
    (out / "keyframes-index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
