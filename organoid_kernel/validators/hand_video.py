"""手-物-接触 Validator(老 organoid 视觉车道的回归,面向 ego 人手视频)。

三个检查项,逐级依赖、各自如实降级:
  hand_quality        21 点手部关键点 13 项质量指标(关键点由本地 MediaPipe 生成,
                      derived;或直接消费数据自带的 human.hand_landmarks)
  object_tracking     种子框光流跟踪的可行性与置信度(无人工种子时运动自动种子,降级信任)
  hand_object_contact 指尖-物体框接触候选 + 事件聚合(老阈值);contact_truth 恒为
                      not_measured —— 这是图像空间近邻,不是真实接触测量
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import handlab
from ..ledger import Claim, ACCEPTED, REJECTED, INCONCLUSIVE, NOT_EVALUATED

REQUIRES = ["camera.rgb"]
CLAIMS = ["hand_quality", "object_tracking", "hand_object_contact"]
MAX_ANALYZED_FRAMES = 600          # 抽样上限:批量质检不整段跑,收据记采样口径
MIN_FRAME_COVERAGE = 0.5           # 手检出覆盖率低于此 → hand_quality 判负
MIN_MEAN_CONFIDENCE = 0.7


def run(pkg, inventory: dict, ctx: dict) -> tuple:
    receipt = {"schema": "organoid-kernel.hand-video.v1", "episode_id": pkg.episode_id}
    claims = []
    rgb = pkg.get("camera.rgb")
    if rgb is None or not rgb.files:
        return receipt, [Claim(c, NOT_EVALUATED, "无视频文件") for c in CLAIMS]
    if not handlab.MODEL_PATH.exists():
        return receipt, [Claim(c, NOT_EVALUATED,
                               "本机无 hand_landmarker 模型(assets/models/)")
                         for c in CLAIMS]
    video = Path(rgb.files[0])
    out = ctx["out_dir"]

    # ---- 手部:生成 21 点关键点 + 13 项质量指标
    try:
        gen = handlab.extract_hands(video, out / "hands.jsonl",
                                    max_frames=MAX_ANALYZED_FRAMES)
        hands_rows = [json.loads(l) for l in (out / "hands.jsonl").open()]
        metrics = handlab.hand_metrics(hands_rows,
                                       frame_stride=gen.get("frame_stride", 1))
        receipt["hand_generation"] = gen
        receipt["hand_metrics"] = metrics
        if not hands_rows:
            claims.append(Claim("hand_quality", NOT_EVALUATED,
                                "画面中未检出人手(可能非人手数据)"))
        else:
            bad = (metrics["frame_coverage"] < MIN_FRAME_COVERAGE
                   or metrics["mean_confidence"] < MIN_MEAN_CONFIDENCE
                   or metrics["nonfinite_landmark_rate"] > 0.001)
            reason = (f"检出覆盖率 {metrics['frame_coverage']:.0%},"
                      f"均值置信度 {metrics['mean_confidence']:.2f},"
                      f"腕点步长 p95 {metrics['wrist_step_p95']}(归一化)")
            claims.append(Claim("hand_quality", REJECTED if bad else ACCEPTED,
                                reason if bad else "", detail=metrics))
    except Exception as e:
        claims.append(Claim("hand_quality", NOT_EVALUATED, f"手部生成失败: {str(e)[:120]}"))
        hands_rows = []

    # ---- 物体跟踪:人工种子(ctx 传入)优先,否则运动自动种子
    seeds = ctx.get("object_seeds")
    seed_note = "observed_manual_seed"
    if not seeds:
        try:
            seeds = [handlab.auto_seed_moving_object(video)]
            seed_note = "auto_motion_seed(降级信任:无人工画框,取帧差最大运动团块)"
        except Exception as e:
            claims.append(Claim("object_tracking", NOT_EVALUATED,
                                f"无人工种子且自动种子失败: {str(e)[:80]}"))
            claims.append(Claim("hand_object_contact", NOT_EVALUATED, "无物体轨迹"))
            return receipt, claims
    try:
        track = handlab.track_objects(video, seeds, out / "objects.jsonl",
                                      max_frames=MAX_ANALYZED_FRAMES * 2)
        receipt["object_tracking"] = {**track, "seed": seed_note}
        st = (ACCEPTED if track["tracked_ratio"] >= 0.7 else
              INCONCLUSIVE if track["tracked_ratio"] >= 0.3 else REJECTED)
        claims.append(Claim("object_tracking", st,
                            f"跟踪保持率 {track['tracked_ratio']:.0%}({seed_note})",
                            detail=track))
    except Exception as e:
        claims.append(Claim("object_tracking", NOT_EVALUATED, str(e)[:120]))
        claims.append(Claim("hand_object_contact", NOT_EVALUATED, "无物体轨迹"))
        return receipt, claims

    # ---- 接触候选:手 + 物都在才有意义
    if not hands_rows:
        claims.append(Claim("hand_object_contact", NOT_EVALUATED, "无手部关键点"))
        return receipt, claims
    objects_rows = [json.loads(l) for l in (out / "objects.jsonl").open()]
    contact = handlab.label_contact_candidates(hands_rows, objects_rows)
    (out / "contact-frames.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in contact["frame_candidates"]),
        encoding="utf-8")
    (out / "contact-events.json").write_text(
        json.dumps(contact["events"], ensure_ascii=False, indent=1), encoding="utf-8")
    receipt["contact"] = {"frame_candidates": len(contact["frame_candidates"]),
                          "events": len(contact["events"]),
                          "contact_truth": "not_measured"}
    claims.append(Claim("hand_object_contact",
                        ACCEPTED if contact["events"] else INCONCLUSIVE,
                        f"接触候选事件 {len(contact['events'])} 个"
                        "(图像空间近邻,非真实接触测量)" if contact["events"] else
                        "无接触候选事件(手与被跟踪物体无近邻交叠)",
                        detail=receipt["contact"]))
    return receipt, claims
