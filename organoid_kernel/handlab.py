"""视频手-物-接触实验台(忠实移植自老 organoid 的 tracking.py / contact.py /
quality.py 手部指标,外加 21 点手部关键点的本地生成)。

五件事,产物沿用老契约的字段与阈值:
  extract_hands            视频 → hands.jsonl(21 点/手,px/py 像素坐标 + 置信度;
                           MediaPipe HandLandmarker 本地模型,derived 并记模型来源;
                           另存 wx/wy/wz 米制手内 3D 关键点,供单目 3D 提升)
  lift_hands_3d            hands.jsonl → hands3d.jsonl(solvePnP 单目 3D 提升:
                           相机系米制腕点/关节/捏合宽度,逐帧重投影 RMS 作质量证据)
  track_objects            种子框 + 光流(LK + 仿射 RANSAC + 合理性闸)或 CamShift
                           逐帧跟踪,置信度显式;支持运动自动种子(降级信任,如实标注)
  label_contact_candidates 指尖(4/8/12/16/20)到物体框距离 → 接触候选帧 + 事件聚合
  hand_metrics             13 项手部质量指标(覆盖率/双手率/置信度/非有限率/越界率/
                           腕点步长与速度分位),与老 quality.py 同口径
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "assets/models/hand_landmarker.task"
FINGERTIP_INDICES = (4, 8, 12, 16, 20)


# ---------------------------------------------------------------- 手部关键点生成

def extract_hands(video_path: Path, output_jsonl: Path, stride: int = 2,
                  max_frames: int = 1200, min_confidence: float = 0.4) -> dict:
    """MediaPipe HandLandmarker → 老 hands.jsonl 契约。生成物是 derived 标签,
    provenance 记模型文件哈希;抽帧步长与上限入收据(大视频不整段跑)。"""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=min_confidence)
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 左右目并排视频(宽高比 > 2.2)只取左目:双目重复检出会让同一只手在两个
    # 半幅间来回跳,腕点步长被虚高一个画幅
    stereo_half = width / max(height, 1) > 2.2
    if stereo_half:
        width = width // 2
    rows, frame_index, processed = [], 0, 0
    while processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride == 0:
            if stereo_half:
                frame = frame[:, :width]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(frame_index / fps * 1000))
            world = result.hand_world_landmarks or [None] * len(result.hand_landmarks)
            for lm, wlm, handed in zip(result.hand_landmarks, world, result.handedness):
                points = [{"x": round(p.x, 5), "y": round(p.y, 5),
                           "z": round(p.z, 5),
                           "px": round(p.x * width, 1),
                           "py": round(p.y * height, 1)} for p in lm]
                if wlm is not None:
                    # world landmarks:米制、以手几何中心为原点、尺度锚定规范手模型
                    for d, w in zip(points, wlm):
                        d.update(wx=round(w.x, 5), wy=round(w.y, 5), wz=round(w.z, 5))
                rows.append({
                    "frame_index": frame_index,
                    "timestamp_sec": round(frame_index / fps, 4),
                    "handedness": handed[0].category_name,
                    "handedness_score": round(float(handed[0].score), 4),
                    "landmarks": points,
                })
            processed += 1
        frame_index += 1
    cap.release()
    landmarker.close()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    import hashlib
    return {"records": len(rows), "frames_processed": processed,
            "frame_stride": stride, "fps": fps, "resolution": [width, height],
            "stereo_left_half_only": stereo_half,
            "world_landmarks": any(r["landmarks"] and "wx" in r["landmarks"][0]
                                   for r in rows),
            "model": {"path": MODEL_PATH.name,
                      "sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()},
            "evidence": "derived_mediapipe_hand_landmarker"}


# ---------------------------------------------------------------- 单目 3D 提升(PnP)

def lift_hands_3d(hands_jsonl: Path, output_jsonl: Path, resolution,
                  focal_px: float | None = None, hfov_deg: float | None = None,
                  principal_point=None, max_reproj_ok_px: float = 12.0) -> dict:
    """hands.jsonl(需含 wx/wy/wz)→ hands3d.jsonl:相机系米制 3D 手部轨迹。

    原理:MediaPipe world landmarks 是米制但以手心为原点的手内 3D(尺度锚定
    规范手模型);以它为物点、px/py 为像点,solvePnP 解出手在相机系的位姿,
    腕点深度随之得出。两条显式假设(写进收据,深度随之线性缩放):
      * 尺度锚 = 规范手尺寸 —— 真实手比规范手大 k 倍,深度就被低估 k 倍;
      * 针孔无畸变 —— 广角/鱼眼源须先去畸变或传标定焦距,否则画面边缘偏差大。
    焦距来源三级:focal_px(标定值)> hfov_deg > 默认 60° 水平视场假设,
    来源写进收据。逐帧记录重投影 RMS;PnP 失败/负深度/超阈值帧如实标 ok=false。
    """
    import cv2

    width, height = resolution
    if focal_px is not None:
        fx, source = float(focal_px), "calibrated_focal_px"
    else:
        hfov = hfov_deg if hfov_deg is not None else 60.0
        fx = 0.5 * width / math.tan(math.radians(hfov) / 2)
        source = f"assumed_hfov_{hfov:g}deg" if hfov_deg is not None \
            else "default_hfov_60deg"
    cx, cy = principal_point if principal_point else (width / 2.0, height / 2.0)
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)

    rows_out, depths, reprojs = [], [], []
    n_in = n_skip = n_bad = 0
    with Path(hands_jsonl).open() as fh:
        for line in fh:
            rec = json.loads(line)
            n_in += 1
            pts = rec.get("landmarks") or []
            if len(pts) < 6 or "wx" not in pts[0]:
                n_skip += 1
                continue
            obj = np.array([[p["wx"], p["wy"], p["wz"]] for p in pts])
            img = np.array([[p["px"], p["py"]] for p in pts])
            success, rvec, tvec = cv2.solvePnP(obj, img, K, None,
                                               flags=cv2.SOLVEPNP_SQPNP)
            row = {"frame_index": rec["frame_index"],
                   "timestamp_sec": rec["timestamp_sec"],
                   "handedness": rec.get("handedness"), "ok": False}
            if success:
                R, _ = cv2.Rodrigues(rvec)
                cam = obj @ R.T + tvec.reshape(3)
                proj = cam @ K.T
                proj = proj[:, :2] / proj[:, 2:3]
                rms = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))
                depth = float(cam[0, 2])              # 腕点 = landmark 0
                row.update(ok=depth > 0 and rms <= max_reproj_ok_px,
                           depth_m=round(depth, 4),
                           wrist_cam_m=[round(v, 4) for v in cam[0]],
                           pinch_width_m=round(float(np.linalg.norm(cam[4] - cam[8])), 4),
                           reproj_rms_px=round(rms, 2),
                           joints_cam_m=[[round(v, 4) for v in p] for p in cam])
            if row["ok"]:
                depths.append(row["depth_m"])
                reprojs.append(row["reproj_rms_px"])
            else:
                n_bad += 1
            rows_out.append(row)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as fh:
        for r in rows_out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"records": n_in, "lifted_ok": len(depths),
            "degenerate_or_failed": n_bad, "skipped_no_world": n_skip,
            "intrinsics": {"fx": round(fx, 2), "fy": round(fx, 2),
                           "cx": round(float(cx), 2), "cy": round(float(cy), 2),
                           "source": source},
            "depth_m_p50": round(float(np.median(depths)), 4) if depths else None,
            "reproj_rms_px_p50": round(float(np.median(reprojs)), 2) if reprojs else None,
            "assumptions": ["尺度锚=MediaPipe 规范手尺寸(深度随真实手尺寸线性缩放)",
                            "针孔无畸变(fy=fx)", f"焦距来源: {source}"],
            "evidence": "derived_pnp_metric_lift"}


# ---------------------------------------------------------------- 物体跟踪(老 tracking.py 口径)

def _clip_box(box, width, height):
    x, y, w, h = box
    x = min(max(x, 0.0), width - 1.0)
    y = min(max(y, 0.0), height - 1.0)
    return [x, y, min(max(w, 1.0), width - x), min(max(h, 1.0), height - y)]


def _features(cv2, gray, box):
    x, y, w, h = [int(round(v)) for v in box]
    mask = np.zeros_like(gray)
    mask[y:y + h, x:x + w] = 255
    return cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=80,
                                   qualityLevel=0.01, minDistance=5, blockSize=7)


def _track_flow(cv2, prev_gray, gray, state, width, height):
    points = state.get("points")
    if points is None or len(points) < 4:
        state["points"] = points = _features(cv2, prev_gray, state["box"])
    if points is None or len(points) < 4:
        state.update(confidence=0.0, status="lost")
        return
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, points, None, winSize=(31, 31), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if nxt is None or status is None:
        state.update(confidence=0.0, status="lost")
        return
    valid = status.reshape(-1) == 1
    old, new = points.reshape(-1, 2)[valid], nxt.reshape(-1, 2)[valid]
    if len(new) < 4:
        state.update(confidence=len(new) / max(len(points), 1), status="lost",
                     points=new.reshape(-1, 1, 2))
        return
    transform, inliers = cv2.estimateAffinePartial2D(
        old, new, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if transform is None:
        state.update(confidence=0.0, status="lost")
        return
    x, y, w, h = state["box"]
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                       dtype=np.float32).reshape(-1, 1, 2)
    t = cv2.transform(corners, transform).reshape(-1, 2)
    nb = [float(t[:, 0].min()), float(t[:, 1].min()),
          float(t[:, 0].max() - t[:, 0].min()), float(t[:, 1].max() - t[:, 1].min())]
    area_ratio = (nb[2] * nb[3]) / max(w * h, 1.0)
    center_shift = math.hypot(nb[0] + nb[2] / 2 - (x + w / 2), nb[1] + nb[3] / 2 - (y + h / 2))
    plausible = 0.6 <= area_ratio <= 1.7 and center_shift <= max(w, h) * 0.8
    inlier_ratio = float(inliers.reshape(-1).mean()) if inliers is not None else 0.0
    state["confidence"] = len(new) / max(len(points), 1) * inlier_ratio if plausible else 0.0
    state["status"] = "tracked" if plausible else "rejected_motion"
    if plausible:
        state["box"] = _clip_box(nb, width, height)
        state["points"] = new.reshape(-1, 1, 2)
        if len(new) < 20:
            rep = _features(cv2, gray, state["box"])
            if rep is not None:
                state["points"] = rep


def auto_seed_moving_object(video_path: Path, warmup_frames: int = 90) -> dict:
    """无人工种子时的降级方案:帧差累积找最大运动团块作初始框。
    信任级别低于人工画框,evidence 如实标 auto_motion_seed。"""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    ok, prev = cap.read()
    if not ok:
        raise ValueError("视频无可读帧")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    acc = np.zeros(prev_gray.shape, dtype=np.float32)
    for _ in range(warmup_frames):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        acc += cv2.absdiff(gray, prev_gray).astype(np.float32)
        prev_gray = gray
    cap.release()
    blur = cv2.GaussianBlur(acc, (21, 21), 0)
    _, mask = cv2.threshold((255 * blur / max(blur.max(), 1)).astype(np.uint8),
                            60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("帧差未找到运动区域,自动种子失败")
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return {"object_id": "auto0", "object_class": "moving_region",
            "bbox_xywh": [float(x), float(y), float(w), float(h)],
            "algorithm": "optical_flow", "evidence": "auto_motion_seed"}


def track_objects(video_path: Path, seeds: list, output_jsonl: Path,
                  max_frames: int = 1200) -> dict:
    """种子框逐帧跟踪(老口径:LK 光流 + 仿射 RANSAC + 面积/位移合理性闸)。"""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    if not ok:
        raise ValueError("视频无可读帧")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    states = []
    for seed in seeds:
        box = _clip_box([float(v) for v in seed["bbox_xywh"]], width, height)
        states.append({"object_id": seed["object_id"],
                       "object_class": seed.get("object_class", "object"),
                       "box": box, "points": _features(cv2, gray, box),
                       "confidence": 1.0, "status": "seeded",
                       "seed_evidence": seed.get("evidence", "observed_manual_seed")})
    rows, frame_index, prev_gray = [], 0, gray
    while ok and frame_index < max_frames:
        if frame_index > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for st in states:
                _track_flow(cv2, prev_gray, gray, st, width, height)
            prev_gray = gray
        for st in states:
            x, y, w, h = st["box"]
            rows.append({"frame_index": frame_index,
                         "timestamp_sec": round(frame_index / fps, 4),
                         "object_id": st["object_id"],
                         "object_class": st["object_class"],
                         "bbox_xywh_px": [round(v, 1) for v in (x, y, w, h)],
                         "tracking_confidence": round(st["confidence"], 4),
                         "tracking_status": st["status"],
                         "evidence": st["seed_evidence"] if frame_index == 0
                         else "derived_optical_flow"})
        frame_index += 1
        ok, frame = cap.read()
    cap.release()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    tracked = sum(1 for r in rows if r["tracking_status"] in ("seeded", "tracked"))
    return {"records": len(rows), "objects": len(states),
            "processed_frames": frame_index,
            "tracked_ratio": round(tracked / max(len(rows), 1), 4)}


# ---------------------------------------------------------------- 接触候选(老 contact.py 口径)

def _point_box_distance(px, py, box):
    x, y, w, h = box
    dx = max(x - px, 0.0, px - (x + w))
    dy = max(y - py, 0.0, py - (y + h))
    return math.hypot(dx, dy)


def label_contact_candidates(hands_rows: list, objects_rows: list,
                             margin_px: float = 24.0,
                             minimum_tracking_confidence: float = 0.2,
                             minimum_event_frames: int = 3,
                             minimum_event_score: float = 0.1) -> dict:
    objects_by_frame = defaultdict(list)
    for row in objects_rows:
        objects_by_frame[int(row["frame_index"])].append(row)
    candidates = []
    for hand in hands_rows:
        frame = int(hand["frame_index"])
        tips = [hand["landmarks"][i] for i in FINGERTIP_INDICES]
        for obj in objects_by_frame.get(frame, []):
            tc = float(obj["tracking_confidence"])
            if tc < minimum_tracking_confidence:
                continue
            box = [float(v) for v in obj["bbox_xywh_px"]]
            dists = [_point_box_distance(float(t["px"]), float(t["py"]), box)
                     for t in tips]
            dmin = min(dists)
            if dmin > margin_px:
                continue
            diagonal = math.hypot(box[2], box[3])
            proximity = math.exp(-dmin / max(diagonal * 0.08, 1.0))
            candidates.append({
                "frame_index": frame, "timestamp_sec": hand["timestamp_sec"],
                "handedness": hand["handedness"], "object_id": obj["object_id"],
                "object_class": obj["object_class"],
                "minimum_fingertip_distance_px": round(dmin, 2),
                "fingertips_inside_bbox": sum(d == 0.0 for d in dists),
                "tracking_confidence": tc,
                "contact_candidate_score": round(
                    proximity * tc * float(hand["handedness_score"]), 4),
                "contact_state": "candidate",
                "evidence": "derived_image_space_proximity"})
    grouped = defaultdict(list)
    for row in candidates:
        grouped[(row["handedness"], row["object_id"])].append(row)
    events, event_id = [], 0
    for (handedness, object_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: int(r["frame_index"]))
        runs, current = [], [rows[0]]
        for row in rows[1:]:
            if int(row["frame_index"]) <= int(current[-1]["frame_index"]) + 2:
                current.append(row)
            else:
                runs.append(current)
                current = [row]
        runs.append(current)
        for run in runs:
            frames = sorted({int(r["frame_index"]) for r in run})
            score = sum(float(r["contact_candidate_score"]) for r in run) / len(run)
            if len(frames) < minimum_event_frames or score < minimum_event_score:
                continue
            event_id += 1
            events.append({"event_id": event_id, "handedness": handedness,
                           "object_id": object_id,
                           "object_class": run[0]["object_class"],
                           "start_frame": frames[0], "end_frame": frames[-1],
                           "frame_count": len(frames),
                           "start_sec": min(float(r["timestamp_sec"]) for r in run),
                           "end_sec": max(float(r["timestamp_sec"]) for r in run),
                           "mean_candidate_score": round(score, 4),
                           "contact_state": "candidate_event",
                           "evidence": "derived_image_space_proximity_persistence"})
    return {"frame_candidates": candidates, "events": events,
            "contact_truth": "not_measured"}


# ---------------------------------------------------------------- 13 项手部质量(老 quality.py 口径)

def _percentile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    return float(s[min(int(q * len(s)), len(s) - 1)])


def hand_metrics(rows: list, frame_stride: int = 1) -> dict:
    by_frame = Counter(int(r["frame_index"]) for r in rows)
    by_hand = defaultdict(list)
    confidence, nonfinite, oob, n_landmarks = [], 0, 0, 0
    for r in rows:
        by_hand[str(r["handedness"])].append(r)
        confidence.append(float(r["handedness_score"]))
        for p in r["landmarks"]:
            n_landmarks += 1
            vals = [float(p[k]) for k in ("x", "y", "z", "px", "py")]
            if not all(math.isfinite(v) for v in vals):
                nonfinite += 1
            elif not (0.0 <= float(p["x"]) <= 1.0 and 0.0 <= float(p["y"]) <= 1.0):
                oob += 1
    frame_ids = sorted(by_frame)
    # 分母 = 实际被分析的帧数(按抽帧步长折算),否则 stride>1 时覆盖率被系统性稀释
    expected = ((frame_ids[-1] - frame_ids[0]) // max(frame_stride, 1) + 1
                if frame_ids else 0)
    steps, speeds = [], []
    for hand_rows in by_hand.values():
        hand_rows.sort(key=lambda r: int(r["frame_index"]))
        for prev, cur in zip(hand_rows, hand_rows[1:]):
            dt = float(cur["timestamp_sec"]) - float(prev["timestamp_sec"])
            if dt <= 0:
                continue
            p0, p1 = prev["landmarks"][0], cur["landmarks"][0]
            d = math.sqrt(sum((float(p1[k]) - float(p0[k])) ** 2 for k in ("x", "y", "z")))
            steps.append(d)
            speeds.append(d / dt)
    return {
        "records": len(rows), "unique_frames": len(frame_ids),
        "expected_frames": expected,
        "frame_coverage": len(frame_ids) / expected if expected else 0.0,
        "two_hand_coverage": (sum(c >= 2 for c in by_frame.values()) / expected
                              if expected else 0.0),
        "mean_confidence": (sum(confidence) / len(confidence)) if confidence else 0.0,
        "p05_confidence": _percentile(confidence, 0.05),
        "min_confidence": min(confidence) if confidence else 0.0,
        "nonfinite_landmark_rate": nonfinite / n_landmarks if n_landmarks else 0.0,
        "out_of_bounds_landmark_rate": oob / n_landmarks if n_landmarks else 0.0,
        "wrist_step_p95": round(_percentile(steps, 0.95), 5),
        "wrist_step_max": round(max(steps), 5) if steps else 0.0,
        "wrist_speed_p95": round(_percentile(speeds, 0.95), 4),
        "wrist_speed_max": round(max(speeds), 4) if speeds else 0.0,
    }
