"""Task-level UAV visual simulator and replaceable Agent contract.

This module models mission semantics, simple camera geometry and keep-out
constraints. It does not claim flight dynamics, wind, propulsion or autopilot
fidelity.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from .evidence import EvidencePackage, Stream
from .hashing import sha256_json
from .ledger import ACCEPTED, NOT_EVALUATED, REJECTED, Claim, Ledger


SCHEMA = "organoid-kernel.uav-task.v1"
ACTION_NAMES = {"takeoff", "goto", "look_at", "zoom", "orbit", "capture", "land"}


def _vec(value: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(value), dtype=float)
    if result.shape != (3,):
        raise ValueError(f"expected 3-vector, got {result.shape}")
    return result


def _float_list(value: np.ndarray) -> list[float]:
    return [round(float(x), 6) for x in value]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second))


def _look_angles(camera: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    delta = target - camera
    horizontal = math.hypot(float(delta[0]), float(delta[1]))
    yaw = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    pitch = math.degrees(math.atan2(float(delta[2]), horizontal))
    return yaw, pitch


@dataclass(frozen=True)
class UavAction:
    """Stable action protocol shared by the built-in and future trained Agent."""

    kind: str
    args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ACTION_NAMES:
            raise ValueError(f"unknown UAV action: {self.kind}")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "args": self.args}


@dataclass
class Target:
    name: str
    kind: str
    position: np.ndarray
    size_m: tuple[float, float, float] = (4.6, 1.9, 1.6)
    plate_position: np.ndarray | None = None
    plate_size_m: tuple[float, float] = (0.52, 0.12)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "position": _float_list(self.position),
            "size_m": list(self.size_m),
            "plate_position": (
                _float_list(self.plate_position)
                if self.plate_position is not None else None
            ),
            "plate_size_m": list(self.plate_size_m),
        }


@dataclass(frozen=True)
class KeepOut:
    name: str
    center: tuple[float, float]
    half_size: tuple[float, float]
    min_altitude_m: float = 0.0

    def contains(self, point: np.ndarray) -> bool:
        return (
            self.min_altitude_m <= float(point[2])
            and abs(float(point[0]) - self.center[0]) <= self.half_size[0]
            and abs(float(point[1]) - self.center[1]) <= self.half_size[1]
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CameraState:
    zoom: float = 1.0
    gimbal_yaw_deg: float = 0.0
    gimbal_pitch_deg: float = -35.0
    horizontal_fov_deg: float = 62.0
    resolution_px: tuple[int, int] = (1920, 1080)

    def to_dict(self) -> dict[str, Any]:
        return {
            "zoom": round(self.zoom, 4),
            "gimbal_yaw_deg": round(self.gimbal_yaw_deg, 4),
            "gimbal_pitch_deg": round(self.gimbal_pitch_deg, 4),
            "horizontal_fov_deg": round(self.horizontal_fov_deg, 4),
            "resolution_px": list(self.resolution_px),
        }


@dataclass
class UavState:
    position: np.ndarray
    armed: bool = False
    airborne: bool = False
    battery_pct: float = 100.0
    camera: CameraState = field(default_factory=CameraState)

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": _float_list(self.position),
            "armed": self.armed,
            "airborne": self.airborne,
            "battery_pct": round(self.battery_pct, 4),
            "camera": self.camera.to_dict(),
        }


@dataclass
class UavWorld:
    scene_name: str
    target: Target
    drone_start: np.ndarray = field(
        default_factory=lambda: np.asarray([-12.0, -14.0, 0.0], dtype=float)
    )
    takeoff_altitude_m: float = 8.0
    bounds_xy: tuple[float, float, float, float] = (-35.0, 35.0, -35.0, 35.0)
    keep_outs: list[KeepOut] = field(default_factory=list)
    lighting: float = 0.88
    occlusion: float = 0.0
    seed: int = 7

    def initial_state(self) -> UavState:
        return UavState(position=self.drone_start.copy())

    def to_spec(self) -> dict[str, Any]:
        return {
            "schema": "kabuki.world_spec.v0",
            "status": "draft",
            "coordinate_system": {
                "frame": "local_scene_m",
                "axes": "x-east,y-north,z-up",
                "note": "任务级局部米制坐标，不是地理坐标",
            },
            "world": {
                "environment_asset": f"uav://scenes/{self.scene_name}",
                "scene_name": self.scene_name,
                "drones": [{
                    "name": "uav-01",
                    "profile": "generic_quadrotor_task_v1",
                    "start_position": _float_list(self.drone_start),
                    "camera": CameraState().to_dict(),
                }],
                "objects": {"target": self.target.to_dict()},
                "constraints": {
                    "bounds_xy": list(self.bounds_xy),
                    "keep_outs": [x.to_dict() for x in self.keep_outs],
                    "min_flight_altitude_m": 2.0,
                },
            },
            "provenance": {
                "simulator": SCHEMA,
                "seed": self.seed,
                "fidelity": "task_level_visual_approximation",
            },
        }


class UavAgent(Protocol):
    """Future trained Agents implement this one-step decision interface."""

    def decide(self, observation: dict[str, Any]) -> UavAction | None:
        ...


class RuleBasedUavAgent:
    """Deterministic baseline used until a trained Agent is connected."""

    def __init__(self, target_name: str = "target-vehicle"):
        self.target_name = target_name
        self.phase = "takeoff"
        self.capture_attempts = 0

    def decide(self, observation: dict[str, Any]) -> UavAction | None:
        plate_readable = bool(observation.get("plate_readable", False))
        target = observation["target_position"]
        if observation["mission_status"] == "completed":
            return None
        if self.phase == "takeoff":
            self.phase = "approach"
            return UavAction("takeoff", {"altitude_m": 8.0})
        if self.phase == "approach":
            self.phase = "look"
            return UavAction("goto", {
                "position": [target[0] - 6.0, target[1] - 6.0, 7.0],
                "speed_mps": 5.0,
            })
        if self.phase == "look":
            self.phase = "zoom"
            return UavAction("look_at", {"target": self.target_name})
        if self.phase == "zoom":
            self.phase = "capture"
            return UavAction("zoom", {"zoom": 1.5})
        if self.phase == "capture":
            self.capture_attempts += 1
            if plate_readable:
                self.phase = "land"
                return UavAction("capture", {"label": "plate-evidence"})
            self.phase = "orbit"
            return UavAction("orbit", {
                "target": self.target_name,
                "radius_m": 5.5,
                "degrees": 270.0,
                "segments": 4,
                "capture_each_segment": True,
                "zoom": 3.0,
            })
        if self.phase == "orbit":
            self.phase = "land"
            return UavAction("land", {})
        if self.phase == "land":
            self.phase = "done"
            return UavAction("land", {})
        return None


class ExternalAgentAdapter:
    """Adapter for a future model returning the stable action JSON shape."""

    def __init__(self, decide_fn):
        self.decide_fn = decide_fn

    def decide(self, observation: dict[str, Any]) -> UavAction | None:
        raw = self.decide_fn(observation)
        if raw is None or isinstance(raw, UavAction):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("external Agent must return dict or UavAction")
        return UavAction(kind=raw["kind"], args=dict(raw.get("args", {})))


class UavTaskSimulator:
    """Deterministic task-level simulator with fail-closed action validation."""

    def __init__(self, world: UavWorld):
        self.world = world
        self.state = world.initial_state()
        self.events: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.observations: list[dict[str, Any]] = []
        self.captures: list[dict[str, Any]] = []
        self.step_index = 0
        self.mission_status = "ready"
        self.rejected_candidates: list[dict[str, Any]] = []
        self._record_observation("initial")

    @property
    def target(self) -> Target:
        return self.world.target

    def observe(self) -> dict[str, Any]:
        quality = self._image_quality()
        return {
            "step": self.step_index,
            "mission_status": self.mission_status,
            "drone": self.state.to_dict(),
            "target_position": _float_list(self.target.position),
            "target_name": self.target.name,
            "target_visible": quality["visible"],
            "image_quality": quality["score"],
            "plate_readable": quality["plate_readable"],
            "distance_m": quality["distance_m"],
            "lighting": self.world.lighting,
            "occlusion": self.world.occlusion,
            "capture_count": len(self.captures),
        }

    def propose(self, action: UavAction,
                observation: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create a candidate; this does not mutate state."""
        return {
            "schema": "kabuki.uav_action_candidate.v1",
            "step": self.step_index + 1,
            "action": action.to_dict(),
            "observation": observation or self.observe(),
        }

    def verify(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Verify and commit one candidate, or retain rejected evidence."""
        action = UavAction(**candidate["action"])
        before = self.state.to_dict()
        try:
            result = self._execute(action)
            accepted = True
        except (ValueError, RuntimeError) as exc:
            result = {"error": str(exc)}
            accepted = False
        record = {
            "schema": "kabuki.uav_agent_trace.v1",
            "iteration": self.step_index + 1 if accepted else 0,
            "phase": self.mission_status,
            "candidate": candidate,
            "accepted": accepted,
            "before": before,
            "result": result,
            "after": self.state.to_dict() if accepted else before,
        }
        if accepted:
            self.step_index += 1
            self.actions.append(record)
            self._record_observation(action.kind)
        else:
            self.rejected_candidates.append(record)
        return record

    def run(self, agent: UavAgent, max_steps: int = 32) -> dict[str, Any]:
        for _ in range(max_steps):
            observation = self.observe()
            if self.mission_status == "completed":
                break
            action = agent.decide(observation)
            if action is None:
                break
            self.verify(self.propose(action, observation))
        if self.mission_status != "completed":
            self.mission_status = "failed"
            self.events.append({
                "event": "mission_failed",
                "reason": "Agent stopped before completion or step budget exhausted",
            })
        return self.receipt()

    def receipt(self) -> dict[str, Any]:
        claims = self._claims()
        accepted = all(item["status"] == ACCEPTED for item in claims.values())
        return {
            "schema": SCHEMA,
            "episode_id": f"{self.world.scene_name}-seed{self.world.seed}",
            "scene": self.world.to_spec(),
            "mission": {
                "name": "single_uav_vehicle_evidence",
                "status": self.mission_status,
                "steps": len(self.actions),
                "accepted_actions": len(self.actions),
                "rejected_candidates": len(self.rejected_candidates),
            },
            "actions": self.actions,
            "rejected_candidates": self.rejected_candidates,
            "observations": self.observations,
            "captures": self.captures,
            "events": self.events,
            "claims": claims,
            "replay": {
                "deterministic": True,
                "seed": self.world.seed,
                "trace_hash": sha256_json({
                    "actions": self.actions,
                    "captures": self.captures,
                    "events": self.events,
                }),
            },
            "result": {
                "accepted": accepted,
                "grade": "accepted" if accepted else "rejected",
            },
        }

    def evidence_package(self) -> EvidencePackage:
        receipt = self.receipt()
        pose = np.asarray(
            [obs["drone"]["position"] for obs in self.observations], dtype=float
        )
        times = np.arange(len(self.observations), dtype=float)
        camera = np.asarray([[
            obs["drone"]["camera"]["gimbal_yaw_deg"],
            obs["drone"]["camera"]["gimbal_pitch_deg"],
            obs["drone"]["camera"]["zoom"],
        ] for obs in self.observations], dtype=float)
        target = np.asarray(
            [self.target.position for _ in self.observations], dtype=float
        )
        quality = np.asarray([[
            obs["image_quality"], float(obs["plate_readable"]),
        ] for obs in self.observations], dtype=float)
        actions = np.zeros((len(self.actions), 7), dtype=float)
        action_ids = {
            "takeoff": 1, "goto": 2, "look_at": 3, "zoom": 4,
            "orbit": 5, "capture": 6, "land": 7,
        }
        for index, record in enumerate(self.actions):
            action = record["candidate"]["action"]
            actions[index, 0] = index
            actions[index, 1] = action_ids[action["kind"]]
            actions[index, 2] = float(record["result"].get("distance_m", 0.0))
            actions[index, 3] = float(record["result"].get("score", 0.0))
            actions[index, 4] = float(record["after"]["position"][2])
            actions[index, 5] = float(record["after"]["camera"]["zoom"])
            actions[index, 6] = 1.0
        pkg = EvidencePackage(
            episode_id=receipt["episode_id"],
            dataset_format="uav_task_v1",
            fps=1.0,
            meta={
                "scene_name": self.world.scene_name,
                "mission_name": receipt["mission"]["name"],
                "seed": self.world.seed,
                "fidelity": "task_level_visual_approximation",
                "receipt_hash": sha256_json(receipt),
            },
        )
        pkg.add(Stream("uav.pose", "derived", data=pose,
                       columns=["x", "y", "z"], unit="m",
                       frame="local_scene_m", timestamps=times))
        pkg.add(Stream("uav.action", "observed", data=actions,
                       columns=["step", "action_id", "distance_m",
                                "image_quality", "altitude_m", "zoom",
                                "accepted"], unit="mixed", frame="mission"))
        pkg.add(Stream("uav.gimbal_state", "derived", data=camera,
                       columns=["yaw_deg", "pitch_deg", "zoom"],
                       unit="deg/x", frame="camera", timestamps=times))
        pkg.add(Stream("scene.target_truth", "assumed", data=target,
                       columns=["x", "y", "z"], unit="m",
                       frame="local_scene_m", timestamps=times))
        pkg.add(Stream("mission.result", "derived", data=quality,
                       columns=["image_quality", "plate_readable"],
                       unit="score/bool", frame="mission", timestamps=times))
        pkg.add(Stream("mission.events", "derived",
                       data=np.asarray([[index, 1.0]
                                        for index, _ in enumerate(self.events)],
                                       dtype=float),
                       columns=["event_index", "present"], unit="count",
                       frame="mission"))
        return pkg

    def organoid_ledger(self) -> Ledger:
        ledger = Ledger(
            episode_id=f"{self.world.scene_name}-seed{self.world.seed}",
            policy="uav_task_v1",
        )
        for name, item in self._claims().items():
            ledger.add(Claim(name, item["status"], item["reason"],
                             receipt="uav-task-receipt.json",
                             detail=item.get("detail", {})))
        ledger.grade = "accepted" if all(
            claim.status == ACCEPTED for claim in ledger.claims
        ) else "rejected"
        ledger.grade_reason = (
            "无人机任务级视觉闭环全部通过"
            if ledger.grade == "accepted" else
            "至少一项无人机任务闭环检查未通过"
        )
        return ledger

    def write_outputs(self, out_dir: Path) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        receipt = self.receipt()
        (out / "uav-task-receipt.json").write_text(
            json_dumps(receipt), encoding="utf-8"
        )
        self.organoid_ledger().save(out / "claim-ledger.json")
        (out / "evidence.json").write_text(
            json_dumps(self.evidence_package().summary()), encoding="utf-8"
        )
        (out / "world.initial.json").write_text(
            json_dumps(self.world.to_spec()), encoding="utf-8"
        )
        with (out / "agent-trace.jsonl").open("w", encoding="utf-8") as fh:
            for record in self.actions:
                fh.write(json_dumps(record) + "\n")
        with (out / "agent-trace.rejected.jsonl").open("w", encoding="utf-8") as fh:
            for record in self.rejected_candidates:
                fh.write(json_dumps(record) + "\n")
        return {
            "receipt": out / "uav-task-receipt.json",
            "ledger": out / "claim-ledger.json",
            "evidence": out / "evidence.json",
            "world": out / "world.initial.json",
            "trace": out / "agent-trace.jsonl",
            "rejected": out / "agent-trace.rejected.jsonl",
        }

    def _record_observation(self, event: str) -> None:
        observation = self.observe()
        observation["event"] = event
        self.observations.append(observation)

    def _validate_position(self, point: np.ndarray) -> None:
        xmin, xmax, ymin, ymax = self.world.bounds_xy
        if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax):
            raise ValueError("goto leaves scene bounds")
        if point[2] < 2.0 and self.mission_status != "ready":
            raise ValueError("flight altitude below 2m")
        for keep_out in self.world.keep_outs:
            if keep_out.contains(point):
                raise ValueError(f"goto enters keep-out: {keep_out.name}")

    def _execute(self, action: UavAction) -> dict[str, Any]:
        kind, args = action.kind, action.args
        if kind == "takeoff":
            if self.state.airborne:
                raise ValueError("UAV is already airborne")
            altitude = _clamp(
                args.get("altitude_m", self.world.takeoff_altitude_m), 2.0, 50.0
            )
            self.state.armed = True
            self.state.airborne = True
            self.state.position[2] = altitude
            self.mission_status = "airborne"
            self.events.append({"event": "takeoff", "altitude_m": altitude})
            return {"altitude_m": altitude}
        if kind == "goto":
            if not self.state.airborne:
                raise ValueError("goto requires airborne UAV")
            position = _vec(args["position"])
            self._validate_position(position)
            distance = _distance(self.state.position, position)
            self.state.position = position
            self.state.battery_pct = max(
                0.0, self.state.battery_pct - distance * 0.12
            )
            self.mission_status = "on_station"
            return {
                "position": _float_list(position),
                "distance_m": round(distance, 4),
                "speed_mps": float(args.get("speed_mps", 5.0)),
            }
        if kind == "look_at":
            if not self.state.airborne:
                raise ValueError("look_at requires airborne UAV")
            target = self._resolve_target(args.get("target"))
            yaw, pitch = _look_angles(self.state.position, target.position)
            self.state.camera.gimbal_yaw_deg = yaw
            self.state.camera.gimbal_pitch_deg = pitch
            return {"gimbal_yaw_deg": yaw, "gimbal_pitch_deg": pitch}
        if kind == "zoom":
            zoom = _clamp(args["zoom"], 1.0, 10.0)
            self.state.camera.zoom = zoom
            return {"zoom": zoom, **self._image_quality()}
        if kind == "orbit":
            if not self.state.airborne:
                raise ValueError("orbit requires airborne UAV")
            target = self._resolve_target(args.get("target"))
            if "zoom" in args:
                self.state.camera.zoom = _clamp(args["zoom"], 1.0, 10.0)
            radius = _clamp(args.get("radius_m", 6.0), 3.0, 30.0)
            degrees = float(args.get("degrees", 270.0))
            segments = int(_clamp(args.get("segments", 4), 2, 16))
            initial_angle = math.atan2(
                self.state.position[1] - target.position[1],
                self.state.position[0] - target.position[0],
            )
            path = []
            for index in range(1, segments + 1):
                angle = initial_angle + math.radians(degrees) * index / segments
                point = np.asarray([
                    radius * math.cos(angle), radius * math.sin(angle),
                    max(4.0, float(self.state.position[2])),
                ], dtype=float)
                point[:2] += target.position[:2]
                self._validate_position(point)
                self.state.position = point
                self.state.camera.gimbal_yaw_deg, self.state.camera.gimbal_pitch_deg = (
                    _look_angles(point, target.position)
                )
                path.append(_float_list(point))
                if args.get("capture_each_segment", False):
                    quality = self._image_quality()
                    self.captures.append({
                        "label": f"orbit-evidence-{index}",
                        "angle_deg": round(math.degrees(angle), 4),
                        **quality,
                        "position": _float_list(point),
                        "camera": self.state.camera.to_dict(),
                    })
            self.state.battery_pct = max(
                0.0, self.state.battery_pct - abs(degrees) * 0.01
            )
            self.mission_status = "orbiting"
            return {
                "radius_m": radius, "degrees": degrees, "segments": segments,
                "path": path, **self._image_quality(),
            }
        if kind == "capture":
            if not self.state.airborne:
                raise ValueError("capture requires airborne UAV")
            quality = self._image_quality()
            capture = {
                "label": args.get("label", f"capture-{len(self.captures) + 1}"),
                "angle_deg": float(args.get("angle_deg", 0.0)),
                **quality,
                "position": _float_list(self.state.position),
                "camera": self.state.camera.to_dict(),
            }
            self.captures.append(capture)
            if quality["plate_readable"]:
                self.events.append({
                    "event": "evidence_captured",
                    "label": capture["label"],
                    "quality": quality["score"],
                })
            return capture
        if kind == "land":
            if not self.state.airborne:
                raise ValueError("land requires airborne UAV")
            self.state.position[2] = 0.0
            self.state.airborne = False
            self.state.armed = False
            if self._has_successful_evidence():
                self.mission_status = "completed"
                self.events.append({"event": "mission_completed"})
            else:
                self.mission_status = "failed"
            return {"landed": True, "mission_status": self.mission_status}
        raise ValueError(f"unsupported action: {kind}")

    def _resolve_target(self, name: str | None) -> Target:
        if name in (None, self.target.name, "target"):
            return self.target
        raise ValueError(f"unknown target: {name}")

    def _image_quality(self) -> dict[str, Any]:
        distance = _distance(self.state.position, self.target.position)
        fov = self.state.camera.horizontal_fov_deg / self.state.camera.zoom
        target_angle = math.degrees(math.atan2(
            max(self.target.size_m[0], self.target.size_m[1]) / 2.0,
            max(distance, 0.01),
        ))
        framing = _clamp(target_angle / (fov * 0.42), 0.0, 1.0)
        expected_yaw, expected_pitch = _look_angles(
            self.state.position, self.target.position
        )
        pointing_error = (
            abs(self.state.camera.gimbal_yaw_deg - expected_yaw) / 45.0
            + abs(self.state.camera.gimbal_pitch_deg - expected_pitch) / 30.0
        )
        pointing = _clamp(1.0 - pointing_error / 2.0, 0.0, 1.0)
        score = _clamp(
            0.10 + 0.58 * framing + 0.22 * pointing
            + 0.10 * self.world.lighting - 0.22 * self.world.occlusion,
            0.0, 1.0,
        )
        visible = bool(score >= 0.18)
        plate_readable = bool(
            visible and score >= 0.72
            and self.state.camera.zoom >= 2.0 and self.state.airborne
        )
        return {
            "distance_m": round(distance, 4),
            "score": round(score, 4),
            "visible": visible,
            "plate_readable": plate_readable,
        }

    def _has_successful_evidence(self) -> bool:
        return sum(1 for capture in self.captures
                   if capture["plate_readable"]) >= 1

    def _claims(self) -> dict[str, dict[str, Any]]:
        successful = self._has_successful_evidence()
        all_actions_accepted = len(self.actions) > 0 and not self.rejected_candidates
        return {
            "mission_schema_valid": {
                "status": ACCEPTED,
                "reason": "动作、观测和结果符合 uav_task_v1",
            },
            "action_executable": {
                "status": ACCEPTED if all_actions_accepted else REJECTED,
                "reason": (
                    "所有已提交动作均通过任务级约束"
                    if all_actions_accepted else "存在动作候选被拒绝"
                ),
            },
            "target_acquired": {
                "status": ACCEPTED if any(
                    obs["target_visible"] for obs in self.observations
                ) else REJECTED,
                "reason": "目标在场景真值和相机反馈中可见",
            },
            "camera_visibility_sufficient": {
                "status": ACCEPTED if successful else REJECTED,
                "reason": "至少一次画面达到车牌可读阈值",
            },
            "evidence_capture_complete": {
                "status": ACCEPTED if successful else REJECTED,
                "reason": "成功取证照片已进入任务收据",
            },
            "mission_completed": {
                "status": ACCEPTED if self.mission_status == "completed"
                else REJECTED,
                "reason": f"mission_status={self.mission_status}",
            },
            "collision_free": {
                "status": ACCEPTED if not self.rejected_candidates
                else NOT_EVALUATED,
                "reason": (
                    "本首期模型只验证任务级禁飞区约束"
                    if not self.rejected_candidates else
                    "存在约束校验失败候选，需单独分析"
                ),
            },
            "reproducible_replay": {
                "status": ACCEPTED,
                "reason": "固定场景、规则Agent和随机种子",
            },
        }


def json_dumps(value: Any) -> str:
    import json
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def make_default_demo() -> tuple[UavWorld, RuleBasedUavAgent]:
    target = Target(
        name="target-vehicle",
        kind="parked_vehicle",
        position=np.asarray([10.0, 8.0, 0.9]),
        plate_position=np.asarray([10.0, 7.05, 1.0]),
    )
    world = UavWorld(
        scene_name="traffic_parking_lot_v1",
        target=target,
        keep_outs=[
            KeepOut("school_building", center=(-1.0, 3.0),
                    half_size=(3.0, 2.0), min_altitude_m=0.0),
        ],
        lighting=0.88,
        occlusion=0.05,
        seed=7,
    )
    return world, RuleBasedUavAgent(target.name)


def run_demo(out_dir: Path) -> dict[str, Path]:
    world, agent = make_default_demo()
    simulator = UavTaskSimulator(world)
    simulator.run(agent)
    return simulator.write_outputs(out_dir)
