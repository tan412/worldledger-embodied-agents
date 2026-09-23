"""Feedback-conditioned continuous action policy and outcome model.

This is reward-weighted conditional imitation, not online RL or a VLA.
The actor learns a distribution of task-space corrections; the critic ranks
samples. Neither receives true scene parameters or authorizes acceptance.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .autonomous_grasp import validate_action
from .hashing import sha256_json
from .kuavo_recovery import write_json

SCHEMA = "organoid.kuavo-recovery-policy.v1"
REASONS = ("no_bilateral_grasp", "grasp_lost", "forbidden_contact",
           "ik_tolerance_exceeded", "visual_placement_error")
FEATURE_NAMES = (
    *("object_" + s for s in ("x", "y", "z")), "width", "table_z",
    *("goal_" + s for s in ("x", "y", "z")),
    *("previous_grasp_" + s for s in ("x", "y", "z")), "previous_close", "previous_lift",
    *("reason_" + r for r in REASONS), "contact_fraction", "contact_observed",
    "front_normal_force", "back_normal_force",
    *("q_" + str(i) for i in range(7)), *("dq_" + str(i) for i in range(7)),
    "tcp_x", "tcp_y", "tcp_z", "claw_angle",
)
PARAMETER_NAMES = ("dx_m", "dy_m", "height_above_table_m", "close_angle_rad", "lift_m")
LOW = np.array([-.008, -.008, .027, -.50, .043])
HIGH = np.array([.008, .008, .061, -.13, .055])
PRIOR = np.array([0., 0., .043, -.25, .049])

RESIDUAL_SCHEMA = "organoid.kuavo-residual-policy.v1"


def residual_features(feedback, previous_feedback=None):
    """Observation-only residual view; missing history stays explicitly absent."""
    current = features(feedback)
    if previous_feedback is None:
        previous = current.copy()
        history_mask = 0.0
    else:
        previous = features(previous_feedback)
        history_mask = 1.0
    delta = current - previous
    result = np.r_[current, delta, history_mask].astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Malformed residual policy input")
    return result


class ResidualCorrectionPolicy(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96), nn.Tanh(),
            nn.Linear(96, 48), nn.Tanh(),
            nn.Linear(48, len(LOW)),
        )

    def forward(self, x):
        return torch.tanh(self.net(x))


def train_residual_policy(rows, output, seed=92614, epochs=400):
    """Train only bounded residuals around PRIOR; no rollout outcomes at test time."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    usable = [r for r in rows if r["split"] in ("train", "validation")
              and r["labels"] is not None]
    train_ids = {r["family"] for r in usable if r["split"] == "train"}
    valid_ids = {r["family"] for r in usable if r["split"] == "validation"}
    if not usable or train_ids & valid_ids:
        raise ValueError("Need nonempty, scene-disjoint train/validation rows")
    x = np.stack([residual_features(r["feedback"]) for r in usable])
    a = np.stack([encode_action(r["feedback"], r["action"]) for r in usable])
    prior = 2 * (PRIOR - LOW) / (HIGH - LOW) - 1
    y = np.clip(a - prior, -.75, .75).astype(np.float32)
    tr = np.asarray([i for i, r in enumerate(usable) if r["split"] == "train"])
    va = np.asarray([i for i, r in enumerate(usable) if r["split"] == "validation"])
    mean, std = x[tr].mean(0), x[tr].std(0)
    std[std < 1e-4] = 1.
    xt = torch.from_numpy((x - mean) / std)
    yt = torch.from_numpy(y)
    model = ResidualCorrectionPolicy(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.003)
    best, best_loss, best_epoch = None, float("inf"), 0
    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(xt[tr])
        loss = nn.functional.smooth_l1_loss(pred, yt[tr])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step()
        model.eval()
        with torch.no_grad():
            val = nn.functional.smooth_l1_loss(model(xt[va]), yt[va])
        if epoch % 10 == 0:
            history.append({"epoch": epoch, "train_loss": float(loss),
                            "validation_loss": float(val)})
        if float(val) < best_loss:
            best_loss, best_epoch = float(val), epoch
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= 80:
            break
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": RESIDUAL_SCHEMA, "model": best, "mean": torch.tensor(mean),
        "std": torch.tensor(std), "input_dim": x.shape[1],
        "feature_names": list(FEATURE_NAMES) + ["delta." + n for n in FEATURE_NAMES] + ["history_mask"],
        "parameter_names": list(PARAMETER_NAMES), "prior": PRIOR.tolist(),
        "train_families": sorted(train_ids), "validation_families": sorted(valid_ids),
        "training_rows_sha256": sha256_json(usable), "execution_authorized": False,
    }
    torch.save(checkpoint, output / "residual_policy.pt")
    write_json(output / "residual-training-report.json", {
        "schema": RESIDUAL_SCHEMA, "train_rows": len(tr), "validation_rows": len(va),
        "selected_epoch": best_epoch, "best_validation_loss": best_loss,
        "history": history, "uses_privileged_scene_truth": False,
        "uses_online_test_updates": False, "scope": "offline bounded residual imitation",
    })
    return checkpoint


def features(feedback):
    o, a = feedback["initial_observation"], feedback["previous_action"]
    if o["status"] != "observed":
        raise ValueError("Unavailable observation must not become a zero-filled feature")
    contact = feedback["last_contact_fraction"]
    values = [
        *o["center_m"], o["width_m"], o["table_z_m"], *o["goal_m"],
        *a["grasp_m"], a["close_angle"], a["lift_m"],
        *(float(feedback["stop_reason"].startswith(r)) for r in REASONS),
        0. if contact is None else contact, float(contact is not None),
        *feedback["finger_normal_force_N"], *feedback["joint_position_rad"],
        *feedback["joint_velocity_rad_s"], *feedback["tcp_position_m"],
        feedback["observed_claw_angle_rad"],
    ]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Malformed policy input")
    return result


def encode_action(feedback, action):
    o = feedback["initial_observation"]
    p = np.array([action["grasp_m"][0] - o["center_m"][0],
                  action["grasp_m"][1] - o["center_m"][1],
                  action["grasp_m"][2] - o["table_z_m"],
                  action["close_angle"], action["lift_m"]])
    return (2 * (p - LOW) / (HIGH - LOW) - 1).astype(np.float32)


def decode_action(feedback, parameters, origin):
    p = np.asarray(parameters)
    if p.shape != LOW.shape or not np.isfinite(p).all():
        raise ValueError("Malformed continuous action")
    p = LOW + (np.clip(p, -1, 1) + 1) * .5 * (HIGH - LOW)
    o = feedback["initial_observation"]
    return validate_action({
        "grasp_m": [float(o["center_m"][0] + p[0]), float(o["center_m"][1] + p[1]),
                    float(o["table_z_m"] + p[2])],
        "goal_m": [float(o["goal_m"][0]), float(o["goal_m"][1]), float(o["table_z_m"] + p[2])],
        "close_angle": float(p[3]), "lift_m": float(p[4]), "via_xy": [], "origin": origin,
    })


def exploration(feedback, rng, index):
    """Fixed data-collection distribution, independent of any test result."""
    p = 2 * (PRIOR - LOW) / (HIGH - LOW) - 1
    if index % 4 == 3:
        p = rng.uniform(-1, 1, len(LOW))
    elif index % 4:
        p += rng.normal(0, [.30, .30, .40, .45, .25])
    return decode_action(feedback, p, "untrained_continuous_prior_search")


def outcome_labels(record):
    if record["status"] == "not_evaluated":
        return None
    receipt = record["receipt"]
    if receipt.get("replay", {}).get("verified") is not True:
        return None
    success = record["status"] == "accepted"
    obj = receipt["objects"]["orange"]
    reason = record["reason"]
    checks = receipt["checks"]
    # Witnessed progress is auxiliary reward, never the success acceptance label.
    reward = 4. * success
    reward += .5 * min(1., max(0., obj["lift_m"]) / .04)
    reward += .5 * (obj["bilateral_transport_fraction"] or 0.)
    reward += .25 * bool(checks["object_in_bin"])
    if not checks["no_forbidden_contact"]:
        reward -= 1.
    if reason.startswith("ik_tolerance_exceeded"):
        reward -= .5
    return {
        "success": float(success), "reward": float(reward),
        "ik_failure": reason.startswith("ik_tolerance_exceeded"),
        "contact_failure": reason in ("no_bilateral_grasp", "grasp_lost"),
        "causal_force_insufficiency": None,
    }


class CorrectionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 64), nn.Tanh(),
                                 nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, len(LOW)))
        self.log_std = nn.Parameter(torch.full((len(LOW),), -.9))

    def forward(self, x):
        return torch.tanh(self.net(x)), self.log_std.clamp(-2.5, -.3).exp()


class OutcomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(FEATURE_NAMES) + len(LOW), 64), nn.Tanh(),
                                 nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, x, action):
        return self.net(torch.cat((x, action), dim=-1))


def train_policy(rows, output, seed=9211, epochs=500):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    usable = [r for r in rows if r["split"] in ("train", "validation") and r["labels"] is not None]
    if {r["split"] for r in usable} != {"train", "validation"}:
        raise ValueError("Independent training and validation scenes required")
    train_ids = {r["family"] for r in usable if r["split"] == "train"}
    validation_ids = {r["family"] for r in usable if r["split"] == "validation"}
    if train_ids & validation_ids:
        raise ValueError("Scene-family leakage")
    x = np.stack([features(r["feedback"]) for r in usable])
    a = np.stack([encode_action(r["feedback"], r["action"]) for r in usable])
    reward = np.array([r["labels"]["reward"] for r in usable], dtype=np.float32)
    success = np.array([r["labels"]["success"] for r in usable], dtype=np.float32)
    tr = np.array([i for i, r in enumerate(usable) if r["split"] == "train"])
    va = np.array([i for i, r in enumerate(usable) if r["split"] == "validation"])
    mean, std = x[tr].mean(0), x[tr].std(0)
    std[std < 1e-4] = 1.
    xt, at = torch.from_numpy((x - mean) / std), torch.from_numpy(a)
    yt = torch.from_numpy(np.column_stack((success, reward)))
    # Successes have greater imitation weight, failures remain outcome-model data.
    weights = torch.from_numpy(np.exp(np.clip(reward, -1, 5)))
    actor, critic = CorrectionPolicy(), OutcomeModel()
    initial = deepcopy(actor.state_dict())
    optimizer = torch.optim.AdamW([*actor.parameters(), *critic.parameters()], lr=.002, weight_decay=.003)

    def losses(indices):
        mu, scale = actor(xt[indices])
        nll = (.5 * ((at[indices] - mu) / scale).square() + scale.log()).mean(-1)
        policy_loss = (nll * weights[indices]).sum() / weights[indices].sum()
        predicted = critic(xt[indices], at[indices])
        critic_loss = nn.functional.binary_cross_entropy_with_logits(predicted[:, 0], yt[indices, 0])
        critic_loss += .25 * nn.functional.mse_loss(predicted[:, 1], yt[indices, 1])
        return policy_loss + critic_loss, policy_loss, critic_loss

    best, best_loss, best_epoch, history = None, float("inf"), 0, []
    for epoch in range(epochs):
        actor.train()
        critic.train()
        optimizer.zero_grad()
        loss, _, _ = losses(tr)
        loss.backward()
        nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], 5.)
        optimizer.step()
        actor.eval()
        critic.eval()
        with torch.no_grad():
            value, pl, cl = losses(va)
        if epoch % 10 == 0:
            history.append({"epoch": epoch, "train_loss": float(loss.detach()),
                            "validation_loss": float(value), "policy_loss": float(pl),
                            "outcome_loss": float(cl)})
        if float(value) < best_loss - 1e-5:
            best_loss, best_epoch = float(value), epoch
            best = deepcopy((actor.state_dict(), critic.state_dict()))
        elif epoch - best_epoch >= 80:
            break
    output = Path(output)
    output.mkdir(exist_ok=True)
    checkpoint = {
        "schema": SCHEMA, "actor": best[0], "critic": best[1],
        "mean": torch.tensor(mean), "std": torch.tensor(std),
        "feature_names": list(FEATURE_NAMES), "parameter_names": list(PARAMETER_NAMES),
        "train_families": sorted(train_ids), "validation_families": sorted(validation_ids),
        "training_rows_sha256": sha256_json(usable), "seed": seed, "selected_epoch": best_epoch,
        "scope": "reward_weighted_conditional_imitation_fixed_weights_at_test",
        "execution_authorized": False,
    }
    torch.save(checkpoint, output / "correction_policy.pt")
    delta = sum(float((best[0][k] - initial[k]).square().sum()) for k in initial)
    report = {
        "train_rows": len(tr), "validation_rows": len(va),
        "train_successes": int(success[tr].sum()), "validation_successes": int(success[va].sum()),
        "selected_epoch": best_epoch, "actor_squared_weight_change": delta,
        "history": history, "test_data_used": False, "force_cause_labels_trained": False,
    }
    write_json(output / "training-report.json", report)
    return report


class LearnedCorrection:
    def __init__(self, path):
        torch.set_num_threads(1)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["schema"] != SCHEMA or saved["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Unsupported policy schema")
        self.actor, self.critic = CorrectionPolicy().eval(), OutcomeModel().eval()
        self.actor.load_state_dict(saved["actor"])
        self.critic.load_state_dict(saved["critic"])
        self.mean, self.std = saved["mean"], saved["std"]

    def propose(self, feedback, rng):
        x = (torch.from_numpy(features(feedback)) - self.mean) / self.std
        with torch.no_grad():
            mean, scale = self.actor(x[None])
            parameters = np.clip(mean.numpy() + rng.normal(size=(32, len(LOW))) * scale.numpy(), -1, 1)
            parameters[0] = mean.numpy()[0]
            tensor = torch.tensor(parameters, dtype=torch.float32)
            outcomes = self.critic(x[None].repeat(len(tensor), 1), tensor).numpy()
        scores = outcomes[:, 0] + .25 * outcomes[:, 1]
        selected = int(np.argmax(scores))
        action = decode_action(feedback, parameters[selected], "learned_conditional_policy")
        return action, {
            "mean": mean.numpy()[0].tolist(), "std": scale.numpy().tolist(),
            "samples": parameters.tolist(), "scores": scores.tolist(), "selected": selected,
            "physics_rollouts_used_for_ranking": 0,
        }


class LearnedResidual:
    """Frozen history-conditioned residual generator for recovery evaluation."""

    def __init__(self, path):
        torch.set_num_threads(1)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["schema"] != RESIDUAL_SCHEMA:
            raise ValueError("Unsupported residual policy schema")
        self.model = ResidualCorrectionPolicy(saved["input_dim"]).eval()
        self.model.load_state_dict(saved["model"])
        self.mean, self.std = saved["mean"], saved["std"]

    def propose(self, feedback, previous_feedback, rng, samples=32):
        x = residual_features(feedback, previous_feedback)
        tensor_x = (torch.from_numpy(x) - self.mean) / self.std
        with torch.no_grad():
            mean = self.model(tensor_x[None]).numpy()[0]
        candidates = np.clip(
            mean[None] + rng.normal(0, .18, size=(samples, len(LOW))), -1, 1)
        candidates[0] = mean
        action = decode_action(feedback, candidates[0], "learned_history_residual")
        return action, {
            "schema": RESIDUAL_SCHEMA,
            "mean": mean.tolist(),
            "samples": candidates.tolist(),
            "selected": 0,
            "history_used": previous_feedback is not None,
            "critic": "embedded residual actor only; Organoid remains acceptance authority",
            "execution_authorized": False,
        }
