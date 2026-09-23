"""Pre-execution, masked multi-head critic and whole-family evaluation.

Input whitelist excludes all rollout metrics. Missing/censored outcomes do not
become negatives. Scores never authorize execution.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .multi_robot_tasks import FEATURE_NAMES, ROBOTS, write_json

HEADS = ("success", "collision_free", "ik_feasible")


class FeasibilityCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 48), nn.Tanh(),
                                 nn.Linear(48, 24), nn.Tanh(), nn.Linear(24, len(HEADS)))

    def forward(self, x):
        return self.net(x)


def family_split(records, seed):
    families = np.asarray(sorted({r["family"] for r in records}))
    if len(families) < 10:
        raise ValueError("At least 10 independent scene families are required")
    np.random.default_rng(seed).shuffle(families)
    ntrain, nval = int(.7 * len(families)), max(1, int(.15 * len(families)))
    assignment = {int(family): name for name, values in (
        ("train", families[:ntrain]), ("validation", families[ntrain:ntrain+nval]),
        ("test", families[ntrain+nval:])) for family in values}
    return assignment


def data_arrays(records):
    x = np.asarray([r["initial_features"] for r in records], dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("Bad pre-execution feature schema")
    y = np.asarray([[float(r["labels"][key]) if r["labels"][key] is not None else 0.
                     for key in HEADS] for r in records], dtype=np.float32)
    mask = np.asarray([[r["labels"][key] is not None for key in HEADS]
                       for r in records], dtype=bool)
    return x, y, mask


def head_metrics(y, mask, probabilities, priors):
    result = {}
    for index, head in enumerate(HEADS):
        valid = mask[:, index] & np.isfinite(probabilities[:, index])
        truth, prob = y[valid, index], probabilities[valid, index]
        if not len(truth):
            result[head] = {"evaluated": 0, "status": "not_evaluated"}
            continue
        pred = prob >= .5
        tp = int(np.sum(pred & (truth == 1)))
        tn = int(np.sum(~pred & (truth == 0)))
        fp = int(np.sum(pred & (truth == 0)))
        fn = int(np.sum(~pred & (truth == 1)))
        balanced = .5 * (tp / (tp + fn) + tn / (tn + fp)) if (tp+fn) and (tn+fp) else None
        result[head] = {
            "evaluated": len(truth), "positive": int(truth.sum()),
            "accuracy": float(np.mean(pred == truth)),
            "balanced_accuracy": balanced, "brier": float(np.mean((prob - truth) ** 2)),
            "train_prior_brier": float(np.mean((priors[index] - truth) ** 2)),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }
    return result


def fit(records, assignment, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    x, y, mask = data_arrays(records)
    indices = {name: np.asarray([i for i, r in enumerate(records) if assignment[r["family"]] == name])
               for name in ("train", "validation", "test")}
    if any(not len(value) for value in indices.values()):
        raise ValueError("All three scene splits need examples")
    tr, va = indices["train"], indices["validation"]
    active, priors = [], []
    for i in range(len(HEADS)):
        truth = y[tr, i][mask[tr, i]]
        active.append(bool(len(np.unique(truth)) == 2))
        priors.append(float(truth.mean()) if len(truth) else 0.)
    active = np.asarray(active)
    if not active.any():
        raise ValueError("No head has both positive and negative training evidence")
    mean, std = x[tr].mean(0), x[tr].std(0)
    std[std < 1e-5] = 1
    inputs = torch.from_numpy((x - mean) / std)
    truth = torch.from_numpy(y)
    weight = torch.from_numpy((mask & active[None, :]).astype(np.float32))
    if not weight[va].sum():
        raise ValueError("Validation split has no supported labels")
    model = FeasibilityCritic()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    best_state, best_loss, selected_epoch = None, float("inf"), None
    def loss_at(idx):
        losses = nn.functional.binary_cross_entropy_with_logits(model(inputs[idx]), truth[idx], reduction="none")
        return (losses * weight[idx]).sum() / weight[idx].sum().clamp_min(1.)
    for epoch in range(400):
        model.train()
        optimizer.zero_grad()
        loss = loss_at(tr)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_at(va))
        if validation_loss < best_loss - 1e-6:
            best_state, best_loss, selected_epoch = copy.deepcopy(model.state_dict()), validation_loss, epoch
        elif epoch - selected_epoch >= 60:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(inputs)).numpy()
    probabilities[:, ~active] = np.nan
    metrics = {name: head_metrics(y[idx], mask[idx], probabilities[idx], priors)
               for name, idx in indices.items()}
    return model, {
        "mean": mean, "std": std, "low": x[tr].min(0), "high": x[tr].max(0),
        "active": active, "priors": priors, "selected_epoch": selected_epoch,
    }, probabilities, metrics, indices


def ranking_metrics(records, probs, indices):
    groups = defaultdict(list)
    for i in indices:
        if records[i]["labels"]["success"] is not None and np.isfinite(probs[i, 0]):
            groups[(records[i]["family"], records[i]["robot"])].append(i)
    selected_success = direct_success = oracle_success = 0
    for group in groups.values():
        selected = max(group, key=lambda i: probs[i, 0])
        direct = next((i for i in group if records[i]["candidate"] == "direct"), None)
        selected_success += records[selected]["labels"]["success"]
        direct_success += records[direct]["labels"]["success"] if direct is not None else 0
        oracle_success += any(records[i]["labels"]["success"] for i in group)
    return {"scene_count": len(groups), "critic_selected_success": int(selected_success),
            "direct_success": int(direct_success), "oracle_any_candidate_success": int(oracle_success),
            "note": "Offline ranking on held-out scenes; physical acceptance is still required."}


def train_critic(records, output, seed):
    output = Path(output)
    usable = [r for r in records if "initial_features" in r and r.get("replay", {}).get("verified")]
    assignment = family_split(usable, seed)
    model, stats, probs, metrics, indices = fit(usable, assignment, seed)
    checkpoint = {
        "schema": "organoid.pre-execution-critic.v2", "state_dict": model.state_dict(),
        "feature_names": FEATURE_NAMES, "heads": list(HEADS),
        "mean": torch.tensor(stats["mean"]), "std": torch.tensor(stats["std"]),
        "low": torch.tensor(stats["low"]), "high": torch.tensor(stats["high"]),
        "active_heads": stats["active"].tolist(), "supported_profiles": list(ROBOTS),
        "scene_split": {str(k): v for k, v in assignment.items()},
        "selected_epoch": stats["selected_epoch"], "execution_authorized": False,
    }
    torch.save(checkpoint, output / "feasibility_critic.pt")
    external_robot = {}
    # Actual leave-one-robot-out test, including scene isolation, never tune on held-out robot.
    for held_out in ROBOTS:
        source = [r for r in usable if r["robot"] != held_out]
        other_model, other_stats, _, _, _ = fit(source, assignment, seed)
        target = [r for r in usable if r["robot"] == held_out and assignment[r["family"]] == "test"]
        x, y, mask = data_arrays(target)
        with torch.no_grad():
            p = torch.sigmoid(other_model(torch.from_numpy((x-other_stats["mean"])/other_stats["std"]))).numpy()
        p[:, ~other_stats["active"]] = np.nan
        external_robot[held_out] = {
            "episodes": len(target), "metrics": head_metrics(y, mask, p, other_stats["priors"]),
            "scope": "Exploratory held-out robot and held-out scene evaluation; not deployment authorization.",
        }
    evaluation = {
        "schema": "organoid.pre-execution-critic-evaluation.v2",
        "input_policy": "Only initial state, scene and candidate geometry; no rollout outcomes.",
        "split_unit": "scene_family_including_all_robots_and_candidates",
        "split_episode_counts": {k: len(v) for k, v in indices.items()},
        "split_family_counts": dict(Counter(assignment.values())),
        "active_heads": dict(zip(HEADS, stats["active"].tolist())),
        "metrics": metrics, "test_ranking": ranking_metrics(usable, probs, indices["test"]),
        "leave_one_robot_out": external_robot,
        "selected_epoch": stats["selected_epoch"],
        "hardware_test": "not_evaluated",
    }
    write_json(output / "critic-evaluation.json", evaluation)
    predictions = []
    for i, record in enumerate(usable):
        predictions.append({
            "trajectory": record["trajectory"], "split": assignment[record["family"]],
            "probabilities": {head: float(probs[i, j]) if np.isfinite(probs[i, j]) else None
                              for j, head in enumerate(HEADS)},
        })
    write_json(output / "critic-predictions.json", predictions)
    write_json(output / "split.json", {str(k): v for k, v in assignment.items()})
    return evaluation


def score_candidate(checkpoint_path, robot, features):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved.get("schema") != "organoid.pre-execution-critic.v2" or saved["feature_names"] != FEATURE_NAMES:
        raise ValueError("Unsupported checkpoint or feature schema")
    x = torch.as_tensor(features, dtype=torch.float32)
    if x.shape != saved["mean"].shape or not torch.isfinite(x).all():
        raise ValueError("Invalid feature vector")
    span = (saved["high"] - saved["low"]).clamp_min(.001)
    outside = (x < saved["low"] - .1 * span) | (x > saved["high"] + .1 * span)
    if robot not in saved["supported_profiles"] or outside.any():
        return {"status": "not_evaluated_ood", "execution_authorized": False,
                "out_of_range": [FEATURE_NAMES[i] for i in torch.where(outside)[0].tolist()]}
    model = FeasibilityCritic()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    with torch.no_grad():
        values = torch.sigmoid(model((x-saved["mean"])/saved["std"])).tolist()
    return {"status": "scored", "execution_authorized": False,
            "probabilities": {head: values[i] if saved["active_heads"][i] else None
                              for i, head in enumerate(HEADS)}}
