"""Low-dimensional action-chunk BC and upstream conditional diffusion training."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

from .hashing import sha256_file
from .kuavo_recovery import write_json
from .kuavo_sequence import ACTION_DIM, ACTION_DT, OBS_NAMES

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/diffusion_policy_upstream"
HORIZON = 16
OBS_STEPS = 2
EXECUTE_STEPS = 4


def backbone(kind):
    if kind == "bc":
        return nn.Sequential(
            nn.Linear(len(OBS_NAMES) * OBS_STEPS, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, HORIZON * ACTION_DIM))
    if kind != "diffusion":
        raise ValueError(kind)
    manifest = json.loads((UPSTREAM / "source-manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        if sha256_file(UPSTREAM / relative) != expected:
            raise ValueError("Diffusion backbone source hash drift")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    return ConditionalUnet1D(
        input_dim=ACTION_DIM, global_cond_dim=len(OBS_NAMES) * OBS_STEPS,
        diffusion_step_embed_dim=32, down_dims=[32, 64],
        kernel_size=3, n_groups=8, cond_predict_scale=True)


def windows(obs, action):
    if len(obs) != len(action) or not len(obs):
        raise ValueError("Nonempty aligned observations and actions required")
    n = len(obs)
    before = np.maximum(np.arange(n)[:, None] + [-1, 0], 0)
    after = np.minimum(np.arange(n)[:, None] + np.arange(HORIZON), n - 1)
    return obs[before], action[after]


def load_samples(root, records, families):
    xs, ys = [], []
    for row in records:
        if row["family"] not in families or row["status"] != "accepted":
            continue
        path = Path(root) / row["directory"] / "sequence.npz"
        if sha256_file(path) != row["sequence_sha256"]:
            raise ValueError("Dataset sequence hash drift")
        with np.load(path, allow_pickle=False) as data:
            x, y = windows(data["obs"], data["action"])
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError("No accepted trajectories for the selected families")
    return np.concatenate(xs), np.concatenate(ys)


def train(root, name, kind, families, validation_families, *, steps=2000, seed=71):
    from diffusers import DDPMScheduler
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if set(families) & set(validation_families):
        raise ValueError("Train/validation family leakage")
    root = Path(root)
    out = root / "models" / name
    out.mkdir(parents=True, exist_ok=False)
    records = json.loads((root / "collection.json").read_text())
    x, y = load_samples(root, records, set(families))
    xv, yv = load_samples(root, records, set(validation_families))
    mean = x[:, -1].mean(0)
    std = np.maximum(x[:, -1].std(0), .01)
    center = (y.reshape(-1, ACTION_DIM).min(0) + y.reshape(-1, ACTION_DIM).max(0)) / 2
    scale = np.maximum((y.reshape(-1, ACTION_DIM).max(0) -
                        y.reshape(-1, ACTION_DIM).min(0)) / 2, .05)
    tx = torch.tensor(np.clip((x - mean) / std, -10, 10).reshape(len(x), -1))
    ty = torch.tensor((y - center) / scale)
    vx = torch.tensor(np.clip((xv - mean) / std, -10, 10).reshape(len(xv), -1))
    vy = torch.tensor((yv - center) / scale)
    net = backbone(kind)
    ema = deepcopy(net)
    opt = torch.optim.AdamW(net.parameters(), lr=.0003, weight_decay=.00001)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
        clip_sample=True, prediction_type="epsilon")
    generator = torch.Generator().manual_seed(seed + 1)
    val_generator = torch.Generator().manual_seed(1994)
    val_ids = torch.randint(len(vx), (min(512, len(vx)),), generator=val_generator)
    vx, vy = vx[val_ids], vy[val_ids]
    val_noise = torch.randn(vy.shape, generator=val_generator)
    val_t = torch.randint(100, (len(vx),), generator=val_generator)
    best_loss, best_weights, best_step = float("inf"), None, None
    log = []
    started = time.monotonic()
    for step in range(steps):
        ids = torch.randint(len(tx), (64,), generator=generator)
        bx, by = tx[ids], ty[ids]
        net.train()
        opt.zero_grad()
        if kind == "bc":
            pred = net(bx).reshape(-1, HORIZON, ACTION_DIM)
            loss = nn.functional.mse_loss(pred, by)
        else:
            noise = torch.randn(by.shape, generator=generator)
            t = torch.randint(100, (len(ids),), generator=generator)
            noisy = noise_scheduler.add_noise(by, noise, t)
            pred = net(noisy, t, global_cond=bx)
            loss = nn.functional.mse_loss(pred, noise)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.)
        opt.step()
        with torch.no_grad():
            for target, source in zip(ema.parameters(), net.parameters()):
                target.lerp_(source, .01)
        if (step + 1) % 100 == 0 or step == steps - 1:
            ema.eval()
            with torch.no_grad():
                if kind == "bc":
                    vp = ema(vx).reshape(-1, HORIZON, ACTION_DIM)
                    vl = nn.functional.mse_loss(vp, vy)
                else:
                    vn = noise_scheduler.add_noise(vy, val_noise, val_t)
                    vp = ema(vn, val_t, global_cond=vx)
                    vl = nn.functional.mse_loss(vp, val_noise)
            value = float(vl)
            log.append({"step": step + 1, "train_loss": float(loss.detach()),
                        "validation_loss": value})
            if value < best_loss:
                best_loss, best_step = value, step + 1
                best_weights = {k: v.detach().clone() for k, v in ema.state_dict().items()}
            print("TRAIN", name, step + 1, round(float(loss.detach()), 5),
                  round(value, 5), flush=True)
    checkpoint = {
        "schema": "organoid.kuavo-action-sequence.v1", "kind": kind, "name": name,
        "model": best_weights, "obs_mean": torch.tensor(mean), "obs_std": torch.tensor(std),
        "action_center": torch.tensor(center), "action_scale": torch.tensor(scale),
        "obs_names": OBS_NAMES, "horizon": HORIZON, "obs_steps": OBS_STEPS,
        "action_dt": ACTION_DT, "execute_steps": EXECUTE_STEPS,
        "families": sorted(families), "validation_families": sorted(validation_families),
        "dataset_sha256": sha256_file(root / "collection.json"),
        "inference_steps": 20, "prediction_type": "epsilon",
        "execution_authorized": False,
    }
    torch.save(checkpoint, out / "policy.pt")
    selected = [r for r in records if r["family"] in families and r["status"] == "accepted"]
    report = {
        "kind": kind, "name": name, "seed": seed, "train_families": sorted(families),
        "validation_families": sorted(validation_families),
        "train_episodes": len(selected), "train_transitions": len(x),
        "validation_transitions": len(xv), "steps": steps, "best_step": best_step,
        "best_validation_loss": best_loss, "parameter_count": sum(p.numel() for p in net.parameters()),
        "seconds": time.monotonic() - started, "history": log,
        "checkpoint_sha256": sha256_file(out / "policy.pt"),
        "uses_raw_image_encoder": False, "uses_phase_or_clock": False,
        "uses_privileged_object_state": False, "online_weight_updates": False,
        "normalization": "training_families_only", "loss_is_success_rate": False,
    }
    write_json(out / "training.json", report)
    return report


class SequencePolicy:
    def __init__(self, path, *, seed=301):
        from diffusers import DDIMScheduler
        torch.set_num_threads(1)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["schema"] != "organoid.kuavo-action-sequence.v1" or saved["obs_names"] != OBS_NAMES:
            raise ValueError("Checkpoint observation contract mismatch")
        self.saved, self.name = saved, saved["name"]
        self.kind = saved["kind"]
        self.net = backbone(self.kind).eval()
        self.net.load_state_dict(saved["model"])
        self.execute_steps = saved["execute_steps"]
        self.generator = torch.Generator().manual_seed(seed)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
            clip_sample=True, prediction_type="epsilon")
        self.scheduler.set_timesteps(saved["inference_steps"])

    @torch.inference_mode()
    def predict(self, history):
        if not history:
            raise ValueError("Missing observation history")
        obs = np.stack(([history[0]] if len(history) == 1 else []) + list(history[-2:]))
        x = torch.tensor(obs).float()
        x = ((x - self.saved["obs_mean"]) / self.saved["obs_std"]).clamp(-10, 10).reshape(1, -1)
        if self.kind == "bc":
            action = self.net(x).reshape(1, HORIZON, ACTION_DIM)
        else:
            action = torch.randn((1, HORIZON, ACTION_DIM), generator=self.generator)
            for t in self.scheduler.timesteps:
                pred = self.net(action, t, global_cond=x)
                action = self.scheduler.step(pred, t, action, eta=0).prev_sample
        return (action[0] * self.saved["action_scale"] + self.saved["action_center"]).numpy()
