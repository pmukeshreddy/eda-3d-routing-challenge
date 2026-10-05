"""Critic-free group-relative policy updates and train-only data access."""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import random
from time import perf_counter

import torch

from .generator import GenConfig
from .routing_data import digest, load_split


class TrainingCases:
    """Seeded shuffled cycles ensure every training case is used, without replacement."""
    def __init__(self, root, *, seed: int):
        self.split = load_split(root, split="train")
        if not len(self.split):
            raise ValueError("training split is empty")
        self.rng = random.Random(seed)
        self.pending = []
        self.seen = []

    def next(self):
        if not self.pending:
            self.pending = list(range(len(self.split)))
            self.rng.shuffle(self.pending)
        index = self.pending.pop()
        self.seen.append(self.split.entries[index]["case_id"])
        return self.split.load(index)

    def state(self):
        return {"rng": self.rng.getstate(), "pending": self.pending.copy(), "seen": self.seen.copy()}


def failure_floor(manifest: dict) -> dict:
    """Bound every legal routing on any configured generated instance.

    A tree has at most V-1 edges. Each sink path costs at most (V-1)*Dmax.
    Reference delay is an integer >=1. The upstream generator adds at most
    6*6 vertices to each XY dimension. This uses configuration metadata only,
    not held-out entries, reference wires, or observed selector performance.
    """
    configuration = manifest["configuration"]
    upper_bounds = []
    for family in configuration["families"].values():
        cfg = {**asdict(GenConfig()), **configuration["defaults"], **family}
        high = lambda key: max(cfg[key]) if isinstance(cfg[key], list) else cfg[key]
        width, height, layers = high("width") + 36, high("height") + 36, high("layers")
        max_delay = max(high("via_delay"), high("center_delay") + high("layer_slope") * (layers - 1))
        upper_bounds.append(high("n_nets") * high("max_fanout") * (width * height * layers - 1) * max_delay)
    upper = max(upper_bounds)
    return {"reward": -math.log(upper) - 1.0, "legal_delay_upper_bound": upper,
            "reference_delay_lower_bound": 1, "strict_margin": 1.0,
            "rule": "log(1 / max_family(sinks_max * (vertices_max - 1) * delay_max)) - 1",
            "configuration_sha256": digest(configuration)}


def group_advantages(rewards) -> torch.Tensor:
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError("a group needs at least two finite terminal rewards")
    std = values.std(unbiased=False)
    return (values - values.mean()) / std if std > 1e-8 else torch.zeros_like(values)


def policy_loss(logits, eligible, actions, old_log_probs, old_all_log_probs, advantages,
                *, clip_ratio, entropy_coefficient, kl_coefficient):
    """Per-state clipped objective, exact behavior KL, and entropy regularizer.

    Old action probabilities and distributions are recorded during rollout and
    remain frozen throughout optimization. Invalid/padded coordinates are zero
    probability and excluded explicitly from entropy/KL arithmetic.
    """
    if not eligible.any(dim=-1).all() or not eligible.gather(1, actions[:, None]).all():
        raise ValueError("selected action must be currently eligible")
    masked = logits.masked_fill(~eligible, -torch.inf)
    log_probs = masked.log_softmax(dim=-1)
    safe_new = torch.where(eligible, log_probs, 0.)
    safe_old = torch.where(eligible, old_all_log_probs.detach(), 0.)
    probabilities = torch.where(eligible, log_probs.exp(), 0.)
    old_probabilities = torch.where(eligible, safe_old.exp(), 0.)
    selected = log_probs.gather(1, actions[:, None]).squeeze(1)
    ratio = (selected - old_log_probs.detach()).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages
    entropy = -(probabilities * safe_new).sum(dim=-1)
    kl = (old_probabilities * (safe_old - safe_new)).sum(dim=-1).clamp_min(0.)
    loss = -torch.minimum(unclipped, clipped) - entropy_coefficient * entropy + kl_coefficient * kl
    if not torch.isfinite(loss).all():
        raise FloatingPointError("nonfinite GRPO loss")
    return {"loss": loss, "entropy": entropy, "kl": kl,
            "clip_fraction": ((ratio - 1).abs() > clip_ratio).float(), "selected_log_probs": selected}


def optimizer_update(policy, optimizer, trajectories, config, *, device, rng, deadline=math.inf):
    from .scheduler_features import batch_states
    rewards = [t["outcome"]["reward"] for t in trajectories]
    advantages = group_advantages(rewards)
    samples = [(step, float(advantages[i]), 1.0 / (len(trajectories) * len(t["steps"])))
               for i, t in enumerate(trajectories) for step in t["steps"]]
    if not samples:
        raise ValueError("cannot update from an empty rollout group")
    rng.shuffle(samples)
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {name: 0. for name in ("loss", "entropy", "kl", "clip_fraction")}
    for offset in range(0, len(samples), config["minibatch_states"]):
        if perf_counter() >= deadline:
            optimizer.zero_grad(set_to_none=True)
            raise TimeoutError("training update exceeded the experiment deadline")
        group = samples[offset:offset + config["minibatch_states"]]
        batch = batch_states([s[0]["state"] for s in group], device=device)
        logits = policy(batch)
        old_all = torch.zeros_like(logits)
        for i, (step, _, _) in enumerate(group):
            old_all[i, :len(step["behavior_all_log_probs"])] = step["behavior_all_log_probs"].to(device)
        tensor = lambda vals, dtype=torch.float32: torch.tensor(vals, dtype=dtype, device=device)
        terms = policy_loss(logits, batch["eligible"], tensor([s[0]["action_index"] for s in group], torch.long),
                            tensor([s[0]["behavior_log_prob"] for s in group]), old_all,
                            tensor([s[1] for s in group]), clip_ratio=config["clip_ratio"],
                            entropy_coefficient=config["entropy_coefficient"], kl_coefficient=config["kl_coefficient"])
        weights = tensor([s[2] for s in group])
        (terms["loss"] * weights).sum().backward()
        for name in totals:
            totals[name] += float((terms[name].detach() * weights).sum().cpu())
    if perf_counter() >= deadline:
        optimizer.zero_grad(set_to_none=True)
        raise TimeoutError("experiment deadline reached before optimizer step")
    gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in policy.parameters()):
        raise FloatingPointError("nonfinite policy parameters")
    policy.eval()
    return {**totals, "gradient_norm": float(gradient_norm.cpu()), "states": len(samples),
            "reward_mean": sum(rewards) / len(rewards),
            "reward_std": float(torch.tensor(rewards).std(unbiased=False)),
            "group_collapsed": bool(torch.count_nonzero(advantages) == 0),
            "advantages": advantages.tolist(), "rewards": rewards}


def save_checkpoint(path, policy, optimizer, update, config, floor, *, extra=None):
    from .scheduler_features import FEATURE_SCHEMA_VERSION
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
               "optimizer": optimizer.state_dict(), "update": update, "config": config,
               "seeds": config["seeds"], "feature_schema_version": FEATURE_SCHEMA_VERSION,
               "failure_floor": floor, "torch_rng": torch.get_rng_state(),
               "python_rng": random.getstate(), "extra": extra or {}}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, *, device="cpu"):
    from .scheduler_features import FEATURE_SCHEMA_VERSION
    from .scheduler_policy import SchedulerPolicy
    # These are locally generated checkpoint artifacts, not downloaded models.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("checkpoint feature schema mismatch")
    policy = SchedulerPolicy(**payload["config"]["model"]).to(device)
    policy.load_state_dict(payload["model"])
    policy.eval()
    return policy, payload
