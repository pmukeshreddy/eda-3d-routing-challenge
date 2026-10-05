"""Concurrent CPU routing environments with one central, batched neural selector."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gzip
import math
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

import torch

from .routing_data import digest
from .routing_env import RoutingBudget, RoutingEnv
from .routing_runner import bbox_order
from .scheduler_features import batch_states, encode_observation


def seeded(base, *parts):
    return int(digest([base, *parts])[:15], 16)


def make_job(instance, metadata, *, trajectory_id, environment_seed, action_seed):
    return {"instance": instance, "metadata": metadata, "trajectory_id": trajectory_id,
            "environment_seed": environment_seed, "action_seed": action_seed}


def _outcome(env, job, steps, floor, failures):
    state = env.observation()
    delay = state["best_total_delay"]
    legal = delay is not None
    reward = math.log(job["metadata"].reference_total_delay / delay) if legal else floor["reward"]
    if not math.isfinite(reward) or (legal and reward <= floor["reward"]):
        raise ValueError("terminal reward violates the dataset-derived failure floor")
    return {"trajectory_id": job["trajectory_id"], "case_id": job["metadata"].case_id,
            "legal": legal, "final_delay": delay, "reward": reward,
            "conflicts": len(state["conflict_vertices"]) + len(state["conflict_edges"]),
            "conflict_vertices": len(state["conflict_vertices"]), "conflict_edges": len(state["conflict_edges"]),
            "missing_nets": len(state["missing_net_ids"]), "steps": steps,
            "passes": state["pass_number"], "engine_calls": state["budget"]["engine_calls"],
            "runtime_s": state["budget"]["elapsed_s"], "termination_reason": state["termination_reason"],
            "engine_failures": failures,
            "final_conflict_ownership": {
                "vertices": [(v, sorted(owners)) for v, owners in sorted(state["conflict_vertices"].items())],
                "edges": [(edge, sorted(owners)) for edge, owners in sorted(state["conflict_edges"].items())]},
            "environment_seed": job["environment_seed"], "action_seed": job["action_seed"]}


def collect_rollouts(jobs, *, selector, policy, config, floor, device, record_steps=False, deadline=math.inf):
    """Each environment step is dispatched exactly once for one selected net.

    Independent env.step calls run in threads, each backed by Block 2's native
    CPU worker process. Neural inference sees all active states in one batch.
    Waves cap resident environments and preserve reproducible job ordering.
    """
    output = []
    for offset in range(0, len(jobs), config["concurrency"]):
        wave = jobs[offset:offset + config["concurrency"]]
        envs = [RoutingEnv() for _ in wave]
        observations, records = [], [[] for _ in wave]
        counts, errors = [0] * len(wave), [[] for _ in wave]
        generators = [torch.Generator(device="cpu").manual_seed(j["action_seed"]) for j in wave]
        orders = [bbox_order(j["instance"]) for j in wave]
        randoms = [random.Random(j["action_seed"]) for j in wave]
        started = perf_counter()
        try:
            for env, job in zip(envs, wave):
                # Admit only a full, unchanged episode budget. Validation may
                # not silently receive a smaller budget near the wall cap.
                if deadline - perf_counter() < config["routing_budget"]["max_seconds"]:
                    raise TimeoutError("insufficient experiment time for a full routing episode")
                observations.append(env.reset(job["instance"], job["environment_seed"], RoutingBudget(**config["routing_budget"])))
            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                while True:
                    if perf_counter() >= deadline:
                        raise TimeoutError("experiment deadline reached during rollout")
                    active = [i for i, env in enumerate(envs) if env.eligible_net_ids()]
                    if not active:
                        break
                    actions, pending = {}, {}
                    if selector.startswith("policy"):
                        encoded = [encode_observation(observations[i]) for i in active]
                        batch = batch_states(encoded, device=device)
                        policy.eval()
                        with torch.no_grad():
                            logits = policy(batch)
                            if not torch.isfinite(logits[~batch["padding_mask"]]).all():
                                raise FloatingPointError("nonfinite policy logits")
                            masked = logits.masked_fill(~batch["eligible"], -torch.inf)
                            logs = masked.log_softmax(-1).cpu()
                            probabilities = logs.exp()
                        for row, i in enumerate(active):
                            state = encoded[row]
                            index = (int(probabilities[row].argmax()) if selector == "policy_deterministic"
                                     else int(torch.multinomial(probabilities[row], 1, generator=generators[i])))
                            if index >= len(state.net_ids) or not state.eligible[index]:
                                raise AssertionError("policy selected an ineligible or padded action")
                            actions[i] = state.net_ids[index]
                            selected = float(logs[row, index])
                            if not math.isfinite(selected):
                                raise FloatingPointError("nonfinite selected action log probability")
                            if record_steps:
                                old_all = torch.where(state.eligible, logs[row, :len(state.net_ids)], 0.).clone()
                                pending[i] = {"state": state, "action": actions[i], "action_index": index,
                                              "behavior_log_prob": selected, "behavior_all_log_probs": old_all,
                                              "intermediate_reward": 0.0}
                    else:
                        for i in active:
                            eligible = observations[i]["eligible_net_ids"]
                            if selector == "bbox":
                                allowed = set(eligible)
                                actions[i] = next(nid for nid in orders[i] if nid in allowed)
                            elif selector == "random":
                                actions[i] = randoms[i].choice(eligible)
                            else:
                                raise ValueError(f"unknown scheduler: {selector}")
                    if perf_counter() >= deadline:
                        raise TimeoutError("experiment deadline reached during policy encoding/inference")
                    futures = {i: pool.submit(envs[i].step, actions[i]) for i in active}
                    for i, future in futures.items():
                        result = future.result()
                        observations[i] = result.observation
                        if result.diagnostics["engine_status"] != "not_called":
                            counts[i] += 1
                            if i in pending:
                                records[i].append(pending[i])
                        if result.diagnostics["engine_status"] not in ("success", "not_called"):
                            errors[i].append({"step": counts[i], "status": result.diagnostics["engine_status"],
                                              "error": result.diagnostics.get("error")})
            for i, job in enumerate(wave):
                output.append({"trajectory_id": job["trajectory_id"], "steps": records[i],
                               "outcome": _outcome(envs[i], job, counts[i], floor, errors[i]),
                               "wave_runtime_s": perf_counter() - started})
        finally:
            for env in envs:
                env.close()
    return output


def save_trajectories(path, trajectories):
    """Retain pre-action features/masks/old probabilities and separate hindsight."""
    from .scheduler_features import FEATURE_SCHEMA_VERSION
    payload = {"feature_schema_version": FEATURE_SCHEMA_VERSION, "trajectories": []}
    for trajectory in trajectories:
        steps = [{**{k: v for k, v in step.items() if k != "state"},
                  "state": {name: getattr(step["state"], name) for name in
                            ("net_ids", "net_features", "global_features", "adjacency", "eligible")}}
                 for step in trajectory["steps"]]
        payload["trajectories"].append({**trajectory, "steps": steps})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=1) as stream:
        torch.save(payload, stream)


def summarize(trajectories):
    outcomes = [t["outcome"] for t in trajectories]
    if not outcomes:
        raise ValueError("cannot summarize an empty evaluation")
    legal = [o for o in outcomes if o["legal"]]
    failed = [o for o in outcomes if not o["legal"]]
    return {"episodes": len(outcomes), "legal_completion_rate": len(legal) / len(outcomes),
            "mean_failed_conflicts": mean(o["conflicts"] for o in failed) if failed else 0.,
            "mean_normalized_reward": mean(o["reward"] for o in outcomes),
            "mean_legal_delay": mean(o["final_delay"] for o in legal) if legal else None,
            "mean_steps": mean(o["steps"] for o in outcomes),
            "mean_runtime_s": mean(o["runtime_s"] for o in outcomes),
            "mean_passes": mean(o["passes"] for o in outcomes),
            "mean_engine_calls": mean(o["engine_calls"] for o in outcomes),
            "termination_counts": {reason: sum(o["termination_reason"] == reason for o in outcomes)
                                   for reason in sorted({o["termination_reason"] for o in outcomes})},
            "outcomes": outcomes}
