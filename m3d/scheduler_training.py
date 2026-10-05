"""One local GRPO experiment with explicit train/validation separation."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
from time import perf_counter

import torch

from .routing_data import file_hash, load_split
from .routing_env import RoutingBudget
from .scheduler_features import FEATURE_SCHEMA_VERSION
from .scheduler_grpo import (TrainingCases, failure_floor, load_checkpoint,
                              optimizer_update, save_checkpoint)
from .scheduler_policy import SchedulerPolicy
from .scheduler_rollout import collect_rollouts, make_job, save_trajectories, seeded, summarize


def choose_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be auto, cpu, mps, or cuda")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but not available in this process")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available in this process")
    return requested


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def validate_config(config):
    for key in ("group_size", "concurrency", "max_updates", "epochs_per_group", "minibatch_states",
                "validation_every_updates", "validation_sample_repeats", "cpu_threads"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["group_size"] < 2 or config["max_updates"] % config["epochs_per_group"]:
        raise ValueError("G must be at least 2 and max_updates a multiple of epochs_per_group")
    for key in ("wall_seconds", "evaluation_reserve_seconds", "learning_rate", "weight_decay", "clip_ratio",
                "entropy_coefficient", "kl_coefficient", "max_grad_norm"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 < config["clip_ratio"] < 1 or config["wall_seconds"] <= 0:
        raise ValueError("invalid clipping/time configuration")
    RoutingBudget(**config["routing_budget"])


def _initialize(data, out, config):
    validate_config(config)
    out = Path(out)
    if (out / "config.json").exists():
        raise ValueError("output already contains an experiment; refusing a silent restart")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["cpu_threads"])
    random.seed(config["seeds"]["model"])
    torch.manual_seed(config["seeds"]["model"])
    device = choose_device(config["device"])
    manifest = json.loads((Path(data) / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Block 4 experiment requires the certified complete dataset")
    floor = failure_floor(manifest)
    policy = SchedulerPolicy(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    count = sum(p.numel() for p in policy.parameters())
    if not 2_000_000 <= count <= 15_000_000:
        raise ValueError(f"scheduler parameter count outside intended range: {count}")
    policy.eval()
    saved = {**config, "resolved_device": device, "parameter_count": count,
             "feature_schema_version": FEATURE_SCHEMA_VERSION, "failure_floor": floor,
             "dataset_manifest_sha256": file_hash(Path(data) / "manifest.json"), "torch_version": torch.__version__}
    write_json(out / "config.json", saved)
    return out, saved, policy, optimizer, device, floor


def validation_jobs(validation, config, *, repeats, prefix):
    jobs = []
    for i in range(len(validation)):
        instance, metadata = validation.load(i)
        for repeat in range(repeats):
            jobs.append(make_job(instance, metadata, trajectory_id=f"{prefix}/{metadata.case_id}/{repeat}",
                                 environment_seed=seeded(config["seeds"]["validation"], metadata.case_id, repeat, "engine"),
                                 action_seed=seeded(config["seeds"]["validation"], metadata.case_id, repeat, "actions")))
    return jobs


def evaluate_policy(policy, validation, config, floor, device, *, label, update, output, deadline=math.inf):
    results = {}
    for mode, repeats in (("deterministic", 1), ("sampled", config["validation_sample_repeats"])):
        jobs = validation_jobs(validation, config, repeats=repeats, prefix=f"{label}/{mode}")
        trajectories = collect_rollouts(jobs, selector=f"policy_{mode}", policy=policy, config=config,
                                        floor=floor, device=device, deadline=deadline)
        results[mode] = summarize(trajectories)
        append_json(output, {"label": label, "update": update, "mode": mode, **results[mode]})
        print(json.dumps({"event": "validation", "label": label, "update": update, "mode": mode,
                          **{k: v for k, v in results[mode].items() if k != "outcomes"}}), flush=True)
    return results


def mechanics(data, out, configuration):
    config = copy.deepcopy(configuration)
    config.update(group_size=config["mechanics"]["group_size"], max_updates=config["mechanics"]["max_updates"],
                  wall_seconds=config["mechanics"]["wall_seconds"])
    start = perf_counter()
    out, config, policy, optimizer, device, floor = _initialize(data, out, config)
    train = load_split(data, split="train")
    index = next(i for i, e in enumerate(train.entries) if e["family"] == "sparse")
    instance, metadata = train.load(index)
    jobs = [make_job(instance, metadata, trajectory_id=f"mechanics/{i}",
                     environment_seed=config["seeds"]["groups"],
                     action_seed=seeded(config["seeds"]["actions"], "mechanics", i))
            for i in range(config["group_size"])]
    trajectories = collect_rollouts(jobs, selector="policy_sampled", policy=policy, config=config,
                                    floor=floor, device=device, record_steps=True, deadline=start + config["wall_seconds"])
    save_trajectories(out / "rollouts/group00000.pt.gz", trajectories)
    steps = [step for trajectory in trajectories for step in trajectory["steps"]]
    if not steps:
        raise AssertionError("mechanics run produced no actions")
    errors = []
    with torch.no_grad():
        for step in steps[:8]:
            actual = float(policy.log_prob(step["state"], step["action"]).cpu())
            errors.append(abs(actual - step["behavior_log_prob"]))
    if max(errors) > 5e-4:
        raise AssertionError("recorded behavior log probabilities do not match selected actions")
    before = {name: p.detach().cpu().clone() for name, p in policy.named_parameters()}
    rng = random.Random(config["seeds"]["optimization"])
    for update in range(1, config["max_updates"] + 1):
        metrics = optimizer_update(policy, optimizer, trajectories, config, device=device, rng=rng,
                                   deadline=start + config["wall_seconds"])
        append_json(out / "training_metrics.jsonl", {"update": update, **metrics})
    change = sum(float((p.detach().cpu() - before[name]).square().sum()) for name, p in policy.named_parameters()) ** .5
    if change <= 0:
        raise AssertionError("optimizer failed to change model parameters")
    path = out / "checkpoints/best.pt"
    save_checkpoint(path, policy, optimizer, config["max_updates"], config, floor)
    restored, _ = load_checkpoint(path, device=device)
    if any(policy.act(s["state"], deterministic=True) != restored.act(s["state"], deterministic=True) for s in steps[:8]):
        raise AssertionError("checkpoint changed deterministic actions")
    runtime = perf_counter() - start
    result = {"status": "passed" if runtime <= config["wall_seconds"] else "partial", "case_id": metadata.case_id, "group_size": len(trajectories),
              "optimizer_updates": config["max_updates"], "parameter_delta_l2": change,
              "max_log_prob_error": max(errors), "masked_actions_valid": True,
              "finite_losses_and_parameters": True, "checkpoint_actions_match": True,
              "runtime_s": runtime, "wall_cap_s": config["wall_seconds"], "within_wall_cap": runtime <= config["wall_seconds"],
              "device": device, "rollouts": summarize(trajectories)}
    write_json(out / "acceptance.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def train(data, out, configuration):
    start = perf_counter()
    out, config, policy, optimizer, device, floor = _initialize(data, out, copy.deepcopy(configuration))
    deadline = start + config["wall_seconds"]
    train_cases = TrainingCases(data, seed=config["seeds"]["dataset"])
    validation = load_split(data, split="val")  # The only held-out instances opened in this block.
    if len(train_cases.split) != 12 or len(validation) != 4:
        raise ValueError("this acceptance configuration expects 12 train and 4 validation instances")
    initial_weights = {name: p.detach().cpu().clone() for name, p in policy.named_parameters()}
    save_checkpoint(out / "checkpoints/initial.pt", policy, optimizer, 0, config, floor)
    validation_log = out / "validation_metrics.jsonl"
    baselines = {}
    for selector in ("bbox", "random"):
        jobs = validation_jobs(validation, config, repeats=config["validation_sample_repeats"], prefix=selector)
        baselines[selector] = summarize(collect_rollouts(jobs, selector=selector, policy=None, config=config,
                                                       floor=floor, device=device, deadline=deadline))
        append_json(validation_log, {"label": selector, "update": 0, "mode": "paired", **baselines[selector]})
        print(json.dumps({"event": "baseline", "selector": selector,
                          **{k: v for k, v in baselines[selector].items() if k != "outcomes"}}), flush=True)
    initial = evaluate_policy(policy, validation, config, floor, device, label="initial", update=0,
                              output=validation_log, deadline=deadline)
    rng = random.Random(config["seeds"]["optimization"])
    update, group = 0, 0
    best = None
    training_rows = []
    last_validation_update = None
    stop_reason = "max_updates"
    while update < config["max_updates"]:
        if deadline - perf_counter() < config["evaluation_reserve_seconds"] + config["routing_budget"]["max_seconds"] + 10:
            stop_reason = "wall_budget_admission"
            break
        instance, metadata = train_cases.next()
        group_seed = seeded(config["seeds"]["groups"], group)
        jobs = [make_job(instance, metadata, trajectory_id=f"train/group{group:05d}/sample{i}",
                         environment_seed=group_seed,
                         action_seed=seeded(config["seeds"]["actions"], group, i)) for i in range(config["group_size"])]
        trajectories = collect_rollouts(jobs, selector="policy_sampled", policy=policy, config=config,
                                        floor=floor, device=device, record_steps=True,
                                        deadline=deadline - config["evaluation_reserve_seconds"])
        save_trajectories(out / f"rollouts/group{group:05d}.pt.gz", trajectories)
        summary = summarize(trajectories)
        for epoch in range(config["epochs_per_group"]):
            try:
                metrics = optimizer_update(policy, optimizer, trajectories, config, device=device, rng=rng,
                                           deadline=deadline - config["evaluation_reserve_seconds"])
            except TimeoutError:
                stop_reason = "wall_budget_update"
                break
            update += 1
            row = {"update": update, "group": group, "epoch": epoch, "case_id": metadata.case_id,
                   "elapsed_s": perf_counter() - start, **metrics,
                   "legal_completion_rate": summary["legal_completion_rate"],
                   "mean_steps": summary["mean_steps"], "mean_runtime_s": summary["mean_runtime_s"],
                   "termination_counts": summary["termination_counts"]}
            training_rows.append(row)
            append_json(out / "training_metrics.jsonl", row)
            print(json.dumps({"event": "update", **row}), flush=True)
        group += 1
        save_checkpoint(out / "checkpoints/last.pt", policy, optimizer, update, config, floor,
                        extra={"training_sampler": train_cases.state(), "optimizer_shuffle_rng": rng.getstate(), "group": group})
        if update % config["validation_every_updates"] == 0 or update == config["max_updates"]:
            evaluated = evaluate_policy(policy, validation, config, floor, device,
                                        label="trained", update=update, output=validation_log, deadline=deadline)
            last_validation_update = update
            if best is None or evaluated["sampled"]["mean_normalized_reward"] > best["metrics"]["sampled"]["mean_normalized_reward"]:
                best = {"update": update, "metrics": evaluated}
                save_checkpoint(out / "checkpoints/best.pt", policy, optimizer, update, config, floor,
                                extra={"validation": evaluated, "training_sampler": train_cases.state()})
        if stop_reason != "max_updates":
            break
    if update and last_validation_update != update:
        evaluated = evaluate_policy(policy, validation, config, floor, device,
                                    label="trained", update=update, output=validation_log, deadline=deadline)
        if best is None or evaluated["sampled"]["mean_normalized_reward"] > best["metrics"]["sampled"]["mean_normalized_reward"]:
            best = {"update": update, "metrics": evaluated}
            save_checkpoint(out / "checkpoints/best.pt", policy, optimizer, update, config, floor,
                            extra={"validation": evaluated, "training_sampler": train_cases.state()})
    change = sum(float((p.detach().cpu() - initial_weights[name]).square().sum()) for name, p in policy.named_parameters()) ** .5
    initial_score = initial["sampled"]["mean_normalized_reward"]
    trained_score = best["metrics"]["sampled"]["mean_normalized_reward"] if best else None
    improved_initial = trained_score is not None and trained_score > initial_score + 1e-8
    improved_baselines = [name for name, stats in baselines.items()
                          if trained_score is not None and trained_score > stats["mean_normalized_reward"] + 1e-8]
    collapsed = sum(row["group_collapsed"] for row in training_rows)
    updated_cases = sorted({row["case_id"] for row in training_rows})
    full_coverage = len(updated_cases) == len(train_cases.split)
    runtime = perf_counter() - start
    within_cap = runtime <= config["wall_seconds"]
    result = {"status": ("partial" if not full_coverage or not within_cap else
                         "passed" if improved_initial and improved_baselines else "not_improved"),
              "primary_metric": "paired sampled validation mean terminal reward; larger is better",
              "improved_over_initial": improved_initial, "improved_over_baselines": improved_baselines,
              "selected_update": best["update"] if best else None,
              "baselines": baselines, "initial_policy": initial,
              "trained_policy": best["metrics"] if best else None,
              "updates": update, "groups": group, "stop_reason": stop_reason,
              "runtime_s": runtime, "wall_cap_s": config["wall_seconds"], "within_wall_cap": within_cap,
              "parameter_count": config["parameter_count"], "parameter_delta_l2": change, "device": device,
              "training_cases_seen": sorted(set(train_cases.seen)), "training_case_count": len(train_cases.split),
              "training_cases_updated": updated_cases, "all_training_cases_updated": full_coverage,
              "test_instances_opened": 0, "failure_floor": floor,
              "collapsed_update_fraction": collapsed / len(training_rows) if training_rows else None,
              "mean_training_reward_std": sum(r["reward_std"] for r in training_rows) / len(training_rows) if training_rows else None,
              "block_2_gap": "Unchanged: official case_05 retained 3 vertex conflicts after 50 passes.",
              "limitations": ["12 training cases and 4 validation cases are a pipeline experiment, not a final policy.",
                              "Validation uses two paired sampled episodes per case; no significance claim.",
                              "All actions share trajectory-level group-relative credit; no critic or hindsight inputs."]}
    write_json(out / "acceptance.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("baselines", "initial_policy", "trained_policy")}, indent=2), flush=True)
    return result
