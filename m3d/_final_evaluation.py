"""Measurement-only adapter around the frozen Block 4 rollout collector.

No training operation is invoked. The observer records actions and exports the
environment's checker-approved best solution without changing its decisions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import shutil
from statistics import mean, pstdev
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .checker import check
from .model import Instance, Submission
from .negotiated import VCONG
from .routing_data import digest, file_hash, load_split
from .routing_env import RoutingEnv
from .scheduler_features import FEATURE_SCHEMA_VERSION
from .scheduler_grpo import load_checkpoint
from .scheduler_rollout import collect_rollouts, make_job, seeded
from .scheduler_training import append_json, write_json
from .scorer import score_case, score_submission_set


def read_json(path):
    return json.loads(Path(path).read_text())


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def history_audit(manifest):
    """Inspect logs/checkpoint provenance; do not open held-out inputs."""
    test_ids = {e["case_id"] for e in manifest["cases"] if e["split"] == "test"}
    train_ids = {e["case_id"] for e in manifest["cases"] if e["split"] == "train"}
    training = [json.loads(line) for line in Path("build/block4/training_metrics.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in Path("build/block4/validation_metrics.jsonl").read_text().splitlines()]
    seen = {r["case_id"] for r in training}
    validation_seen = {o["case_id"] for r in validation for o in r["outcomes"]}
    mechanics = read_json("build/block4/mechanics/acceptance.json")["case_id"]
    assert seen == train_ids and not test_ids.intersection(seen | validation_seen | {mechanics})
    assert all(case.startswith("val-") for case in validation_seen)
    assert read_json("build/block4/acceptance.json")["test_instances_opened"] == 0
    return {"passed": True, "training_updates_checked": len(training),
            "training_cases": sorted(seen), "validation_cases": sorted(validation_seen),
            "test_case_mentions_in_prior_episode_logs": 0,
            "scope": "recorded training, validation and mechanics episode logs plus Block 4 acceptance"}


def freeze(out, *, started=None):
    out = Path(out)
    if (out / "frozen_config.json").exists() or (out / "run_started.json").exists():
        raise ValueError("Block 5 is already frozen; refusing replacement")
    config = read_json("build/block4/config.json")
    acceptance = read_json("build/block4/acceptance.json")
    manifest = read_json("build/data/routing_v1/manifest.json")
    assert acceptance["selected_update"] == 48 and manifest["status"] == "complete"
    assert file_hash("build/data/routing_v1/manifest.json") == config["dataset_manifest_sha256"]
    prior = read_json("build/block4/provenance.json")
    for path, sha in {**prior["protected"], **prior["scheduler_before_run"]}.items():
        assert file_hash(path) == sha, path
    audit = history_audit(manifest)
    if not torch.backends.mps.is_available():
        raise RuntimeError("Frozen Block 4 device is MPS; run with the tested interpreter and GPU access")
    torch.set_num_threads(config["cpu_threads"])
    paths = {"trained": "build/block4/checkpoints/best.pt", "initial": "build/block4/checkpoints/initial.pt"}
    checkpoints = {}
    for label, path in paths.items():
        model, payload = load_checkpoint(path)
        assert payload["config"]["model"] == config["model"]
        assert payload["config"]["routing_budget"] == config["routing_budget"]
        assert payload["failure_floor"] == config["failure_floor"]
        assert payload["feature_schema_version"] == FEATURE_SCHEMA_VERSION
        assert payload["update"] == (acceptance["selected_update"] if label == "trained" else 0)
        checkpoints[label] = {"path": path, "sha256": file_hash(path), "update": payload["update"],
                              "parameters": sum(p.numel() for p in model.parameters())}
        del model, payload
    source_paths = sorted(set(Path("m3d").glob("*.py")) | set(Path("native").rglob("*.cpp"))
                          | set(Path("native").rglob("*.h")) | set(Path("m3d").glob("_wire_native*.so"))
                          | {Path("scripts/verify_submissions.py"), Path("configs/scheduler_grpo_v1.json")})
    sources = {str(p): file_hash(p) for p in source_paths}
    test_entries = sorted((e for e in manifest["cases"] if e["split"] == "test"), key=lambda e: e["case_id"])
    small = next(e["case_id"] for e in test_entries if e["family"] == "sparse")
    hard = read_json("benchmarks_hard/suite.json")
    assert len(hard["cases"]) == 9
    protected = {**prior["protected"], "build/block4/config.json": file_hash("build/block4/config.json"),
                 "build/block4/acceptance.json": file_hash("build/block4/acceptance.json"),
                 "benchmarks_hard/suite.json": file_hash("benchmarks_hard/suite.json")}
    variants = ["bbox", "random", "initial_deterministic", "initial_sampled", "trained_deterministic", "trained_sampled"]
    frozen = {"format": "m3d-block5-frozen-v1", "created_utc": utc_now(), "checkpoints": checkpoints,
              "source_commit": command("git", "rev-parse", "HEAD"),
              "working_tree_status": command("git", "status", "--short"),
              "source_files": sources, "source_sha256": digest(sources), "protected_files": protected,
              "dataset": {"root": "build/data/routing_v1", "manifest_sha256": file_hash("build/data/routing_v1/manifest.json"),
                          "test_slots": [{"case_id": e["case_id"], "instance_file": e["instance_file"],
                                          "instance_sha256": e["instance_file_sha256"]} for e in test_entries]},
              "prior_split_audit": audit, "block4_config": config,
              "engine": {"method": "best", "native_call_seed": "(episode_seed + engine_call_index) modulo 2^64"},
              "environment": {"empty_reset": True, "initial_present_factor": .5, "present_multiplier": 1.7,
                              "present_cap": 1e6, "history_increment": "0.5 * (distinct_owners - 1)",
                              "vertex_congestion_weight": VCONG,
                              "vertex_price_mapping": "half of vertex pressure on each incident undirected edge",
                              "foreign_pins": "hard blocked", "transition_source": "m3d/routing_env.py"},
              "routing_budget": config["routing_budget"], "feature_schema_version": FEATURE_SCHEMA_VERSION,
              "model": config["model"], "failure_floor": config["failure_floor"],
              "device": "mps", "concurrency": config["concurrency"], "cpu_threads": config["cpu_threads"],
              "seeds": [5101, 5102], "seed_derivation": "scheduler_rollout.seeded(seed, phase, case_id, 'engine'/'actions')",
              "test_variants": variants, "deterministic_seed": 5101,
              "variant_seeds": {v: ([5101] if v.endswith("_deterministic") else [5101, 5102]) for v in variants},
              "hard_variants": ["bbox", "trained_deterministic"], "hard_primary": "trained_deterministic",
              "hard_suite": "benchmarks_hard", "optional_all_tiers": False,
              "evaluation_wall_cap_s": 2700,
              "reproducibility": {"case_id": small, "selector": "trained_deterministic", "seed": 5101,
                                  "extra_episodes": 1, "excluded_from_test_aggregates": True},
              "runtime_measurement": "Per-episode reset through terminal observation, decision trace and route serialization, plus amortized shared model/input/setup and collector bookkeeping. Exact full selector-batch wall time is also recorded. Independent checker/scorer time is separate; environment-internal checking stays included.",
              "comparison": "Paired seeds; score deltas only on jointly legal case/seed pairs. Failure reward is frozen. No complete-tier aggregate unless every hard case is legal.",
              "hardware": {"cpu": command("sysctl", "-n", "machdep.cpu.brand_string"),
                           "model": command("sysctl", "-n", "hw.model"), "cpu_count": int(command("sysctl", "-n", "hw.ncpu")),
                           "ram_bytes": int(command("sysctl", "-n", "hw.memsize")),
                           "accelerator": "Apple MPS", "python_executable": sys.executable,
                           "python_version": platform.python_version(), "torch_version": torch.__version__,
                           "os": platform.platform()}, "correctness_fixes": []}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "frozen_config.json", frozen)
    sha = file_hash(out / "frozen_config.json")
    (out / "frozen_config.sha256").write_text(sha + "\n")
    (out / "frozen_config.json").chmod(0o444)
    print(json.dumps({"frozen": str(out / "frozen_config.json"), "sha256": sha,
                      "checkpoint": checkpoints["trained"], "prior_split_audit": audit, "hardware": frozen["hardware"]}), flush=True)
    return 0


def verify_frozen(out):
    out = Path(out)
    assert file_hash(out / "frozen_config.json") == (out / "frozen_config.sha256").read_text().strip()
    frozen = read_json(out / "frozen_config.json")
    for path, sha in {**frozen["source_files"], **frozen["protected_files"]}.items():
        assert file_hash(path) == sha, f"frozen file changed: {path}"
    for value in frozen["checkpoints"].values():
        assert file_hash(value["path"]) == value["sha256"]
    assert file_hash(Path(frozen["dataset"]["root"]) / "manifest.json") == frozen["dataset"]["manifest_sha256"]
    return frozen


def stats(values):
    return {"n": len(values), "mean": mean(values) if values else None,
            "std": pstdev(values) if values else None, "values": values}


def geomean(values):
    return math.exp(math.fsum(math.log(v) for v in values) / len(values)) if values else None


def aggregate(rows):
    legal = [r for r in rows if r["legal"]]
    failed = [r for r in rows if not r["legal"]]
    per_seed = []
    for seed in sorted({r["seed"] for r in rows}):
        subset = [r for r in rows if r["seed"] == seed]
        per_seed.append({"seed": seed, "legal_rate": mean(int(r["legal"]) for r in subset),
                         "mean_reward": mean(r["reward"] for r in subset),
                         "legal_geomean_score": geomean([r["normalized_score"] for r in subset if r["legal"]]),
                         "mean_runtime_s": mean(r["runtime_s"] for r in subset),
                         "mean_engine_calls": mean(r["engine_calls"] for r in subset)})
    return {"episodes": len(rows), "n_legal": len(legal), "legal_rate": len(legal) / len(rows),
            "legal_geomean_score": geomean([r["normalized_score"] for r in legal]),
            "mean_reward": mean(r["reward"] for r in rows),
            "mean_failed_conflicts": mean(r["conflicts"] for r in failed) if failed else None,
            "mean_legal_delay": mean(r["total_delay"] for r in legal) if legal else None,
            "runtime_s": stats([r["runtime_s"] for r in rows]),
            "reward": stats([r["reward"] for r in rows]),
            "legal_score": stats([r["normalized_score"] for r in legal]),
            "steps": stats([r["scheduling_actions"] for r in rows]),
            "engine_calls": stats([r["engine_calls"] for r in rows]),
            "stop_reasons": {s: sum(r["stop_reason"] == s for r in rows) for s in sorted({r["stop_reason"] for r in rows})},
            "per_seed": per_seed,
            "seed_mean_reward": stats([r["mean_reward"] for r in per_seed]),
            "seed_legal_rate": stats([r["legal_rate"] for r in per_seed])}


def delta(rows, trained, other):
    left = {(r["case"], r["seed"]): r for r in rows if r["selector"] == trained}
    right = {(r["case"], r["seed"]): r for r in rows if r["selector"] == other}
    keys = sorted(left.keys() & right.keys())
    common = [k for k in keys if left[k]["legal"] and right[k]["legal"]]
    lt = geomean([left[k]["normalized_score"] for k in common])
    rt = geomean([right[k]["normalized_score"] for k in common])
    return {"trained": trained, "comparison": other, "paired_episodes": len(keys),
            "delta_legal_rate": mean(int(left[k]["legal"]) - int(right[k]["legal"]) for k in keys),
            "jointly_legal_pairs": [list(k) for k in common],
            "trained_common_geomean": lt, "comparison_common_geomean": rt,
            "delta_common_geomean_score": lt - rt if common else None,
            "common_score_ratio": lt / rt if common else None,
            "delta_mean_reward": mean(left[k]["reward"] - right[k]["reward"] for k in keys),
            "delta_runtime_s": mean(left[k]["runtime_s"] - right[k]["runtime_s"] for k in keys),
            "delta_engine_calls": mean(left[k]["engine_calls"] - right[k]["engine_calls"] for k in keys)}


class ObservedEnv(RoutingEnv):
    """Passive recording; routing decisions remain in the frozen collector."""
    def __init__(self, job, out, phase, selector):
        super().__init__(method="best")
        self.job, self.out, self.phase, self.selector = job, Path(out), phase, selector
        self.actions = []
        self.started = self.finished = None
        self.solution_path = self.trace_path = None
        self.bookkeeping_s = 0.
        self.final_state = None

    def reset(self, *args, **kwargs):
        self.started = perf_counter()
        observation = super().reset(*args, **kwargs)
        assert not observation["routes"] and observation["best_total_delay"] is None
        return observation

    def step(self, net_id):
        result = super().step(net_id)
        self.actions.append({"net_id": net_id, "engine_status": result.diagnostics["engine_status"],
                             "engine_calls": result.observation["budget"]["engine_calls"],
                             "pass": result.observation["pass_number"]})
        if result.done:
            self._export()
        return result

    def eligible_net_ids(self):
        ids = super().eligible_net_ids()
        if not ids:
            self._export()
        return ids

    def _export(self):
        if self.finished is not None:
            return
        start = perf_counter()
        self.final_state = self.observation()
        stem = (self.job['metadata'].case_id if self.phase == "hard" else
                f"{self.job['metadata'].case_id}__seed{self.job['_seed']}")
        folder = self.out / "solutions" / self.phase / self.selector
        solution = self.best_solution()
        if solution is not None:
            folder.mkdir(parents=True, exist_ok=True)
            self.solution_path = folder / f"{stem}.sol.json"
            solution.save(str(self.solution_path))
        self.trace_path = self.out / "decisions" / self.phase / self.selector / f"{stem}.json"
        write_json(self.trace_path, {"trajectory_id": self.job["trajectory_id"], "actions": self.actions})
        self.bookkeeping_s = perf_counter() - start
        self.finished = perf_counter()


def test_jobs(frozen, selector, *, only_case=None, phase="test"):
    split = load_split(frozen["dataset"]["root"], split="test")
    jobs = []
    for i, entry in enumerate(split.entries):
        if only_case is not None and entry["case_id"] != only_case:
            continue
        for seed in frozen["variant_seeds"][selector]:
            instance, meta = split.load(i)
            job = make_job(instance, meta, trajectory_id=f"{phase}/{selector}/{meta.case_id}/{seed}",
                           environment_seed=seeded(seed, "test", meta.case_id, "engine"),
                           action_seed=seeded(seed, "test", meta.case_id, "actions"))
            job.update(_seed=seed, _input_path=str(Path(frozen["dataset"]["root"]) / entry["instance_file"]))
            jobs.append(job)
    return jobs


def hard_jobs(frozen, selector):
    suite = read_json(Path(frozen["hard_suite"]) / "suite.json")
    jobs = []
    seed = frozen["deterministic_seed"]
    for entry in sorted(suite["cases"], key=lambda e: e["name"]):
        path = Path(frozen["hard_suite"]) / entry["instance_file"]
        instance = Instance.load(str(path))
        meta = SimpleNamespace(case_id=entry["name"], reference_total_delay=entry["baseline_total"])
        job = make_job(instance, meta, trajectory_id=f"hard/{selector}/{meta.case_id}/{seed}",
                       environment_seed=seeded(seed, "hard", meta.case_id, "engine"),
                       action_seed=seeded(seed, "hard", meta.case_id, "actions"))
        job.update(_seed=seed, _input_path=str(path))
        jobs.append(job)
    return jobs


def run_variant(out, frozen, phase, selector, deadline, *, only_case=None):
    started = perf_counter()
    verify_frozen(out)
    if perf_counter() >= deadline:
        raise TimeoutError("Block 5 total wall cap reached")
    policy = None
    if selector not in ("bbox", "random"):
        label = "initial" if selector.startswith("initial") else "trained"
        policy, payload = load_checkpoint(frozen["checkpoints"][label]["path"], device=frozen["device"])
        policy.requires_grad_(False)
        assert payload["update"] == frozen["checkpoints"][label]["update"]
        del payload
    jobs = (hard_jobs(frozen, selector) if phase == "hard" else
            test_jobs(frozen, selector, only_case=only_case, phase=phase))
    observers = []

    def factory():
        env = ObservedEnv(jobs[len(observers)], out, phase, selector)
        observers.append(env)
        return env

    shared_startup_s = perf_counter() - started
    selection = selector if selector in ("bbox", "random") else ("policy_deterministic" if selector.endswith("deterministic") else "policy_sampled")
    collection_start = perf_counter()
    with patch("m3d.scheduler_rollout.RoutingEnv", factory):
        trajectories = collect_rollouts(jobs, selector=selection, policy=policy, config=frozen["block4_config"],
                                        floor=frozen["failure_floor"], device=frozen["device"], deadline=deadline)
    collection_s = perf_counter() - collection_start
    # Each wave runs until its longest episode finishes. Residual collector
    # bookkeeping outside those intervals is shared evenly, never hidden.
    wave_spans = []
    for offset in range(0, len(observers), frozen["concurrency"]):
        wave = observers[offset:offset + frozen["concurrency"]]
        wave_spans.append(max(e.finished for e in wave) - min(e.started for e in wave))
    shared_bookkeeping_s = max(0., collection_s - sum(wave_spans))
    rows = []
    checking_s = 0.
    for job, env, trajectory in zip(jobs, observers, trajectories):
        result = trajectory["outcome"]
        row_start = perf_counter()
        runtime = env.finished - env.started + (shared_startup_s + shared_bookkeeping_s) / len(jobs)
        row = {**result, "phase": phase, "case": job["metadata"].case_id, "selector": selector,
               "seed": job["_seed"], "instance_file": job["_input_path"],
               "instance_sha256": file_hash(job["_input_path"]),
               "total_delay": result["final_delay"], "reference_delay": job["metadata"].reference_total_delay,
               "normalized_score": None, "terminal_reward": result["reward"],
               "scheduling_actions": len(env.actions), "steps": len(env.actions),
               "missing_net_ids": list(env.final_state["missing_net_ids"]),
               "expansions_charged": env.final_state["budget"]["expansions_charged"],
               "stop_reason": result["termination_reason"], "environment_runtime_s": result["runtime_s"],
               "runtime_s": runtime, "shared_startup_allocated_s": shared_startup_s / len(jobs),
               "shared_bookkeeping_allocated_s": shared_bookkeeping_s / len(jobs),
               "solution_file": str(env.solution_path) if env.solution_path else None,
               "solution_sha256": file_hash(env.solution_path) if env.solution_path else None,
               "decisions_file": str(env.trace_path), "decisions_sha256": file_hash(env.trace_path),
               "frozen_config_sha256": file_hash(Path(out) / "frozen_config.json")}
        row["runtime_s"] += perf_counter() - row_start
        check_start = perf_counter()
        if row["legal"]:
            saved = Submission.load(str(env.solution_path))
            checked = check(job["instance"], saved)
            scored = score_case(job["instance"], saved, row["reference_delay"], row["runtime_s"])
            assert checked.legal and scored.legal and checked.total_delay == row["total_delay"]
            assert math.isclose(math.log(scored.ratio), row["reward"], abs_tol=1e-12)
            row.update(normalized_score=scored.ratio, checker_legal=True, scorer_total_delay=scored.total_delay)
        else:
            assert env.solution_path is None and row["total_delay"] is None
            row["checker_legal"] = False
        row["checker_scorer_s"] = perf_counter() - check_start
        checking_s += row["checker_scorer_s"]
        append_json(Path(out) / "raw_episodes.jsonl", row)
        rows.append(row)
    elapsed = perf_counter() - started
    batch = {"phase": phase, "selector": selector, "episodes": len(rows),
             "end_to_end_batch_wall_s": elapsed, "solver_batch_wall_s": elapsed - checking_s,
             "external_checker_scorer_s": checking_s, "shared_startup_s": shared_startup_s,
             "shared_collector_bookkeeping_s": shared_bookkeeping_s, "concurrency": frozen["concurrency"]}
    append_json(Path(out) / "batch_runtime.jsonl", batch)
    print(json.dumps({"event": "completed_selector", **batch, "aggregate": aggregate(rows)}), flush=True)
    return rows, batch


def seal_test(out, result):
    out = Path(out)
    write_json(out / "test_results.json", result)
    files = [out / "test_results.json"]
    for row in result["episodes"]:
        files.append(Path(row["decisions_file"]))
        if row["solution_file"]:
            files.append(Path(row["solution_file"]))
    hashes = {str(p): file_hash(p) for p in files}
    write_json(out / "test_seal.json", {"sealed_utc": utc_now(), "files": hashes})
    for path in [*files, out / "test_seal.json"]:
        path.chmod(0o444)


def verify_test_seal(out):
    for path, sha in read_json(Path(out) / "test_seal.json")["files"].items():
        assert file_hash(path) == sha, f"sealed test artifact changed: {path}"


def package_submission(out, frozen, hard):
    selected = [r for r in hard["episodes"] if r["selector"] == frozen["hard_primary"]]
    if len(selected) != 9 or not all(r["legal"] for r in selected):
        return {"created": False, "reason": f"trained primary system completed {sum(r['legal'] for r in selected)}/9 hard cases legally"}
    target = Path("submissions/hard/grpo_scheduler")
    if target.exists():
        raise ValueError("submission path already exists; refusing overwrite")
    target.mkdir(parents=True)
    for row in selected:
        shutil.copyfile(row["solution_file"], target / f"{row['case']}.sol.json")
    runtimes = {r["case"]: r["runtime_s"] for r in selected}
    write_json(target / "runtime.json", runtimes)
    write_json(target / "meta.json", {"author": "local experiment", "method": "grpo_scheduler",
               "date": utc_now()[:10], "description": "RL selects which net to process; Block 1 constructs routes in the Block 2 environment.",
               "training": "Only generated Block 3 training cases; validation selected the frozen checkpoint.",
               "public_hard_evaluation": "This checkpoint was evaluated on public hard-tier cases only in Block 5. The earlier Block 2 bbox case_05 check remains separately documented.",
               "warm_starts": "No reference or participant routes used as warm starts.",
               "checkpoint_sha256": frozen["checkpoints"]["trained"]["sha256"],
               "frozen_config_sha256": file_hash(Path(out) / "frozen_config.json")})
    scored = score_submission_set(read_json("benchmarks_hard/suite.json"), "benchmarks_hard", str(target), "grpo_scheduler", runtimes)
    assert scored.complete
    # Verify only this package with the unchanged repository script.
    verification_root = Path(out) / "submission_verification"
    (verification_root / "hard").mkdir(parents=True)
    (verification_root / "hard/grpo_scheduler").symlink_to(target.resolve(), target_is_directory=True)
    result = subprocess.run([sys.executable, "scripts/verify_submissions.py", str(verification_root)], text=True, capture_output=True, check=True)
    (Path(out) / "submission_verification.txt").write_text(result.stdout)
    return {"created": True, "path": str(target), "official_score": scored.to_dict(), "repository_verification": "passed"}


def evaluation_result(rows, batches, variants):
    return {"created_utc": utc_now(), "episodes": rows, "batches": batches,
            "aggregates": {v: aggregate([r for r in rows if r["selector"] == v]) for v in variants},
            "per_case": {v: {case: aggregate([r for r in rows if r["selector"] == v and r["case"] == case])
                              for case in sorted({r["case"] for r in rows})} for v in variants}}


def evaluate(out, *, started=None):
    out = Path(out)
    started = perf_counter() if started is None else started
    frozen = verify_frozen(out)
    if not torch.backends.mps.is_available():
        raise RuntimeError("Frozen MPS device unavailable; no held-out input was opened")
    torch.set_num_threads(frozen["cpu_threads"])
    deadline = started + frozen["evaluation_wall_cap_s"]
    # Exclusive marker is deliberately never removed: a campaign cannot restart
    # silently after outcomes have been inspected, including after a crash.
    with (out / "run_started.json").open("x") as stream:
        json.dump({"started_utc": utc_now(), "frozen_config_sha256": file_hash(out / "frozen_config.json")}, stream)
    try:
        test_rows, test_batches = [], []
        for selector in frozen["test_variants"]:
            rows, batch = run_variant(out, frozen, "test", selector, deadline)
            test_rows.extend(rows)
            test_batches.append(batch)
        test = evaluation_result(test_rows, test_batches, frozen["test_variants"])
        test["deltas"] = [delta(test_rows, trained, other) for trained, others in (
            ("trained_deterministic", ["bbox", "random", "initial_deterministic"]),
            ("trained_sampled", ["bbox", "random", "initial_sampled"])) for other in others]
        test["routing_budget"] = frozen["routing_budget"]
        test["seeds"] = frozen["variant_seeds"]
        seal_test(out, test)
        print(json.dumps({"event": "test_sealed", "sha256": file_hash(out / "test_results.json"),
                          "episodes": len(test_rows)}), flush=True)

        # Exactly one predeclared small-case replay; it is never pooled into the
        # held-out comparison or used to change seeds/checkpoint/configuration.
        spec = frozen["reproducibility"]
        replay_rows, _ = run_variant(out, frozen, "reproducibility", spec["selector"], deadline, only_case=spec["case_id"])
        original = next(r for r in test_rows if r["case"] == spec["case_id"] and r["selector"] == spec["selector"])
        replay = replay_rows[0]
        original_actions = read_json(original["decisions_file"])["actions"]
        replay_actions = read_json(replay["decisions_file"])["actions"]
        fields = ("legal", "total_delay", "reward", "conflicts", "missing_net_ids", "engine_calls", "passes", "stop_reason")
        matches = {key: original[key] == replay[key] for key in fields}
        repro = {**spec, "decisions_match": original_actions == replay_actions, "result_fields_match": matches,
                 "solution_file_hash_match": original["solution_sha256"] == replay["solution_sha256"],
                 "original_runtime_s": original["runtime_s"], "replay_runtime_s": replay["runtime_s"],
                 "original_trajectory": original["trajectory_id"], "replay_trajectory": replay["trajectory_id"],
                 "passed": original_actions == replay_actions and all(matches.values()) and original["solution_sha256"] == replay["solution_sha256"]}
        write_json(out / "reproducibility.json", repro)
        print(json.dumps({"event": "reproducibility", **repro}), flush=True)

        verify_test_seal(out)
        hard_rows, hard_batches = [], []
        for selector in frozen["hard_variants"]:
            rows, batch = run_variant(out, frozen, "hard", selector, deadline)
            hard_rows.extend(rows)
            hard_batches.append(batch)
        hard = evaluation_result(hard_rows, hard_batches, frozen["hard_variants"])
        hard["deltas"] = [delta(hard_rows, frozen["hard_primary"], "bbox")]
        hard["official"] = {}
        suite = read_json(Path(frozen["hard_suite"]) / "suite.json")
        for selector in frozen["hard_variants"]:
            rows = [r for r in hard_rows if r["selector"] == selector]
            complete = len(rows) == len(suite["cases"]) and all(r["legal"] for r in rows)
            official = {"complete": complete, "n_legal": sum(r["legal"] for r in rows), "n_cases": 9,
                        "aggregate": None, "reason": None if complete else "Incomplete; no competitive aggregate calculated"}
            if complete:
                scored = score_submission_set(suite, frozen["hard_suite"], str(out / "solutions/hard" / selector), selector,
                                               {r["case"]: r["runtime_s"] for r in rows})
                assert scored.complete
                official.update(aggregate=scored.aggregate, scorer=scored.to_dict())
            hard["official"][selector] = official
        write_json(out / "hard_results.json", hard)
        packaging = package_submission(out, frozen, hard)
        verify_test_seal(out)
        verify_frozen(out)
        initial_delta = next(d for d in test["deltas"] if d["trained"] == "trained_sampled" and d["comparison"] == "initial_sampled")
        baseline_deltas = [d for d in test["deltas"] if d["trained"] == "trained_sampled" and d["comparison"] in ("bbox", "random")]
        hard_delta = hard["deltas"][0]
        # Primary overall evidence is mean terminal reward (frozen failure
        # floor); paired legal score deltas remain visible as conditional data.
        beats_initial = initial_delta["delta_mean_reward"] > 1e-8
        beats_baseline = any(d["delta_mean_reward"] > 1e-8 for d in baseline_deltas)
        beats_hard = hard_delta["delta_legal_rate"] > 0 or (hard_delta["delta_legal_rate"] == 0 and
                      (hard_delta["delta_common_geomean_score"] or 0) > 1e-8)
        conclusion = ("RL improved scheduling" if beats_initial and beats_baseline and beats_hard else
                      "RL learned something but did not beat the nonlearned scheduler" if beats_initial else
                      "RL did not produce measurable improvement")
        summary = {"status": "complete", "evaluation_completed_utc": utc_now(), "evaluation_runtime_s": perf_counter() - started,
                   "wall_cap_s": frozen["evaluation_wall_cap_s"], "frozen_config_sha256": file_hash(out / "frozen_config.json"),
                   "checkpoint": frozen["checkpoints"]["trained"], "test": test["aggregates"], "test_deltas": test["deltas"],
                   "hard": hard["aggregates"], "hard_official": hard["official"], "hard_deltas": hard["deltas"],
                   "reproducibility_passed": repro["passed"], "submission": packaging, "conclusion": conclusion,
                   "optional_all_tiers_run": False, "correctness_fixes": frozen["correctness_fixes"],
                   "block2_gap": "Preserved: earlier case_05 had 3 vertex conflicts after 50 passes.",
                   "block4_acceptance": "not_improved", "training_updates_in_block5": 0,
                   "limitations": ["Four generated test cases and two stochastic seeds; no statistical significance claim.",
                                   "Timing comparisons are internal to this machine and concurrency setting.",
                                   "Legal-only scores are conditional; incomplete hard-tier runs have no competitive aggregate."]}
        if summary["evaluation_runtime_s"] > summary["wall_cap_s"]:
            summary["status"] = "wall_cap_exceeded"
        write_json(out / "summary.json", summary)
        verify(out)
        render_report(out)
        print(json.dumps({"event": "complete", "status": summary["status"], "conclusion": conclusion,
                          "runtime_s": summary["evaluation_runtime_s"], "hard_official": hard["official"],
                          "submission": packaging}), flush=True)
        return 0 if summary["status"] == "complete" else 2
    except Exception as exc:
        write_json(out / "evaluation_error.json", {"status": "partial", "error": repr(exc), "elapsed_s": perf_counter() - started,
                   "policy": "Existing episodes are retained. The campaign cannot restart automatically."})
        raise


def verify(out, *, started=None):
    out = Path(out)
    frozen = verify_frozen(out)
    verify_test_seal(out)
    history_audit(read_json(Path(frozen["dataset"]["root"]) / "manifest.json"))
    rows = [json.loads(line) for line in (out / "raw_episodes.jsonl").read_text().splitlines()]
    assert len({r["trajectory_id"] for r in rows}) == len(rows)
    expected = {("test", v, slot["case_id"], seed) for v in frozen["test_variants"]
                for slot in frozen["dataset"]["test_slots"] for seed in frozen["variant_seeds"][v]}
    assert {(r["phase"], r["selector"], r["case"], r["seed"]) for r in rows if r["phase"] == "test"} == expected
    replay = [r for r in rows if r["phase"] == "reproducibility"]
    assert len(replay) == 1 and replay[0]["case"] == frozen["reproducibility"]["case_id"]
    hard = read_json(out / "hard_results.json")
    assert len(rows) == len(expected) + 1 + 18
    assert read_json(out / "test_results.json")["episodes"] == [r for r in rows if r["phase"] == "test"]
    assert hard["episodes"] == [r for r in rows if r["phase"] == "hard"]
    assert len([r for r in rows if r["phase"] == "hard"]) == 18
    checked = 0
    for row in rows:
        assert file_hash(row["instance_file"]) == row["instance_sha256"]
        assert file_hash(row["decisions_file"]) == row["decisions_sha256"]
        assert len(read_json(row["decisions_file"])["actions"]) == row["scheduling_actions"]
        if row["legal"]:
            assert file_hash(row["solution_file"]) == row["solution_sha256"]
            inst, sub = Instance.load(row["instance_file"]), Submission.load(row["solution_file"])
            c = check(inst, sub)
            s = score_case(inst, sub, row["reference_delay"], row["runtime_s"])
            assert c.legal and s.legal and s.total_delay == row["total_delay"]
            assert s.ratio == row["normalized_score"] and math.isclose(math.log(s.ratio), row["reward"], abs_tol=1e-12)
            checked += 1
        else:
            assert row["solution_file"] is None and row["total_delay"] is None and row["normalized_score"] is None
            assert row["reward"] == frozen["failure_floor"]["reward"]
    for label, official in hard["official"].items():
        if official["complete"]:
            selected = [r for r in hard["episodes"] if r["selector"] == label]
            s = score_submission_set(read_json("benchmarks_hard/suite.json"), "benchmarks_hard", str(out / "solutions/hard" / label), label,
                                     {r["case"]: r["runtime_s"] for r in selected})
            assert s.complete and s.aggregate == official["aggregate"]
        else:
            assert official["aggregate"] is None
    result = {"passed": True, "frozen_hashes_verified": True, "prior_test_use_audit_passed": True,
              "sealed_test_results_unchanged": True, "episode_count": len(rows), "test_episodes": len(expected),
              "explicit_small_case_replays": 1, "hard_episodes": 18, "legal_saved_outputs_checked": checked,
              "scorer_values_reproduced": True, "training_updates": 0}
    write_json(out / "verification.json", result)
    print(json.dumps({"event": "verification", **result}), flush=True)
    return 0


def render_report(out):
    out = Path(out)
    frozen, summary = read_json(out / "frozen_config.json"), read_json(out / "summary.json")
    test, hard, repro = (read_json(out / f"{name}.json") for name in ("test_results", "hard_results", "reproducibility"))
    verify_result = read_json(out / "verification.json")
    fmt = lambda value, digits=4: "—" if value is None else f"{value:.{digits}f}"
    lines = ["# Block 5 final evaluation", "", f"**{summary['conclusion']}.**", "",
             "## A. System and frozen experiment", "",
             f"The validation-selected checkpoint is update {summary['checkpoint']['update']}, with {summary['checkpoint']['parameters']:,} parameters.",
             f"Checkpoint SHA-256: `{summary['checkpoint']['sha256']}`.",
             f"Source commit: `{frozen['source_commit']}`; the working tree contains uncommitted files, so the authoritative source fingerprint is `{frozen['source_sha256']}`.",
             "Block 1 constructs each selected net's route using method `best`. Block 2 owns the empty-start layout, negotiated congestion, rollback and legality snapshots. The Block 4 graph/Transformer policy chooses eligible net IDs.",
             f"Frozen budgets: `{json.dumps(frozen['routing_budget'], sort_keys=True)}`. Concurrency: {frozen['concurrency']}. No training, parameter changes, congestion changes, reward changes, or routing changes were made.",
             f"Hardware/runtime: `{json.dumps(frozen['hardware'], sort_keys=True)}`.",
             f"Evaluation runtime: {summary['evaluation_runtime_s']:.2f}s under a {summary['wall_cap_s']}s cap. External verification time is recorded separately in the raw and batch results.",
             frozen['runtime_measurement'], "",
             "## B. First held-out generated test evaluation", "",
             "Four cases; stochastic seeds 5101 and 5102. Deterministic policies use 5101. Bbox uses both engine seeds to provide paired comparisons. The sole additional small-case replay is excluded from aggregates.",
             "Scores are reference delay / our legal delay; terminal reward is its logarithm. Failed episodes receive the unchanged dataset-derived floor. Conditional scores/delays exclude failures; failed episodes still contribute to legality, reward, runtime and action counts.", "",
             "| Selector | Legal | Legal-only score GM | Mean reward | Failed conflicts | Mean legal delay | Runtime mean ± SD (s) | Mean actions | Mean calls |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, a in test["aggregates"].items():
        lines.append(f"| {name} | {a['n_legal']}/{a['episodes']} | {fmt(a['legal_geomean_score'])} | {fmt(a['mean_reward'])} | {fmt(a['mean_failed_conflicts'],1)} | {fmt(a['mean_legal_delay'],1)} | {fmt(a['runtime_s']['mean'],2)} ± {fmt(a['runtime_s']['std'],2)} | {fmt(a['steps']['mean'],1)} | {fmt(a['engine_calls']['mean'],1)} |")
    lines += ["", "The standard deviations above describe episodes, including case difficulty. Per-case stochastic statistics, each seed's aggregates, and individual episodes are preserved in `test_results.json`; they are not independent samples of a large benchmark population.", "", "## C. Public hard tier", "",
              "All nine cases started with empty wiring. Trained deterministic inference was selected before test access as the sole primary challenge system. Reference delay scalars come from `benchmarks_hard/suite.json`; reference and participant wires were never loaded.", "",
              "| Case | Selector | Legal | Delay | Baseline | Ratio | Runtime (s) | Passes | Calls | Stop | Conflicts | Missing |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for r in sorted(hard["episodes"], key=lambda r: (r["case"], r["selector"])):
        lines.append(f"| {r['case']} | {r['selector']} | {r['legal']} | {r['total_delay'] if r['legal'] else '—'} | {r['reference_delay']} | {fmt(r['normalized_score'])} | {fmt(r['runtime_s'],2)} | {r['passes']} | {r['engine_calls']} | {r['stop_reason']} | {r['conflicts']} | {r['missing_nets']} |")
    for name, result in hard["official"].items():
        lines += ["", f"{name}: {result['n_legal']}/9 legal. Official aggregate: {fmt(result['aggregate'])}. {result['reason'] or 'Complete and scored with the unchanged official scorer.'}"]
    lines += ["", "## D. RL ablation", "", "Only net selection differs. Positive legality/score/reward deltas favor RL; positive runtime/call deltas mean RL costs more. Score deltas use the explicitly recorded jointly legal case/seed pairs, never an unmatched mixture of solved cases.", "",
              "| Dataset | Trained mode vs selector | Paired episodes | Δ legality | Common legal pairs | Δ score GM | Δ mean reward | Δ runtime (s) | Δ calls |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for phase, ds in (("test", test["deltas"]), ("hard", hard["deltas"])):
        for d in ds:
            lines.append(f"| {phase} | {d['trained']} vs {d['comparison']} | {d['paired_episodes']} | {fmt(d['delta_legal_rate'])} | {len(d['jointly_legal_pairs'])} | {fmt(d['delta_common_geomean_score'])} | {fmt(d['delta_mean_reward'])} | {fmt(d['delta_runtime_s'],2)} | {fmt(d['delta_engine_calls'],1)} |")
    lines += ["", "## E. Failures, reproducibility and scope", "",
              f"Reproducibility on `{repro['case_id']}`: decisions match = {repro['decisions_match']}; final fields = {all(repro['result_fields_match'].values())}; solution hash = {repro['solution_file_hash_match']}. Runtime is not expected to match exactly.",
              f"Saved-output verification: {verify_result['legal_saved_outputs_checked']} legal files independently checked, scorer values reproduced, frozen hashes verified, and the test seal remained unchanged through the hard-tier evaluation.",
              "Block 2's earlier case_05 failure (three vertex conflicts after 50 passes) remains unchanged. Block 4's learning acceptance remained unmet. Block 5 uses the frozen 20-pass budget; its results do not revise the earlier 50-pass acceptance record.",
              "Detailed failure ownership, missing net IDs, native failure diagnostics and stop reasons are retained in `raw_episodes.jsonl`. No failed layout receives a legal score or a delay average contribution. Conditional delay/call reductions can reflect early termination and are not an overall success claim.",
              "No all-tier campaign was run. Absolute times are not compared to public leaderboard times from other hardware.", "",
              "## F. Conclusion and packaging", "", f"**{summary['conclusion']}.** See the paired deltas above for the exact supported changes; four test cases and two stochastic seeds support no significance claim.",
              f"Submission: `{json.dumps(summary['submission'], sort_keys=True)}`.",
              "No leaderboard edit, push, or PR was made. Full configuration, checksums, raw episodes, decision traces, legal route files and summaries are under `build/block5/`.", "",
              "Reproduce artifact verification without rerouting:", "", "```sh",
              "/Users/mukeshreddy/anaconda3/bin/python3 -m m3d.final_evaluation --phase verify --out build/block5", "```", "",
              "The campaign commands were `--phase freeze` followed by `--phase evaluate` using that interpreter. Existing freeze/run markers prevent a silent second test campaign.", ""]
    Path("docs/BLOCK_5_RESULTS.md").write_text("\n".join(lines))
