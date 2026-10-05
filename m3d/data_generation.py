"""Block 3 generation/resume and certificate audit. No environment success filter."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .baseline import Baseline
from .checker import check
from .model import Instance, Submission
from .negotiated import Negotiated
from .routing_data import (FAMILIES, REPO, SPLITS, dataset_path, digest, file_hash,
                           fingerprints, load_config, plan_cases, requested_config, validate_config,
                           validate_entry_slot)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def _locked(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".generation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another generator is writing this dataset") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def released_inputs() -> dict:
    entries = []
    for directory in sorted(REPO.glob("benchmarks*")):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            raw = json.loads(path.read_text())
            if raw.get("format") == "m3d-instance":
                entries.append({"path": str(path.relative_to(REPO)), "file_sha256": file_hash(path),
                                "geometry": fingerprints(Instance.from_dict(raw))["geometry"]})
    if not entries:
        raise ValueError("no released benchmark inputs found for exclusion audit")
    return {"count": len(entries), "catalog_sha256": digest(entries), "inputs": entries}


def source_provenance() -> dict:
    names = ("generator", "suite", "model", "grid", "baseline", "negotiated", "checker",
             "routing_data", "data_generation", "_data_worker")
    files = {f"m3d/{name}.py": file_hash(REPO / "m3d" / f"{name}.py") for name in names}
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True, timeout=2)
    return {"git_head": result.stdout.strip() if result.returncode == 0 else None,
            "files": files, "source_sha256": digest(files), "python": sys.version}


def _router_config(cfg):
    cls = Negotiated if cfg.router == "negotiated" else Baseline
    defaults = {name: p.default for name, p in inspect.signature(cls).parameters.items()
                if p.default is not inspect.Parameter.empty}
    defaults["order"] = cfg.baseline_order
    return {"identity": f"{cls.__module__}.{cls.__name__}", "configuration": defaults}


def _worker(cfg, directory: Path, timeout_s: float) -> dict:
    start = perf_counter()
    deadline = start + timeout_s
    directory.mkdir(parents=True, exist_ok=True)
    request, output, journal = (directory / name for name in ("request.json", "result.json", "events.jsonl"))
    _write_json(request, asdict(cfg))
    # A new retry has its own directory. Only an interrupted, unpublished
    # attempt could leave these files; the journal is retained in the report.
    with (directory / "stderr.txt").open("w") as errors:
        process = subprocess.Popen([sys.executable, "-m", "m3d._data_worker",
                                    str(request.resolve()), str(output.resolve()), str(journal.resolve())],
                                   cwd=REPO, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            code = process.wait(timeout=max(0.0, deadline - perf_counter()))
            if code == 0 and output.exists():
                result = json.loads(output.read_text())
            else:
                result = {"status": "not_certified", "reason": f"worker exit {code}"}
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)
            result = {"status": "timeout", "reason": "not certified within budget; not proof of infeasibility"}
        except BaseException:
            process.kill()
            process.wait(timeout=1)
            raise
    result["elapsed_s"] = perf_counter() - start
    result["events"] = []
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                result["events"].append(json.loads(line))
            except json.JSONDecodeError:
                # A killed worker can leave only its final line incomplete.
                result["journal_truncated"] = True
    errors = (directory / "stderr.txt").read_text()
    if errors:
        result["stderr"] = errors
    return result


def _bounded_audit(root: Path, released: dict, deadline: float) -> dict:
    if perf_counter() >= deadline:
        raise TimeoutError("no remaining time for certificate audit")
    directory = root / ".staging" / "audit"
    directory.mkdir(parents=True, exist_ok=True)
    request, output = directory / "request.json", directory / "result.json"
    _write_json(request, released)
    with (directory / "stderr.txt").open("w") as errors:
        process = subprocess.Popen([sys.executable, "-m", "m3d._data_worker", "--audit", str(root.resolve()),
                                    str(request.resolve()), str(output.resolve())], cwd=REPO,
                                   stdout=subprocess.DEVNULL, stderr=errors)
        try:
            code = process.wait(timeout=max(0.0, deadline - perf_counter()))
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait(timeout=1)
            raise TimeoutError("certificate audit exceeded the total time cap") from exc
        except BaseException:
            process.kill()
            process.wait(timeout=1)
            raise
    if code != 0:
        raise ValueError(f"certificate audit worker exited {code}")
    result = json.loads(output.read_text())
    if result["status"] != "passed":
        raise ValueError(result["reason"])
    if perf_counter() >= deadline:
        raise TimeoutError("certificate audit result arrived after the total deadline")
    return result["audit"]


def _effective(instance: Instance, requested: dict) -> dict:
    pins = instance.pin_by_id()
    destinations = [len(n.sinks) for n in instance.nets]
    cross = sum(len({pins[p].die for p in n.pins()}) > 1 for n in instance.nets)
    distances = [sum(abs(a - b) for a, b in zip(pins[n.driver].vertex(), pins[s].vertex()))
                 for n in instance.nets for s in n.sinks]
    return {"seed": instance.seed, "parameters": instance.params,
            "width": instance.width, "height": instance.height, "layers": instance.layers,
            "n_nets": len(instance.nets), "n_pins": len(instance.pins),
            "destination_count_min": min(destinations), "destination_count_max": max(destinations),
            "destination_count_mean": sum(destinations) / len(destinations),
            "multidestination_fraction": sum(n > 1 for n in destinations) / len(destinations),
            "cross_die_fraction": cross / len(instance.nets),
            "mean_driver_sink_manhattan": sum(distances) / len(distances),
            "net_density_per_xy_vertex": len(instance.nets) / (instance.width * instance.height),
            "requested_net_density_per_xy_vertex": requested["n_nets"] / (requested["width"] * requested["height"]),
            "grid_enlarged": (instance.width, instance.height) != (requested["width"], requested["height"]),
            "grid_growth": [instance.width - requested["width"], instance.height - requested["height"]]}


def audit_dataset(root, *, released=None) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "m3d-routing-data-v1":
        raise ValueError("unknown manifest format")
    released = released or released_inputs()
    public = {entry["geometry"] for entry in released["inputs"]}
    slots = plan_cases(manifest["configuration"])
    if slots != manifest["planned_cases"]:
        raise ValueError("manifest split slots differ from the configured plan")
    plan = {slot["case_id"]: slot for slot in slots}
    seen, ids = set(), set()
    for entry in manifest["cases"]:
        validate_entry_slot(entry, plan)
        if entry["case_id"] in ids:
            raise ValueError("duplicate case ID in manifest")
        ids.add(entry["case_id"])
        instance_path = dataset_path(root, entry["instance_file"])
        certificate_path = dataset_path(root, entry["certificate_file"])
        if file_hash(instance_path) != entry["instance_file_sha256"] or file_hash(certificate_path) != entry["certificate_file_sha256"]:
            raise ValueError(f"file hash mismatch: {entry['case_id']}")
        instance = Instance.load(str(instance_path))
        fp = fingerprints(instance)
        if fp != entry["fingerprints"]:
            raise ValueError("fingerprint mismatch")
        if fp["geometry"] in seen or fp["geometry"] in public:
            raise ValueError(f"duplicate routing geometry: {entry['case_id']}")
        seen.add(fp["geometry"])
        certificate = Submission.load(str(certificate_path))
        checked = check(instance, certificate)
        if (certificate.instance != instance.name or not checked.legal
                or checked.total_delay != entry["reference_total_delay"] or entry["checker_legal"] is not True):
            raise ValueError(f"invalid feasibility certificate or reference delay: {entry['case_id']}")
    expected_status = "complete" if len(ids) == len(plan) and not manifest.get("audit_pending", False) else "partial"
    if manifest["status"] != expected_status:
        raise ValueError("manifest completion status does not match accepted split slots")
    return {"checked_certificates": len(ids), "duplicate_count": 0,
            "released_inputs_checked": released["count"], "all_legal": True,
            "reference_delays_match": True}


def _coverage(manifest, attempts):
    coverage = {}
    metrics = ("width", "height", "n_nets", "destination_count_min", "destination_count_max",
               "destination_count_mean", "cross_die_fraction", "multidestination_fraction",
               "net_density_per_xy_vertex", "mean_driver_sink_manhattan")
    for split in SPLITS:
        coverage[split] = {}
        for family in FAMILIES:
            accepted = [e for e in manifest["cases"] if e["split"] == split and e["family"] == family]
            coverage[split][family] = {
                "target": manifest["configuration"]["counts"][split] // 4,
                "accepted": len(accepted),
                "rejected_attempts": sum(a["split"] == split and a["family"] == family
                                         and a["status"] not in ("running", "accepted") for a in attempts),
                "enlarged_cases": sum(e["effective"]["grid_enlarged"] for e in accepted),
                "requested_ranges": {key: [min(e["requested_parameters"][key] for e in accepted),
                                            max(e["requested_parameters"][key] for e in accepted)]
                                     for key in ("width", "height", "n_nets", "max_fanout", "frac_local", "frac_cross")}
                                     if accepted else {},
                "effective_ranges": {key: [min(e["effective"][key] for e in accepted),
                                            max(e["effective"][key] for e in accepted)] for key in metrics}
                                     if accepted else {},
            }
    return coverage


def generate_dataset(config: dict, root, *, progress=None) -> dict:
    validate_config(config)
    root = Path(root)
    with _locked(root):
        return _generate(config, root, progress)


def _generate(config, root, progress):
    start = perf_counter()
    deadline = start + config["limits"]["total_seconds"]
    released = released_inputs()
    source = source_provenance()
    compatibility = digest({"configuration": {k: v for k, v in config.items() if k != "counts"},
                            "source": source["source_sha256"], "released": released["catalog_sha256"]})
    manifest_path, report_path = root / "manifest.json", root / "generation_report.json"
    previous_audit = None
    resume_audit_timed_out = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        previous = manifest["configuration"]["counts"]
        if (manifest["compatibility_sha256"] != compatibility
                or any(config["counts"][s] != previous[s] for s in ("val", "test"))
                or config["counts"]["train"] < previous["train"]):
            raise ValueError("incompatible dataset configuration/source; only training count increases are supported")
        try:
            previous_audit = _bounded_audit(root, released, deadline)
        except TimeoutError:
            resume_audit_timed_out = True
        if report_path.exists():
            report = json.loads(report_path.read_text())
        elif not manifest["cases"]:
            # Recover initialization interrupted before a journal was created.
            report = {"format": "m3d-generation-report-v1", "attempts": [], "runs": []}
        else:
            raise ValueError("missing generation journal; cannot safely restore retry accounting")
        manifest["configuration"] = config
    else:
        manifest = {"format": "m3d-routing-data-v1", "status": "partial", "configuration": config,
                    "compatibility_sha256": compatibility, "source": source,
                    "released_exclusion": released, "cases": []}
        report = {"format": "m3d-generation-report-v1", "attempts": [], "runs": []}
    plan = plan_cases(config)
    manifest["planned_cases"] = plan
    accepted = {e["case_id"] for e in manifest["cases"]}
    initial_count = len(accepted)
    geometry = {e["fingerprints"]["geometry"] for e in manifest["cases"]}
    public = {e["geometry"] for e in released["inputs"]}
    for attempt in report["attempts"]:
        if attempt["status"] == "running":
            attempt["status"] = "accepted" if attempt["case_id"] in accepted else "interrupted"
            attempt["reason"] = "recovered interrupted journal; never inferred infeasibility"
            # Reserve a lost attempt's full time allowance on resume.
            attempt["elapsed_s"] = attempt["time_allowance_s"]
    manifest["audit_pending"] = True
    manifest["status"] = "partial"
    _write_json(report_path, report)
    _write_json(manifest_path, manifest)
    cap_reached = False
    for slot in plan:
        if resume_audit_timed_out:
            cap_reached = True
            break
        if slot["case_id"] in accepted:
            continue
        prior = [a for a in report["attempts"] if a["case_id"] == slot["case_id"]]
        spent = sum(a["elapsed_s"] for a in prior)
        case_deadline = perf_counter() + max(0.0, config["limits"]["case_seconds"] - spent)
        for retry in range(len(prior), config["limits"]["case_retries"]):
            remaining = deadline - perf_counter()
            if remaining <= 0:
                cap_reached = True
                break
            allowance = min(remaining, case_deadline - perf_counter(),
                            config["limits"]["attempt_seconds"])
            if allowance <= 0:
                break
            cfg = requested_config(config, slot, retry)
            requested = asdict(cfg)
            attempt = {**slot, "retry_index": retry, "requested_seed": cfg.seed,
                       "requested_parameters": requested, "status": "running",
                       "time_allowance_s": allowance, "elapsed_s": 0.0}
            report["attempts"].append(attempt)
            _write_json(report_path, report)
            allowance = max(0.0, min(allowance, case_deadline - perf_counter(), deadline - perf_counter()))
            result = _worker(cfg, root / ".staging" / slot["case_id"] / str(retry), allowance)
            if result["status"] == "certified" and perf_counter() >= min(case_deadline, deadline):
                result.update(status="timeout", reason="certificate arrived after case/total deadline; not proof of infeasibility")
            attempt.update({k: v for k, v in result.items() if k not in ("instance", "certificate")})
            spent += result["elapsed_s"]
            if result["status"] == "certified":
                instance = Instance.from_dict(result["instance"])
                certificate = Submission.from_dict(result["certificate"])
                fp = fingerprints(instance)
                # The worker's unchanged official checker has already checked
                # these exact serialized objects, inside the wall-time bound.
                if result.get("checker_legal") is not True or certificate.instance != instance.name:
                    attempt.update(status="invalid_certificate", reason="official checker rejected certificate or delay")
                elif fp["geometry"] in public or fp["geometry"] in geometry:
                    attempt.update(status="duplicate", reason="released geometry" if fp["geometry"] in public else "dataset geometry",
                                   fingerprints=fp)
                else:
                    inst_file = f"instances/{slot['split']}/{slot['case_id']}.json"
                    cert_file = f"certificates/{slot['split']}/{slot['case_id']}.sol.json"
                    _write_json(root / inst_file, instance.to_dict())
                    _write_json(root / cert_file, certificate.to_dict())
                    candidate_attempts = sum(sum(e["event"] == "candidate_started" for e in a.get("events", []))
                                             for a in report["attempts"] if a["case_id"] == slot["case_id"])
                    entry = {**slot, "requested_seed": cfg.seed, "effective_seed": instance.seed,
                             "retry_index": retry, "requested_parameters": requested,
                             "effective": _effective(instance, requested), "fingerprints": fp,
                             "instance_file": inst_file, "certificate_file": cert_file,
                             "instance_file_sha256": file_hash(root / inst_file),
                             "certificate_file_sha256": file_hash(root / cert_file),
                             "reference_router": {**_router_config(cfg), "source_sha256": source["source_sha256"]},
                             "checker_legal": True, "reference_total_delay": result["reference_total_delay"],
                             "attempt_count": candidate_attempts, "outer_attempt_count": retry + 1,
                             "generation_certification_seconds": spent,
                             "successful_attempt_timings": result["timings"]}
                    manifest["cases"].append(entry)
                    accepted.add(slot["case_id"])
                    geometry.add(fp["geometry"])
                    attempt["status"] = "accepted"
                    # Files and certificate are durable before the acceptance
                    # commit. Unlisted staging/orphan files are never loadable.
                    _write_json(manifest_path, manifest)
            _write_json(report_path, report)
            if progress:
                progress(f"{slot['case_id']} retry={retry}: {attempt['status']} ({result['elapsed_s']:.2f}s)")
            if slot["case_id"] in accepted:
                break
        if cap_reached:
            break
    missing = [slot["case_id"] for slot in plan if slot["case_id"] not in accepted]
    _write_json(manifest_path, manifest)
    try:
        audit = previous_audit if previous_audit is not None and len(accepted) == initial_count else _bounded_audit(root, released, deadline)
        manifest["audit_pending"] = False
    except TimeoutError as exc:
        audit = {"status": "pending", "reason": str(exc)}
    manifest["status"] = "complete" if not missing and not manifest["audit_pending"] else "partial"
    _write_json(manifest_path, manifest)
    run = {"elapsed_s": perf_counter() - start, "cap_seconds": config["limits"]["total_seconds"],
           "cap_reached": cap_reached or perf_counter() >= deadline}
    report["runs"].append(run)
    report.update(status=manifest["status"], accepted=len(accepted), target=len(plan),
                  shortfalls=missing, rejected_attempts=sum(a["status"] != "accepted" for a in report["attempts"]),
                  coverage=_coverage(manifest, report["attempts"]), audit=audit,
                  total_generation_seconds=sum(r["elapsed_s"] for r in report["runs"]))
    _write_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/routing_data_v1.json")
    parser.add_argument("--out", default="build/data/routing_v1")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    if args.audit_only:
        print(json.dumps(audit_dataset(args.out), indent=2))
        return 0
    report = generate_dataset(load_config(args.config), args.out, progress=lambda line: print(line, flush=True))
    print(json.dumps({k: v for k, v in report.items() if k != "attempts"}, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
