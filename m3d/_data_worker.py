"""Bounded subprocess entrypoint for upstream feasible generation, with a journal.

The wrappers observe upstream calls; they do not change generation or routing.
They exist only inside this process, never in the training environment.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

from . import baseline, generator, negotiated
from .checker import check


def run(request: Path, output: Path, journal: Path):
    cfg = generator.GenConfig(**json.loads(request.read_text()))
    start = perf_counter()
    serial = 0
    times = {"generation_s": 0.0, "reference_routing_s": 0.0, "checking_s": 0.0}
    candidate_fn, baseline_fn, negotiated_fn = generator.generate_candidate, baseline.route, negotiated.route_negotiated
    check_fn = generator.check

    def record(event, **fields):
        with journal.open("a") as stream:
            stream.write(json.dumps({"event": event, "candidate_index": serial,
                                     "elapsed_s": perf_counter() - start, **fields}) + "\n")

    def candidate(config, seed):
        nonlocal serial
        serial += 1
        record("candidate_started", effective_seed=seed, effective_parameters=asdict(config))
        t0 = perf_counter()
        try:
            result = candidate_fn(config, seed)
            record("candidate_generated")
            return result
        except Exception as exc:
            record("placement_failed", reason=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            times["generation_s"] += perf_counter() - t0

    def route(function, instance, **kwargs):
        record("reference_started", router=cfg.router)
        t0 = perf_counter()
        try:
            result, stats = function(instance, **kwargs)
            record("reference_finished", produced_certificate=result is not None, stats=asdict(stats))
            return result, stats
        finally:
            times["reference_routing_s"] += perf_counter() - t0

    def checked(instance, submission):
        t0 = perf_counter()
        result = check_fn(instance, submission)
        times["checking_s"] += perf_counter() - t0
        record("candidate_checked", legal=result.legal, total_delay=result.total_delay,
               reasons=result.reasons + [reason for n in result.nets for reason in n.reasons])
        return result

    generator.generate_candidate = candidate
    baseline.route = lambda inst, **kw: route(baseline_fn, inst, **kw)
    negotiated.route_negotiated = lambda inst, **kw: route(negotiated_fn, inst, **kw)
    generator.check = checked
    try:
        result = generator.generate_feasible(cfg)
        checked_at = perf_counter()
        final = check(result.instance, result.reference)
        times["checking_s"] += perf_counter() - checked_at
        if not final.legal or final.total_delay != result.baseline_total:
            raise ValueError("reference failed final official checker verification")
        payload = {"status": "certified", "instance": result.instance.to_dict(),
                   "certificate": result.reference.to_dict(), "reference_total_delay": final.total_delay,
                   "checker_legal": True, "upstream_attempts": result.attempts, "timings": times}
    except Exception as exc:
        record("generation_failed", reason=f"{type(exc).__name__}: {exc}")
        payload = {"status": "not_certified", "reason": f"{type(exc).__name__}: {exc}",
                   "timings": times}
    payload["worker_seconds"] = perf_counter() - start
    output.write_text(json.dumps(payload))


if __name__ == "__main__":
    if sys.argv[1] == "--audit":
        from .data_generation import audit_dataset
        root, request, output = (Path(arg) for arg in sys.argv[2:])
        try:
            payload = {"status": "passed", "audit": audit_dataset(root, released=json.loads(request.read_text()))}
        except Exception as exc:
            payload = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        output.write_text(json.dumps(payload))
    else:
        run(*(Path(arg) for arg in sys.argv[1:]))
