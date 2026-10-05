"""Thin, deterministic external selector for Block 2. No reference routes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .checker import check
from .model import Instance, Submission
from .routing_env import RoutingBudget, RoutingEnv


def bbox_order(instance: Instance) -> tuple[int, ...]:
    pins = instance.pin_vertex()

    def key(net):
        coordinates = [pins[p] for p in net.pins()]
        span = sum(max(v[d] for v in coordinates) - min(v[d] for v in coordinates) for d in range(3))
        return -span, net.id

    return tuple(n.id for n in sorted(instance.nets, key=key))


def drive_episode(env: RoutingEnv, *, progress=None) -> dict:
    """Each iteration selects exactly one currently eligible net externally."""
    order = bbox_order(env.observation()["instance"])
    failures = []
    while eligible := env.eligible_net_ids():
        allowed = set(eligible)
        nid = next(nid for nid in order if nid in allowed)
        result = env.step(nid)
        if result.diagnostics["engine_status"] not in ("success", "not_called"):
            failures.append(result.diagnostics)
        if progress is not None:
            progress(result)
    state = env.observation()
    return {
        "instance": state["instance"].name,
        "status": state["status"], "legal": state["best_total_delay"] is not None,
        "total_delay": state["best_total_delay"],
        "runtime_s": state["budget"]["elapsed_s"],
        "passes": state["pass_number"], "passes_completed": state["passes_completed"],
        "engine_calls": state["budget"]["engine_calls"],
        "expansions_charged": state["budget"]["expansions_charged"],
        "termination_reason": state["termination_reason"],
        "missing_net_ids": list(state["missing_net_ids"]),
        "conflict_vertices": [{"vertex": list(env._grid.coord(v)), "nets": sorted(ns)}
                              for v, ns in sorted(state["conflict_vertices"].items())],
        "conflict_edges": [{"edge": [list(env._grid.coord(v)) for v in e], "nets": sorted(ns)}
                           for e, ns in sorted(state["conflict_edges"].items())],
        "failed_calls": failures,
        "budget": state["budget"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance")
    parser.add_argument("--out", required=True, help="best legal m3d-submission JSON")
    parser.add_argument("--report", help="diagnostics JSON, including explicit failure status")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=("best", "shortest_path", "cost_distance"), default="best")
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--passes", type=int, default=50)
    parser.add_argument("--calls", type=int, default=5000)
    parser.add_argument("--expansions", type=int, default=200_000_000)
    parser.add_argument("--call-expansions", type=int, default=1_000_000)
    parser.add_argument("--call-seconds", type=float, default=10)
    args = parser.parse_args(argv)
    instance = Instance.load(args.instance)
    budget = RoutingBudget(args.seconds, args.passes, args.calls, args.expansions,
                           args.call_expansions, args.call_seconds)
    start = perf_counter()
    with RoutingEnv(method=args.method) as env:
        env.reset(instance, args.seed, budget)
        report = drive_episode(env)
        solution = env.best_solution()
        report["output"] = None
        if solution is not None:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            solution.save(str(path))
            # Verify the actual serialized artifact with the unchanged checker.
            checked = check(instance, Submission.load(str(path)))
            report["checker"] = checked.to_dict()
            report["legal"] = checked.legal
            report["total_delay"] = checked.total_delay
            report["status"] = "success" if checked.legal else "failure"
            report["output"] = str(path)
        report["wall_runtime_s"] = perf_counter() - start
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
