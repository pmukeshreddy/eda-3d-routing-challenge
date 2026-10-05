"""ONE removed-net rebuild in a reference layout; not a full-case solver."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m3d.checker import check
from m3d.grid import Grid
from m3d.model import Instance, Submission
from m3d.wire_engine import WireEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=str(ROOT / "benchmarks/case_01.json"))
    parser.add_argument("--reference", default=str(ROOT / "benchmarks/reference/case_01.sol.json"))
    parser.add_argument("--net", type=int, default=5)
    parser.add_argument("--method", choices=["best", "shortest_path", "cost_distance"], default="best")
    parser.add_argument("--edge-price", type=float, default=0.25,
                        help="Uniform illustrative resource price; default 0.25 per edge")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-expansions", type=int, default=1_000_000)
    parser.add_argument("--out", help="Optional assembled fixture JSON (contains reference routes)")
    args = parser.parse_args()
    instance = Instance.load(args.case)
    reference = Submission.load(args.reference)
    before = check(instance, reference)
    if not before.legal:
        raise ValueError("integration reference is not a legal complete layout")
    fixed = [r for r in reference.routes if r.net != args.net]
    if len(fixed) != len(reference.routes) - 1:
        raise ValueError("selected net must occur exactly once in the reference")
    blocked_edges = {edge for route in fixed for edge in route.edges}
    blocked_vertices = {v for edge in blocked_edges for v in edge}
    grid = Grid(instance)
    prices = {(u, v): args.edge_price for u in range(grid.wh * grid.l)
              for v, _ in grid.neighbors(u) if u < v}
    engine = WireEngine(instance)
    result = engine.route_net(args.net, blocked_vertices, blocked_edges, prices,
                              method=args.method, seed=args.seed, max_expansions=args.max_expansions)
    output = {
        "label": "REFERENCE-LAYOUT INTEGRATION FIXTURE: only one net rebuilt; other routes are reference data",
        "instance": instance.name, "net": args.net, "status": result.status,
        "method": result.method, "net_delay": result.total_delay,
        "resource_cost": result.resource_cost, "objective": result.objective,
        "uniform_edge_price": args.edge_price,
        "expansions": result.expansions, "elapsed_s": result.elapsed_s,
        "budget_exhausted": result.budget_exhausted,
        "candidates": result.candidates, "reference_total_delay": before.total_delay,
    }
    if result.status != "success":
        print(json.dumps(output, indent=2))
        return 1
    assembled = Submission(instance.name, sorted([*fixed, result.to_net_route()], key=lambda r: r.net))
    after = check(instance, assembled)
    output.update(legal=after.legal, assembled_total_delay=after.total_delay,
                  reference_net_delay=next(n.delay for n in before.nets if n.net == args.net))
    if args.out and after.legal:
        assembled.save(args.out)
        output["assembled_fixture_file"] = args.out
    print(json.dumps(output, indent=2))
    return 0 if after.legal else 1


if __name__ == "__main__":
    raise SystemExit(main())
