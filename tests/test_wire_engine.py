"""Block 1 only: tiny independent oracles and official-checker fixtures.

Run: python3 -m unittest tests.test_wire_engine -v
The separate demo is the sole official-case/reference integration run.
"""
import copy
import itertools
import math
import unittest

from m3d.checker import check
from m3d.model import Cell, Instance, Net, NetRoute, Pin, Submission, canon_edge
from m3d.wire_engine import WireEngine


def fixture(width, height, terminals, layers=1, delays=None, via=3):
    pins = [Pin(i, i, int(z == layers - 1 and layers > 1), x, y, z)
            for i, (x, y, z) in enumerate(terminals)]
    return Instance("fixture", width, height, layers, delays or [1] * layers,
                    via, [Cell(p.id, p.die, p.x, p.y, 1, 1) for p in pins],
                    pins, [Net(0, 0, list(range(1, len(pins))))])


def oracle(inst, source, blocked=(), blocked_edges=(), prices=None, weight=1):
    """Bellman-Ford over explicit coordinate pairs; no engine/grid helpers."""
    vertices = [v for v in itertools.product(range(inst.width),
                range(inst.height), range(inst.layers)) if v not in blocked]
    distances = {v: math.inf for v in vertices}
    if source not in distances:
        return distances
    distances[source] = 0
    edges = []
    for u in vertices:
        for v in vertices:
            if sum(abs(a - b) for a, b in zip(u, v)) != 1:
                continue
            edge = canon_edge(u, v)
            if edge in blocked_edges:
                continue
            delay = inst.via_delay if u[2] != v[2] else inst.layer_delay[u[2]]
            edges.append((u, v, weight * delay + (prices or {}).get(edge, 0)))
    for _ in range(len(vertices) - 1):
        changed = False
        for u, v, cost in edges:
            if distances[u] + cost < distances[v]:
                distances[v] = distances[u] + cost
                changed = True
        if not changed:
            break
    return distances


class WireEngineTests(unittest.TestCase):
    def legal(self, inst, result, other_routes=()):
        self.assertEqual(result.status, "success")
        self.assertEqual(list(result.edges), sorted(set(result.edges)))
        submission = Submission(inst.name, [result.to_net_route(), *other_routes])
        # Exercise the repository's serialization contract as well.
        report = check(inst, Submission.from_dict(submission.to_dict()))
        self.assertTrue(report.legal, report.to_dict())
        self.assertEqual(next(n.delay for n in report.nets if n.net == 0),
                         result.total_delay)
        self.assertEqual(result.objective, result.total_delay + result.resource_cost)
        self.assertGreaterEqual(result.elapsed_s, 0)
        return report

    def test_two_pin_minimum_and_integer_delay(self):
        inst = fixture(5, 1, [(0, 0, 0), (4, 0, 0)], 3, [6, 1, 6], 2)
        for method in ("shortest_path", "cost_distance", "best"):
            result = WireEngine(inst).route_net(0, method=method)
            self.legal(inst, result)
            self.assertEqual(result.total_delay, 8)  # two vias + four cheap steps
            self.assertEqual(result.total_delay, oracle(inst, (0, 0, 0))[(4, 0, 0)])
        huge = fixture(3, 1, [(0, 0, 0), (2, 0, 0)], delays=[2**53 + 1])
        result = WireEngine(huge).route_net(0)
        self.legal(huge, result)
        self.assertEqual(result.total_delay, 2 * (2**53 + 1))

    def test_multisink_driver_distances_and_branching(self):
        inst = fixture(4, 3, [(0, 1, 0), (3, 0, 0), (3, 2, 0)])
        engine = WireEngine(inst)
        result = engine.route_net(0, blocked_vertices=[(0, 0, 0), (0, 2, 0)])
        self.legal(inst, result)
        self.assertEqual(result.total_delay, 8)
        distances = oracle(inst, (0, 1, 0), [(0, 0, 0), (0, 2, 0)])
        self.assertEqual(result.total_delay, sum(distances[p.vertex()] for p in inst.pins[1:]))
        degree = {}
        for u, v in result.edges:
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
        self.assertTrue(any(n >= 3 for n in degree.values()))
        self.assertEqual(len(result.edges), len(degree) - 1)

    def test_blockers_foreign_pins_and_unreachable(self):
        inst = fixture(5, 3, [(0, 1, 0), (4, 1, 0), (2, 1, 0), (2, 0, 0)])
        inst.nets = [Net(0, 0, [1]), Net(1, 2, [3])]
        foreign = NetRoute(1, [canon_edge((2, 1, 0), (2, 0, 0))])
        blocked = {(1, 1, 0)}
        blocked_edges = {canon_edge((3, 2, 0), (4, 2, 0))}
        snapshot = copy.deepcopy((inst, blocked, blocked_edges))
        engine = WireEngine(inst)
        for method in ("shortest_path", "cost_distance", "best"):
            result = engine.route_net(0, blocked_vertices=blocked,
                                      blocked_edges=blocked_edges, method=method)
            self.legal(inst, result, [foreign])
            self.assertFalse(set(result.edges) & blocked_edges)
            self.assertFalse({v for e in result.edges for v in e} & blocked)
            distances = oracle(inst, (0, 1, 0), blocked | {(2, 1, 0), (2, 0, 0)}, blocked_edges)
            self.assertEqual(result.total_delay, distances[(4, 1, 0)])
            failed = engine.route_net(0, blocked_vertices=[(1, y, 0) for y in range(3)], method=method)
            self.assertEqual(failed.status, "unreachable")
            self.assertFalse(failed.edges)
            self.assertIsNone(failed.total_delay)
        self.assertEqual((inst, blocked, blocked_edges), snapshot)

    def test_prices_change_paths_but_not_reported_delay(self):
        inst = fixture(3, 2, [(0, 0, 0), (2, 0, 0)])
        engine = WireEngine(inst)
        direct = engine.route_net(0)
        prices = {canon_edge((0, 0, 0), (1, 0, 0)): 10,
                  canon_edge((0, 0, 0), (0, 1, 0)): 0.5}
        for method in ("shortest_path", "cost_distance", "best"):
            result = engine.route_net(0, edge_prices=prices, method=method)
            self.legal(inst, result)
            self.assertEqual((direct.total_delay, result.total_delay), (2, 4))
            self.assertEqual(result.resource_cost, 0.5)
            self.assertEqual(result.objective, 4.5)
            self.assertEqual(result.objective, oracle(inst, (0, 0, 0), prices=prices)[(2, 0, 0)])
        self.assertEqual(len(result.candidates), 2)
        self.assertTrue(all(c["status"] == "success" for c in result.candidates))

    def test_component_merge_distances_and_unique_resource_cost(self):
        inst = fixture(5, 3, [(0, 1, 0), (3, 0, 0), (4, 0, 0), (4, 2, 0)])
        # Uniform price makes merging nearby sinks preferable to separate root paths.
        prices = {canon_edge(u, v): 3.0 for u in itertools.product(range(5), range(3), [0])
                  for v in itertools.product(range(5), range(3), [0])
                  if sum(abs(a-b) for a, b in zip(u, v)) == 1}
        engine = WireEngine(inst)
        result = engine.route_net(0, edge_prices=prices, method="cost_distance", seed=7)
        self.legal(inst, result)
        self.assertEqual(result.resource_cost, 3 * len(result.edges))
        root = engine.vertex_id(inst.pins[0].vertex())
        active = {engine.vertex_id(p.vertex()): 1 for p in inst.pins[1:]}
        self.assertEqual(len(result.merges), 3)
        for merge in result.merges:
            best = math.inf
            for u, weight in active.items():
                distances = oracle(inst, engine.vertex(u), prices=prices, weight=weight)
                best = min(best, *(distances[engine.vertex(v)] for v in [*active, root] if v != u))
            self.assertEqual(merge["distance"], best)
            u, v = merge["u"], merge["v"]
            self.assertEqual(merge["weight_u"], active[u])
            if v == root:
                self.assertEqual(merge["representative"], root)
                del active[u]
            else:
                self.assertEqual(merge["weight_v"], active[v])
                weight = active.pop(u) + active.pop(v)
                self.assertIn(merge["representative"], (u, v))
                active[merge["representative"]] = weight
        self.assertFalse(active)
        shortest = engine.route_net(0, edge_prices=prices, method="shortest_path")
        best = engine.route_net(0, edge_prices=prices, method="best", seed=7)
        self.legal(inst, best)
        self.assertEqual(best.objective, min(shortest.objective, result.objective))

        # A hand-checkable sharing benefit: the far-side vertical connection
        # and middle-row trunk use five edges while keeping both delays at four.
        sharing = fixture(4, 3, [(0, 1, 0), (3, 0, 0), (3, 2, 0)])
        sharing_prices = {edge: cost for edge, cost in prices.items()
                          if all(v[0] < 4 for v in edge)}
        shared_engine = WireEngine(sharing)
        spt = shared_engine.route_net(0, method="shortest_path", edge_prices=sharing_prices)
        cd = shared_engine.route_net(0, method="cost_distance", edge_prices=sharing_prices, seed=7)
        chosen = shared_engine.route_net(0, edge_prices=sharing_prices, seed=7)
        for candidate in (spt, cd, chosen):
            self.legal(sharing, candidate)
        self.assertEqual((spt.total_delay, spt.resource_cost, spt.objective), (8, 24, 32))
        self.assertEqual((cd.total_delay, cd.resource_cost, cd.objective), (8, 15, 23))
        self.assertEqual((chosen.method, chosen.objective), ("cost_distance", 23))

    def test_repeatability_and_shared_budget(self):
        inst = fixture(5, 3, [(0, 1, 0), (4, 0, 0), (4, 2, 0)])
        engine = WireEngine(inst)
        prices = {canon_edge((0, 1, 0), (1, 1, 0)): 3}
        for method in ("shortest_path", "cost_distance", "best"):
            a = engine.route_net(0, edge_prices=prices, method=method, seed=42, max_expansions=10000)
            b = engine.route_net(0, edge_prices=prices, method=method, seed=42, max_expansions=10000)
            self.assertEqual((a.edges, a.total_delay, a.resource_cost, a.expansions, a.merges),
                             (b.edges, b.total_delay, b.resource_cost, b.expansions, b.merges))
            failed = engine.route_net(0, edge_prices=prices, method=method, max_expansions=1)
            self.assertEqual(failed.status, "budget_exhausted")
            self.assertFalse(failed.edges)
            self.assertIsNone(failed.objective)
            self.assertLessEqual(failed.expansions, 1)
        shortest = engine.route_net(0, edge_prices=prices, method="shortest_path")
        limited = engine.route_net(0, edge_prices=prices, max_expansions=2 * shortest.expansions)
        self.legal(inst, limited)
        self.assertLessEqual(limited.expansions, 2 * shortest.expansions)
        self.assertEqual(limited.expansions, sum(c["expansions"] for c in limited.candidates))
        zero = engine.route_net(0, max_expansions=0)
        self.assertEqual(zero.status, "budget_exhausted")
        fast = engine.route_net(0)
        self.assertEqual(fast.method, "shortest_path")
        self.assertEqual(len(fast.candidates), 1)

    def test_rejects_malformed_inputs(self):
        inst = fixture(3, 2, [(0, 0, 0), (2, 0, 0)])
        engine = WireEngine(inst)
        edge = ((0, 0, 0), (1, 0, 0))
        for value in (-1, math.nan, math.inf, "3", True):
            with self.assertRaises((ValueError, TypeError)):
                engine.route_net(0, edge_prices={edge: value})
        for kwargs in ({"blocked_vertices": [(0.5, 0, 0)]},
                       {"blocked_vertices": [-1]}, {"blocked_vertices": [True]},
                       {"blocked_edges": [((0, 0, 0), (2, 0, 0))]},
                       {"edge_prices": {(edge[1], edge[0]): 1, edge: 2}},
                       {"method": "unknown"}, {"max_expansions": -1}, {"seed": -1}):
            with self.assertRaises((ValueError, TypeError)):
                engine.route_net(0, **kwargs)
        with self.assertRaises(ValueError):
            engine.route_net(99)
        malformed = copy.deepcopy(inst)
        malformed.nets[0].sinks.append(1)
        with self.assertRaises(ValueError):
            WireEngine(malformed)


if __name__ == "__main__":
    unittest.main()
