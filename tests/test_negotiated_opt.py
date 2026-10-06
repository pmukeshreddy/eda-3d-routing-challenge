"""Behavioral checks for transactional group optimization and its CLI."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from m3d.checker import check
from m3d.cli import main
from m3d.model import Instance, Net, NetRoute, Pin, Submission
from m3d.negotiated import route_negotiated
from m3d import negotiated_opt as opt


def path(*points):
    """Expand axis-aligned waypoints; expected delays are hand calculated."""
    edges = []
    for start, end in zip(points, points[1:]):
        current = start
        while current != end:
            following = list(current)
            axis = next(i for i in range(3) if current[i] != end[i])
            following[axis] += 1 if end[axis] > current[axis] else -1
            following = tuple(following)
            edges.append((current, following))
            current = following
    return edges


def fixture():
    points = [(1, 2, 0), (5, 2, 0), (3, 1, 0), (3, 3, 0)]
    inst = Instance("group", 7, 7, 1, [1], 1, [],
                    [Pin(i, i, 0, *p) for i, p in enumerate(points)],
                    [Net(0, 0, [1]), Net(1, 2, [3])])
    original = Submission(inst.name, [
        NetRoute(0, path(points[0], (1, 6, 0), (5, 6, 0), points[1])),
        NetRoute(1, path(points[2], points[3])),
    ])  # total 12 + 2 = 14
    improved = Submission(inst.name, [
        NetRoute(0, path(points[0], points[1])),
        NetRoute(1, path(points[2], (0, 1, 0), (0, 3, 0), points[3])),
    ])  # total 4 + 8 = 12
    return inst, original, improved


class TestNegotiatedOpt(unittest.TestCase):
    def test_accepts_improvement_and_keeps_best_after_worse_legal_move(self):
        inst, original, improved = fixture()
        self.assertEqual(check(inst, original).total_delay, 14)
        self.assertEqual(check(inst, improved).total_delay, 12)
        optimizer = opt.DelayOptimizer(inst, original, seed=0)
        self.assertTrue(optimizer.consider(improved, temperature=0))
        self.assertTrue(optimizer.consider(original, temperature=1000))
        self.assertEqual(optimizer.current.to_dict(), original.to_dict())
        self.assertEqual(optimizer.best.to_dict(), improved.to_dict())
        self.assertEqual(optimizer.best_delay, 12)
        returned, stats = optimizer.run(time_budget=0)
        self.assertEqual(returned.to_dict(), improved.to_dict())
        self.assertEqual(stats.optimized_delay, 12)

    def test_illegal_candidate_never_changes_current_or_best(self):
        inst, original, improved = fixture()
        invalid = Submission(inst.name, [improved.routes[0], original.routes[1]])
        self.assertFalse(check(inst, invalid).legal)
        optimizer = opt.DelayOptimizer(inst, original)
        before = original.to_dict()
        self.assertFalse(optimizer.consider(invalid, temperature=1000))
        self.assertEqual(optimizer.current.to_dict(), before)
        self.assertEqual(optimizer.best.to_dict(), before)

    def test_zero_temperature_rejects_worse_routes_exactly(self):
        inst, original, improved = fixture()
        optimizer = opt.DelayOptimizer(inst, improved)
        self.assertFalse(optimizer.consider(original, temperature=0))
        self.assertEqual(optimizer.current.to_dict(), improved.to_dict())

    def test_depth_two_blockers_are_included(self):
        # Use actual legal routes: target's free path meets net 1; net 1's
        # free path meets net 2 at (3,1), away from other-net pins.
        inst = Instance("depth", 7, 7, 1, [1], 1, [],
                        [Pin(i, i, 0, *p) for i, p in enumerate([
                            (0, 3, 0), (6, 3, 0), (2, 1, 0), (4, 1, 0),
                            (3, 0, 0), (3, 2, 0)])],
                        [Net(0, 0, [1]), Net(1, 2, [3]), Net(2, 4, [5])])
        sub = Submission(inst.name, [
            NetRoute(0, path((0,3,0), (0,6,0), (6,6,0), (6,3,0))),
            NetRoute(1, path((2,1,0), (2,4,0), (4,4,0), (4,1,0))),
            NetRoute(2, path((3,0,0), (3,2,0))),
        ])
        self.assertTrue(check(inst, sub).legal)
        optimizer = opt.DelayOptimizer(inst, sub)
        self.assertEqual(optimizer.build_group(0, float("inf")), {0, 1, 2})

    def test_group_solver_keeps_outside_routes_as_hard_obstacles(self):
        inst, original, _ = fixture()
        optimizer = opt.DelayOptimizer(inst, original)
        candidate = optimizer.reroute_group({0}, [0], float("inf"), 0)
        self.assertIsNotNone(candidate)
        self.assertTrue(check(inst, candidate).legal)
        self.assertEqual(candidate.routes[1].edges, original.routes[1].edges)

    def test_expired_group_attempt_preserves_exact_routes(self):
        inst, original, _ = fixture()
        optimizer = opt.DelayOptimizer(inst, original)
        before = original.to_dict()
        with self.assertRaises(opt.BudgetExpired):
            optimizer.reroute_group({0, 1}, [0, 1], -1, 0)
        self.assertEqual(optimizer.current.to_dict(), before)
        self.assertEqual(optimizer.best.to_dict(), before)

    def test_fixed_seed_replays_group_orders_and_results(self):
        inst, original, _ = fixture()
        a, b = opt.DelayOptimizer(inst, original), opt.DelayOptimizer(inst, original)
        orders_a = a.orders({0, 1}, 0)
        orders_b = b.orders({0, 1}, 0)
        self.assertEqual(orders_a, orders_b)
        left = a.reroute_group({0, 1}, orders_a[0], float("inf"), 17)
        right = b.reroute_group({0, 1}, orders_b[0], float("inf"), 17)
        self.assertIsNotNone(left)
        self.assertEqual(left.to_dict(), right.to_dict())

    def test_equal_search_steps_ignore_machine_speed_for_annealing(self):
        inst, worse, initial = fixture()
        outcomes = []
        for elapsed in [0.0, 0.9]:
            optimizer = opt.DelayOptimizer(inst, initial, seed=0)
            clock = {"now": 0.0, "finished": False}
            original_consider = optimizer.consider

            def trial(*args):
                clock["now"] = elapsed
                return worse

            def consider(candidate, temperature):
                accepted = original_consider(candidate, temperature)
                clock["finished"] = True
                return accepted

            with mock.patch.object(optimizer, "build_group", return_value={0, 1}), \
                    mock.patch.object(optimizer, "orders", return_value=[[0, 1]]), \
                    mock.patch.object(optimizer, "reroute_group", side_effect=trial), \
                    mock.patch.object(optimizer, "consider", side_effect=consider), \
                    mock.patch.object(opt.time, "monotonic", side_effect=lambda:
                                      2.0 if clock["finished"] else clock["now"]):
                optimizer.run(time_budget=1)
            outcomes.append((optimizer.stats.accepted_worse, optimizer.current.to_dict()))
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0][0], 1)

    def test_timeout_after_routing_one_group_net_restores_exact_snapshot(self):
        inst, initial, _ = fixture()
        optimizer = opt.DelayOptimizer(inst, initial)
        route_net = opt._GroupNegotiated._route_net
        calls = []

        def interrupted(solver, nid, pres_fac):
            if calls:
                raise opt.BudgetExpired
            result = route_net(solver, nid, pres_fac)
            self.assertIsNotNone(result)
            calls.append(nid)
            return result

        with mock.patch.object(opt._GroupNegotiated, "_route_net", interrupted):
            with self.assertRaises(opt.BudgetExpired):
                optimizer.reroute_group({0, 1}, [0, 1], float("inf"), 0)
        self.assertEqual(optimizer.current.to_dict(), initial.to_dict())
        self.assertEqual(optimizer.best.to_dict(), initial.to_dict())

    def test_completed_better_trial_survives_later_order_timeout(self):
        inst, initial, improved = fixture()
        optimizer = opt.DelayOptimizer(inst, initial)
        clock = {"finished": False}

        def trial(group, order, deadline, seed):
            if order == [0, 1]:
                return improved
            clock["finished"] = True
            raise opt.BudgetExpired

        with mock.patch.object(optimizer, "build_group", return_value={0, 1}), \
                mock.patch.object(optimizer, "orders", return_value=[[0, 1], [1, 0]]), \
                mock.patch.object(optimizer, "reroute_group", side_effect=trial), \
                mock.patch.object(opt.time, "monotonic", side_effect=lambda:
                                  2.0 if clock["finished"] else 0.0):
            result, stats = optimizer.run(time_budget=1)
        self.assertEqual(result.to_dict(), improved.to_dict())
        self.assertEqual(stats.timed_out_orders, 1)

    def test_zero_budget_returns_original_exactly(self):
        inst, original, _ = fixture()
        sub, stats = opt.optimize_delay(inst, original, time_budget=0)
        self.assertEqual(sub.to_dict(), original.to_dict())
        self.assertEqual(stats.accepted_improvements, 0)

    def test_invalid_budgets_are_rejected(self):
        inst, original, _ = fixture()
        for budget in [-1, float("nan"), float("inf")]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                opt.optimize_delay(inst, original, time_budget=budget)

    def test_wrapper_starts_from_real_negotiated_solution(self):
        inst, _, _ = fixture()
        baseline, _ = route_negotiated(inst)
        candidate, stats = opt.route_negotiated_opt(inst, time_budget=0)
        self.assertEqual(candidate.to_dict(), baseline.to_dict())
        self.assertTrue(stats.success)

    def test_initial_router_failure_is_reported_without_a_solution(self):
        inst, _, _ = fixture()
        with mock.patch.object(opt, "route_negotiated", return_value=(None, None)):
            candidate, stats = opt.route_negotiated_opt(inst, time_budget=1)
        self.assertIsNone(candidate)
        self.assertFalse(stats.success)
        self.assertEqual(stats.stop_reason, "initial_routing_failed")

    def test_short_optimization_run_never_loses_original_legal_solution(self):
        inst, original, _ = fixture()
        candidate, stats = opt.optimize_delay(inst, original, time_budget=0.02)
        result = check(inst, candidate)
        self.assertTrue(result.legal, result.reasons)
        self.assertLessEqual(result.total_delay, 14)
        self.assertEqual(result.total_delay, stats.optimized_delay)

    def test_cli_rejects_nonfinite_and_negative_time_budgets(self):
        for budget in ["-1", "nan", "inf"]:
            with self.subTest(budget=budget), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["run-suite", "--router", "negotiated_opt",
                          "--out-dir", "unused", "--time-budget", budget])
                self.assertEqual(raised.exception.code, 2)

    def test_cli_writes_checked_solution_and_optimizer_stats(self):
        inst, _, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            inst.save(directory / "case.json")
            (directory / "suite.json").write_text(json.dumps({"cases": [{
                "name": inst.name, "instance_file": "case.json"}]}))
            output = directory / "out"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["run-suite", "--suite", str(directory),
                             "--out-dir", str(output), "--router", "negotiated_opt",
                             "--time-budget", "0"])
            self.assertEqual(code, 0)
            self.assertTrue(check(inst, Submission.load(output / "group.sol.json")).legal)
            stats = json.loads((output / "optimizer_stats.json").read_text())[inst.name]
            self.assertEqual(stats["time_budget_s"], 0)
            self.assertIn(inst.name, json.loads((output / "runtime.json").read_text()))


if __name__ == "__main__":
    unittest.main()
