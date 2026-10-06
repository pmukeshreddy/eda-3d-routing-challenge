"""Native behavior is checked independently with the unchanged Python checker."""
import json
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from m3d.checker import check
from m3d.model import Instance, Net, NetRoute, Pin, Submission
from m3d.negotiated_opt import optimize_delay
from tests.test_negotiated_opt import fixture, path


@unittest.skipUnless(shutil.which('clang++') or shutil.which('g++'), 'C++17 compiler required')
class TestNativeOptimizer(unittest.TestCase):
    def run_native(self, inst, initial, steps=20, seed=0):
        result, stats = optimize_delay(inst, initial, time_budget=10, seed=seed,
                                       _work_limit=steps)
        self.assertEqual(stats.backend, 'native_cpp17')
        checked = check(inst, result)
        self.assertTrue(checked.legal, checked.reasons)
        self.assertEqual(checked.total_delay, stats.optimized_delay)
        return result, stats

    def test_native_search_improves_constructed_coordinated_case(self):
        points = [(0, 1, 0), (4, 1, 0), (5, 1, 0), (2, 0, 0), (2, 2, 0)]
        inst = Instance('coordinated', 6, 3, 2, [1, 1], 1, [],
                        [Pin(i, i, 0, *p) for i, p in enumerate(points)],
                        [Net(0, 0, [1, 2]), Net(1, 3, [4])])
        initial = Submission(inst.name, [
            NetRoute(0, path(points[0], (1, 1, 0), (1, 1, 1),
                             (3, 1, 1), (3, 1, 0), points[2])),
            NetRoute(1, path(points[3], points[4])),
        ])
        self.assertEqual(check(inst, initial).total_delay, 15)
        _, cleanup = self.run_native(inst, initial, steps=2)
        self.assertEqual(cleanup.optimized_delay, 15)
        result, stats = self.run_native(inst, initial)
        self.assertLessEqual(stats.optimized_delay, 13)
        self.assertGreater(stats.completed_orders, 0)
        self.assertGreater(stats.legal_candidates, 0)

    def test_multisink_uses_true_driver_delay(self):
        points = [(0, 1, 0), (4, 1, 0), (4, 3, 0)]
        inst = Instance('sinks', 6, 5, 1, [1], 1, [],
                        [Pin(i, i, 0, *p) for i, p in enumerate(points)], [Net(5, 0, [1, 2])])
        initial = Submission(inst.name, [NetRoute(5, path(points[0], (0, 4, 0),
                    (5, 4, 0), (5, 1, 0), points[1]) + path((5, 3, 0), points[2]))])
        self.assertTrue(check(inst, initial).legal)
        _, stats = self.run_native(inst, initial, steps=1)
        self.assertEqual(stats.optimized_delay, 10)
        self.assertEqual(stats.single_net_improvements, 1)

    def test_fixed_work_seed_replays_routes_and_counters(self):
        inst, initial, _ = fixture()
        a, sa = self.run_native(inst, initial, steps=15, seed=17)
        b, sb = self.run_native(inst, initial, steps=15, seed=17)
        self.assertEqual(a.to_dict(), b.to_dict())
        for name in ('attempted_groups', 'completed_orders', 'timed_out_orders',
                     'accepted_improvements', 'accepted_equal', 'accepted_worse',
                     'legal_candidates', 'optimized_delay', 'work_steps'):
            self.assertEqual(getattr(sa, name), getattr(sb, name), name)
        self.assertEqual(sa.work_steps, 15)

    def test_short_deadline_keeps_legal_baseline_or_better(self):
        inst, initial, _ = fixture()
        result, stats = optimize_delay(inst, initial, time_budget=1e-9)
        self.assertTrue(check(inst, result).legal)
        self.assertLessEqual(stats.optimized_delay, 14)
        self.assertEqual(stats.work_steps, 0)

    def test_zero_budget_avoids_native_setup_preserves_identity(self):
        from m3d import negotiated_native as native
        inst, initial, _ = fixture()
        with mock.patch.object(native, '_binary', side_effect=AssertionError('compiled')):
            result, stats = optimize_delay(inst, initial, time_budget=0)
        self.assertIs(result, initial)
        self.assertEqual(stats.backend_setup_seconds, 0)

    def run_harness(self, mode, inst, initial, improved=None):
        from m3d import negotiated_native as native
        payload = native._encode_input(inst, initial, 10, 0, 1)
        if improved is not None:
            payload += native._encode_input(inst, improved, 10, 0, 1)
        with tempfile.TemporaryDirectory() as tmp:
            executable = str(Path(tmp) / 'harness')
            compiled = subprocess.run([shutil.which('clang++') or shutil.which('g++'),
                '-std=c++17', '-O2', str(Path(__file__).with_name('native_opt_harness.cpp')),
                '-o', executable], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            process = subprocess.run([executable, mode], input=payload,
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
        return native._decode_output(inst, initial, process.stdout)

    def test_native_group_respects_outside_routes_and_foreign_pins(self):
        inst, initial, _ = fixture()
        candidate, stats, delay = self.run_harness('outside', inst, initial)
        self.assertEqual(candidate.routes[1].edges, initial.routes[1].edges)
        self.assertTrue(check(inst, candidate).legal)
        self.assertEqual(delay, 10)

    def test_legal_candidate_is_polished_before_acceptance(self):
        inst, initial, _ = fixture()
        candidate, _, delay = self.run_harness('polish', inst, initial)
        self.assertTrue(check(inst, candidate).legal)
        self.assertEqual(delay, 10)

    def test_warm_repair_preserves_unaffected_blocker_tree(self):
        points = [(0, 0, 0), (2, 0, 0), (5, 0, 0), (7, 0, 0)]
        inst = Instance('warm_preserves', 8, 3, 1, [1], 1, [],
                        [Pin(i, i, 0, *p) for i, p in enumerate(points)],
                        [Net(0, 0, [1]), Net(1, 2, [3])])
        initial = Submission(inst.name, [
            NetRoute(0, path(points[0], (0, 1, 0), (2, 1, 0), points[1])),
            NetRoute(1, path(points[2], (5, 1, 0), (7, 1, 0), points[3])),
        ])
        candidate, _, delay = self.run_harness('warm_preserves', inst, initial)
        self.assertEqual(delay, 6)
        self.assertEqual(candidate.routes[1].edges, initial.routes[1].edges)

    def test_partial_polish_timeout_keeps_completed_legal_improvements(self):
        points = [(0, 0, 0), (2, 0, 0), (5, 0, 0), (7, 0, 0)]
        inst = Instance('partial_polish', 8, 3, 1, [1], 1, [],
                        [Pin(i, i, 0, *p) for i, p in enumerate(points)],
                        [Net(0, 0, [1]), Net(1, 2, [3])])
        initial = Submission(inst.name, [
            NetRoute(0, path(points[0], (0, 1, 0), (2, 1, 0), points[1])),
            NetRoute(1, path(points[2], (5, 1, 0), (7, 1, 0), points[3])),
        ])
        candidate, _, delay = self.run_harness('partial_polish_timeout', inst, initial)
        self.assertEqual(delay, 6)
        self.assertTrue(check(inst, candidate).legal)
        self.assertEqual(candidate.routes[1].edges, initial.routes[1].edges)

    def test_periodic_global_repair_replays_completed_work(self):
        inst, initial, _ = fixture()
        a, sa, da = self.run_harness('global_order', inst, initial)
        b, sb, db = self.run_harness('global_order', inst, initial)
        self.assertEqual(sa['reroute_orders'], 5)
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual(da, db)
        for field in ('completed_orders', 'bounded_orders', 'legal_candidates',
                      'accepted_improvements', 'accepted_worse', 'expansions', 'work_steps'):
            self.assertEqual(sa[field], sb[field], field)

    def test_branch_constructor_prices_true_root_distance(self):
        points = [(0, 2, 0), (3, 1, 0), (3, 4, 0)]
        inst = Instance('root_distance', 4, 5, 1, [1], 1, [],
                        [Pin(i, i, 0, *p) for i, p in enumerate(points)], [Net(0, 0, [1, 2])])
        initial = Submission(inst.name, [NetRoute(0,
            path(points[0], (0, 1, 0), points[1], points[2]))])
        candidate, _, delay = self.run_harness('root_distance', inst, initial)
        self.assertTrue(check(inst, candidate).legal)
        self.assertEqual(delay, 9)

    def test_branch_constructor_does_not_recharge_selected_trunk_congestion(self):
        points = [(0, 0, 0), (4, 0, 0), (5, 1, 0)]
        inst = Instance('sunk_congestion', 6, 3, 1, [1], 1, [],
                        [Pin(i, i, 0, *p) for i, p in enumerate(points)], [Net(0, 0, [1, 2])])
        initial = Submission(inst.name, [NetRoute(0,
            path(points[0], (0, 2, 0), (5, 2, 0), points[2], (5, 0, 0), points[1]))])
        candidate, _, delay = self.run_harness('sunk_congestion', inst, initial)
        self.assertTrue(check(inst, candidate).legal)
        self.assertEqual(delay, 10)

    def test_native_retains_best_after_uphill_move_and_timeout(self):
        inst, initial, improved = fixture()
        candidate, stats, delay = self.run_harness('retention', inst, initial, improved)
        self.assertEqual(delay, 12)
        self.assertEqual(stats['accepted_worse'], 1)
        self.assertEqual(check(inst, candidate).total_delay, 12)

    def test_completed_better_order_survives_next_order_partial_timeout(self):
        points = [(0, 1, 0), (4, 1, 0), (5, 1, 0), (2, 0, 0), (2, 2, 0)]
        inst = Instance('partial_timeout', 6, 3, 2, [1, 1], 1, [],
                        [Pin(i, i, 0, *p) for i, p in enumerate(points)],
                        [Net(0, 0, [1, 2]), Net(1, 3, [4])])
        initial = Submission(inst.name, [
            NetRoute(0, path(points[0], (1, 1, 0), (1, 1, 1),
                             (3, 1, 1), (3, 1, 0), points[2])),
            NetRoute(1, path(points[3], points[4])),
        ])
        candidate, stats, delay = self.run_harness('partial_order_timeout', inst, initial)
        self.assertEqual(stats['completed_orders'], 1)
        self.assertEqual(stats['timed_out_orders'], 1)
        self.assertEqual(delay, 13)
        self.assertEqual(check(inst, candidate).total_delay, 13)

    def test_invalid_worker_output_is_rejected(self):
        from m3d import negotiated_native as native
        inst, initial, improved = fixture()
        for output in ('not json', '{}', json.dumps({'delay': 0, 'routes': [], 'stats': {}})):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                native._decode_output(inst, initial, output)

    def test_worker_delay_disagreement_is_rejected(self):
        from m3d import negotiated_native as native
        from m3d.grid import Grid
        inst, initial, _ = fixture()
        grid = Grid(inst)
        out = {'delay': 1, 'stats': {}, 'routes': [
            {'net': r.net, 'edges': [[grid.vid(a), grid.vid(b)] for a, b in r.edges]}
            for r in initial.routes]}
        with self.assertRaisesRegex(RuntimeError, 'delay'):
            native._decode_output(inst, initial, json.dumps(out))


if __name__ == '__main__':
    unittest.main()
