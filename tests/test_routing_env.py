"""Focused Block 2 checks; no benchmark/reference fixtures or upstream sweeps."""
import copy
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from threading import Thread
from time import perf_counter, sleep
import unittest
from unittest.mock import patch

from m3d.checker import check
from m3d._routing_worker import BoundedWireEngine
from m3d.model import Cell, Instance, Net, Pin, Submission
from m3d.routing_env import RoutingBudget, RoutingEnv
from m3d.routing_runner import bbox_order, drive_episode
from m3d.wire_engine import WireEngine, WireResult


def fixture(crossing=False):
    points = ([(0, 2, 0), (4, 2, 0), (2, 0, 0), (2, 4, 0)] if crossing
              else [(0, 0, 0), (4, 0, 0), (0, 4, 0), (4, 4, 0)])
    pins = [Pin(i, i, 0, *v) for i, v in enumerate(points)]
    return Instance("environment-fixture", 5, 5, 2, [1, 1], 1,
                    [Cell(p.id, 0, p.x, p.y, 1, 1) for p in pins],
                    pins, [Net(42, 0, [1]), Net(9001, 2, [3])])


def failure(nid):
    return WireResult(nid, "budget_exhausted", (), None, None, None,
                      "best", 1, 0.0, True, (), ())


def stalled_worker(connection, instance):
    connection.send((True, None))
    connection.recv()
    sleep(10)


class RoutingEnvTests(unittest.TestCase):
    def env(self, inst=None, **limits):
        env = RoutingEnv()
        self.addCleanup(env.close)
        env.reset(inst or fixture(), seed=13, budget=RoutingBudget(**limits))
        return env

    def test_empty_reset_arbitrary_ids_and_invalid_action_is_atomic(self):
        env = self.env()
        self.assertEqual(env.eligible_net_ids(), (42, 9001))
        state = env.observation()
        self.assertEqual(state["routes"], {})
        self.assertEqual(state["vertex_owners"], {})
        self.assertIsNone(env.best_solution())
        for action in (0, -1, True, 42.0, "42", []):
            with self.assertRaises(ValueError):
                env.step(action)
        after = env.observation()
        for key in state.keys() - {"budget"}:
            self.assertEqual(after[key], state[key], key)
        self.assertEqual(after["budget"]["engine_calls"], 0)
        env.step(42)
        with self.assertRaises(ValueError):
            env.step(42)

    def test_vertex_overlap_is_priced_then_failed_replacement_rolls_back(self):
        inst = fixture(crossing=True)
        env = self.env(inst)
        # Force genuine engine-built shortest trees to create a vertex-only
        # crossing; subsequent failure injects the hard-to-trigger error path.
        engine = WireEngine(inst)
        with patch.object(env._worker, "route_net", side_effect=lambda nid, **kw:
                          engine.route_net(nid, method="shortest_path")):
            env.step(42)
            env.step(9001)
        before = env.observation()
        center = (0 * 5 + 2) * 5 + 2
        self.assertEqual(before["conflict_vertices"], {center: frozenset({42, 9001})})
        self.assertEqual(before["conflict_edges"], {})
        self.assertEqual(before["history_vertex"][center], 0.5)
        self.assertEqual(before["present_factor"], 0.85)
        self.assertEqual(env.eligible_net_ids(), (42, 9001))
        self.assertIsNone(env.best_solution())

        def fail(nid, **kwargs):
            during = env.observation()
            self.assertNotIn(nid, during["routes"])
            self.assertEqual(during["vertex_owners"][center], frozenset({9001}))
            self.assertNotIn(center, kwargs["blocked_vertices"])
            # Half each endpoint's vertex pressure; selected net was removed.
            self.assertAlmostEqual(kwargs["edge_prices"][(11, 12)], 0.675)
            return failure(nid)

        with patch.object(env._worker, "route_net", side_effect=fail):
            result = env.step(42)
        self.assertTrue(result.diagnostics["rolled_back"])
        after = env.observation()
        for key in ("routes", "vertex_owners", "edge_owners", "delays",
                    "route_search_costs", "conflict_vertices", "conflict_edges"):
            self.assertEqual(after[key], before[key], key)
        self.assertEqual(after["budget"]["engine_calls"], 3)

    def test_shared_branch_counts_once_and_unrouted_foreign_pins_blocked(self):
        inst = fixture()
        inst.pins[1].x, inst.pins[1].y = 4, 2
        inst.pins[2].x, inst.pins[2].y = 2, 0
        inst.pins.append(Pin(4, 4, 0, 4, 1, 0))
        inst.nets[0].sinks.append(4)
        env = self.env(inst)
        env.step(42)
        state = env.observation()
        foreign = {env._grid.vid(inst.pins[i].vertex()) for i in (2, 3)}
        self.assertFalse(foreign & state["vertex_owners"].keys())
        self.assertTrue(all(owners == frozenset({42})
                            for owners in state["edge_owners"].values()))
        self.assertEqual(len(state["edge_owners"]), len(state["routes"][42]))
        self.assertEqual(len(state["vertex_owners"]), len(state["routes"][42]) + 1)

    def test_complete_episode_and_reset_snapshot_isolation(self):
        inst = fixture(crossing=True)
        env = self.env(inst)
        result = drive_episode(env)
        self.assertEqual(result["status"], "success")
        solution = env.best_solution()
        report = check(inst, Submission.from_dict(solution.to_dict()))
        self.assertTrue(report.legal, report.to_dict())
        self.assertEqual(report.total_delay, result["total_delay"])
        saved = copy.deepcopy(solution)
        solution.routes[0].edges.clear()
        self.assertEqual(env.best_solution(), saved)
        env.close()
        self.assertEqual(env.best_solution(), saved)
        with self.assertRaises(ValueError):
            env.step(42)
        env.reset(fixture(), seed=3, budget=RoutingBudget(max_calls=1))
        reset = env.observation()
        for key in ("routes", "history_vertex", "history_edge", "vertex_owners", "edge_owners"):
            self.assertEqual(reset[key], {})
        self.assertIsNone(env.best_solution())
        terminal = env.step(9001)
        self.assertTrue(terminal.done)
        self.assertEqual(terminal.observation["missing_net_ids"], (42,))
        self.assertEqual(terminal.observation["termination_reason"], "call_budget")

    def test_missing_retries_expansion_budget_and_timeout(self):
        env = self.env(max_passes=2)
        with patch.object(env._worker, "route_net", side_effect=lambda nid, **kw: failure(nid)):
            env.step(42)
            env.step(9001)
            self.assertEqual(env.eligible_net_ids(), (42, 9001))
            env.step(42)
            last = env.step(9001)
        self.assertTrue(last.done)
        self.assertEqual(last.observation["termination_reason"], "pass_budget")
        self.assertIsNone(env.best_solution())
        env.reset(fixture(), 0, RoutingBudget(max_expansions=1))
        last = env.step(42)
        self.assertTrue(last.done)
        self.assertEqual(last.observation["termination_reason"], "expansion_budget")
        self.assertEqual(last.observation["routes"], {})

        env.reset(fixture(), 0, RoutingBudget())
        with patch.object(env._worker, "route_net", side_effect=TimeoutError("native timeout")):
            result = env.step(42)
        self.assertEqual(result.diagnostics["engine_status"], "timeout")
        self.assertEqual(result.observation["routes"], {})

    def test_budget_validation_and_external_bbox_ties(self):
        for kw in ({"max_seconds": float("inf")}, {"max_calls": -1},
                   {"max_passes": True}, {"max_call_seconds": 0}):
            with self.assertRaises(ValueError):
                RoutingBudget(**kw)
        inst = fixture()
        inst.nets.reverse()
        self.assertEqual(bbox_order(inst), (42, 9001))
        env = self.env(inst, max_calls=0)
        report = drive_episode(env)
        self.assertEqual(report["status"], "failure")
        self.assertEqual(report["missing_net_ids"], [42, 9001])
        self.assertIsNone(report["total_delay"])

    def test_best_survives_later_failed_repair_and_budget_termination(self):
        inst = fixture(crossing=True)
        env = self.env(inst, max_calls=4)
        engine = WireEngine(inst)
        with patch.object(env._worker, "route_net", side_effect=lambda nid, **kw:
                          engine.route_net(nid, method="shortest_path")):
            env.step(42)
            env.step(9001)
        occupied = set(v for e in env.observation()["routes"][9001] for v in e)
        replacement = engine.route_net(42, blocked_vertices=occupied)
        with patch.object(env._worker, "route_net", return_value=replacement):
            repaired = env.step(42)
        self.assertFalse(repaired.done)
        saved = env.best_solution()
        self.assertTrue(check(inst, saved).legal)
        with patch.object(env._worker, "route_net", return_value=failure(9001)):
            last = env.step(9001)
        self.assertTrue(last.done)
        self.assertTrue(last.diagnostics["rolled_back"])
        self.assertEqual(env.best_solution(), saved)
        self.assertTrue(check(inst, env.best_solution()).legal)

    def test_wall_timeout_kills_worker_and_next_call_can_recover(self):
        worker = BoundedWireEngine(fixture())
        self.addCleanup(worker.close)
        start = perf_counter()
        with patch("m3d._routing_worker._serve", stalled_worker):
            with self.assertRaises(TimeoutError):
                worker.route_net(42, timeout_s=0.2, max_expansions=1000)
        self.assertLess(perf_counter() - start, 2.0)
        self.assertIsNone(worker._process)
        result = worker.route_net(42, timeout_s=2, max_expansions=1000)
        self.assertEqual(result.status, "success")

    def test_observation_notices_time_expiry_and_reset_clears_congestion(self):
        inst = fixture(crossing=True)
        env = self.env(inst)
        engine = WireEngine(inst)
        with patch.object(env._worker, "route_net", side_effect=lambda nid, **kw:
                          engine.route_net(nid, method="shortest_path")):
            env.step(42)
            env.step(9001)
        self.assertTrue(env.observation()["history_vertex"])
        env.reset(fixture(), 0, RoutingBudget(max_seconds=0.05))
        self.assertEqual(env.observation()["history_vertex"], {})
        sleep(0.06)
        expired = env.observation()
        self.assertTrue(expired["done"])
        self.assertEqual(expired["termination_reason"], "time_budget")
        self.assertEqual(expired["eligible_net_ids"], ())

    def test_invalid_reset_preserves_whole_previous_episode(self):
        env = self.env()
        drive_episode(env)
        before = env.observation()
        saved = env.best_solution()
        broken = fixture()
        broken.name = "invalid-new-instance"
        broken.nets.append(copy.deepcopy(broken.nets[0]))
        with self.assertRaises(ValueError):
            env.reset(broken)
        self.assertEqual(env.observation(), before)
        self.assertEqual(env.best_solution(), saved)
        broken = fixture()
        broken.nets[0].sinks = [999]
        with self.assertRaises(ValueError):
            env.reset(broken)
        self.assertEqual(env.observation(), before)

    def test_call_deadline_covers_incomplete_pipe_frame(self):
        # A readable header is not a complete message. This catches a blocking
        # recv after poll incorrectly being treated as a wall-clock timeout.
        parent, child = mp.Pipe()
        worker = BoundedWireEngine(fixture())
        worker._connection = parent
        self.addCleanup(worker.close)
        payload = pickle.dumps((True, None))
        os.write(child.fileno(), struct.pack("!i", len(payload)))

        def send_later():
            sleep(0.3)
            try:
                os.write(child.fileno(), payload)
            except OSError:
                pass
            child.close()

        thread = Thread(target=send_later, daemon=True)
        thread.start()
        start = perf_counter()
        try:
            with self.assertRaises(TimeoutError):
                worker._receive(start + 0.03)
            self.assertLess(perf_counter() - start, 0.2)
        finally:
            thread.join(timeout=1)

    def test_best_is_separate_when_later_route_conflicts_at_budget_end(self):
        inst = fixture(crossing=True)
        env = self.env(inst, max_calls=4)
        engine = WireEngine(inst)
        with patch.object(env._worker, "route_net", side_effect=lambda nid, **kw:
                          engine.route_net(nid, method="shortest_path")):
            env.step(42)
            env.step(9001)
        for nid in (42, 9001):
            own = {inst.pin_vertex()[p] for p in next(n for n in inst.nets if n.id == nid).pins()}
            blocked = {(x, y, 0) for x in range(5) for y in range(5)} - own
            upper_route = engine.route_net(nid, blocked_vertices=blocked)
            with patch.object(env._worker, "route_net", return_value=upper_route):
                result = env.step(nid)
            if nid == 42:
                saved = env.best_solution()
                self.assertTrue(check(inst, saved).legal)
        self.assertTrue(result.done)
        self.assertEqual(result.observation["termination_reason"], "call_budget")
        self.assertTrue(result.observation["conflict_vertices"])
        self.assertEqual(env.best_solution(), saved)
        self.assertTrue(check(inst, env.best_solution()).legal)

    def test_cli_exports_checked_solution_and_failure_writes_no_solution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inst = fixture()
            inst.save(str(root / "input.json"))
            base = [sys.executable, "-m", "m3d.routing_runner", str(root / "input.json"),
                    "--seconds", "5", "--report", str(root / "report.json")]
            success = subprocess.run(base + ["--out", str(root / "legal.sol.json")],
                                     capture_output=True, text=True, timeout=10)
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertTrue(check(inst, Submission.load(str(root / "legal.sol.json"))).legal)
            failed = subprocess.run(base + ["--out", str(root / "failed.sol.json"), "--calls", "0"],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(failed.returncode, 2, failed.stderr)
            self.assertFalse((root / "failed.sol.json").exists())
            import json
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["status"], "failure")
            self.assertEqual(report["missing_net_ids"], [42, 9001])
            self.assertIsNone(report["output"])


if __name__ == "__main__":
    unittest.main()
