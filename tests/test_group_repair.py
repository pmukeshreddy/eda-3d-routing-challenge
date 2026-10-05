"""Five focused checks for coordinated Block 2 destroy-and-repair."""
import copy
import unittest
from unittest.mock import patch

from m3d.checker import check
from m3d.grid import edge_key
from m3d.model import Instance, Net, Pin
from m3d.routing_env import RoutingBudget, RoutingEnv, _Route
from m3d.wire_engine import WireEngine
from tests.test_routing_env import failure


def trap_fixture(outside=False):
    points = [(0, 2, 0), (4, 2, 0), (2, 0, 0), (2, 4, 0), (1, 4, 0), (3, 4, 0)]
    nets = [Net(42, 0, [1]), Net(9001, 2, [3, 4, 5])]
    if outside:
        points += [(0, 4, 1), (4, 4, 1)]
        nets.append(Net(17, 6, [7]))
    return Instance('joint-delay-trap', 5, 5, 2, [1, 1], 1, [],
                    [Pin(k, k, p[2], *p) for k, p in enumerate(points)], nets)


def install_result(env, result):
    edges = frozenset(edge_key(env._grid.vid(a), env._grid.vid(b)) for a, b in result.edges)
    vertices = frozenset(v for e in edges for v in e) | frozenset(env._net_pins[result.net_id])
    env._install(result.net_id, _Route(result.edges, vertices, edges,
                                     result.total_delay, result.resource_cost))


class GroupRepairTests(unittest.TestCase):
    def env(self, instance=None, **limits):
        env = RoutingEnv()
        self.addCleanup(env.close)
        env.reset(instance or trap_fixture(), 13, RoutingBudget(**limits))
        return env

    def initial_trap(self, env):
        engine = WireEngine(env._instance)
        for nid in (17, 42, 9001) if 17 in env._nets else (42, 9001):
            install_result(env, engine.route_net(nid, blocked_vertices=env._owners_v))
        env._consider_snapshot()
        return engine

    def test_two_net_destroy_escapes_single_net_delay_trap(self):
        env = self.env()
        engine = self.initial_trap(env)
        before = env.observation()
        self.assertEqual(before['delays'], {42: 4, 9001: 20})
        # With its blocker frozen, neither individual route can improve.
        for nid, other in ((42, 9001), (9001, 42)):
            alone = engine.route_net(nid, blocked_vertices=env._routes[other].vertices)
            self.assertGreaterEqual(alone.total_delay, before['delays'][nid])
        result = env._repair_group((42, 9001))
        self.assertTrue(result['accepted'])
        after = env.observation()
        self.assertEqual(after['delays'], {42: 6, 9001: 14})
        self.assertEqual(after['best_total_delay'], 20)
        self.assertTrue(check(env._instance, env.best_solution()).legal)
        self.assertEqual(len(result['candidates']), 2)

    def test_remove_all_group_routes_and_rollback_partial_candidates(self):
        env = self.env(trap_fixture(outside=True))
        engine = self.initial_trap(env)
        before = env.observation()
        saved = env.best_solution()
        calls = []

        def route(nid, **kwargs):
            calls.append(nid)
            state = env.observation()
            self.assertEqual(state['routes'][17], before['routes'][17])
            if len(calls) % 2:
                self.assertEqual(set(state['routes']), {17})
                self.assertEqual(set().union(*state['vertex_owners'].values()), {17})
                self.assertEqual(set().union(*state['edge_owners'].values()), {17})
                kwargs.pop('timeout_s')
                return engine.route_net(nid, **kwargs)
            return failure(nid)

        with patch.object(env._worker, 'route_net', side_effect=route):
            result = env._repair_group((42, 9001))
        self.assertTrue(result['rolled_back'])
        self.assertFalse(result['accepted'])
        self.assertEqual(len(calls), 4)
        after = env.observation()
        for key in ('routes', 'vertex_owners', 'edge_owners', 'delays',
                    'route_search_costs', 'conflict_vertices', 'conflict_edges',
                    'history_vertex', 'history_edge', 'present_factor', 'best_total_delay'):
            self.assertEqual(after[key], before[key], key)
        self.assertEqual(env.best_solution(), saved)

    def test_order_search_keeps_lowest_complete_valid_candidate(self):
        for orders in (((42, 9001), (9001, 42)), ((9001, 42), (42, 9001))):
            with self.subTest(orders=orders):
                env = self.env()
                self.initial_trap(env)
                with patch.object(env, '_candidate_orders', return_value=orders):
                    result = env._repair_group((42, 9001))
                objectives = [c['objective'] for c in result['candidates'] if c['complete']]
                self.assertEqual(result['after_objective'], min(objectives))
                self.assertEqual(result['after_objective'], (0, 0, 0, 20))
                self.assertEqual(check(env._instance, env.best_solution()).total_delay, 20)

        # In one layer A separates B's pins. With A fixed, every B route must
        # cross it, but joint {B,C} repair can reduce five conflicts to one.
        points = [(0, 3, 0), (6, 3, 0), (3, 0, 0), (3, 6, 0), (0, 6, 0), (1, 6, 0)]
        inst = Instance('congestion-stepping-stone', 7, 7, 1, [1], 1, [],
                        [Pin(k, k, 0, *p) for k, p in enumerate(points)],
                        [Net(0, 0, [1]), Net(1, 2, [3]), Net(2, 4, [5])])
        env = self.env(inst)
        engine = WireEngine(inst)
        corridor = {(3, 0, 0), (3, 1, 0), (3, 2, 0), (2, 2, 0), (1, 2, 0),
                    (1, 3, 0), (2, 3, 0), (3, 3, 0), (3, 4, 0), (3, 5, 0), (3, 6, 0)}
        # Exclude the shortcut between (3,2) and (3,3) as well.
        detour = engine.route_net(1, blocked_vertices={
            (x, y, 0) for x in range(7) for y in range(7)}-corridor,
            blocked_edges=(((3, 2, 0), (3, 3, 0)), ((2, 2, 0), (2, 3, 0))))
        for route in (engine.route_net(0), detour, engine.route_net(2)):
            install_result(env, route)
        outside = env._routes[0]
        self.assertEqual(env._objective()[:3], (0, 5, 5))
        result = env._repair_group((1, 2))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['after_objective'][:3], (0, 1, 1))
        self.assertEqual(env._routes[0], outside)
        self.assertIsNone(env.best_solution())
        self.assertEqual(result['after_objective'], min(
            c['objective'] for c in result['candidates'] if c['complete']))

    def test_neighborhood_expands_after_repeated_failed_repairs(self):
        pins = [Pin(2*y+x, 2*y+x, 0, 11*x, y, 0) for y in range(10) for x in range(2)]
        inst = Instance('expanding', 12, 12, 2, [1, 1], 1, [], pins,
                        [Net(y, 2*y, [2*y+1]) for y in range(10)])
        env = self.env(inst)
        with patch.object(env._worker, 'route_net', side_effect=lambda nid, **kw: failure(nid)):
            # Initial construction fails, so every net needs repair.
            for nid in range(10):
                env.step(nid)
            sizes = []
            for _ in range(10):
                env._eligible = {0}
                result = env.step(0)
                sizes.append(len(result.diagnostics['group_net_ids']))
        self.assertEqual(sizes[:8], [2, 2, 3, 3, 5, 5, 8, 8])
        self.assertEqual(sizes[-1], 10)
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(env.observation()['group_repairs'], 10)

    def test_best_legal_candidate_survives_later_partial_attempt_and_budget(self):
        env = self.env(max_calls=3)
        self.initial_trap(env)
        original = copy.deepcopy(env.best_solution())
        with patch.object(env, '_candidate_orders', return_value=((9001, 42), (42, 9001))):
            result = env._repair_group((42, 9001))
        self.assertTrue(result['accepted'])
        self.assertTrue(env.observation()['done'])
        self.assertEqual(env.observation()['termination_reason'], 'call_budget')
        self.assertEqual(sum(c['block1_calls'] for c in env._repair_log[-1]['candidates']), 3)
        self.assertEqual(env.observation()['best_total_delay'], 20)
        best = env.best_solution()
        self.assertEqual(check(env._instance, original).total_delay, 24)
        self.assertTrue(check(env._instance, best).legal)
        self.assertEqual(check(env._instance, best).total_delay, 20)
        self.assertEqual(set(env.observation()['routes']), {42, 9001})
        self.assertFalse(env.observation()['conflict_vertices'])
        best.routes[0].edges.clear()
        env.close()
        self.assertTrue(check(env._instance, env.best_solution()).legal)


if __name__ == '__main__':
    unittest.main()
