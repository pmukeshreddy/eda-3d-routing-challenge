"""Focused Block 4 policy checks using only hand-built current observations."""
import copy
import importlib.util
import unittest

import torch

from m3d.model import Instance, Net, Pin


def observation(net_count=3):
    points = [(1, 1, 0), (4, 1, 0), (3, 2, 0), (6, 2, 1),
              (15, 15, 0), (18, 15, 0)][:2 * net_count]
    pins = [Pin(i, i, int(z > 0), x, y, z)
            for i, (x, y, z) in enumerate(points)]
    nets = [Net(10 * (i + 1), 2 * i, [2 * i + 1])
            for i in range(net_count)]
    instance = Instance("synthetic-policy-state", 20, 20, 2, [2, 2], 3,
                        [], pins, nets)
    packed = lambda xyz: (xyz[2] * 20 + xyz[1]) * 20 + xyz[0]
    route = tuple(((x, 1, 0), (x + 1, 1, 0)) for x in range(1, 4))
    ids = tuple(n.id for n in nets)
    vertices = {packed(v): frozenset({10}) for e in route for v in e}
    edges = {tuple(sorted((packed(a), packed(b)))): frozenset({10})
             for a, b in route}
    return {
        "instance": instance, "grid": (20, 20, 2),
        "net_pins": {n.id: tuple(packed(points[p]) for p in n.pins()) for n in nets},
        "pin_owners": {packed(points[p]): n.id for n in nets for p in n.pins()},
        "routes": {10: route}, "vertex_owners": vertices, "edge_owners": edges,
        "conflict_vertices": {}, "conflict_edges": {},
        "history_vertex": {packed((2, 1, 0)): 0.5},
        "history_edge": {(packed((2, 1, 0)), packed((3, 1, 0))): 1.0},
        "present_factor": 0.85, "eligible_net_ids": ids[1:] or ids,
        "missing_net_ids": ids[1:], "delays": {10: 6},
        "current_routed_delay": 6, "route_search_costs": {10: 8.5},
        "best_total_delay": None, "pass_number": 2, "passes_completed": 1,
        "done": False, "status": "running", "termination_reason": None,
        "checker_report": None,
        "budget": {"limits": {"max_seconds": 120.0, "max_passes": 20,
                                "max_calls": 1500, "max_expansions": 20_000_000,
                                "max_expansions_per_call": 1_000_000,
                                "max_call_seconds": 10.0},
                   "elapsed_s": 0.1, "remaining_seconds": 119.9,
                   "engine_calls": 4, "remaining_calls": 1496,
                   "expansions_charged": 100, "remaining_expansions": 19_999_900,
                   "remaining_passes": 19},
    }


class SchedulerPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("m3d.scheduler_features"),
                             "the current-state encoder must exist")
        self.assertIsNotNone(importlib.util.find_spec("m3d.scheduler_policy"),
                             "the typed scheduler policy must exist")
        from m3d.scheduler_features import encode_observation, batch_states
        from m3d.scheduler_policy import SchedulerPolicy
        self.encode, self.batch = encode_observation, batch_states
        torch.manual_seed(123)
        self.policy = SchedulerPolicy().eval()

    def test_distribution_masks_invalid_ids_and_keeps_training_gradients(self):
        obs = observation()
        dist = self.policy.distribution(obs)
        self.assertEqual(dist.net_ids, (10, 20, 30))
        self.assertEqual(tuple(dist.logits.shape), (3,))
        self.assertTrue(torch.isneginf(dist.logits[0]))
        self.assertEqual(dist.probs[0].item(), 0.0)
        self.assertAlmostEqual(dist.probs.sum().item(), 1.0, places=6)
        self.assertIn(dist.mode(), (20, 30))
        self.assertEqual(self.policy.act(obs, deterministic=True), dist.mode())
        first, second = torch.Generator().manual_seed(17), torch.Generator().manual_seed(17)
        actions = [dist.sample(first) for _ in range(20)]
        self.assertEqual(actions, [dist.sample(second) for _ in range(20)])
        self.assertTrue(set(actions) <= {20, 30})
        expected = torch.log_softmax(dist.logits, dim=0)[1]
        torch.testing.assert_close(dist.log_prob(20), expected)
        torch.testing.assert_close(self.policy.log_prob(obs, 20), expected)
        self.assertTrue(torch.isneginf(dist.log_prob(10)))
        for invalid in (999, True, 20.0):
            with self.assertRaises(ValueError):
                dist.log_prob(invalid)
        (-dist.log_prob(20)).backward()
        self.assertTrue(any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            and p.grad.abs().sum().item() > 0 for p in self.policy.parameters()))
        obs["eligible_net_ids"] = ()
        with self.assertRaises(ValueError):
            self.policy.distribution(obs)

    def test_renaming_and_permuting_ids_preserves_logits_and_ignores_metadata(self):
        obs = observation()
        changed = copy.deepcopy(obs)
        rename = {10: 801, 20: -54, 30: 7}
        for net in changed["instance"].nets:
            net.id = rename[net.id]
        changed["instance"].nets.reverse()
        for key in ("net_pins", "routes", "delays", "route_search_costs"):
            changed[key] = {rename[nid]: value for nid, value in changed[key].items()}
        for key in ("eligible_net_ids", "missing_net_ids"):
            changed[key] = tuple(rename[nid] for nid in changed[key])
        for key in ("vertex_owners", "edge_owners", "conflict_vertices", "conflict_edges"):
            changed[key] = {resource: frozenset(rename[nid] for nid in owners)
                            for resource, owners in changed[key].items()}
        changed["pin_owners"] = {v: rename[nid] for v, nid in changed["pin_owners"].items()}
        changed["instance"].name = "hidden-evaluation-label"
        changed["instance"].params = {"reference_delay": 10**99, "split": "test"}
        changed["instance"].seed = changed["instance"].master_seed = 123456789
        changed["checker_report"] = {"total_delay": 10**99}
        changed["best_total_delay"] = 1
        changed["budget"]["elapsed_s"] = 111
        changed["budget"]["remaining_seconds"] = 9
        a, b = self.encode(obs), self.encode(changed)
        permutation = [b.net_ids.index(rename[nid]) for nid in a.net_ids]
        torch.testing.assert_close(a.net_features, b.net_features[permutation], rtol=0, atol=0)
        torch.testing.assert_close(a.global_features, b.global_features, rtol=0, atol=0)
        torch.testing.assert_close(a.adjacency, b.adjacency[permutation][:, permutation], rtol=0, atol=0)
        with torch.no_grad():
            original = self.policy(self.batch([a], device="cpu"))[0]
            permuted = self.policy(self.batch([b], device="cpu"))[0, permutation]
        torch.testing.assert_close(original, permuted, rtol=1e-5, atol=1e-6)

    def test_padding_never_changes_real_net_logits_or_gradients(self):
        short, full = self.encode(observation(1)), self.encode(observation())
        batch = self.batch([short, full], device="cpu")
        self.assertEqual(batch["net_ids"], [short.net_ids, full.net_ids])
        self.assertEqual(batch["padding_mask"].tolist(), [[False, True, True], [False] * 3])
        self.assertEqual(batch["eligible"].tolist(), [[True, False, False], [False, True, True]])
        together = self.policy(batch)
        alone = self.policy(self.batch([short], device="cpu"))
        torch.testing.assert_close(together[0, :1], alone[0], rtol=1e-5, atol=1e-6)
        # Padded feature values and adjacency entries must be incapable of affecting real tokens.
        poisoned = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        poisoned["net_features"][0, 1:] = 1e5
        poisoned["adjacency"][0, 1:, :] = 1
        poisoned["adjacency"][0, :, 1:] = 1
        poisoned["net_features"].requires_grad_()
        result = self.policy(poisoned)
        torch.testing.assert_close(result[0, :1], alone[0], rtol=1e-5, atol=1e-6)
        result[0, 0].backward()
        self.assertEqual(poisoned["net_features"].grad[0, 1:].abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(together).all())

    def test_features_track_current_resources_and_graph_connects_conflicting_nets(self):
        from m3d.scheduler_features import NET_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, FEATURE_SCHEMA_VERSION
        obs = observation()
        state = self.encode(obs)
        self.assertIsInstance(FEATURE_SCHEMA_VERSION, str)
        self.assertEqual(state.net_features.shape, (3, len(NET_FEATURE_NAMES)))
        self.assertEqual(state.global_features.shape, (len(GLOBAL_FEATURE_NAMES),))
        self.assertEqual(state.net_features.dtype, torch.float32)
        self.assertEqual(state.global_features.dtype, torch.float32)
        self.assertEqual(state.net_features.device.type, "cpu")
        self.assertEqual(state.eligible.dtype, torch.bool)
        self.assertEqual(state.adjacency.tolist(), [[False, True, False], [True, False, False], [False, False, False]])
        at = {name: i for i, name in enumerate(NET_FEATURE_NAMES)}
        self.assertAlmostEqual(state.net_features[0, at["driver_x"]].item(), 1 / 19)
        self.assertAlmostEqual(state.net_features[0, at["bbox_dx"]].item(), 3 / 19)
        self.assertEqual(state.net_features[0, at["routed"]].item(), 1)
        self.assertEqual(state.net_features[1, at["missing"]].item(), 1)
        self.assertEqual(state.net_features[1, at["cross_die"]].item(), 1)
        self.assertGreater(state.net_features[0, at["route_delay"]].item(), 0)
        self.assertGreater(state.net_features[0, at["bbox_vertex_occupancy"]].item(), 0)
        self.assertGreater(state.net_features[0, at["bbox_vertex_history"]].item(), 0)
        self.assertGreater(state.net_features[0, at["bbox_edge_history"]].item(), 0)
        obs["conflict_vertices"] = {22: frozenset({10, 30})}
        obs["vertex_owners"][22] = frozenset({10, 30})
        conflicts = self.encode(obs)
        self.assertTrue(conflicts.adjacency[0, 2])
        self.assertGreater(conflicts.net_features[0, at["conflict_vertices"]].item(), 0)
        self.assertGreater(conflicts.net_features[2, at["conflict_neighbors"]].item(), 0)
        self.assertTrue(torch.isfinite(conflicts.net_features).all())
        parameters = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        self.assertGreaterEqual(parameters, 2_000_000)
        self.assertLessEqual(parameters, 15_000_000)


if __name__ == "__main__":
    unittest.main()
