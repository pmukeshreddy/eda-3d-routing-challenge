import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from m3d.checker import check
from m3d.model import Instance, Submission
from m3d.negotiated import Negotiated, route_negotiated
from m3d.rl_policy import LinearPolicy, PolicySelector, candidate_features, FEATURE_DIM


ROOT = Path(__file__).resolve().parents[1]


class TestSelector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = Instance.load(str(ROOT / 'benchmarks_hard/case_01.json'))

    def test_default_selector_preserves_reference(self):
        sub, _ = route_negotiated(self.inst, selector=None)
        reference = Submission.load(str(ROOT / 'benchmarks_hard/reference/case_01.sol.json'))
        self.assertEqual(check(self.inst, sub).total_delay, 11938)
        self.assertEqual({r.net: set(r.edges) for r in sub.routes},
                         {r.net: set(r.edges) for r in reference.routes})

    def test_callback_sees_current_state_and_each_net_once_per_round(self):
        seen = set()
        calls = []
        def select(router, eligible, pressure, iteration):
            nid = eligible[0]
            self.assertNotIn((iteration, nid), seen)
            self.assertIn(nid, router.routes)
            self.assertEqual(router.reroute_counts[nid], sum(n == nid for _, n in seen))
            self.assertGreater(pressure, 0.5)
            seen.add((iteration, nid))
            calls.append(nid)
            return nid
        sub, _ = route_negotiated(self.inst, selector=select)
        self.assertTrue(calls)
        self.assertEqual(check(self.inst, sub).total_delay, 11938)

    def test_ineligible_action_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'eligible'):
            route_negotiated(self.inst, selector=lambda *args: -1)

    def test_features_include_real_delay_and_finite_normalization(self):
        router = Negotiated(self.inst)
        for nid in router._order_nets():
            router._add(nid, *router._route_net(nid, router.pres_fac0))
        eligible = router._order_nets()[:4]
        features = candidate_features(router, eligible, 0.85, 1)
        self.assertEqual(len(features), 4)
        for row in features:
            self.assertEqual(len(row), FEATURE_DIM)
            self.assertTrue(all(math.isfinite(x) and 0 <= x <= 1 for x in row))
        self.assertEqual(max(row[2] for row in features), 1.0)
        # The one-candidate decision has exactly probability one and no gradient.
        selector = PolicySelector(LinearPolicy(), random.Random(42), training=True)
        self.assertEqual(selector(router, eligible[:1], 0.85, 1), eligible[0])
        self.assertEqual(selector.gradient, [0.0] * FEATURE_DIM)


class TestPolicy(unittest.TestCase):
    def test_log_probability_gradient_matches_finite_differences(self):
        policy = LinearPolicy([0.12 * (i - 3) for i in range(FEATURE_DIM)])
        rows = [[((i * 3 + j) % 7) / 7 for i in range(FEATURE_DIM)] for j in range(3)]
        analytic = policy.log_gradient(rows, 1)
        eps = 1e-5
        for i in range(FEATURE_DIM):
            original = policy.weights[i]
            policy.weights[i] = original + eps
            plus = math.log(policy.probabilities(rows)[1])
            policy.weights[i] = original - eps
            minus = math.log(policy.probabilities(rows)[1])
            policy.weights[i] = original
            self.assertAlmostEqual(analytic[i], (plus - minus) / (2 * eps), places=7)

    def test_softmax_is_stable_and_greedy_ties_use_net_id(self):
        policy = LinearPolicy([1000.0] * FEATURE_DIM)
        rows = [[1.0] * FEATURE_DIM, [0.0] * FEATURE_DIM]
        self.assertEqual(policy.probabilities(rows), [1.0, 0.0])
        zero = LinearPolicy([0.0] * FEATURE_DIM)
        self.assertEqual(zero.choose(rows, [8, 3]), 1)

    def test_positive_update_increases_selected_probability(self):
        policy = LinearPolicy([0.0] * FEATURE_DIM)
        rows = [[1.0] * FEATURE_DIM, [0.0] * FEATURE_DIM]
        before = policy.probabilities(rows)[0]
        policy.update([policy.log_gradient(rows, 0)], learning_rate=0.05)
        self.assertGreater(policy.probabilities(rows)[0], before)
        self.assertAlmostEqual(math.sqrt(sum(w*w for w in policy.weights)), 0.05)

    def test_checkpoint_roundtrip_and_invalid_weights(self):
        policy = LinearPolicy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.json'
            policy.save(path, {'seed': 42})
            loaded = LinearPolicy.load(path)
            self.assertEqual(loaded.weights, policy.weights)
            data = json.loads(path.read_text())
            data['weights'][0] = float('nan')
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                LinearPolicy.load(path)


if __name__ == '__main__':
    unittest.main()
