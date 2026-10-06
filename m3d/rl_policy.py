"""Small, dependency-free REINFORCE policy for negotiated rerouting order.

Only net selection is learned; Dijkstra and the congestion schedule are fixed.
Feature extraction uses current routing state, never reference solutions.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path


BASE_FEATURES = (
    "bbox_span", "sink_count", "sink_delay", "edge_count",
    "congested_edge_fraction", "congested_vertex_fraction", "excess_occupancy",
    "mean_history", "conflicting_nets", "reroute_count",
)
FEATURE_NAMES = list(BASE_FEATURES) + [f"pressure:{x}" for x in BASE_FEATURES] + [
    f"iteration:{x}" for x in BASE_FEATURES]
FEATURE_DIM = len(FEATURE_NAMES)
FEATURE_VERSION = 1


def _static_features(router, nid):
    pins = router.net_pins[nid]
    coordinates = [router.g.coord(v) for v in pins]
    bbox = sum(max(c[i] for c in coordinates) - min(c[i] for c in coordinates)
               for i in range(3))
    vertices, edges = router.routes[nid]
    adjacency = {v: [] for v in vertices}
    for a, b in edges:
        weight = router.inst.edge_delay(router.g.coord(a), router.g.coord(b))
        adjacency[a].append((b, weight))
        adjacency[b].append((a, weight))
    distances = {pins[0]: 0}
    stack = [pins[0]]
    while stack:
        u = stack.pop()
        for v, weight in adjacency[u]:
            if v not in distances:
                distances[v] = distances[u] + weight
                stack.append(v)
    return [bbox, len(pins) - 1, sum(distances[v] for v in pins[1:]), len(edges)]


def excess_occupancy(router, nid):
    vertices, edges = router.routes[nid]
    return (sum(max(0, len(router.owners_v[v]) - 1) for v in vertices) +
            sum(max(0, len(router.owners_e[e]) - 1) for e in edges))


def candidate_features(router, eligible, pressure, iteration, cache=None):
    """Current features in eligible order; cache only unchanged tree information."""
    cache = {} if cache is None else cache
    rows = []
    for nid in eligible:
        route = router.routes[nid]
        if nid not in cache or cache[nid][0] is not route:
            cache[nid] = (route, _static_features(router, nid))
        vertices, edges = route
        vertex_users = [router.owners_v[v] for v in vertices]
        edge_users = [router.owners_e[e] for e in edges]
        conflicting = set()
        for users in vertex_users:
            if len(users) > 1:
                conflicting.update(users)
        conflicting.discard(nid)
        excess = sum(max(0, len(users) - 1) for users in vertex_users + edge_users)
        history = (sum(router.h_v.get(v, 0.0) for v in vertices) +
                   sum(router.h_e.get(e, 0.0) for e in edges))
        rows.append(cache[nid][1] + [
            sum(len(users) > 1 for users in edge_users) / max(1, len(edges)),
            sum(len(users) > 1 for users in vertex_users) / max(1, len(vertices)),
            excess, history / max(1, len(vertices) + len(edges)),
            len(conflicting), router.reroute_counts[nid],
        ])
    if not rows:
        raise ValueError("at least one eligible net is required")
    maxima = [max(abs(row[j]) for row in rows) or 1.0 for j in range(len(BASE_FEATURES))]
    pressure_scale = pressure / (1.0 + pressure)
    iteration_scale = iteration / max(1, router.max_iters)
    result = []
    for row in rows:
        normalized = [value / scale for value, scale in zip(row, maxima)]
        result.append(normalized + [x * pressure_scale for x in normalized] +
                      [x * iteration_scale for x in normalized])
    return result


class LinearPolicy:
    def __init__(self, weights=None):
        self.weights = list(weights) if weights is not None else [2.0] + [0.0] * (FEATURE_DIM - 1)
        if len(self.weights) != FEATURE_DIM or not all(
                isinstance(w, (int, float)) and math.isfinite(w) for w in self.weights):
            raise ValueError(f"policy requires {FEATURE_DIM} finite weights")

    def logits(self, rows):
        return [sum(w * x for w, x in zip(self.weights, row)) for row in rows]

    def probabilities(self, rows):
        logits = self.logits(rows)
        largest = max(logits)
        values = [math.exp(value - largest) for value in logits]
        total = sum(values)
        return [value / total for value in values]

    def log_gradient(self, rows, selected, probabilities=None):
        probabilities = self.probabilities(rows) if probabilities is None else probabilities
        return [rows[selected][j] - sum(p * row[j] for p, row in zip(probabilities, rows))
                for j in range(FEATURE_DIM)]

    def choose(self, rows, ids, rng=None):
        if rng is None:
            scores = self.logits(rows)
            return max(range(len(ids)), key=lambda i: (scores[i], -ids[i]))
        return rng.choices(range(len(ids)), weights=self.probabilities(rows), k=1)[0]

    def update(self, gradients, learning_rate=0.05):
        if not gradients:
            return
        mean = [sum(g[j] for g in gradients) / len(gradients) for j in range(FEATURE_DIM)]
        norm = math.sqrt(sum(x * x for x in mean))
        scale = learning_rate / max(1.0, norm)
        self.weights = [w + scale * g for w, g in zip(self.weights, mean)]

    def save(self, path, metadata=None):
        data = {"format": "m3d-rl-policy", "feature_version": FEATURE_VERSION,
                "features": FEATURE_NAMES, "weights": self.weights,
                "metadata": metadata or {}}
        Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if (data.get("format") != "m3d-rl-policy" or
                data.get("feature_version") != FEATURE_VERSION or
                data.get("features") != FEATURE_NAMES):
            raise ValueError("incompatible RL policy checkpoint")
        return cls(data["weights"])


class PolicySelector:
    """A fresh selector is required for each episode (cache and gradient are local)."""
    def __init__(self, policy, rng=None, training=False):
        self.policy = policy
        self.rng = rng if rng is not None else random.Random(42)
        self.training = training
        self.gradient = [0.0] * FEATURE_DIM
        self.cache = {}
        self.decisions = 0

    def __call__(self, router, eligible, pressure, iteration):
        if len(eligible) == 1:
            return eligible[0]
        rows = candidate_features(router, eligible, pressure, iteration, self.cache)
        selected = self.policy.choose(rows, eligible, self.rng if self.training else None)
        if self.training:
            gradient = self.policy.log_gradient(rows, selected)
            self.gradient = [a + b for a, b in zip(self.gradient, gradient)]
        self.decisions += 1
        return eligible[selected]
