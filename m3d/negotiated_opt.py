"""Post-legalization delay optimization with a native search backend.

The unchanged negotiated router supplies the legal baseline. The C++17 worker
uses hard-obstacle cleanup, depth-two/regional group negotiation and annealing,
retaining the best independently checked legal snapshot. ``DelayOptimizer`` is
also kept as a portable Python fallback and for existing callers.

The search deadline starts after baseline routing and native setup. Compilation,
serialization, verification and process overhead are reported but excluded from
the search budget. Fixed seed/work reproduces search decisions; a wall-clock
cutoff can stop at different steps on different machines.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math
import random
import time
from typing import Optional

from .checker import check
from .grid import Grid, edge_key
from .model import Instance, NetRoute, Submission
from .negotiated import Negotiated, VCONG, route_negotiated


class BudgetExpired(Exception):
    """A speculative search hit its monotonic deadline."""


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExpired


def _validate_budget(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("time budget must be finite and non-negative")


@dataclass
class OptStats:
    success: bool = False
    seed: int = 0
    time_budget_s: float = 60.0
    baseline_delay: Optional[int] = None
    optimized_delay: Optional[int] = None
    initial_seconds: float = 0.0
    optimization_seconds: float = 0.0
    attempted_groups: int = 0
    reroute_orders: int = 0
    timed_out_orders: int = 0
    accepted_improvements: int = 0
    accepted_worse: int = 0
    accepted_equal: int = 0
    rejected_illegal: int = 0
    stop_reason: str = "not_started"
    backend: str = "python_portable"
    backend_setup_seconds: float = 0.0
    backend_seconds: float = 0.0
    native_setup_seconds: float = 0.0
    native_search_seconds: float = 0.0
    completed_orders: int = 0
    bounded_orders: int = 0
    legal_candidates: int = 0
    single_net_improvements: int = 0
    single_net_gain: int = 0
    work_steps: int = 0
    expansions: int = 0


class _GroupNegotiated(Negotiated):
    """Original negotiation schedule restricted to a group and hard exterior."""

    def __init__(self, inst, order, outside, pin_owner, deadline, tie_seed):
        super().__init__(inst, max_iters=35)
        self.group_order = list(order)
        self.outside = outside
        self.pin_owner = pin_owner
        self.deadline = deadline
        self.tie_seed = tie_seed

    def _order_nets(self):
        return list(self.group_order)

    def _tie(self, vertex):
        return ((vertex * 1103515245) ^ self.tie_seed) & 0x7fffffff

    def _route_net(self, nid, pres_fac):
        _deadline(self.deadline)
        pins = self.net_pins[nid]
        tree_v, tree_e = {pins[0]}, set()
        physical = {pins[0]: 0}
        remaining = set(pins[1:])
        pops = 0
        while remaining:
            # The existing trunk's true driver delay matters to the score.
            # Congestion penalties apply to the proposed extension only.
            dist = {s: float(physical[s]) for s in tree_v}
            previous = {}
            queue = [(dist[s], self._tie(s), s) for s in sorted(tree_v)]
            heapq.heapify(queue)
            found = None
            while queue:
                cost, _, u = heapq.heappop(queue)
                pops += 1
                if pops % 128 == 0:
                    _deadline(self.deadline)
                if cost != dist[u]:
                    continue
                if u in remaining:
                    found = u
                    break
                for v, weight in self.g.neighbors(u):
                    if v in self.outside or v in tree_v:
                        continue
                    owner = self.pin_owner.get(v)
                    if owner is not None and owner != nid:
                        continue
                    edge = edge_key(u, v)
                    ec = weight * (1.0 + self.h_e.get(edge, 0.0)
                                   + pres_fac * len(self.owners_e.get(edge, ())))
                    vc = VCONG * (self.h_v.get(v, 0.0)
                                 + pres_fac * len(self.owners_v.get(v, ())))
                    candidate = cost + ec + vc
                    if candidate < dist.get(v, math.inf):
                        dist[v] = candidate
                        previous[v] = u
                        heapq.heappush(queue, (candidate, self._tie(v), v))
            if found is None:
                return None
            chain = [found]
            while chain[-1] in previous:
                chain.append(previous[chain[-1]])
            for a, b in zip(reversed(chain), reversed(chain[:-1])):
                physical[b] = physical[a] + self.inst.edge_delay(
                    self.g.coord(a), self.g.coord(b))
                tree_v.add(b)
                tree_e.add(edge_key(a, b))
            remaining.discard(found)
            _deadline(self.deadline)
        return tree_v, tree_e


class DelayOptimizer:
    def __init__(self, inst: Instance, initial: Submission, seed: int = 0):
        evaluated = check(inst, initial)
        if not evaluated.legal:
            raise ValueError("delay optimization requires a legal initial solution")
        self.inst = inst
        self.g = Grid(inst)
        self.current = self.best = initial
        self.current_delay = self.best_delay = evaluated.total_delay
        self.net_delays = {n.net: n.delay for n in evaluated.nets}
        self.nets = {n.id: n for n in inst.nets}
        pins = inst.pin_vertex()
        self.pin_vid = {p: self.g.vid(v) for p, v in pins.items()}
        self.pin_owner = {self.pin_vid[p]: n.id for n in inst.nets for p in n.pins()}
        self.lower = {}
        for net in inst.nets:
            driver = pins[net.driver]
            total = 0
            for sink_id in net.sinks:
                sink = pins[sink_id]
                xy = abs(driver[0] - sink[0]) + abs(driver[1] - sink[1])
                total += min(xy * delay + inst.via_delay * (
                    abs(driver[2] - z) + abs(sink[2] - z)
                ) for z, delay in enumerate(inst.layer_delay))
            self.lower[net.id] = total
        self.rng = random.Random(seed)
        self.stats = OptStats(success=True, seed=seed, baseline_delay=self.best_delay,
                              optimized_delay=self.best_delay)
        self.relaxed = {}
        self._refresh_owners()

    def _refresh_owners(self):
        self.vertices = {}
        self.owners = {}
        for route in self.current.routes:
            vertices = {self.pin_vid[p] for p in self.nets[route.net].pins()}
            vertices.update(self.g.vid(v) for edge in route.edges for v in edge)
            self.vertices[route.net] = vertices
            for vertex in vertices:
                self.owners[vertex] = route.net

    def _relaxed_tree(self, nid, deadline):
        """Exact driver-rooted tree with other routes removed, pins retained."""
        _deadline(deadline)
        if nid in self.relaxed:
            return self.relaxed[nid]
        net = self.nets[nid]
        driver = self.pin_vid[net.driver]
        sinks = [self.pin_vid[p] for p in net.sinks]
        remaining = set(sinks)
        dist, prev, queue = {driver: 0}, {}, [(0, driver)]
        pops = 0
        while queue and remaining:
            cost, u = heapq.heappop(queue)
            pops += 1
            if pops % 128 == 0:
                _deadline(deadline)
            if cost != dist[u]:
                continue
            remaining.discard(u)
            if not remaining:
                break
            for v, weight in self.g.neighbors(u):
                owner = self.pin_owner.get(v)
                if owner is not None and owner != nid:
                    continue
                candidate = cost + weight
                if candidate < dist.get(v, math.inf):
                    dist[v] = candidate
                    prev[v] = u
                    heapq.heappush(queue, (candidate, v))
        if remaining:
            raise RuntimeError("a net in a legal solution has no relaxed route")
        vertices = {driver}
        for sink in sinks:
            current = sink
            while current not in vertices:
                vertices.add(current)
                current = prev[current]
        result = (vertices, sum(dist[s] for s in sinks))
        self.relaxed[nid] = result
        return result

    def build_group(self, target, deadline):
        group, frontier = {target}, {target}
        for _ in range(2):
            following = set()
            for nid in sorted(frontier):
                vertices, _ = self._relaxed_tree(nid, deadline)
                following.update(self.owners[v] for v in vertices
                                 if v in self.owners and self.owners[v] not in group)
            group.update(following)
            frontier = following
        return group

    def orders(self, group, target):
        others = sorted(group - {target},
                        key=lambda n: (-(self.net_delays[n] - self.lower[n]), n))
        first = [target] + others
        shuffled = sorted(group)
        self.rng.shuffle(shuffled)
        orders = []
        for order in [first, list(reversed(first)), shuffled]:
            if order not in orders:
                orders.append(order)
        return orders

    def reroute_group(self, group, order, deadline, tie_seed):
        _deadline(deadline)
        outside = {v for v, owner in self.owners.items() if owner not in group}
        group_inst = replace(self.inst, nets=[self.nets[n] for n in sorted(group)])
        solver = _GroupNegotiated(group_inst, order, outside, self.pin_owner,
                                  deadline, tie_seed)
        routed, stats = solver.run()
        if routed is None or not stats.success:
            return None
        replacements = {
            r.net: NetRoute(r.net, sorted(r.edges)) for r in routed.routes
        }
        candidate = Submission(self.inst.name, [
            replacements.get(r.net, r) for r in self.current.routes
        ])
        # In particular this checks outside-group capacity against the FULL case.
        if not check(self.inst, candidate).legal:
            self.stats.rejected_illegal += 1
            return None
        return candidate

    def consider(self, candidate, temperature):
        evaluated = check(self.inst, candidate)
        if not evaluated.legal:
            self.stats.rejected_illegal += 1
            return False
        delta = evaluated.total_delay - self.current_delay
        if delta > 0:
            # Allow a two-unit detour on tiny integer grids, or 1% of the
            # complete-case delay on larger instances.
            if (temperature <= 0 or delta > max(2, 0.01 * self.current_delay)
                    or self.rng.random() >= math.exp(-delta / temperature)):
                return False
        self.current = candidate
        self.current_delay = evaluated.total_delay
        self.net_delays = {n.net: n.delay for n in evaluated.nets}
        self._refresh_owners()
        if delta < 0:
            self.stats.accepted_improvements += 1
        elif delta > 0:
            self.stats.accepted_worse += 1
        else:
            self.stats.accepted_equal += 1
        if evaluated.total_delay < self.best_delay:
            self.best, self.best_delay = candidate, evaluated.total_delay
        return True

    def run(self, time_budget=60.0):
        _validate_budget(time_budget)
        self.stats.time_budget_s = time_budget
        started = time.monotonic()
        deadline = started + time_budget
        temperature0 = max(1.0, 0.002 * self.stats.baseline_delay)
        self.stats.stop_reason = "time_budget"
        while time.monotonic() < deadline:
            ranked = sorted(self.nets, key=lambda n: (
                -(self.net_delays[n] - self.lower[n]), n))
            ranked = [n for n in ranked if self.net_delays[n] > self.lower[n]]
            if not ranked:
                self.stats.stop_reason = "delay_lower_bound_reached"
                break
            # Sample among the largest gaps, avoiding repeated selection of one
            # intractable group while still prioritizing excess physical delay.
            pool = ranked[:8]
            target = self.rng.choices(pool, weights=[
                self.net_delays[n] - self.lower[n] for n in pool], k=1)[0]
            try:
                group = self.build_group(target, deadline)
            except BudgetExpired:
                break
            self.stats.attempted_groups += 1
            orders = self.orders(group, target)
            best_trial, best_trial_delay = None, math.inf
            for index, order in enumerate(orders):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                trial_deadline = min(deadline, time.monotonic() + min(
                    10.0, remaining / (len(orders) - index)))
                self.stats.reroute_orders += 1
                try:
                    candidate = self.reroute_group(
                        group, order, trial_deadline, self.rng.randrange(1 << 31))
                except BudgetExpired:
                    self.stats.timed_out_orders += 1
                    continue
                if candidate is not None:
                    trial_delay = check(self.inst, candidate).total_delay
                    if trial_delay < best_trial_delay:
                        best_trial, best_trial_delay = candidate, trial_delay
            if best_trial is not None:
                # Cooling follows deterministic search steps, not machine speed.
                # Wall-clock time controls deadlines only.
                temperature = temperature0 * 0.97 ** self.stats.attempted_groups
                self.consider(best_trial, temperature)
        self.stats.optimization_seconds = time.monotonic() - started
        self.stats.optimized_delay = self.best_delay
        final = check(self.inst, self.best)
        assert final.legal and final.total_delay <= self.stats.baseline_delay
        return self.best, self.stats


def optimize_delay(inst: Instance, initial: Submission, time_budget: float = 60.0,
                   seed: int = 0, *, _work_limit: Optional[int] = None):
    """Use native search, or the portable Python path if no compiler exists.

    ``_work_limit`` bounds deterministic outer steps for tests and experiments;
    the wall-clock deadline remains a safety cap. Zero budget never compiles.
    """
    _validate_budget(time_budget)
    if _work_limit is not None and (type(_work_limit) is not int or _work_limit < 0):
        raise ValueError("work limit must be a non-negative integer")
    if time_budget == 0:
        return DelayOptimizer(inst, initial, seed=seed).run(0)
    from .negotiated_native import NativeUnavailable, optimize_native
    try:
        return optimize_native(inst, initial, time_budget, seed,
                               -1 if _work_limit is None else _work_limit)
    except NativeUnavailable:
        if _work_limit is not None:
            raise RuntimeError("fixed-work mode requires the native compiler")
        return DelayOptimizer(inst, initial, seed=seed).run(time_budget)


def route_negotiated_opt(inst: Instance, time_budget: float = 60.0, seed: int = 0):
    _validate_budget(time_budget)
    started = time.monotonic()
    initial, _ = route_negotiated(inst)
    initial_seconds = time.monotonic() - started
    if initial is None:
        return None, OptStats(seed=seed, time_budget_s=time_budget,
                              initial_seconds=initial_seconds,
                              stop_reason="initial_routing_failed")
    best, stats = optimize_delay(inst, initial, time_budget=time_budget, seed=seed)
    stats.initial_seconds = initial_seconds
    return best, stats
