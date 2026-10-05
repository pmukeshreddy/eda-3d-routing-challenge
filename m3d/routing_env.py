"""Block 2: budgeted, transactional multi-net repair around unchanged Block 1.

Initial actions construct nets; subsequent actions select a neighborhood seed.
Every repair removes and reconstructs the whole interacting group.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import combinations
import math
import random
from time import perf_counter

from ._routing_worker import BoundedWireEngine
from .checker import check
from .grid import Grid, edge_key
from .model import Instance, NetRoute, Submission
from .negotiated import VCONG
from .wire_engine import WireEngine


@dataclass(frozen=True)
class RoutingBudget:
    max_seconds: float = 180.0
    max_passes: int = 50
    max_calls: int = 5000
    max_expansions: int = 200_000_000
    max_expansions_per_call: int = 1_000_000
    max_call_seconds: float = 10.0

    def __post_init__(self):
        for name in ("max_seconds", "max_call_seconds"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_passes", "max_calls", "max_expansions", "max_expansions_per_call"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 2**64 - 1:
                raise ValueError(f"{name} must be a nonnegative uint64 integer")


@dataclass(frozen=True)
class StepResult:
    observation: dict
    diagnostics: dict
    done: bool


@dataclass(frozen=True)
class _Route:
    edges: tuple
    vertices: frozenset
    packed_edges: frozenset
    delay: int
    search_cost: float


class RoutingEnv:
    def __init__(self, *, method: str = "best"):
        if method not in ("best", "shortest_path", "cost_distance"):
            raise ValueError("unknown Block 1 method")
        self.method = method
        self._worker = None
        self._instance = None

    def reset(self, instance: Instance, seed: int = 0,
              budget: RoutingBudget | None = None) -> dict:
        if not isinstance(instance, Instance):
            raise TypeError("instance must be an Instance")
        if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
            raise ValueError("seed must be a uint64 integer")
        if budget is not None and not isinstance(budget, RoutingBudget):
            raise TypeError("budget must be a RoutingBudget")
        # Reuse Block 1's complete input validation before publishing any new
        # episode state. A rejected reset leaves the previous episode intact.
        started = perf_counter()
        snapshot = deepcopy(instance)
        WireEngine(snapshot)
        self.close()
        self._started = started
        self._ended = None
        self._instance = snapshot
        self._budget = budget or RoutingBudget()
        self._deadline = self._started + self._budget.max_seconds
        self._seed = seed
        self._grid = Grid(self._instance)
        self._nets = {n.id: n for n in self._instance.nets}
        if len(self._nets) != len(self._instance.nets):
            raise ValueError("net IDs must be unique")
        pins = self._instance.pin_vertex()
        self._net_pins = {n.id: tuple(self._grid.vid(pins[p]) for p in n.pins())
                          for n in self._instance.nets}
        self._pin_owner = {v: nid for nid, vs in self._net_pins.items() for v in vs}
        self._routes = {}
        self._owners_v, self._owners_e = {}, {}
        self._h_v, self._h_e = {}, {}
        self._present = 0.5
        self._eligible = set(self._nets)
        self._pass = 1
        self._passes_completed = 0
        self._calls = self._expansions = 0
        self._phase = "initial"
        self._repair_failures = {nid: 0 for nid in self._nets}
        self._group_repairs = self._accepted_repairs = 0
        self._group_sizes = {}
        self._repair_log = []
        self._exhausted_seeds = set()
        self._prepare_regions()
        self._interactions = {}
        self._best = self._best_delay = None
        self._checker_report = None
        self._done = False
        self._in_step = False
        self._reason = None
        self._worker = BoundedWireEngine(self._instance)
        if not self._nets:
            self._consider_snapshot()
            self._finish("legal")
        elif self._budget.max_passes == 0:
            self._finish("pass_budget")
        else:
            self._check_budget()
        return self.observation()

    def _require_reset(self):
        if self._instance is None:
            raise RuntimeError("reset the environment before use")

    def _remaining_seconds(self):
        return max(0.0, self._deadline - perf_counter())

    def _finish(self, reason):
        if not self._done:
            self._done, self._reason = True, reason
            self._worker.close()
            self._ended = perf_counter()

    def _check_budget(self):
        if self._done:
            return
        if self._remaining_seconds() <= 0:
            self._finish("time_budget")
        elif self._calls >= self._budget.max_calls:
            self._finish("call_budget")
        elif (self._expansions >= self._budget.max_expansions
              or self._budget.max_expansions_per_call == 0):
            self._finish("expansion_budget")

    def _conflicts(self):
        return ({v: frozenset(owners) for v, owners in self._owners_v.items() if len(owners) > 1},
                {e: frozenset(owners) for e, owners in self._owners_e.items() if len(owners) > 1})

    def eligible_net_ids(self) -> tuple[int, ...]:
        self._require_reset()
        self._check_budget()
        return () if self._done else tuple(sorted(self._eligible))

    def observation(self) -> dict:
        """Return detached raw state. Packed IDs use Grid's (z*h+y)*w+x."""
        self._require_reset()
        if not self._in_step:
            self._check_budget()
        conflicts_v, conflicts_e = self._conflicts()
        elapsed = (self._ended if self._ended is not None else perf_counter()) - self._started
        return {
            "instance": deepcopy(self._instance),
            "grid": (self._grid.w, self._grid.h, self._grid.l),
            "pin_owners": dict(self._pin_owner),
            "net_pins": dict(self._net_pins),
            "routes": {nid: r.edges for nid, r in self._routes.items()},
            "vertex_owners": {v: frozenset(ns) for v, ns in self._owners_v.items()},
            "edge_owners": {e: frozenset(ns) for e, ns in self._owners_e.items()},
            "conflict_vertices": conflicts_v,
            "conflict_edges": conflicts_e,
            "history_vertex": dict(self._h_v),
            "history_edge": dict(self._h_e),
            "present_factor": self._present,
            "phase": self._phase,
            "interactions": {nid: dict(scores) for nid, scores in self._interactions.items()},
            "repair_failures": dict(self._repair_failures),
            "group_repairs": self._group_repairs,
            "accepted_group_repairs": self._accepted_repairs,
            "group_sizes_used": dict(self._group_sizes),
            "eligible_net_ids": () if self._done else tuple(sorted(self._eligible)),
            "missing_net_ids": tuple(sorted(self._nets.keys() - self._routes.keys())),
            "delays": {nid: r.delay for nid, r in self._routes.items()},
            "current_routed_delay": sum(r.delay for r in self._routes.values()),
            "route_search_costs": {nid: r.search_cost for nid, r in self._routes.items()},
            "best_total_delay": self._best_delay,
            "pass_number": self._pass,
            "passes_completed": self._passes_completed,
            "done": self._done,
            "status": ("success" if self._best is not None else "failure") if self._done else "running",
            "termination_reason": self._reason,
            "checker_report": deepcopy(self._checker_report),
            "budget": {
                "limits": asdict(self._budget), "elapsed_s": elapsed,
                "remaining_seconds": max(0.0, self._budget.max_seconds - elapsed),
                "engine_calls": self._calls, "remaining_calls": self._budget.max_calls - self._calls,
                "expansions_charged": self._expansions,
                "remaining_expansions": self._budget.max_expansions - self._expansions,
                "remaining_passes": max(0, self._budget.max_passes - self._passes_completed),
            },
        }

    def _remove(self, nid):
        route = self._routes.pop(nid)
        for owners, resources in ((self._owners_v, route.vertices), (self._owners_e, route.packed_edges)):
            for resource in resources:
                owners[resource].remove(nid)
                if not owners[resource]:
                    del owners[resource]
        return route

    def _install(self, nid, route):
        for owners, resources in ((self._owners_v, route.vertices), (self._owners_e, route.packed_edges)):
            for resource in resources:
                owners.setdefault(resource, set()).add(nid)
        self._routes[nid] = route

    def _prices(self, congestion=None):
        present, h_v, h_e = congestion or (self._present, self._h_v, self._h_e)
        prices = {}
        # Native search already adds the base delay. Only send the surcharge.
        for a, b in h_e.keys() | self._owners_e.keys():
            delay = (self._grid.via_delay if a // self._grid.wh != b // self._grid.wh
                     else self._grid.layer_delay[a // self._grid.wh])
            prices[a, b] = delay * (h_e.get((a, b), 0.0)
                                    + present * len(self._owners_e.get((a, b), ())))
        # Undirected edge API: split each vertex's pressure equally onto every
        # incident edge. A path's internal vertex is then charged exactly once.
        for v in h_v.keys() | self._owners_v.keys():
            pressure = 0.5 * VCONG * (h_v.get(v, 0.0)
                                       + present * len(self._owners_v.get(v, ())))
            for neighbor, _ in self._grid.neighbors(v):
                e = edge_key(v, neighbor)
                prices[e] = prices.get(e, 0.0) + pressure
        return prices

    def _consider_snapshot(self):
        if self._nets.keys() != self._routes.keys() or any(self._conflicts()):
            return
        candidate = Submission(self._instance.name,
                               [NetRoute(nid, list(self._routes[nid].edges)) for nid in sorted(self._nets)])
        report = check(self._instance, candidate)
        self._checker_report = report.to_dict()
        if not report.legal:
            self._finish("checker_rejected")
        elif self._best_delay is None or report.total_delay < self._best_delay:
            self._best, self._best_delay = deepcopy(candidate), report.total_delay

    def _prepare_regions(self):
        """Static pin bboxes and a one-vertex halo; no route search here."""
        self._bboxes, self._spans, self._lower_delays = {}, {}, {}
        self._region_nets = {}
        for nid, pins in self._net_pins.items():
            coords = [self._grid.coord(v) for v in pins]
            lo = tuple(min(p[d] for p in coords) for d in range(3))
            hi = tuple(max(p[d] for p in coords) for d in range(3))
            self._bboxes[nid] = lo, hi
            self._spans[nid] = sum(b-a for a, b in zip(lo, hi))
            self._lower_delays[nid] = max(1, sum(
                (abs(p[0]-coords[0][0]) + abs(p[1]-coords[0][1])) * min(self._grid.layer_delay)
                + abs(p[2]-coords[0][2]) * self._grid.via_delay for p in coords[1:]))
            for z in range(max(0, lo[2]-1), min(self._grid.l, hi[2]+2)):
                for y in range(max(0, lo[1]-1), min(self._grid.h, hi[1]+2)):
                    for x in range(max(0, lo[0]-1), min(self._grid.w, hi[0]+2)):
                        self._region_nets.setdefault(self._grid.vid((x, y, z)), []).append(nid)

    def _recompute_interactions(self):
        scores = {nid: {} for nid in self._nets}

        def add(a, b, value):
            if a != b:
                scores[a][b] = scores[a].get(b, 0.0) + value
                scores[b][a] = scores[b].get(a, 0.0) + value

        for conflicts in self._conflicts():
            for owners in conflicts.values():
                for a, b in combinations(sorted(owners), 2):
                    add(a, b, 100.0)
        # Occupied escape vertices next to pins are especially useful blockers.
        for nid, pins in self._net_pins.items():
            for pin in pins:
                for vertex, _ in self._grid.neighbors(pin):
                    for owner in self._owners_v.get(vertex, ()):
                        add(nid, owner, 10.0)
        detour = {nid: min(4.0, self._routes[nid].delay / self._lower_delays[nid])
                  if nid in self._routes else 4.0 for nid in self._nets}
        for vertex, owners in self._owners_v.items():
            for nid in self._region_nets.get(vertex, ()):
                for owner in owners:
                    add(nid, owner, 0.1 * detour[nid] / max(1, self._spans[nid]))
        for a, b in combinations(sorted(self._nets), 2):
            alo, ahi = self._bboxes[a]
            blo, bhi = self._bboxes[b]
            overlap = math.prod(max(0, min(ahi[d]+1, bhi[d]+1)-max(alo[d]-1, blo[d]-1)+1)
                                for d in range(3))
            if overlap:
                size = min(math.prod(ahi[d]-alo[d]+3 for d in range(3)),
                           math.prod(bhi[d]-blo[d]+3 for d in range(3)))
                add(a, b, overlap / size)
        self._interactions = scores

    def _conflict_component(self, seed):
        adjacency = {nid: set() for nid in self._nets}
        for conflicts in self._conflicts():
            for owners in conflicts.values():
                for nid in owners:
                    adjacency[nid].update(owners - {nid})
        component, frontier = {seed}, [seed]
        while frontier:
            for nid in adjacency[frontier.pop()] - component:
                component.add(nid)
                frontier.append(nid)
        return component

    def _select_group(self, seed):
        stage = self._repair_failures[seed] // 2
        component = self._conflict_component(seed)
        target = (2, 3, 5, 8)[stage] if stage < 4 else max(len(component), 8 * 2**(stage-3))
        target = min(len(self._nets), target)
        group = {seed} if stage < 4 else set(component)
        while len(group) < target:
            choices = self._nets.keys() - group
            # Strongest aggregate interaction with the whole growing group.
            neighbor = max(choices, key=lambda nid: (
                sum(self._interactions.get(nid, {}).get(other, 0.0) for other in group),
                -nid))
            group.add(neighbor)
        return (seed, *sorted(group - {seed}))

    def _candidate_orders(self, group):
        """At most eight distinct orders, never factorial enumeration."""
        conflict_strength = {nid: 0 for nid in group}
        for conflicts in self._conflicts():
            for owners in conflicts.values():
                for nid in set(group) & owners:
                    conflict_strength[nid] += len(owners)-1
        orders = []

        def add(order):
            order = tuple(order)
            if order not in orders and len(orders) < 8:
                orders.append(order)

        add(sorted(group, key=lambda n: (-self._spans[n], n)))
        add(sorted(group, key=lambda n: (-len(self._nets[n].sinks), n)))
        add(sorted(group, key=lambda n: (
            -(self._routes[n].delay / self._lower_delays[n] if n in self._routes else math.inf),
            -(self._routes[n].delay if n in self._routes else 0), n)))
        add(sorted(group, key=lambda n: (-conflict_strength[n], n)))
        base = tuple(group)
        for shift in range(min(3, len(base))):
            add(base[shift:] + base[:shift])
        add(reversed(orders[0]))
        # Deterministic variation when a neighborhood is revisited.
        rng = random.Random(self._seed + self._group_repairs)
        for _ in range(8):
            shuffled = list(group)
            rng.shuffle(shuffled)
            add(shuffled)
        return tuple(orders)

    def _objective(self):
        conflicts = self._conflicts()
        return (len(self._nets.keys() - self._routes.keys()),
                sum(len(resources) for resources in conflicts),
                sum(len(owners)-1 for resources in conflicts for owners in resources.values()),
                sum(route.delay for route in self._routes.values()))

    def _repair_group(self, group):
        """Rebuild complete joint candidates from one identical outside state.

        Routes are immutable. Candidate mutations touch only routes/ownership;
        penalties and the separately copied legal incumbent are read-only until
        a winning complete candidate is committed. Work budgets never roll back.
        """
        group = tuple(group)
        if not group or len(set(group)) != len(group) or not set(group) <= self._nets.keys():
            raise ValueError("invalid repair group")
        if len(group) < min(2, len(self._nets)):
            raise ValueError("repair must include interacting nets together")
        orders = self._candidate_orders(group)
        original = {nid: self._routes[nid] for nid in group if nid in self._routes}
        before = best_objective = self._objective()
        best_routes = None
        best_congestion = None
        # Negotiated candidates can reduce congestion even when the fixed
        # outside layout makes an entirely exclusive reconstruction impossible.
        # All orders use the same proposed prices; rejected attempts do not
        # mutate history or present pressure.
        h_v, h_e = dict(self._h_v), dict(self._h_e)
        for history, conflicts in zip((h_v, h_e), self._conflicts()):
            for resource, owners in conflicts.items():
                history[resource] = history.get(resource, 0.0) + 0.5 * (len(owners)-1)
        pressure = min(1e6, max(8.0, self._present * 1.7)
                       * 2**min(6, self._repair_failures[group[0]] // 2))
        congestion = pressure, h_v, h_e
        variants = (False, True) if any(before[:3]) else (True,)
        diagnostics = {"group_net_ids": group, "before_objective": before,
                       "candidates": [], "engine_status": "not_called"}
        calls_before = self._calls
        self._group_repairs += 1
        self._group_sizes[len(group)] = self._group_sizes.get(len(group), 0) + 1
        self._in_step = True

        def clear_group():
            for nid in group:
                if nid in self._routes:
                    self._remove(nid)

        try:
            for order, exclusive in ((o, e) for o in orders for e in variants):
                self._check_budget()
                if self._done:
                    break
                clear_group()
                candidate_start = self._calls
                candidate = {"order": order, "mode": "exclusive" if exclusive else "negotiated",
                             "complete": False, "objective": None, "calls": []}
                for nid in order:
                    route, call = self._build_route(nid, exclusive=exclusive, congestion=congestion)
                    candidate["calls"].append(call)
                    diagnostics["engine_status"] = call["engine_status"]
                    if route is None:
                        break
                    self._install(nid, route)
                else:
                    candidate["complete"] = True
                    candidate["objective"] = self._objective()
                    if candidate["objective"] < best_objective:
                        best_objective = candidate["objective"]
                        best_routes = {nid: self._routes[nid] for nid in group}
                        best_congestion = None if exclusive else congestion
                candidate["block1_calls"] = self._calls - candidate_start
                diagnostics["candidates"].append(candidate)
        finally:
            # Includes time/call/expansion exhaustion halfway through a candidate.
            clear_group()
            for nid, route in (best_routes if best_routes is not None else original).items():
                self._install(nid, route)
            self._in_step = False
        accepted = best_routes is not None
        if accepted:
            self._accepted_repairs += 1
            if best_congestion is not None:
                self._present, self._h_v, self._h_e = best_congestion
            self._consider_snapshot()
        diagnostics.update(accepted=accepted, route_installed=accepted, rolled_back=not accepted,
                           after_objective=self._objective(), block1_calls=self._calls-calls_before)
        record = {k: deepcopy(v) for k, v in diagnostics.items() if k != "candidates"}
        record["candidates"] = [
            {"order": c["order"], "mode": c["mode"], "complete": c["complete"],
             "objective": c["objective"], "block1_calls": c["block1_calls"],
             "last_status": c["calls"][-1]["engine_status"]}
            for c in diagnostics["candidates"]]
        self._repair_log.append(record)
        self._recompute_interactions()
        self._check_budget()
        return diagnostics

    def _affected_nets(self):
        affected = set(self._nets.keys() - self._routes.keys())
        for conflicts in self._conflicts():
            for owners in conflicts.values():
                affected.update(owners)
        return affected

    def _end_pass(self):
        self._passes_completed += 1
        affected = self._affected_nets()
        if not affected:
            self._finish("legal" if self._best is not None else "checker_rejected")
            return
        self._check_budget()
        if self._done:
            return
        if self._passes_completed >= self._budget.max_passes:
            self._finish("pass_budget")
            return
        if affected <= self._exhausted_seeds:
            self._finish("no_improvement")
            return
        if self._phase == "initial":
            for history, conflicts in zip((self._h_v, self._h_e), self._conflicts()):
                for resource, owners in conflicts.items():
                    history[resource] = history.get(resource, 0.0) + 0.5 * (len(owners)-1)
            self._present = min(self._present * 1.7, 1e6)
        self._phase = "repair"
        self._eligible = affected - self._exhausted_seeds
        self._pass += 1
        self._recompute_interactions()

    def _build_route(self, net_id, *, exclusive=False, congestion=None):
        """The only Block 1 dispatch path; construction never repairs a net alone."""
        self._check_budget()
        diagnostics = {"net_id": net_id, "engine_status": "not_called"}
        if self._done:
            return None, diagnostics
        allowance = min(self._budget.max_expansions_per_call,
                        self._budget.max_expansions - self._expansions)
        called = False
        try:
            # Outside wires stay fixed in both variants. Negotiated candidates
            # price their occupancy; exclusive candidates hard-block it.
            prices = {} if exclusive else self._prices(congestion)
            blocked = {v for v, owner in self._pin_owner.items() if owner != net_id}
            if exclusive:
                blocked.update(self._owners_v)
            if self._remaining_seconds() <= 0:
                raise TimeoutError("episode expired while constructing search state")
            self._calls += 1
            called = True
            self._expansions += allowance
            result = self._worker.route_net(
                net_id, timeout_s=min(self._budget.max_call_seconds, self._remaining_seconds()),
                blocked_vertices=tuple(sorted(blocked)),
                blocked_edges=tuple(sorted(self._owners_e)) if exclusive else (),
                edge_prices=prices, method=self.method,
                seed=(self._seed + self._calls - 1) % (2**64), max_expansions=allowance)
            self._expansions -= allowance - result.expansions
            diagnostics.update(engine_status=result.status, expansions=result.expansions,
                               engine_elapsed_s=result.elapsed_s, method=result.method,
                               budget_exhausted=result.budget_exhausted,
                               total_delay=result.total_delay, resource_cost=result.resource_cost,
                               objective=result.objective, candidates=result.candidates)
            if result.status == "success" and self._remaining_seconds() > 0:
                edges = frozenset(edge_key(self._grid.vid(a), self._grid.vid(b)) for a, b in result.edges)
                vertices = frozenset(v for e in edges for v in e) | frozenset(self._net_pins[net_id])
                return _Route(tuple(result.edges), vertices, edges,
                              result.total_delay, result.resource_cost), diagnostics
        except (TimeoutError, RuntimeError, ValueError, OverflowError) as exc:
            diagnostics.update(engine_status="timeout" if isinstance(exc, TimeoutError) else "error",
                               error=str(exc), expansions_charged=allowance if called else 0)
        return None, diagnostics

    def step(self, net_id: int) -> StepResult:
        self._require_reset()
        if type(net_id) is not int or net_id not in self._eligible or self._done:
            raise ValueError(f"net is not eligible: {net_id!r}")
        self._check_budget()
        if self._done:
            return StepResult(self.observation(), {"net_id": net_id, "engine_status": "not_called"}, True)
        if self._phase == "initial":
            self._in_step = True
            try:
                route, diagnostics = self._build_route(net_id)
                if route is not None:
                    self._install(net_id, route)
                    self._consider_snapshot()
            finally:
                self._in_step = False
            diagnostics.update(route_installed=route is not None, rolled_back=False)
            self._eligible.remove(net_id)
        else:
            group = self._select_group(net_id)
            diagnostics = self._repair_group(group)
            if diagnostics["accepted"]:
                for nid in group:
                    self._repair_failures[nid] = 0
                self._exhausted_seeds.clear()
                self._eligible.difference_update(group)
                self._eligible.intersection_update(self._affected_nets())
            else:
                self._repair_failures[net_id] += 1
                self._eligible.discard(net_id)
                if len(group) == len(self._nets) and self._repair_failures[net_id] >= 10:
                    self._exhausted_seeds.add(net_id)
        diagnostics.update(net_id=net_id, pass_number=self._pass)
        if not self._done and not self._eligible:
            self._end_pass()
        self._check_budget()
        return StepResult(self.observation(), diagnostics, self._done)

    def best_solution(self) -> Submission | None:
        self._require_reset()
        return deepcopy(self._best)

    def close(self):
        if self._worker is not None:
            if not self._done:
                self._finish("closed")
            else:
                self._worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
