"""Block 2: transactional, budgeted negotiated routing around Block 1.

An action selects one entire net/connection, including all of its sinks.
The environment never selects another net or imports reference wires.
See docs/ROUTING_ENV.md for observations, pass semantics and price mapping.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
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

    def _prices(self):
        prices = {}
        # Native search already adds the base delay. Only send the surcharge.
        for a, b in self._h_e.keys() | self._owners_e.keys():
            delay = (self._grid.via_delay if a // self._grid.wh != b // self._grid.wh
                     else self._grid.layer_delay[a // self._grid.wh])
            prices[a, b] = delay * (self._h_e.get((a, b), 0.0)
                                    + self._present * len(self._owners_e.get((a, b), ())))
        # Undirected edge API: split each vertex's pressure equally onto every
        # incident edge. A path's internal vertex is then charged exactly once.
        for v in self._h_v.keys() | self._owners_v.keys():
            pressure = 0.5 * VCONG * (self._h_v.get(v, 0.0)
                                       + self._present * len(self._owners_v.get(v, ())))
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

    def _end_pass(self):
        self._passes_completed += 1
        conflicts_v, conflicts_e = self._conflicts()
        missing = self._nets.keys() - self._routes.keys()
        if not missing and not conflicts_v and not conflicts_e:
            self._finish("legal" if self._best is not None else "checker_rejected")
            return
        self._check_budget()
        if self._done:
            return
        if self._passes_completed >= self._budget.max_passes:
            self._finish("pass_budget")
            return
        affected = set(missing)
        for history, conflicts in ((self._h_v, conflicts_v), (self._h_e, conflicts_e)):
            for resource, owners in conflicts.items():
                history[resource] = history.get(resource, 0.0) + 0.5 * (len(owners) - 1)
                affected.update(owners)
        self._present = min(self._present * 1.7, 1e6)
        self._eligible = affected
        self._pass += 1

    def step(self, net_id: int) -> StepResult:
        self._require_reset()
        # Validate before touching even pass/termination counters. In particular,
        # bools/floats must not alias integer IDs in Python's set membership.
        if type(net_id) is not int or net_id not in self._eligible or self._done:
            raise ValueError(f"net is not eligible: {net_id!r}")
        self._check_budget()
        if self._done:
            return StepResult(self.observation(), {"net_id": net_id, "engine_status": "not_called"}, True)
        old = self._remove(net_id) if net_id in self._routes else None
        installed = False
        diagnostics = {"net_id": net_id, "pass_number": self._pass, "rolled_back": False}
        allowance = min(self._budget.max_expansions_per_call,
                        self._budget.max_expansions - self._expansions)
        called = False
        self._in_step = True
        try:
            prices = self._prices()
            if self._remaining_seconds() <= 0:
                raise TimeoutError("episode expired while constructing search prices")
            self._calls += 1
            called = True
            # Reserve up front: a killed worker has no trustworthy final count.
            self._expansions += allowance
            result = self._worker.route_net(
                net_id, timeout_s=min(self._budget.max_call_seconds, self._remaining_seconds()),
                blocked_vertices=tuple(v for v, owner in self._pin_owner.items() if owner != net_id),
                blocked_edges=(), edge_prices=prices, method=self.method,
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
                route = _Route(tuple(result.edges), vertices, edges, result.total_delay, result.resource_cost)
                self._install(net_id, route)
                installed = True
        except (TimeoutError, RuntimeError, ValueError, OverflowError) as exc:
            diagnostics.update(engine_status="timeout" if isinstance(exc, TimeoutError) else "error",
                               error=str(exc), expansions_charged=allowance if called else 0)
        finally:
            if not installed and old is not None:
                self._install(net_id, old)
                diagnostics["rolled_back"] = True
            self._in_step = False
        diagnostics["route_installed"] = installed
        self._eligible.remove(net_id)
        if installed:
            self._consider_snapshot()
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
