"""Negotiated-congestion router (PathFinder-style), used for the harder tiers.

Routes every net as a minimum-cost tree, allows temporary resource overuse, and
raises a per-resource history penalty on anything overused so nets negotiate away
from contention. After the first pass it only rips up and reroutes the nets that
currently touch an overused resource (the standard PathFinder speedup), so it is
fast even on contended instances. It stops when no 3D vertex or edge is shared by
more than one net (a legal capacity-1 routing) or when iterations run out.

Cost model (uncongested = exactly the true delay, so it reduces to per-net
shortest paths on sparse instances and preserves the delay objective):

    edge_cost(e)   = edge_delay(e) * (1 + h_e[e]  + pres_fac * pres_e[e])
    vertex_cost(v) = VCONG        * (h_v[v] + pres_fac * pres_v[v])   # 0 uncongested

``pres_*`` = number of other nets on the resource; ``h_*`` = accumulated history.
Deterministic. Certifies feasibility of the dense tier and provides its baseline.
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .grid import Grid, edge_key
from .model import Instance, NetRoute, Submission

VCONG = 1.0


@dataclass
class NegStats:
    iterations: int
    overused: int
    success: bool


class Negotiated:
    def __init__(self, inst: Instance, max_iters: int = 50,
                 pres_fac0: float = 0.5, pres_mult: float = 1.7,
                 hist_fac: float = 0.5, order: str = "bbox_desc"):
        self.inst = inst
        self.g = Grid(inst)
        self.max_iters = max_iters
        self.pres_fac0 = pres_fac0
        self.pres_mult = pres_mult
        self.hist_fac = hist_fac
        self.order = order

        self.pin_vid = {p.id: self.g.vid(p.vertex()) for p in inst.pins}
        self.net_pins = {n.id: [self.pin_vid[p] for p in n.pins()] for n in inst.nets}
        self.pin_owner = {self.pin_vid[p]: n.id for n in inst.nets for p in n.pins()}

        self.h_v: Dict[int, float] = defaultdict(float)
        self.h_e: Dict[Tuple[int, int], float] = defaultdict(float)
        self.owners_v: Dict[int, Set[int]] = defaultdict(set)
        self.owners_e: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self.routes: Dict[int, Tuple[Set[int], Set[Tuple[int, int]]]] = {}

    def _order_nets(self) -> List[int]:
        def bbox(nid):
            cs = [self.g.coord(v) for v in self.net_pins[nid]]
            xs = [c[0] for c in cs]; ys = [c[1] for c in cs]; zs = [c[2] for c in cs]
            return (max(xs) - min(xs)) + (max(ys) - min(ys)) + (max(zs) - min(zs))
        nids = [n.id for n in self.inst.nets]
        nids.sort(key=lambda i: (-bbox(i), i)) if self.order == "bbox_desc" else nids.sort()
        return nids

    def _add(self, nid, tv, te):
        for v in tv:
            self.owners_v[v].add(nid)
        for e in te:
            self.owners_e[e].add(nid)
        self.routes[nid] = (tv, te)

    def _remove(self, nid):
        tv, te = self.routes.pop(nid)
        for v in tv:
            s = self.owners_v[v]; s.discard(nid)
            if not s:
                del self.owners_v[v]
        for e in te:
            s = self.owners_e[e]; s.discard(nid)
            if not s:
                del self.owners_e[e]

    def _route_net(self, nid, pres_fac):
        g = self.g
        pins = self.net_pins[nid]
        forbidden_pins = {v for v, o in self.pin_owner.items() if o != nid}
        tree_v: Set[int] = {pins[0]}
        tree_e: Set[Tuple[int, int]] = set()
        remaining = set(pins[1:])
        h_v, h_e, ov, oe = self.h_v, self.h_e, self.owners_v, self.owners_e
        while remaining:
            dist: Dict[int, float] = {s: 0.0 for s in tree_v}
            prev: Dict[int, int] = {}
            heap: List[Tuple[float, int]] = [(0.0, s) for s in tree_v]
            heapq.heapify(heap)
            found = None
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist.get(u, d):
                    continue
                if u in remaining:
                    found = u
                    break
                for v, w in g.neighbors(u):
                    if v in forbidden_pins or v in tree_v:
                        continue
                    ek = edge_key(u, v)
                    ec = w * (1.0 + h_e.get(ek, 0.0) + pres_fac * len(oe.get(ek, ())))
                    vc = VCONG * (h_v.get(v, 0.0) + pres_fac * len(ov.get(v, ())))
                    nd = d + ec + vc
                    if nd < dist.get(v, 1e30):
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(heap, (nd, v))
            if found is None:
                return None
            path = [found]
            cur = found
            while cur in prev:
                cur = prev[cur]
                path.append(cur)
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                tree_v.add(a); tree_v.add(b)
                tree_e.add(edge_key(a, b))
            remaining.discard(found)
        return tree_v, tree_e

    def _overused(self):
        ov = [v for v, o in self.owners_v.items() if len(o) > 1]
        oe = [e for e, o in self.owners_e.items() if len(o) > 1]
        return ov, oe

    def run(self) -> Tuple[Optional[Submission], NegStats]:
        order = self._order_nets()
        pres_fac = self.pres_fac0
        for nid in order:
            r = self._route_net(nid, pres_fac)
            if r is None:
                return None, NegStats(0, -1, False)
            self._add(nid, *r)

        it = 0
        while it < self.max_iters:
            it += 1
            ov, oe = self._overused()
            if not ov and not oe:
                break
            affected: Set[int] = set()
            for v in ov:
                self.h_v[v] += self.hist_fac * (len(self.owners_v[v]) - 1)
                affected |= self.owners_v[v]
            for e in oe:
                self.h_e[e] += self.hist_fac * (len(self.owners_e[e]) - 1)
                affected |= self.owners_e[e]
            pres_fac = min(pres_fac * self.pres_mult, 1e6)
            for nid in order:
                if nid not in affected:
                    continue
                self._remove(nid)
                r = self._route_net(nid, pres_fac)
                if r is None:
                    return None, NegStats(it, -1, False)
                self._add(nid, *r)

        ov, oe = self._overused()
        success = not ov and not oe
        stats = NegStats(it, len(ov) + len(oe), success)
        if not success:
            return None, stats
        routes = [NetRoute(net=n.id,
                           edges=[(self.g.coord(a), self.g.coord(b))
                                  for (a, b) in self.routes[n.id][1]])
                  for n in self.inst.nets]
        return Submission(instance=self.inst.name, routes=routes), stats


def route_negotiated(inst: Instance, **kw) -> Tuple[Optional[Submission], NegStats]:
    return Negotiated(inst, **kw).run()
