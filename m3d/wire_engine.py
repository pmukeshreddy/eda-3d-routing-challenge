"""Single-net native wire placement. See docs/WIRE_ENGINE.md for semantics."""
from dataclasses import dataclass
import math
from numbers import Real
from time import perf_counter
from typing import Optional, Tuple

from .model import Edge, Instance, NetRoute, canon_edge


def _integer(value, name, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class WireResult:
    net_id: int
    status: str
    edges: Tuple[Edge, ...]
    total_delay: Optional[int]
    resource_cost: Optional[float]
    objective: Optional[float]
    method: str
    expansions: int
    elapsed_s: float
    budget_exhausted: bool
    candidates: tuple
    merges: tuple

    def to_net_route(self):
        if self.status != "success":
            raise ValueError(f"No complete wire tree: {self.status}")
        return NetRoute(self.net_id, list(self.edges))


class WireEngine:
    """Snapshot an Instance; calls never modify it or caller-owned occupancy.

    Vertices accept (x,y,z) or packed Grid-compatible integer IDs. Edges are
    endpoint pairs. edge_prices is a mapping from undirected edges to prices;
    omitted prices are zero. Reversed duplicate price keys are rejected.
    """

    def __init__(self, instance: Instance):
        if not isinstance(instance, Instance):
            raise TypeError("instance must be an m3d.model.Instance")
        self.width = _integer(instance.width, "width", 1, 2**31 - 1)
        self.height = _integer(instance.height, "height", 1, 2**31 - 1)
        self.layers = _integer(instance.layers, "layers", 1, 2**31 - 1)
        self.size = self.width * self.height * self.layers
        if self.size > 2**31 - 1:
            raise ValueError("grid exceeds the native 32-bit vertex-ID range")
        if len(instance.layer_delay) != self.layers:
            raise ValueError("layer_delay must have exactly one value per layer")
        delays = [_integer(v, "layer delay", 1) for v in instance.layer_delay]
        via = _integer(instance.via_delay, "via delay", 1)
        self._max_delay = max(*delays, via)
        pins = {}
        vertices = set()
        for pin in instance.pins:
            pid = _integer(pin.id, "pin id")
            vid = self.vertex_id(pin.vertex())
            if pid in pins or vid in vertices:
                raise ValueError("pin IDs and pin vertices must be unique")
            if pin.die not in (0, 1) or type(pin.die) is not int:
                raise ValueError("pin die must be 0 or 1")
            if pin.z != (0 if pin.die == 0 else self.layers - 1):
                raise ValueError("pin layer does not match its die")
            pins[pid] = vid
            vertices.add(vid)
        self._nets = {}
        self._owners = {}
        used_pins = set()
        for net in instance.nets:
            nid = _integer(net.id, "net id")
            if nid in self._nets or not net.sinks:
                raise ValueError("net IDs must be unique and each net needs a sink")
            ids = [net.driver, *net.sinks]
            for pid in ids:
                _integer(pid, "net pin id")
                if pid not in pins or pid in used_pins:
                    raise ValueError("net contains an unknown, repeated, or shared pin")
                used_pins.add(pid)
                self._owners[pins[pid]] = nid
            self._nets[nid] = (pins[net.driver], tuple(pins[p] for p in net.sinks))
        if used_pins != set(pins):
            raise ValueError("every pin must belong to exactly one net")
        try:
            from ._wire_native import NativeEngine
        except ImportError as exc:
            raise ImportError("Build the native engine: python3 scripts/build_wire_engine.py") from exc
        self._native = NativeEngine(self.width, self.height, self.layers, delays, via)

    def vertex_id(self, vertex):
        if type(vertex) is int:
            return _integer(vertex, "vertex ID", 0, self.size - 1)
        if not isinstance(vertex, (tuple, list)) or len(vertex) != 3:
            raise ValueError("vertex must be an integer ID or three integer coordinates")
        x, y, z = vertex
        _integer(x, "x", 0, self.width - 1)
        _integer(y, "y", 0, self.height - 1)
        _integer(z, "z", 0, self.layers - 1)
        return (z * self.height + y) * self.width + x

    def vertex(self, vertex_id):
        vid = _integer(vertex_id, "vertex ID", 0, self.size - 1)
        z, remainder = divmod(vid, self.width * self.height)
        y, x = divmod(remainder, self.width)
        return x, y, z

    def _edge(self, edge):
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise ValueError("edge must contain two adjacent vertices")
        a, b = (self.vertex_id(v) for v in edge)
        if sum(abs(x - y) for x, y in zip(self.vertex(a), self.vertex(b))) != 1:
            raise ValueError("edge must be one adjacent axis-aligned grid step")
        return min(a, b), max(a, b)

    def route_net(self, net_id, blocked_vertices=(), blocked_edges=(),
                  edge_prices=None, method="best", seed=0, max_expansions=1_000_000):
        start = perf_counter()
        _integer(net_id, "net_id")
        if net_id not in self._nets:
            raise ValueError(f"unknown net_id: {net_id}")
        if method not in ("shortest_path", "cost_distance", "best"):
            raise ValueError("method must be shortest_path, cost_distance, or best")
        _integer(seed, "seed", 0, 2**64 - 1)
        _integer(max_expansions, "max_expansions", 0, 2**64 - 1)
        source, sinks = self._nets[net_id]
        # Conservative bound makes every possible tree-delay sum fit int64.
        if (self.size - 1) * self._max_delay * len(sinks) >= 2**63 - 1:
            raise OverflowError("possible routing delay exceeds the int64 range")
        blocked = {self.vertex_id(v) for v in blocked_vertices}
        blocked.update(v for v, owner in self._owners.items() if owner != net_id)
        edges = sorted({self._edge(e) for e in blocked_edges})
        prices = {}
        if edge_prices is not None:
            if not hasattr(edge_prices, "items"):
                raise TypeError("edge_prices must be a mapping")
            for edge, value in edge_prices.items():
                key = self._edge(edge)
                if key in prices:
                    raise ValueError("duplicate undirected edge price")
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError("edge prices must be finite nonnegative numbers")
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("edge prices must be finite and nonnegative")
                prices[key] = value
        raw = self._native.route(source, list(sinks), sorted(blocked), edges,
                                 [(a, b, c) for (a, b), c in sorted(prices.items())],
                                 method, seed, max_expansions)
        wire = tuple(sorted(canon_edge(self.vertex(a), self.vertex(b)) for a, b in raw["edges"]))
        return WireResult(net_id, raw["status"], wire, raw["total_delay"],
                          raw["resource_cost"], raw["objective"], raw["method"],
                          raw["expansions"], perf_counter() - start,
                          raw["budget_exhausted"], tuple(raw["candidates"]), tuple(raw["merges"]))
