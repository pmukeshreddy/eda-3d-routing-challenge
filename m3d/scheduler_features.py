"""Versioned, metadata-free current-state features for the net scheduler.

The only dense interaction structure is an N-net by N-net adjacency matrix.
Grid-resource summaries iterate the sparse dictionaries in an observation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch


FEATURE_SCHEMA_VERSION = "scheduler-current-state-v1"
NET_FEATURE_NAMES = (
    "routed", "missing", "eligible",
    "driver_x", "driver_y", "driver_z", "sink_count",
    "sink_mean_x", "sink_mean_y", "sink_mean_z",
    "sink_min_x", "sink_min_y", "sink_min_z",
    "sink_max_x", "sink_max_y", "sink_max_z",
    "sink_std_x", "sink_std_y", "sink_std_z",
    "bbox_min_x", "bbox_min_y", "bbox_min_z",
    "bbox_max_x", "bbox_max_y", "bbox_max_z",
    "bbox_dx", "bbox_dy", "bbox_dz", "bbox_volume", "bbox_hpwl",
    "cross_die", "opposite_die_sink_fraction", "top_die_pin_fraction",
    "driver_sink_mean_distance", "driver_sink_max_distance",
    "route_delay", "route_search_cost", "route_edge_count", "route_vertex_count",
    "route_via_count", "route_planar_count",
    "conflict_vertices", "conflict_edges", "conflict_neighbors",
    "route_vertex_overuse", "route_edge_overuse",
    "route_vertex_history", "route_vertex_history_max",
    "route_edge_history", "route_edge_history_max",
    "bbox_vertex_occupancy", "bbox_edge_occupancy",
    "bbox_vertex_load", "bbox_edge_load",
    "bbox_vertex_conflicts", "bbox_edge_conflicts",
    "bbox_vertex_history", "bbox_edge_history",
    "bbox_vertex_history_max", "bbox_edge_history_max",
)
GLOBAL_FEATURE_NAMES = (
    "width", "height", "layers", "net_count", "pin_count",
    "routed_fraction", "missing_fraction", "eligible_fraction",
    "conflict_vertex_density", "conflict_edge_density", "current_routed_delay",
    "pass_number", "passes_completed", "pass_progress", "remaining_pass_fraction",
    "present_factor", "present_next_factor",
    "max_seconds", "max_passes", "max_calls", "max_expansions",
    "max_expansions_per_call", "max_call_seconds",
    "calls_used_fraction", "remaining_call_fraction",
    "expansions_used_fraction", "remaining_expansion_fraction",
    "layer_delay_min", "layer_delay_mean", "layer_delay_max", "via_delay",
    "vertex_occupancy", "edge_occupancy", "vertex_load", "edge_load",
    "vertex_history", "edge_history",
)
# Short aliases are part of the checkpoint metadata contract.
NET_FEATURES = NET_FEATURE_NAMES
GLOBAL_FEATURES = GLOBAL_FEATURE_NAMES
NET_FEATURE_DIM = len(NET_FEATURE_NAMES)
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURE_NAMES)


@dataclass(frozen=True)
class EncodedState:
    net_ids: tuple[int, ...]
    net_features: torch.Tensor
    global_features: torch.Tensor
    adjacency: torch.Tensor
    eligible: torch.Tensor


def _bounded(value: float, scale: float = 1.0) -> float:
    """Fixed, finite nonnegative compression; no fitted dataset statistics."""
    value = max(0.0, float(value))
    if not math.isfinite(value):
        raise ValueError("scheduler features require finite current-state values")
    return value / (max(float(scale), 1.0) + value)


def _fraction(value: float, total: float) -> float:
    return min(1.0, max(0.0, float(value) / max(1.0, float(total))))


def encode_observation(observation: Mapping) -> EncodedState:
    """Encode a RoutingEnv observation without reading labels or metadata.

    Token order follows instance.nets. IDs are used only as dictionary keys and
    action labels; renaming or permuting them never changes a physical feature.
    Wall-clock budget consumption is deliberately excluded.
    """
    instance = observation["instance"]
    width, height, layers = (int(v) for v in observation["grid"])
    dims = (width, height, layers)
    if min(dims) < 1:
        raise ValueError("grid dimensions must be positive")
    axes = tuple(max(1, d - 1) for d in dims)
    span = max(1, sum(d - 1 for d in dims))
    volume = width * height * layers
    edge_capacity = ((width - 1) * height * layers
                     + width * (height - 1) * layers
                     + width * height * (layers - 1))
    delay_scale = span * max(max(instance.layer_delay), instance.via_delay)
    nets = instance.nets
    ids = tuple(net.id for net in nets)
    if any(type(nid) is not int for nid in ids) or len(set(ids)) != len(ids):
        raise ValueError("net IDs must be unique Python integers")
    net_count = len(ids)
    id_index = {nid: i for i, nid in enumerate(ids)}
    pins = {pin.id: pin for pin in instance.pins}
    eligible = set(observation["eligible_net_ids"])
    missing = set(observation["missing_net_ids"])
    if not eligible <= id_index.keys() or not missing <= id_index.keys():
        raise ValueError("eligible/missing IDs must identify an observation net")
    routes = observation["routes"]
    vertex_owners, edge_owners = observation["vertex_owners"], observation["edge_owners"]
    conflicts_v, conflicts_e = observation["conflict_vertices"], observation["conflict_edges"]
    history_v, history_e = observation["history_vertex"], observation["history_edge"]

    def unpack(vertex):
        return (vertex % width, (vertex // width) % height, vertex // (width * height))

    def pack(vertex):
        x, y, z = vertex
        return (z * height + y) * width + x

    # Unpack each sparse resource only once, then summarize its bbox membership.
    vertex_resources = [(unpack(v), len(vertex_owners.get(v, ())),
                         max(0.0, float(history_v.get(v, 0.0))))
                        for v in sorted(vertex_owners.keys() | history_v.keys())]
    edge_resources = [(unpack(a), unpack(b), len(edge_owners.get((a, b), ())),
                       max(0.0, float(history_e.get((a, b), 0.0))))
                      for a, b in sorted(edge_owners.keys() | history_e.keys())]
    conflict_counts_v = dict.fromkeys(ids, 0)
    conflict_counts_e = dict.fromkeys(ids, 0)
    conflict_neighbors = {nid: set() for nid in ids}
    adjacency = torch.zeros((net_count, net_count), dtype=torch.bool)
    for resources, counts in ((conflicts_v, conflict_counts_v), (conflicts_e, conflict_counts_e)):
        for owners in resources.values():
            known = [nid for nid in owners if nid in id_index]
            for nid in known:
                counts[nid] += 1
                conflict_neighbors[nid].update(other for other in known if other != nid)
                for other in known:
                    if other != nid:
                        adjacency[id_index[nid], id_index[other]] = True

    rows, boxes = [], []
    for net in nets:
        nid = net.id
        driver_pin = pins[net.driver]
        sink_pins = [pins[pid] for pid in net.sinks]
        driver = driver_pin.vertex()
        sinks = [pin.vertex() for pin in sink_pins]
        points = [driver] + sinks
        low = tuple(min(point[i] for point in points) for i in range(3))
        high = tuple(max(point[i] for point in points) for i in range(3))
        extent = tuple(high[i] - low[i] for i in range(3))
        boxes.append((low, high))
        # A one-vertex halo samples competition immediately outside a pin bbox.
        region_low = tuple(max(0, low[i] - 1) for i in range(3))
        region_high = tuple(min(dims[i] - 1, high[i] + 1) for i in range(3))
        region_dims = tuple(region_high[i] - region_low[i] + 1 for i in range(3))
        region_volume = math.prod(region_dims)
        rx, ry, rz = region_dims
        region_edges = (rx - 1) * ry * rz + rx * (ry - 1) * rz + rx * ry * (rz - 1)

        def inside(point):
            return all(region_low[i] <= point[i] <= region_high[i] for i in range(3))

        regional_v = [(load, history) for point, load, history in vertex_resources if inside(point)]
        regional_e = [(load, history) for a, b, load, history in edge_resources
                      if inside(a) and inside(b)]
        # Empty sinks are supported for structural robustness, though the engine
        # requires at least one sink on nonempty challenge nets.
        sink_points = sinks or [driver]
        mean = tuple(sum(point[i] for point in sink_points) / len(sink_points) for i in range(3))
        std = tuple(math.sqrt(sum((point[i] - mean[i]) ** 2 for point in sink_points)
                              / len(sink_points)) for i in range(3))
        distances = [sum(abs(point[i] - driver[i]) for i in range(3)) for point in sink_points]
        edges = {tuple(sorted((pack(a), pack(b)))) for a, b in routes.get(nid, ())}
        vertices = ({v for edge in edges for v in edge} | {pack(point) for point in points}
                    if nid in routes else set())
        via_count = sum(a // (width * height) != b // (width * height) for a, b in edges)
        route_history_v = [float(history_v.get(v, 0.0)) for v in sorted(vertices)]
        route_history_e = [float(history_e.get(e, 0.0)) for e in sorted(edges)]

        row = [float(nid in routes), float(nid in missing), float(nid in eligible)]
        row.extend(driver[i] / axes[i] for i in range(3))
        row.append(_bounded(len(sinks), 8))
        for values in (mean, tuple(min(p[i] for p in sink_points) for i in range(3)),
                       tuple(max(p[i] for p in sink_points) for i in range(3)), std, low, high, extent):
            row.extend(values[i] / axes[i] for i in range(3))
        row.extend((math.prod(e + 1 for e in extent) / volume, sum(extent) / span,
                    float(any(pin.die != driver_pin.die for pin in sink_pins)),
                    _fraction(sum(pin.die != driver_pin.die for pin in sink_pins), len(sinks)),
                    _fraction(sum(pin.die == 1 for pin in [driver_pin] + sink_pins), len(points)),
                    sum(distances) / len(distances) / span, max(distances) / span,
                    _bounded(observation["delays"].get(nid, 0), delay_scale),
                    _bounded(observation["route_search_costs"].get(nid, 0), delay_scale),
                    _bounded(len(edges), span), _bounded(len(vertices), span),
                    _bounded(via_count, span), _bounded(len(edges) - via_count, span),
                    _bounded(conflict_counts_v[nid], span), _bounded(conflict_counts_e[nid], span),
                    _fraction(len(conflict_neighbors[nid]), net_count - 1),
                    _bounded(sum(max(0, len(vertex_owners.get(v, ())) - 1) for v in sorted(vertices)), span),
                    _bounded(sum(max(0, len(edge_owners.get(e, ())) - 1) for e in sorted(edges)), span),
                    _bounded(sum(route_history_v) / max(1, len(vertices))),
                    _bounded(max(route_history_v, default=0)),
                    _bounded(sum(route_history_e) / max(1, len(edges))),
                    _bounded(max(route_history_e, default=0)),
                    _fraction(sum(load > 0 for load, _ in regional_v), region_volume),
                    _fraction(sum(load > 0 for load, _ in regional_e), region_edges),
                    _bounded(sum(load for load, _ in regional_v), region_volume),
                    _bounded(sum(load for load, _ in regional_e), region_edges),
                    _fraction(sum(load > 1 for load, _ in regional_v), region_volume),
                    _fraction(sum(load > 1 for load, _ in regional_e), region_edges),
                    _bounded(sum(history for _, history in regional_v), region_volume),
                    _bounded(sum(history for _, history in regional_e), region_edges),
                    _bounded(max((history for _, history in regional_v), default=0)),
                    _bounded(max((history for _, history in regional_e), default=0))))
        rows.append(row)

    for i, (low_i, high_i) in enumerate(boxes):
        for j in range(i):
            low_j, high_j = boxes[j]
            gap = sum(max(0, low_i[k] - high_j[k], low_j[k] - high_i[k]) for k in range(3))
            if gap <= 2:
                adjacency[i, j] = adjacency[j, i] = True

    budget = observation["budget"]
    limits = budget["limits"]
    maximum_passes = limits["max_passes"]
    present = observation["present_factor"]
    global_values = [
        _bounded(width, 64), _bounded(height, 64), _bounded(layers, 16),
        _bounded(net_count, 128), _bounded(len(pins), 256),
        _fraction(len(routes), net_count), _fraction(len(missing), net_count),
        _fraction(len(eligible), net_count), _fraction(len(conflicts_v), volume),
        _fraction(len(conflicts_e), edge_capacity),
        _bounded(observation["current_routed_delay"], delay_scale * max(1, net_count)),
        _bounded(observation["pass_number"], 50), _bounded(observation["passes_completed"], 50),
        _fraction(observation["passes_completed"], maximum_passes),
        _fraction(budget["remaining_passes"], maximum_passes),
        _bounded(present), _bounded(min(present * 1.7, 1e6)),
        _bounded(limits["max_seconds"], 180), _bounded(maximum_passes, 50),
        _bounded(limits["max_calls"], 5000), _bounded(limits["max_expansions"], 200_000_000),
        _bounded(limits["max_expansions_per_call"], 1_000_000),
        _bounded(limits["max_call_seconds"], 10),
        _fraction(budget["engine_calls"], limits["max_calls"]),
        _fraction(budget["remaining_calls"], limits["max_calls"]),
        _fraction(budget["expansions_charged"], limits["max_expansions"]),
        _fraction(budget["remaining_expansions"], limits["max_expansions"]),
        _bounded(min(instance.layer_delay), 16),
        _bounded(sum(instance.layer_delay) / len(instance.layer_delay), 16),
        _bounded(max(instance.layer_delay), 16), _bounded(instance.via_delay, 16),
        _fraction(len(vertex_owners), volume), _fraction(len(edge_owners), edge_capacity),
        _bounded(sum(len(owners) for owners in vertex_owners.values()), volume),
        _bounded(sum(len(owners) for owners in edge_owners.values()), edge_capacity),
        _bounded(sum(value for _, _, value in vertex_resources), volume),
        _bounded(sum(value for _, _, _, value in edge_resources), edge_capacity),
    ]
    features = torch.tensor(rows, dtype=torch.float32).reshape(net_count, NET_FEATURE_DIM)
    globals_tensor = torch.tensor(global_values, dtype=torch.float32)
    if not bool(torch.isfinite(features).all() and torch.isfinite(globals_tensor).all()):
        raise ValueError("scheduler features must be finite")
    return EncodedState(ids, features, globals_tensor, adjacency,
                        torch.tensor([nid in eligible for nid in ids], dtype=torch.bool))


def batch_states(states: Sequence[EncodedState], device: str | torch.device = "cpu") -> dict:
    """Pad CPU encodings and transfer every tensor to the requested device.

    padding_mask is True only for padded tokens. IDs stay Python tuples and are
    never copied into a numeric model input. Eligibility is separate from padding.
    """
    if not states:
        raise ValueError("cannot batch an empty list of states")
    size, max_nets = len(states), max(len(state.net_ids) for state in states)
    features = torch.zeros((size, max_nets, NET_FEATURE_DIM), dtype=torch.float32)
    globals_tensor = torch.stack([state.global_features for state in states])
    adjacency = torch.zeros((size, max_nets, max_nets), dtype=torch.bool)
    eligible = torch.zeros((size, max_nets), dtype=torch.bool)
    padding = torch.ones((size, max_nets), dtype=torch.bool)
    for row, state in enumerate(states):
        count = len(state.net_ids)
        features[row, :count] = state.net_features
        adjacency[row, :count, :count] = state.adjacency.bool()
        eligible[row, :count] = state.eligible
        padding[row, :count] = False
    return {"net_features": features.to(device), "global_features": globals_tensor.to(device),
            "adjacency": adjacency.to(device), "eligible": eligible.to(device),
            "padding_mask": padding.to(device), "net_ids": [state.net_ids for state in states]}
