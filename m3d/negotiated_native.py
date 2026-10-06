"""Private C++17 backend for delay optimization.

Compilation is cached only for this Python process in an automatically removed
private temporary directory. Only absence of a compiler selects the portable
Python fallback; compilation, worker and correctness failures are surfaced.
The worker starts its monotonic search deadline after input/graph setup. Python
serialization, process startup, verification, and output are timed but excluded
from that deadline. A deadline is checked every 256 heap pops and between nets;
validation and snapshot copies can cause small wall-clock overruns.
"""
from __future__ import annotations

import atexit
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

from .checker import check
from .grid import Grid
from .model import NetRoute, Submission

_BINARY = None
_TEMP = None
_LOCK = threading.Lock()


class NativeUnavailable(Exception):
    """No supported compiler is installed; use the labeled Python fallback."""


def _binary():
    global _BINARY, _TEMP
    with _LOCK:
        if _BINARY is not None:
            return _BINARY
        compiler = shutil.which('clang++') or shutil.which('g++')
        if compiler is None:
            raise NativeUnavailable('neither clang++ nor g++ is available')
        directory = tempfile.TemporaryDirectory(prefix='m3d-negotiated-')
        executable = Path(directory.name) / 'negotiated-opt'
        source = Path(__file__).with_name('native') / 'negotiated_opt.cpp'
        result = subprocess.run([compiler, '-std=c++17', '-O3', '-DNDEBUG',
                                 str(source), '-o', str(executable)],
                                capture_output=True, text=True)
        if result.returncode:
            directory.cleanup()
            raise RuntimeError(f'native optimizer compilation failed: {result.stderr}')
        _TEMP, _BINARY = directory, str(executable)
        atexit.register(directory.cleanup)
        return _BINARY


def _encode_input(inst, initial, time_budget, seed, work_limit):
    grid = Grid(inst)
    pins = inst.pin_vertex()
    routes = {r.net: r for r in initial.routes}
    lines = [f'{inst.width} {inst.height} {inst.layers} {inst.via_delay} '
             f'{len(inst.nets)} {seed & 0xffffffff} {time_budget:.17g} {work_limit}',
             ' '.join(map(str, inst.layer_delay))]
    for net in inst.nets:
        vertices = [grid.vid(pins[p]) for p in net.pins()]
        edges = routes[net.id].edges
        lines.append(f'{net.id} {len(vertices)} ' + ' '.join(map(str, vertices)))
        lines.append(str(len(edges)))
        lines.extend(f'{grid.vid(a)} {grid.vid(b)}' for a, b in edges)
    return '\n'.join(lines) + '\n'


def _decode_output(inst, initial, output):
    """Treat worker output as untrusted, checking schema, legality and objective."""
    try:
        data = json.loads(output)
        if not isinstance(data, dict) or type(data.get('delay')) is not int:
            raise ValueError('missing integer delay')
        if not isinstance(data.get('stats'), dict) or not isinstance(data.get('routes'), list):
            raise ValueError('missing stats or routes')
        grid = Grid(inst)
        routes = []
        for route in data['routes']:
            if not isinstance(route, dict) or type(route.get('net')) is not int:
                raise ValueError('invalid route')
            if not isinstance(route.get('edges'), list):
                raise ValueError('invalid edges')
            edges = []
            for edge in route['edges']:
                if (not isinstance(edge, list) or len(edge) != 2
                        or any(type(v) is not int or not 0 <= v < grid.wh * grid.l for v in edge)):
                    raise ValueError('invalid vertex id')
                edges.append((grid.coord(edge[0]), grid.coord(edge[1])))
            routes.append(NetRoute(route['net'], edges))
        candidate = Submission(inst.name, routes)
        evaluated = check(inst, candidate)
        if not evaluated.legal:
            raise ValueError(f'illegal native solution: {evaluated.reasons}')
        if data['delay'] != evaluated.total_delay:
            raise ValueError('native delay disagrees with independent checker')
        if evaluated.total_delay > check(inst, initial).total_delay:
            raise ValueError('native best is worse than initial solution')
        return candidate, data['stats'], evaluated.total_delay
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise RuntimeError(f'invalid native optimizer output: {exc}') from exc


def optimize_native(inst, initial, time_budget, seed, work_limit):
    from .negotiated_opt import OptStats
    started = time.monotonic()
    evaluated = check(inst, initial)
    if not evaluated.legal:
        raise ValueError('delay optimization requires a legal initial solution')
    setup_started = time.monotonic()
    binary = _binary()
    payload = _encode_input(inst, initial, time_budget, seed, work_limit)
    setup_seconds = time.monotonic() - setup_started
    worker_started = time.monotonic()
    process = subprocess.run([binary], input=payload, capture_output=True, text=True)
    worker_seconds = time.monotonic() - worker_started
    if process.returncode:
        raise RuntimeError(f'native optimizer failed ({process.returncode}): {process.stderr}')
    candidate, counters, final_delay = _decode_output(inst, initial, process.stdout)
    stats = OptStats(success=True, seed=seed, time_budget_s=time_budget,
                     baseline_delay=evaluated.total_delay, optimized_delay=final_delay,
                     backend='native_cpp17', backend_seconds=worker_seconds)
    allowed = {'attempted_groups', 'reroute_orders', 'completed_orders', 'timed_out_orders',
               'bounded_orders', 'accepted_improvements', 'accepted_equal', 'accepted_worse',
               'legal_candidates', 'single_net_improvements', 'single_net_gain',
               'work_steps', 'expansions', 'stop_reason', 'native_search_seconds',
               'native_setup_seconds'}
    if set(counters) != allowed:
        raise RuntimeError('invalid native optimizer statistics schema')
    for name, value in counters.items():
        if name == 'stop_reason':
            if value not in {'work_limit', 'time_budget', 'delay_lower_bound_reached'}:
                raise RuntimeError('invalid native stop reason')
        elif name.endswith('_seconds'):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise RuntimeError('invalid native timing')
        elif type(value) is not int or value < 0:
            raise RuntimeError('invalid native counter')
        setattr(stats, name, value)
    stats.backend_setup_seconds = setup_seconds + stats.native_setup_seconds
    stats.optimization_seconds = time.monotonic() - started
    # Equal best snapshots preserve the original serialization and object.
    return (initial if final_delay == evaluated.total_delay else candidate), stats
