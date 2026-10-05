# Block 1: native single-net wire placement

Build with the current Python (C++17 compiler and Python headers required):

```sh
python3 -m pip install --target build/wire_deps pybind11==3.0.1
python3 scripts/build_wire_engine.py
python3 -m unittest tests.test_wire_engine -v
python3 examples/wire_engine_demo.py
```

The last command removes net 5 from official `case_01`'s reference layout,
holds every other route fixed, rebuilds that net, and checks the assembled
layout. Its uniform edge price of 0.25 is illustrative. **This is a reference
integration fixture, not an independently solved full case.** `--edge-price 0`
demonstrates the zero-price fast path; `--out /tmp/rebuilt-fixture.sol.json`
optionally serializes the assembled fixture using the repository's `Submission`.

```python
from m3d.model import Instance
from m3d.wire_engine import WireEngine

engine = WireEngine(Instance.load("benchmarks/case_01.json"))
result = engine.route_net(
    5, blocked_vertices=(), blocked_edges=(), edge_prices={},
    method="best", seed=0, max_expansions=1_000_000,
)
if result.status == "success":
    route = result.to_net_route()  # m3d.model.NetRoute; only this net
```

Callers must provide occupancy **excluding the selected net's old route**.
Other nets' pins are always forbidden, even when blockers are omitted. Vertices
accept `(x,y,z)` or `Grid`-compatible IDs `(z*height+y)*width+x`; edges are pairs
of endpoints. Prices map undirected edges to finite nonnegative numbers, with
missing prices zero. Reverse duplicate price entries are errors. Outputs use
sorted, canonical coordinate edges. The engine snapshots routing-relevant
instance data; no caller input or other route is mutated.

## Methods and objective

Both methods report
`J = sum_sink delay(driver,sink) + sum_unique_edge price(edge)`.
`total_delay` is an exact signed 64-bit integer and excludes prices;
`resource_cost` and `objective` are separate floating-point values. Routing
instances whose conservative total-delay bound exceeds int64 are rejected.

`shortest_path` runs one driver-rooted search with fixed costs `delay + price`.
Every sink path comes from the same predecessor tree. Its search costs charge
prices along each sink path; the emitted tree is rescored with prices charged
only once per unique edge. With zero prices, integer Dijkstra attains each
sink's shortest delay under the fixed blockers, minimizing their sum.

`cost_distance` adapts [Held and Perner, Algorithm 1, equations (4)–(5)](https://arxiv.org/html/2503.04419v1):

1. Start with separate destination components, each of weight one, and the root.
2. Select the minimum component-pair distance under `c + min(wu,wv)*d`, or
   `c + wu*d` for a root connection. Searches from each active representative
   using `c + wu*d`, stopped at another representative/root, realize this
   minimum over directed pairs.
3. Add the selected physical path. For two destination components, sum their
   weights and retain a representative with probability proportional to its
   weight, using seeded `mt19937_64` and unbiased bounded sampling. Root merges
   keep the original driver position and retire that destination component.
4. Repeat until every destination component has joined the root.

Here `c` is the caller's edge price, `d` is the instance delay, and all electrical
bifurcation penalties are zero. We implement the base construction, without
Section III's component discounts, two-level heaps, A*, or relocated Steiner
points. For simplicity, nearest-component searches are recomputed after each
merge using reusable buffers; unlike the paper's retained-label implementation,
this can require O(k²) searches for k sinks.

Physical paths can overlap and form cycles. We extract an integer-delay shortest
tree rooted at the driver **within their union**, prune branches unused by any
sink, and recompute J. Unique physical-edge charging and this extraction are
benchmark adaptations; we do not assert the paper's approximation guarantee for
the resulting implementation. Merge diagnostics expose IDs, weights, distance,
and chosen representative for comparison and verification.

## Work budget and results

An expansion is one settled search label, including union-to-tree extraction.
In priced `best` mode, the shortest-path candidate can use at most half the
budget. Cost-Distance can use the remaining shared budget, including the first
candidate's unused allowance. Select the complete candidate with lower J;
ties favor shortest-path. With zero prices, `best` runs only shortest-path.
Explicit `cost_distance` remains available even at zero prices for comparison.

`status` is `success`, `unreachable`, or `budget_exhausted`. Failed results have
no edges and `None` metrics. A valid incumbent survives another candidate's
budget exhaustion: status stays `success`, while `budget_exhausted=True` and
per-candidate statuses disclose the truncated comparison. No partial union is
returned as success. A budget can prevent either/both methods from finishing.

`elapsed_s` covers validation, conversion, **all** candidates, extraction,
metrics, and conversion back to Python; it excludes engine construction and
the external checker. Routes, metrics, merge choices and expansion counts are
repeatable for fixed inputs/seed/budget on the same build; wall time varies.
The C++ search releases the GIL and serializes calls on an engine's reusable
buffers. Memory is O(grid vertices) plus the candidate paths, with implicit
neighbors. Fractional price comparisons use native floating point, so tied
choices need not be identical across architectures.

Only the compact test file is intended for this block. It uses a coordinate-based
Bellman–Ford oracle, hand-derived costs, and the unchanged official checker.
No scheduler or multi-net optimization is included. The minimal build script
supports macOS/Linux; rebuild the ignored extension after changing Python ABI.
