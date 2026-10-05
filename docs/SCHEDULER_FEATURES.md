# Scheduler current-state schema v1

`FEATURE_SCHEMA_VERSION = "scheduler-current-state-v1"`. The encoder consumes
the detached dictionary returned by `RoutingEnv.observation()`. It returns an
`EncodedState` with Python integer `net_ids`, CPU float32 `net_features [N,60]`
and `global_features [37]`, CPU bool `adjacency [N,N]`, and CPU bool
`eligible [N]`. Token order follows `instance.nets`; IDs are action labels and
dictionary keys only. `NET_FEATURES` / `GLOBAL_FEATURES` export the exact ordered
feature-name tuples (also available as `*_FEATURE_NAMES`).

Normalization uses only current physical geometry and fixed constants. Let
`S = max(1, width + height + layers - 3)`, `V = width*height*layers`, `E` be the
number of possible undirected lattice edges, and
`D = S * max(max(layer_delay), via_delay)`. A bounded value with scale `s` is
`x / (max(1,s) + x)` for nonnegative `x`; fractions are clamped to `[0,1]` with
denominator at least one. Coordinates and coordinate differences divide by
`max(1, axis_length-1)`. No fitted normalization statistics are used.

| Net features, in tuple order | Encoding |
| --- | --- |
| routed, missing, eligible | Current membership flags |
| driver xyz, sink count | Normalized coordinates; count bounded at scale 8 |
| sink mean/min/max/std xyz | Normalized per-axis sink summaries |
| bbox min/max xyz, dimensions xyz | Pin bbox over driver and all sinks |
| bbox volume, HPWL | Inclusive bbox volume / V; dimension sum / S |
| cross die, opposite-die sink fraction, top-die pin fraction | Current pin die assignments |
| mean/max driver-to-sink distance | Manhattan distance / S |
| route delay, route search cost | Installed route values bounded at scale D; zero when absent |
| route edge/vertex/via/planar counts | Counts bounded at scale S; unique resources, pins included in routed vertex count |
| conflict vertices/edges/neighbors | Resource counts bounded at scale S; other conflicting nets / max(1,N-1) |
| route vertex/edge overuse | Sum of excess owners over installed resources, bounded at scale S |
| route vertex history mean/max, edge history mean/max | Means include zero-history installed resources; bounded at scale 1 |
| bbox vertex/edge occupancy, load, conflicts, history, history max | Sparse summaries in the pin bbox plus a one-vertex halo, clipped to the grid |

For bbox summaries, occupancy and conflicts divide by regional vertex/edge
capacity. Load and summed history use that capacity as their bounded scale;
history maxima use scale 1. An edge is regional only when both endpoints are
inside the region. Resource counts include all current owners, including the
net itself; no speculative reroute is performed.

The 37 globals contain:

- Width/height/layers (bounded scales 64/64/16), net/pin counts (128/256),
  routed/missing/eligible fractions, conflict vertex/edge densities, and current
  total routed delay (bounded scale D*max(1,N)).
- Pass number and completed passes (scale 50), completed/remaining pass
  fractions, present factor and its next scheduled value `min(1.7*factor,1e6)`
  (scale 1).
- Configured maximum seconds/passes/calls/expansions/per-call expansions/
  per-call seconds (scales 180/50/5000/200000000/1000000/10), then used/remaining
  call fractions and used/remaining expansion fractions.
- Minimum/mean/maximum layer delay and via delay (scale 16), global
  vertex/edge occupancy fractions, and vertex/edge load and summed history
  (bounded by V/E).

Elapsed and remaining wall-clock time are excluded to avoid scheduling jitter.
The encoder does not read instance name, params, seeds, split labels, reference
routes or delays, certificates, checker reports, best-so-far delay, terminal
status, or future rollout results. Geometry, physical delays, and current
routing state are the only content sources.

Adjacency is symmetric, has no self loops, and connects pin bboxes whose
minimum Manhattan separation is at most two lattice steps, plus every pair
sharing a current conflict vertex or edge. Construction uses sparse resource
maps and an N-by-N matrix; it never allocates a dense 3D routing grid.

`batch_states(states, device)` pads to the largest N, transfers all tensors to
the requested device, and retains IDs as a list of tuples. `padding_mask` is
True for padding; eligibility is independent. The default policy has a per-net
MLP, global conditioning, two residual mean-neighbor graph layers, four
Transformer layers (width 192, six heads, feed-forward width 1024, zero dropout),
and one scalar head per net: **2,453,441 trainable parameters**. No positional or
numeric ID embedding is used. Padded tokens never contribute to graph means or
attention; their returned logits are zero. Real-token `forward` logits remain
raw and finite so training can mask eligibility externally.

`distribution(observation_or_encoded)` masks ineligible logits with negative
infinity and returns a typed `NetDistribution`; no eligible net raises
`ValueError`. Sampling uses `torch.multinomial` on CPU with an optional CPU
generator and returns an actual net ID. `mode()` and `log_prob(net_id)` use that
same label mapping. Known ineligible IDs have log probability negative infinity;
unknown IDs and non-integer actions raise `ValueError`. Log probabilities retain
gradients on the model's device.
