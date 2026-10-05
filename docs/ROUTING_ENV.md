# Block 2: complete routing environment

An external selector can repeatedly choose connections while this environment
manages the shared layout and uses Block 1 to construct their wires.

The roadmap remains:

1. Wire engine — implemented previously.
2. Complete environment — this implementation.
3. Training data — later.
4. RL scheduler — later.
5. Evaluation and submission — later.

**Acceptance limitation:** the focused environment checks pass, but the single
official hard-tier acceptance run did not converge to a legal layout within its
declared 50-pass budget. This does not yet demonstrate legal hard-tier output.
See the actual results below.

## Run

Use the Python installation with the existing Block 1 native extension. If it
needs rebuilding, follow [WIRE_ENGINE.md](WIRE_ENGINE.md). No new dependencies
are required.

```sh
python3 -m m3d.routing_runner benchmarks_hard/case_05.json \
  --out build/block2/case_05.sol.json \
  --report build/block2/case_05.report.json \
  --seconds 180 --passes 50 --calls 5000 \
  --expansions 200000000 --call-expansions 1000000 --call-seconds 10
```

The runner starts with empty wiring. It reads only the supplied instance and
calls the environment once per selected net. It orders eligible nets by
descending sum of x/y/z bounding-box spans, then ascending net ID, matching
the repository's `bbox_desc` convention. It calls neither the old negotiated
router nor a reference solver.

Success writes the best layout as an ordinary `m3d-submission`, reloads that
file, and checks it with the unchanged `m3d.checker.check`. Failure exits with
code 2 and reports missing connections, vertex/edge conflicts and failed calls.
It writes no new solution file, reports `output: null`, and leaves any existing
file at `--out` untouched. A partial route collection is never labeled success.

## API

```python
from m3d.model import Instance
from m3d.routing_env import RoutingBudget, RoutingEnv

def main():
    instance = Instance.load("benchmarks_hard/case_05.json")
    with RoutingEnv() as env:
        state = env.reset(instance, seed=0, budget=RoutingBudget())
        while eligible := env.eligible_net_ids():
            net_id = eligible[0]  # replace with any external selector
            result = env.step(net_id)
            state = result.observation
            diagnostics = result.diagnostics
        solution = env.best_solution()  # Submission or None

if __name__ == "__main__":
    main()
```

Use a guarded main in executable scripts because the worker uses Python's
`spawn` process model. `RoutingEnv` owns the worker; use its context manager or
call `close()` to release it. Calls on one environment are sequential.

One connection/action means one whole net, including every sink, exactly as in
Block 1. IDs need not be contiguous or correspond to array positions; they
follow Block 1's nonnegative integer ID contract. Invalid, already consumed,
or terminal actions raise `ValueError` without changing routing state.

`StepResult` contains `observation`, `diagnostics`, and `done`. Diagnostics
include the selected net/pass, native status/method/expansions, delay, resource
price, objective, candidate diagnostics, installation status, rollback, and
any worker error. A valid action after elapsed-time expiry returns a terminal
`not_called` result. It does not dispatch another net.

Each observation is detached from mutable internal state and includes:

- The instance snapshot (fixed pins/nets/delays), grid dimensions, pin owners,
  and per-net pin vertices.
- Per-net wire trees (coordinate edges), distinct-net vertex/edge ownership,
  and conflict maps containing their owners.
- Per-resource history penalties and the current present-congestion factor.
- Eligible/missing IDs, individual actual delays, the current routed-delay sum,
  and per-route search-price sums from their most recent construction.
- Current pass, completed passes, status, termination reason, checker report,
  best legal delay, and elapsed/remaining time, calls, expansions, and passes.

Resource maps use packed `Grid` vertex IDs `(z*height+y)*width+x` and canonical
integer edge pairs. Prices saved with a route describe its construction, not a
revaluation under today's congestion. `current_routed_delay` can describe an
incomplete or conflicting layout; only `best_total_delay` is checker-certified.

A successful reset copies and validates the supplied instance using Block 1,
then clears every route, owner, history, eligibility set, counter, and best
snapshot. Invalid resets leave the previous episode intact. Mutating the
original instance, an observation, or a returned solution cannot edit the
environment.

## Transaction and passes

A step first validates eligibility, removes only the selected net's old route,
constructs current prices and foreign-pin blockers, and calls `WireEngine` for
that net. It installs a complete returned tree or restores the exact old tree
and its delay/search metrics on unreachable, exhausted, timed-out, or failed
replacement. Native calls use finite expansion allowances. A net owns each
physical vertex/edge once regardless of sink-path sharing or branch degree.

All nets are eligible on pass 1. At each incomplete/conflicting pass boundary,
only missing nets and owners of overused vertices or edges become eligible.
Eligibility is fixed for that pass; each eligible net can be selected once.
The transition itself does not call the engine. Initial present pressure is
0.5; between repair passes it grows by 1.7, capped at 1e6. Each resource's
history grows by `0.5 * (distinct_owner_count - 1)` when overused. These are the
initial settings and mechanics in `m3d/negotiated.py`.

Other nets' non-pin wires remain traversable during negotiation. All foreign
pins are blocked even before their owners are routed, both explicitly by the
environment and independently by Block 1.

The native API has additive, undirected **edge** prices. After rip-up, define:

```text
vertex_pressure(v) = VCONG * (history_v(v) + present * other_owners_v(v))
edge_price(u,v) = delay(u,v) * (history_e(u,v) + present * other_owners_e(u,v))
                  + (vertex_pressure(u) + vertex_pressure(v)) / 2
```

Block 1 already adds physical edge delay. These are surcharges only. Splitting
vertex pressure across incident edges charges an internal path vertex once.
For a tree junction, this undirected surrogate charges `degree/2` times its
vertex pressure; a leaf pays half. This is an explicit approximation to the
old router's directed vertex-entry pricing. Ownership and legality always
count each distinct net once. Search prices never enter actual routing delay.

Whenever a step leaves every net routed without any vertex or edge conflict,
the official checker validates a candidate. Only an approved layout can update
the separately copied best snapshot, using actual total delay. A mid-pass
snapshot survives subsequent worse routes, failed replacements, and budget
termination. A legal pass terminates naturally; there are no extra optimization
passes. `best_solution()` always returns a fresh copy or `None`.

## Finite budgets

`RoutingBudget` caps elapsed seconds, total passes (including the initial
pass), total calls, total expansions, per-call expansions, and per-call seconds.
All limits must be finite; zero work limits terminate without a native call.
The default limits are the values shown in the runnable command.

An episode-owned process reuses the unchanged native engine. IPC and native
waiting are bounded by the lesser of remaining episode time and per-call time.
A timed-out worker is terminated, so it cannot install a late result; a later
eligible action may start a new worker. A failed/killed call with no expansion
count is conservatively charged its full reserved allowance. Completed calls
are charged their reported count, even when a valid incumbent survives an
exhausted internal candidate. Calls count dispatch attempts, including failures.

Deadline checks surround dispatch and observe expiry between external actions.
Python bookkeeping/checking and bounded worker cleanup can add small overhead;
this is not a hard real-time process. Cleanup waits at most 0.2 seconds after
termination and another 0.2 seconds after killing if needed. Reading a terminal
observation or closing the environment does not erase the incumbent.

## Focused verification and actual acceptance

Run only the new environment checks with a 60-second outer limit:

```sh
python3 -c 'import subprocess; subprocess.run(["python3", "-m", "unittest", "tests.test_routing_env", "-v"], timeout=60, check=True)'
```

Checks cover distinct occupancy, foreign pins, vertex-only pricing, exact
rollback, invalid actions, reset/observation/snapshot isolation, missing-net
repair, finite work and wall limits, incomplete IPC frames, worker recovery,
best preservation after a later conflicting layout, an actual native multi-net
episode, and CLI serialization checked with the official checker.

One official hard-tier run was made from scratch using the command above:

| Measurement | Actual result |
| --- | --- |
| Case | `benchmarks_hard/case_05.json`, 32×32×6, 83 nets |
| Method / seed | Block 1 `best` / 0 |
| Legality | Failure: no fully legal snapshot found |
| Official total delay | Unavailable; no legal output |
| Episode runtime | 41.8456 s |
| Runtime including runner bookkeeping | 41.8526 s |
| Completed passes | 50 / 50 |
| Engine calls | 1,021 / 5,000 |
| Expansions | 7,149,570 / 200,000,000 |
| Missing nets / edge conflicts | 0 / 0 |
| Remaining vertex conflicts | 3 |
| Failed native calls | 0 |
| Termination | Pass budget; exit code 2; no solution exported |

The remaining conflicts were `(17,16,2)` between nets 12/29, `(18,20,3)`
between 8/28, and `(13,15,4)` between 26/69. All nets had complete trees, but
the shared layout was illegal. No complete legal output existed to send to
the official checker. The native multi-net focused test and successful CLI
fixture did produce complete outputs approved by that checker.

The raw hard-run report is preserved in
[BLOCK_2_ACCEPTANCE.json](BLOCK_2_ACCEPTANCE.json). That single run preceded the
subsequent reset/IPC/observation lifecycle fixes; routing prices, pass mechanics,
selector, seed, and native engine were unchanged afterward. Focused checks were
rerun on the final implementation. No second hard attempt, reference warm start,
penalty retuning, larger budget, or upstream suite was used. Review found no
verified price/pass defect explaining the remaining conflicts. Legal hard-tier
acceptance remains an explicit gap.
