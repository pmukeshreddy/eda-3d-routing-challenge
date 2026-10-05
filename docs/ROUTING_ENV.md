# Block 2 V2: coordinated routing environment

`RoutingEnv` starts from empty wires, constructs an initial layout with unchanged
Block 1, then repairs interacting nets together. The former single-net repair
loop has been replaced; there is no alternative single-net repair path.

## Intended controller and evaluation

RL is the intended V2 controller for routing and repair-group decisions.
Block 2 supplies state, interaction measurements, transactional group repair,
joint scoring and budgets; Block 1 supplies pathfinding. Bbox and heuristic
seed/group selection are optional baseline comparisons only. There is no
requirement to achieve 9/9 with bbox, or improve a bbox score, before RL work.

The current runnable prototype still uses a seed-ID action with heuristic
neighbor selection. Its bbox harness is a baseline tool, not an RL-controlled
V2 evaluator. The RL group-action integration and primary evaluation remain
pending, as recorded in [the V2 plan](BLOCK_2_GROUP_PLAN.md). Missing RL support
must not silently substitute a heuristic controller. Focused environment
correctness checks establish mechanics readiness independently of bbox results.

## Optional bbox baseline

The following command explicitly runs the bbox baseline harness. It is not a
required V2 evaluation step. Use the Python installation containing the native
extension:

```sh
/Users/mukeshreddy/anaconda3/bin/python3 -m m3d.routing_runner benchmarks_hard/case_05.json \
  --out build/block2_group/case_05.sol.json \
  --report build/block2_group/case_05.report.json \
  --seed 5101 --method best --seconds 180 --passes 50 --calls 5000 \
  --expansions 200000000 --call-expansions 1000000 --call-seconds 10
```

In this optional baseline, the external bbox selector uses descending sum of pin-bbox spans,
then ascending net ID. It selects initial nets and, subsequently, problematic
seeds among missing nets and owners of conflicts. The prototype selects each
seed's interacting group deterministically. These heuristic controller choices
are baseline behavior. No policy training or RL evaluation is involved. Reference and submission routes are never inputs.

Success saves a normal `m3d-submission`, reloads it, and independently checks the
serialized layout with `m3d.checker.check`. Failure exits 2 and writes a report
with no new solution. See `BLOCK_2_GROUP_RESULTS.md` for the historical bbox baseline run.
The historical single-net acceptance report remains in `BLOCK_2_ACCEPTANCE.json`;
frozen Block 5 records remain unchanged.

## Current prototype search mechanics

The heuristic seed/neighborhood choices below describe the baseline prototype.
The V2 plan assigns controller decisions to RL; the group transaction, candidate
comparison, ownership and budget guarantees remain environment responsibilities.

1. Initial construction uses the existing soft occupancy prices and always
   blocks foreign pins. A dispatch constructs a whole net, including its sinks.
2. Interaction scores combine shared vertex/edge conflicts (100 per resource),
   occupancy next to pins (10), route occupancy inside another net's pin bbox
   plus a one-vertex halo, and normalized bbox-halo overlap. Regional occupancy
   receives a bounded detour multiplier from current delay divided by a simple
   driver-to-sink delay lower bound. These scores use the current layout.
3. The selected seed grows a neighborhood by strongest aggregate interaction
   with nets already selected, with net ID as the stable tie breaker. Two
   unsuccessful attempts at a seed advance its group size: 2, 3, 5, 8. Later
   attempts include the entire current conflict component and double the target
   neighborhood size up to all nets. There is no permanent small-group cap.
4. Every candidate removes **all** selected routes before any Block 1 call.
   Outside routes remain fixed. All currently occupied vertices and edges are
   hard blockers in exclusive candidates. Negotiated candidates instead price
   their occupancy, allowing a complete group to reduce conflicts even when
   fixed outside wires make an exclusive group impossible. Each rebuilt net
   updates temporary occupancy before constructing the next net.
5. At most eight distinct orders are considered. They include descending bbox,
   fanout, delay/detour and conflict strength, rotations, reversal, and bounded
   seeded permutations when there is room. Two-net groups have just two orders.
   Each order has at most two variants (negotiated and exclusive), so at most
   sixteen complete arrangements are attempted. Already legal states use only
   exclusive variants. The code never enumerates permutations factorially.
6. An order fails if any group net is unreachable, fails, or runs out of budget.
   Complete candidates are compared using the full-state tuple:
   `(missing nets, conflict vertices + edges, total excess owners, total delay)`.
   For two legal states this reduces to total driver-to-sink delay. Individual
   nets may get longer. Equal or worse candidates are not committed.
7. The best improving complete group is committed. Otherwise the exact prior
   routes and ownership are restored. Interaction information is recomputed.
   Failed seeds grow their neighborhoods; successful groups reset their failure
   counters. Search stops at a legal layout, exhausted bounded search, or budget.

A pass is an initial construction sweep or a sweep of eligible repair seeds.
A successful group consumes its members' current eligibility; other eligible
seeds remain available. Failed attempts consume only their seed's eligibility.
At a pass boundary eligibility is rebuilt from the actual remaining problems.
A seed that repeatedly fails after reaching the whole layout is exhausted; if
all remaining problematic seeds are exhausted, termination is `no_improvement`.

## Transactions and incumbent

Per-net route records are immutable. A repair saves every original group route.
Removing/reinstalling those records updates distinct-net vertex/edge ownership;
shared tree branches count once per net. Every order starts from the same
outside-group state. A `finally` block removes any temporary partial group and
restores the original group or installs the best complete candidate, including
when a later order exhausts a time/call/expansion budget.

History penalties, present pressure, conflicts and the legal incumbent do not
leak from rejected candidates. Initial congestion history is recorded at the
initial pass boundary. Negotiated candidates share one proposed history update
(+0.5 per excess owner) and occupancy pressure (at least 8, growing by 1.7 from
the last committed pressure, with additional bounded growth on failed seeds;
capped at 1e6). These penalties commit only with a winning negotiated candidate.
Rejected and exclusive candidates leave stored penalties unchanged. Conflicts are derived
from ownership. Work counters and failure counts deliberately do not roll back.

Every accepted globally conflict-free complete layout is checked by the official
checker. Only checker-approved improvements update the separately copied best
legal solution. A later failed or interrupted candidate cannot erase it.
`best_solution()` returns a fresh copy. Initial construction or any candidate
that lacks a net is never reported as legal.

## Current baseline API and diagnostics

This is the optional bbox harness, not the pending RL evaluation entrypoint:

```python
with RoutingEnv(method="best") as env:
    env.reset(instance, seed=5101, budget=RoutingBudget())
    report = drive_episode(env)  # optional bbox baseline only
    solution = env.best_solution()
```

Use a guarded `main` in scripts: the unchanged bounded worker uses multiprocessing
`spawn`. Call `close()` or use the context manager. Invalid actions and invalid
resets preserve the previous routing state. IDs may be sparse.

`step(net_id)` constructs that net during the initial phase; during repair it
selects the seed for a complete group transaction. Existing observation fields
remain, plus phase, interaction scores, seed failure counts, group repair and
acceptance counts, and the histogram of group sizes. One repair step may make
multiple Block 1 calls; callers must use the call counter rather than action
count to measure routing work. RL scheduler code is unchanged and untrained for
these new action semantics.

Runner reports include missing IDs, conflict vertices/edges, overuse, legal
incumbent delay, runtime, group counts/sizes, total Block 1 calls, and a repair
trace with candidate orders, completeness, objectives and final call status.
Reported routing delay contains physical driver-to-sink delay only.

## Budgets and focused verification

`RoutingBudget` caps elapsed time, passes, calls, total expansions, per-call
expansions and per-call time. The defaults are the command above. Wall deadlines
and worker termination apply to every call. Dispatch attempts consume calls;
a killed call without trustworthy expansion counts consumes its reserved
allowance. Bookkeeping and worker cleanup can add small wall overhead.

The recorded bbox baseline used 180 seconds / 50 passes / 5,000 calls / 200M expansions
per case, compared with the prior bbox comparison's 120 seconds / 20 passes /
1,500 calls / 20M expansions. Multiple complete group orders need a larger work
allowance. Actual usage and execution concurrency are reported with results.
These baseline settings do not prescribe RL evaluation budgets or impose a
legality gate on RL integration or training.

Run only the focused tests:

```sh
/Users/mukeshreddy/anaconda3/bin/python3 -m unittest \
  tests.test_group_repair tests.test_routing_env -v
```

Five new checks cover a native two-net delay trap (24 -> 20, while one net's
delay grows 4 -> 6), whole-group removal and exact rollback with an outside net,
lowest complete joint candidate selection independent of order (including a
5-conflict to 1-conflict improvement when exclusive reconstruction is impossible), repeated-failure
expansion beyond eight nets, and preservation of the best legal candidate when
a later order is interrupted. Existing lifecycle, budgets, foreign pins,
IPC timeout, checker/export and snapshot-isolation checks remain. Tests that
required the obsolete single-net repair semantics were replaced.
