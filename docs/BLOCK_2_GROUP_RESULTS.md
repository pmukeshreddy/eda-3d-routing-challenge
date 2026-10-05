# Block 2 V2: optional bbox baseline results

**Recorded bbox baseline: 1/9 legal, versus the previous bbox result of 4/9.**

The old one-net repair loop was replaced by one coordinated group transaction.
Its focused mechanics checks pass. This run remains a measured bbox baseline
regression; it is not an RL result or a V2 acceptance decision. No failed
layout receives a legal delay or an official aggregate score.

RL is the intended V2 controller. RL-controlled evaluation has not been run.
Bbox is an optional comparison only: neither 9/9 with bbox nor improvement
over 4/9 is a prerequisite or gate for RL integration, training or evaluation.
See [the current V2 plan](BLOCK_2_GROUP_PLAN.md).

## Recorded optional baseline run

All nine cases started from empty routes, using the unchanged bbox selector,
Block 1 `best`, and seed 5101. Cases ran serially. Per-case limits were 180 s,
50 passes, 5,000 Block 1 calls, 200M expansions, 1M expansions/call, and 10 s/call.
These are the existing Block 2 defaults. The prior 4/9 comparison used 120 s,
20 passes, 1,500 calls and 20M expansions, with eight concurrent cases.
The larger bounded allowance accommodates multiple complete group candidates;
this is not a comparison at identical budgets or concurrency.

| Case | Legal | Conflicts V+E | Missing | Legal delay | Runtime (s) | Group repairs | Accepted | Group sizes used | Block 1 calls | Stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| case_01 | no | 1 | 0 | — | 84.36 | 125 | 60 | 2, 3, 5, 8, 16, 32, 62 | 5000 | call_budget |
| case_02 | no | 10 | 0 | — | 110.91 | 226 | 90 | 2, 3, 5, 8 | 5000 | call_budget |
| case_03 | no | 1 | 0 | — | 104.01 | 103 | 64 | 2, 3, 5, 8, 16, 32, 64 | 5000 | call_budget |
| case_04 | no | 2 | 0 | — | 132.96 | 162 | 74 | 2, 3, 5, 8, 16, 32 | 5000 | call_budget |
| case_05 | no | 2 | 0 | — | 147.67 | 142 | 72 | 2, 3, 5, 8, 16, 32, 64 | 5000 | call_budget |
| case_06 | no | 9 | 0 | — | 180.03 | 225 | 109 | 2, 3, 5, 8, 16 | 4773 | time_budget |
| case_07 | no | 3 | 0 | — | 180.06 | 168 | 97 | 2, 3, 5, 8, 16 | 4285 | time_budget |
| case_08 | no | 8 | 0 | — | 180.06 | 248 | 128 | 2, 3, 5, 8 | 3806 | time_budget |
| case_09 | yes | 0 | 0 | 38834 | 35.73 | 85 | 77 | 2, 3 | 764 | legal |

Runtime includes reset, routing, cleanup and serialization. Conflicts count
vertices plus edges with multiple distinct owners. Every final net has a
complete route. Remaining ownership conflicts make each failed case illegal.

Final batch wall time: 1155.95 s. The JSON record includes
per-size repair frequencies, expansion counts, precise runtimes, and source hashes.

## Verification and observed limitation

The 16 focused checks pass, including all five requested checks: a real
two-net trap improves legal total delay 24 -> 20 while one net grows 4 -> 6;
all group routes are removed together and partial candidates roll back;
the lowest complete joint candidate wins; failed neighborhoods grow
2 -> 3 -> 5 -> 8 -> 10; and a legal incumbent survives a later interrupted order.
The joint-ranking check also proves a 5-conflict -> 1-conflict improvement
where outside wiring makes exclusive group reconstruction impossible.

Artifact verification checked all 9 reports, 1484 group transactions and
9709 candidate arrangements. Actual call totals, group-size frequencies,
and every accepted/restored objective match the saved candidate traces.
Runtime source hashes match the final source.
Independently checked legal hard-tier files: 1.

The observed limitation is stagnation after early congestion reduction.
Exclusive candidates frequently cannot finish; later complete negotiated
candidates often retain or add conflicts and are correctly rejected by the
full-state objective. Case 01 grew to the entire 62-net layout; completed
whole-layout candidates still had 77, 89 and 31 conflicts versus the saved
one-conflict layout. Group growth alone did not remove that final conflict.

## Diagnostic attempt and scope

An earlier exclusive-only diagnostic run failed case_01 with 193 conflicts
and case_02 with 526 conflicts, each at 5,000 calls, and was stopped during
case_03. Those reports remain under `build/block2_group/diagnostic_exclusive/`.
This concrete failure led to negotiated whole-group candidates in the same
transaction. A review also found and corrected a counter that included
`not_called` entries as dispatches; the focused budget check verifies the fix.
The final nine-case run above had no further tuning or retries.

Block 1 algorithms, RL scheduler/training, training data, and frozen Block 5
results were not modified. No reference or public submission routes were used.
No branches, worktrees, SAT/CBS/ILP solver, or broad test campaign were added.

Detailed candidate/ownership failure reports and the run log are under
`build/block2_group/`. The portable summary is
[BLOCK_2_GROUP_RESULTS.json](BLOCK_2_GROUP_RESULTS.json).

Recheck the saved evidence without rerouting:

```sh
PYTHONPATH=. /Users/mukeshreddy/anaconda3/bin/python3 build/block2_group/finalize.py
```
