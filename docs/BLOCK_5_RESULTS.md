# Block 5 final evaluation

**RL did not produce measurable improvement.**

## A. System and frozen experiment

The validation-selected checkpoint is update 48, with 2,453,441 parameters.
Checkpoint SHA-256: `90c9c08d0daf735a03275d212c00bcfc074746dcb3bf3225792d9ef934e26d25`.
Source commit: `0a8e052944432de0c657f3d45d6838394e627d63`; the working tree contains uncommitted files, so the authoritative source fingerprint is `1453ddb8852020f7e4fb8b60c95ee478c57b3022fbe2279f5a79c4dbe574aae6`.
Block 1 constructs each selected net's route using method `best`. Block 2 owns the empty-start layout, negotiated congestion, rollback and legality snapshots. The Block 4 graph/Transformer policy chooses eligible net IDs.
The model has width 192, two graph layers, four Transformer layers, six attention heads and a 1,024-unit feed-forward layer. The unchanged feature schema is `scheduler-current-state-v1`. The initial comparison checkpoint is update 0; both checkpoint hashes and the dataset manifest hash are recorded in `build/block5/frozen_config.json`.
Frozen budgets: `{"max_call_seconds": 10, "max_calls": 1500, "max_expansions": 20000000, "max_expansions_per_call": 1000000, "max_passes": 20, "max_seconds": 120}`. Concurrency: 8. No training, parameter changes, congestion changes, reward changes, or routing changes were made.
Hardware/runtime: `{"accelerator": "Apple MPS", "cpu": "Apple M2", "cpu_count": 8, "model": "Mac14,7", "os": "macOS-26.5.2-arm64-arm-64bit", "python_executable": "/Users/mukeshreddy/anaconda3/bin/python3", "python_version": "3.12.7", "ram_bytes": 8589934592, "torch_version": "2.9.1"}`.
Evaluation runtime: 910.37s under a 2700s cap. External verification time is recorded separately in the raw and batch results.
Per-episode reset through terminal observation, decision trace and route serialization, plus amortized shared model/input/setup and collector bookkeeping. Exact full selector-batch wall time is also recorded. Independent checker/scorer time is separate; environment-internal checking stays included.

## B. First held-out generated test evaluation

Four cases; stochastic seeds 5101 and 5102. Deterministic policies use 5101. Bbox uses both engine seeds to provide paired comparisons. The sole additional small-case replay is excluded from aggregates.
Scores are reference delay / our legal delay; terminal reward is its logarithm. Failed episodes receive the unchanged dataset-derived floor. Conditional scores/delays exclude failures; failed episodes still contribute to legality, reward, runtime and action counts.
The frozen failure reward is −19.26540913611998. Test access comprised one predefined 40-episode comparison, followed by the explicitly requested single small-case replay. The test results, decisions and saved solutions were made read-only and hash-sealed before hard-tier routing.

| Selector | Legal | Legal-only score GM | Mean reward | Failed conflicts | Mean legal delay | Runtime mean ± SD (s) | Mean actions | Mean calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bbox | 8/8 | 1.0604 | 0.0587 | — | 4771.5 | 11.47 ± 8.08 | 202.6 | 202.6 |
| random | 8/8 | 1.0666 | 0.0645 | — | 4705.2 | 9.67 ± 6.83 | 197.5 | 197.5 |
| initial_deterministic | 4/4 | 1.0541 | 0.0527 | — | 4797.0 | 45.13 ± 35.11 | 206.2 | 206.2 |
| initial_sampled | 6/8 | 1.0631 | -4.7704 | 139.0 | 2939.7 | 66.71 ± 39.94 | 153.5 | 153.2 |
| trained_deterministic | 4/4 | 1.0570 | 0.0554 | — | 4779.0 | 46.77 ± 37.32 | 206.8 | 206.8 |
| trained_sampled | 6/8 | 1.0545 | -4.7766 | 138.5 | 2973.0 | 67.54 ± 39.29 | 154.6 | 154.4 |

The standard deviations above describe episodes, including case difficulty. Per-case stochastic statistics, each seed's aggregates, and individual episodes are preserved in `test_results.json`; they are not independent samples of a large benchmark population.

For the two-seed comparisons, the individual mean rewards and population standard deviation across seeds are:

| Selector | Seed 5101 mean reward | Seed 5102 mean reward | Mean ± seed SD | Legality at each seed |
| --- | ---: | ---: | ---: | --- |
| bbox | 0.062569 | 0.054782 | 0.058675 ± 0.003894 | 4/4, 4/4 |
| random | 0.061717 | 0.067259 | 0.064488 ± 0.002771 | 4/4, 4/4 |
| initial_sampled | -4.772545 | -4.768320 | -4.770432 ± 0.002113 | 3/4, 3/4 |
| trained_sampled | -4.776869 | -4.776268 | -4.776568 ± 0.000301 | 3/4, 3/4 |

All individual case/seed results are in the sealed JSON and raw episode log. Deterministic policy test batches contain four episodes; the two-seed selector batches contain eight. Runtime includes the resulting batch contention. The hard-tier comparison uses identical eight-plus-one episode waves for both selectors.

## C. Public hard tier

All nine cases started with empty wiring. Trained deterministic inference was selected before test access as the sole primary challenge system. Reference delay scalars come from `benchmarks_hard/suite.json`; reference and participant wires were never loaded.

| Case | Selector | Legal | Delay | Baseline | Ratio | Runtime (s) | Passes | Calls | Stop | Conflicts | Missing |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| case_01 | bbox | True | 10888 | 11938 | 1.0964 | 74.79 | 12 | 482 | legal | 0 | 0 |
| case_01 | trained_deterministic | False | — | 11938 | — | 121.16 | 2 | 89 | time_budget | 665 | 0 |
| case_02 | bbox | False | — | 18910 | — | 119.52 | 20 | 777 | pass_budget | 8 | 0 |
| case_02 | trained_deterministic | False | — | 18910 | — | 121.16 | 2 | 89 | time_budget | 938 | 0 |
| case_03 | bbox | True | 14186 | 14724 | 1.0379 | 98.46 | 13 | 629 | legal | 0 | 0 |
| case_03 | trained_deterministic | False | — | 14724 | — | 121.15 | 2 | 89 | time_budget | 901 | 0 |
| case_04 | bbox | False | — | 18077 | — | 113.89 | 20 | 731 | pass_budget | 2 | 0 |
| case_04 | trained_deterministic | False | — | 18077 | — | 121.15 | 2 | 89 | time_budget | 1108 | 0 |
| case_05 | bbox | True | 20227 | 21051 | 1.0407 | 111.92 | 14 | 717 | legal | 0 | 0 |
| case_05 | trained_deterministic | False | — | 21051 | — | 121.14 | 2 | 89 | time_budget | 1179 | 0 |
| case_06 | bbox | False | — | 27134 | — | 120.15 | 11 | 782 | time_budget | 28 | 0 |
| case_06 | trained_deterministic | False | — | 27134 | — | 121.14 | 2 | 89 | time_budget | 1609 | 0 |
| case_07 | bbox | False | — | 31341 | — | 120.14 | 10 | 782 | time_budget | 61 | 0 |
| case_07 | trained_deterministic | False | — | 31341 | — | 121.14 | 1 | 89 | time_budget | 2075 | 5 |
| case_08 | bbox | False | — | 30886 | — | 120.14 | 9 | 782 | time_budget | 66 | 0 |
| case_08 | trained_deterministic | False | — | 30886 | — | 121.13 | 1 | 89 | time_budget | 1838 | 10 |
| case_09 | bbox | True | 26178 | 27802 | 1.0620 | 52.45 | 13 | 815 | legal | 0 | 0 |
| case_09 | trained_deterministic | False | — | 27802 | — | 120.33 | 3 | 269 | time_budget | 768 | 0 |

bbox: 4/9 legal. Official aggregate: —. Incomplete; no competitive aggregate calculated

trained_deterministic: 0/9 legal. Official aggregate: —. Incomplete; no competitive aggregate calculated

| Selector | Legal rate | Mean reward | Mean failed conflicts | Mean runtime (s) | Mean scheduling actions | Mean engine calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bbox | 44.44% | -10.6775 | 33.0 | 103.50 | 721.9 | 721.9 |
| trained_deterministic | 0% | -19.2654 | 1231.2 | 121.06 | 110.0 | 109.0 |

There are no jointly legal hard-tier pairs, so the RL-versus-bbox score delta is undefined. RL loses 44.44 percentage points of legality and adds 17.56 seconds per episode. Its lower engine-call count reflects time-budget termination, not successful routing with less work. End-to-end times can exceed the 120-second environment allowance slightly because they also include shared setup allocation, termination and output bookkeeping.

## D. RL ablation

Only net selection differs. Positive legality/score/reward deltas favor RL; positive runtime/call deltas mean RL costs more. Score deltas use the explicitly recorded jointly legal case/seed pairs, never an unmatched mixture of solved cases.

| Dataset | Trained mode vs selector | Paired episodes | Δ legality | Common legal pairs | Δ score GM | Δ mean reward | Δ runtime (s) | Δ calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | trained_deterministic vs bbox | 4 | 0.0000 | 4 | -0.0076 | -0.0071 | 35.28 | 4.0 |
| test | trained_deterministic vs random | 4 | 0.0000 | 4 | -0.0066 | -0.0063 | 37.05 | 8.8 |
| test | trained_deterministic vs initial_deterministic | 4 | 0.0000 | 4 | 0.0029 | 0.0027 | 1.64 | 0.5 |
| test | trained_sampled vs bbox | 8 | -0.2500 | 6 | -0.0041 | -4.8352 | 56.07 | -48.2 |
| test | trained_sampled vs random | 8 | -0.2500 | 6 | -0.0036 | -4.8411 | 57.87 | -43.1 |
| test | trained_sampled vs initial_sampled | 8 | 0.0000 | 6 | -0.0087 | -0.0061 | 0.84 | 1.1 |
| hard | trained_deterministic vs bbox | 9 | -0.4444 | 0 | — | -8.5879 | 17.56 | -612.9 |

## E. Failures, reproducibility and scope

Reproducibility on `test-sparse-00000`: decisions match = True; final fields = True; solution hash = True. Runtime is not expected to match exactly.
Saved-output verification: 41 legal files independently checked, scorer values reproduced, frozen hashes verified, and the test seal remained unchanged through the hard-tier evaluation.
Block 2's earlier case_05 failure (three vertex conflicts after 50 passes) remains unchanged. Block 4's learning acceptance remained unmet. Block 5 uses the frozen 20-pass budget; its results do not revise the earlier 50-pass acceptance record.
The current bbox case_05 solution used the predeclared Block 5 seed and configuration. It does not erase the earlier failure or demonstrate an environment fix.

All four failed test episodes were sampled-policy runs of `test-dense-00000`, with no missing nets. Initial-policy seeds 5101/5102 ended after five passes with 140/138 combined vertex-and-edge conflicts; trained-policy seeds ended after five passes with 131/146 conflicts. All stopped on the time budget.

On the hard tier, bbox exhausted the pass budget on case_02 and case_04, and the time budget on case_06 through case_08. Every trained-policy episode exhausted the time budget. The first eight reached only 89 native calls each; case_07 and case_08 still had five and ten missing nets. The single-episode case_09 wave reached 269 calls. This establishes a throughput and termination limitation of the frozen policy system on this machine. The run did not separately profile feature encoding, inference and route-search time, so it cannot assign the slowdown to one component.
Detailed failure ownership, missing net IDs, native failure diagnostics and stop reasons are retained in `raw_episodes.jsonl`. No failed layout receives a legal score or a delay average contribution. Conditional delay/call reductions can reflect early termination and are not an overall success claim.
No all-tier campaign was run. Absolute times are not compared to public leaderboard times from other hardware.

## F. Conclusion and packaging

**RL did not produce measurable improvement.** See the paired deltas above for the exact supported changes; four test cases and two stochastic seeds support no significance claim.
This conclusion uses the frozen comparison's sampled terminal reward and baseline comparisons. Trained deterministic selection did improve the test score over deterministic initialization by 0.0029 (about 0.275%), with unchanged legality, but remained below both simple selectors and took longer. Trained sampling had unchanged legality and lower reward than initialization. Neither learned mode beat the nonlearned selectors, and the primary trained hard-tier system produced no legal output. The small deterministic initialization gain is retained as a conditional observation, not an overall learning-success claim.
Submission: `{"created": false, "reason": "trained primary system completed 0/9 hard cases legally"}`.
No leaderboard edit, push, or PR was made. Full configuration, checksums, raw episodes, decision traces, legal route files and summaries are under `build/block5/`.

Reproduce artifact verification without rerouting:

```sh
/Users/mukeshreddy/anaconda3/bin/python3 -m m3d.final_evaluation --phase verify --out build/block5
```

The campaign commands were `--phase freeze` followed by `--phase evaluate` using that interpreter. Existing freeze/run markers prevent a silent second test campaign.
