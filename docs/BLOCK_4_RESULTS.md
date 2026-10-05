# Block 4 acceptance result

Implementation and mechanics checks passed. The required validation improvement
criterion **did not pass**: the trained scheduler improved sampled reward over
its initialization but did not outperform either nonlearned selector.

One fixed experiment ran for **2,152.97 seconds (35.88 minutes)** under its
3,600-second cap. It completed 48 optimizer updates from 24 groups (G=8), using
all 12 training cases twice. The 2,453,441-parameter policy ran on Apple MPS
(Mac14,7, eight CPU cores, 8 GB shared memory; Python 3.12.7, PyTorch 2.9.1). No configuration
or scheduler source changed during this experiment. Test instances were not
opened or routed. This 12-case shard is not sufficient evidence for a final
policy.

## Validation evidence

Each sampled/baseline row contains two paired seeded episodes on each of the
four validation cases. Deterministic rows contain one episode per case. All
selectors use the same environment: 20 passes, 1,500 calls, 20 million
expansions, 120 seconds per episode, one million expansions and 10 seconds per
call. Wall time includes scheduler overhead.

| Scheduler | Episodes | Legal rate | Mean conflicts on failures | Mean terminal reward | Mean legal delay | Mean steps | Mean runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bbox | 8 | 100% | — | 0.061392 | 5663.00 | 225.13 | 11.35 |
| Uniform random | 8 | 100% | — | 0.071094 | 5550.50 | 219.00 | 12.04 |
| Initial, sampled | 8 | 75% | 281.0 | -4.766131 | 2731.67 | 151.25 | 72.93 |
| Trained update 48, sampled | 8 | 75% | 255.5 | -4.754076 | 2665.67 | 147.38 | 70.06 |
| Initial, deterministic | 4 | 75% | 17.0 | -4.755312 | 2670.33 | 211.00 | 53.57 |
| Trained update 48, deterministic | 4 | 75% | 21.0 | -4.763053 | 2713.00 | 214.50 | 52.24 |

No-failure conflict means are stored as zero in machine-readable metrics; the
table shows a dash because there are no failed episodes to average. Conflicts
count overused vertices plus edges. Delay averages include only legal episodes;
the failed dense cases are excluded from policy delay averages. Steps include
timed-out episodes. These conditional values should not be interpreted as an
overall delay or efficiency win against the fully legal baselines.

The fixed checkpoint-selection metric is paired sampled mean terminal reward.
Update 24 scored -4.757884 and update 48 scored -4.754076, so `best.pt` is update
48. Initial sampled reward was -4.766131: an improvement of 0.012056, with no
completion-rate improvement. Deterministic reward became slightly worse.
Eight sampled episodes provide limited validation evidence, not a statistical
claim or a final benchmark evaluation.

## Measured diagnosis

All policy validation failures hit the time budget on the dense case. Bbox and
random solved that same case in roughly 27–29 seconds. The learned scheduler's
runtime is therefore a practical limitation under the equal episode ceilings;
the neural rollouts complete less routing work before their deadlines. The run
does not isolate feature encoding versus neural inference time, so it cannot
attribute that overhead precisely.

All six dense training groups timed out in all eight episodes. Those 48 failed
episodes received the same metadata-derived floor, -19.265409, and gave zero
GRPO task advantage. One sparse group also had identical legal rewards. In
total **7/24 groups (29.17%)** had zero task advantage; the other 17 groups had
reward variation. All 144 nondense training episodes were legal. Entropy
regularization still acts on constant-reward groups.

Losses, gradients and parameters remained finite. Maximum pre-clipping gradient
norm was 0.56493; maximum measured behavior KL was 0.03119. The final parameter
change from initialization had L2 norm 3.8478. These checks establish working
updates, not scheduler quality. Model under-capacity is not established by this
run. There were no architecture changes, hyperparameter restarts, or extra
training experiments after seeing these results.

## Verification and artifacts

- Nine focused policy/GRPO checks passed in 0.742 seconds under a 60-second cap.
- The separate tiny mechanics run used one training instance, G=4, and two
  updates. All four episodes were legal. It took 14.05 seconds on MPS; action
  log-probability error was at most 2.39e-7, weights changed, and checkpoint
  reload preserved deterministic actions.
- The saved real checkpoint loads with its optimizer state and finite weights.
  An artifact audit checked all 192 training trajectories and 19,945 stored
  actions: IDs/masks agree, selected behavior log probabilities match their
  stored distributions, distributions normalize, and hindsight is separate
  from pre-action features.
- Hash checks confirmed 63 protected files unchanged, including Blocks 1–3
  code, official inputs, the data manifest and prior acceptance reports.

Configuration, checkpoints, metrics, compressed trajectories, provenance and
full acceptance evidence are in `build/block4/`. A compact tracked report is
[BLOCK_4_ACCEPTANCE.json](BLOCK_4_ACCEPTANCE.json). Commands and interface details
are in [SCHEDULER_GRPO.md](SCHEDULER_GRPO.md).

Block 2's independent acceptance gap remains open and unchanged: official
`case_05` ended with three vertex conflicts after 50 passes. This Block 4 run
does not resolve that gap. Block 5 was not started.
