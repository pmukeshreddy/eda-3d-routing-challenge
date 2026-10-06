# PPO for negotiated rerouting order

The neural controller chooses the next pending net in each conflict round. The initial routing pass, Dijkstra tree builder, congestion formula, history updates, and 50-round limit are unchanged. Each affected net is rerouted at most once per round. The controller observes all nets, including those that cannot currently be selected.

The policy has 85,570 parameters: a 30-feature node encoder, two conflict-graph attention blocks of width 64 with four heads, and separate actor/value heads. A graph edge means two current routes share a vertex. Self-loops keep isolated nets well-defined. Action masking excludes nets that are not pending. Net IDs are not model inputs; they determine stable ordering and greedy tie-breaking only.

## Training

```sh
uv venv .venv --python python3
uv pip install --python .venv/bin/python -r requirements-rl-lock.txt
.venv/bin/python -m m3d.ppo_train run --out-dir experiments/ppo_hard
```

The run generates 144 training and 36 validation instances, balanced over the nine hard-tier sizes. The generator may enlarge a case when its original size cannot be certified feasible; actual widths and attempts are recorded. Baseline reference routes are independently checked. Training examples also include the baseline's decisions for five imitation-learning epochs. The nine released hard cases are excluded from training and validation using geometry/topology fingerprints, including when names or seeds change.

PPO uses 16 complete episodes per update, four optimization epochs, minibatches of 128 decisions, clipping 0.2, gradient norm limit 0.5, and KL early stopping at 0.02. The learning rate decreases from 3e-4 to 3e-5, and the entropy coefficient from 0.01 to 0.001. With gamma=lambda=1, the advantage before normalization is the final reward minus the value estimate at the decision. The final reward is -1 for failure/illegality, or `1 + baseline_delay / (baseline_delay + routed_delay)` for a legal route.

Three runs use seeds 42, 43, and 44. Validation occurs initially and every ten updates. After at least 50 updates, eight validation checks without improvement stop that seed; 200 updates is the maximum. Selection uses validation legal-case count first, then mean reward. A baseline-imitation checkpoint can win if PPO makes it worse; checkpoint metadata identifies that outcome explicitly.

Four worker processes collect episodes, each with one PyTorch thread. This is ordinary parallel collection; it does not replace the routing algorithm. Checkpoints allow interrupted training to resume, and completed seeds are reused when rerunning the same command. Dataset preparation also resumes completed cases. Separate output directories are required for different experiment configurations.

## Evaluation

The `run` command freezes the selected policy and sequentially evaluates negotiated baseline, neural policy, random order, most-conflicted-first, and the existing best-of-two router on all nine held-out cases. Each method's individual attempt has the same congestion settings and 50-round limit. Best-of-two is explicitly two attempts. All submitted solutions pass through the independent checker; no fallback substitutes baseline results for neural failures.

```sh
.venv/bin/python -m m3d.rl evaluate --suite benchmarks_hard \
  --policy experiments/ppo_hard/training/policy.pt \
  --out-dir experiments/ppo_hard/evaluation

.venv/bin/python -m m3d.cli run-suite --suite benchmarks_hard \
  --router negotiated_rl --policy experiments/ppo_hard/training/policy.pt \
  --out-dir /tmp/ppo-hard-solutions
```

`training/` contains configuration, seed-specific warm-start and PPO checkpoints, update/validation logs, and the selected `policy.pt`. `evaluation/results.json` records per-case real delays, legality, runtime, percentage change, and the official geometric-mean score (zero if any case fails). Full-suite total-delay improvement is only reported when every case is legal. Training/data generation time is separate from routing runtime.

## Verification

```sh
.venv/bin/python -m unittest discover -s tests -t .
```

Neural tests cover permutation/padding behavior, valid-action masking, return calculation, clipping, checkpoint round trips, real imitation/rollout/update behavior, and the training driver. The pre-existing linear REINFORCE implementation remains available as an earlier experiment, but is not used by `m3d.ppo_train`.

## Experiment status

The first experiment was stopped at the user's request after seed 42 logged PPO update 10, before that update's validation completed. Data preparation produced 144 training and 36 validation instances with 76,374 expert decisions. Initial validation after imitation routed 31 of 36 cases legally; its aggregate score was therefore zero. Seeds 43 and 44 and the final nine-case hard evaluation were not run. No improvement over the baseline has been demonstrated.

The incomplete correctness audit identified two modeling limitations: expert labels use net IDs to break equal-bounding-box ties although IDs are not input features, and pending eligibility masks actions without entering the actor's embeddings or critic. Early training also hit its KL stopping threshold after only 1–8 minibatches per update. These observations warrant investigation; they do not establish a PPO equation error or explain the individual validation failures.

Generated data, logs, and checkpoints remain local under the ignored `experiments/` directory. The experiment is stopped and should only be restarted deliberately. Resume checks require identical source hashes and configuration; checkpoints from a different source tree cannot silently resume in this checkout.
