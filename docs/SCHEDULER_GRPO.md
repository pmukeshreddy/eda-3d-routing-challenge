# Block 4: local GRPO net scheduler

The scheduler chooses one currently eligible net ID. `RoutingEnv.step()` still
owns the transition and calls the existing native wire engine. Blocks 1–3,
their data splits, the official checker, and benchmark inputs are unchanged.
Block 2 hard-tier acceptance remains open: `case_05` ended with three vertex
conflicts after 50 passes. This experiment does not revisit that case.

## Running

PyTorch is the only neural runtime. The command selects CUDA, then Apple MPS,
then CPU; `device` can also be set explicitly in the configuration. CPU routing
workers do not import PyTorch. No service, downloaded model, or cluster is used.

The recorded experiment used Python 3.12.7 and PyTorch 2.9.1 from
`/Users/mukeshreddy/anaconda3/bin/python3`. This machine's current default
`python3` resolves to Homebrew Python 3.13 without PyTorch. Select the tested
interpreter explicitly below; on another machine, use a Python environment
with PyTorch and the existing Block 1 native extension.

```sh
SCHEDULER_PYTHON=/Users/mukeshreddy/anaconda3/bin/python3

"$SCHEDULER_PYTHON" -m unittest tests.test_scheduler_policy tests.test_scheduler_grpo -v

"$SCHEDULER_PYTHON" -m m3d.train_scheduler \
  --data build/data/routing_v1 --out build/block4/mechanics \
  --config configs/scheduler_grpo_v1.json --mechanics

"$SCHEDULER_PYTHON" -m m3d.train_scheduler \
  --data build/data/routing_v1 --out build/block4 \
  --config configs/scheduler_grpo_v1.json

"$SCHEDULER_PYTHON" -m m3d.eval_scheduler \
  --data build/data/routing_v1 --split val \
  --checkpoint build/block4/checkpoints/best.pt
```

Training refuses to overwrite an existing experiment. A completed experiment
returns exit code 2 if the learning acceptance criterion is not met; its results
and checkpoints are still saved. The evaluation CLI accepts only `val` in this
block. Test routing is reserved for Block 5.

## Policy interface

See [SCHEDULER_FEATURES.md](SCHEDULER_FEATURES.md) for the exact versioned schema,
normalization, interaction graph, and padding rules. The model has 2,453,441
parameters, width 192, two graph layers and four Transformer layers. It consumes
60 features per net and 37 global features; IDs are labels, never embeddings.
The single-observation distribution has one masked logit per real net and zero
probability for every ineligible action. Batched forward passes pad only across
the net dimension, with graph, attention, and action masks applied separately.

```python
from m3d.routing_data import load_split
from m3d.routing_env import RoutingEnv, RoutingBudget
from m3d.scheduler_grpo import load_checkpoint

instance, reward_metadata = load_split("build/data/routing_v1", split="train").load(0)
policy, checkpoint = load_checkpoint("build/block4/checkpoints/best.pt")
env = RoutingEnv()
try:
    observation = env.reset(instance, seed=7, budget=RoutingBudget(max_passes=20))
    action = policy.act(observation, deterministic=True)
    log_probability = policy.log_prob(observation, action)
    distribution = policy.distribution(observation)
    result = env.step(action)
finally:
    env.close()
```

`reward_metadata` stays outside the observation. Every environment reset starts
with empty wiring. No reference certificate is opened by the scheduler.

## Update and reward

Training samples shuffled cycles of training cases only. For each selected
instance, eight independent environments use the same native engine seed and
different action-sampling seeds. Their CPU routing calls run concurrently;
inference batches all active environments on the neural device. Completed
environments leave the batch. All runs have identical environment work limits.

Intermediate reward is zero. A checker-approved legal layout receives
`log(reference_total_delay / solution_total_delay)`. Failure receives one fixed
floor derived from the dataset's recorded generation configuration. If `U` is
the largest configured upper bound on total legal delay, the floor is
`-log(U)-1`. The bound is `sinks_max * (vertices_max-1) * max_edge_delay`, including
the upstream generator's maximum grid enlargement. A legal tree's sink path
contains at most `vertices_max-1` edges, and reference delay is a positive
integer. Thus this floor is strictly below every possible legal reward for the
configured dataset, independent of observed scheduler success. The bound,
rule, and metadata fingerprint are saved in config and checkpoints.

Rewards are standardized within each same-instance group with population
standard deviation. A constant-reward group receives zero advantages. All
actions in a trajectory share that trajectory's advantage: this is terminal,
trajectory-level credit assignment, with no critic and no intermediate labels.

The loss minimizes the negative clipped ratio objective (clip 0.2), minus
0.01 times categorical entropy, plus 0.02 times exact categorical
`KL(behavior || current)` on the stored pre-action state. Both the selected
action log probability and the entire masked behavior distribution are frozen
at rollout time. Two optimizer updates use that same rollout group. Each
trajectory has weight `1/G`, divided equally among its actions, so long failed
episodes do not receive extra weight just for being long. State minibatches
accumulate gradients for one full-group update; gradient norm is clipped at 1.

## Fixed acceptance experiment

`configs/scheduler_grpo_v1.json` declares one initial configuration: G=8,
at most 48 optimizer updates (24 groups, two cycles of the 12 training cases),
AdamW at 0.0003, state minibatches of 32, and a 3,600-second total wall cap.
Model, dataset cycling, action sampling, rollout groups, optimizer shuffling,
and validation have separate fixed seeds. There is no hyperparameter search.

Every scheduler uses the same episode limits: 20 passes, 1,500 native calls,
20 million expansions, one million expansions per call, 120 seconds per
episode, and 10 seconds per call. Work limits are reproducible; wall limits
are a backstop and include inference/encoding time. Timing comparisons include
the actual scheduler cost and concurrent worker contention.

Validation contains four cases. Bbox, random, and sampled policies use two
paired seeded episodes per case. Initial and trained policies also have a
separate deterministic four-case evaluation. Validation occurs initially and
after updates 24 and 48. It never supplies gradients. `best.pt` selects among
trained checkpoints by sampled mean terminal reward. The predeclared primary
acceptance measure is improvement in that reward over both the initial policy
and at least one nonlearned baseline. All completion, failure-conflict, legal
delay, step, and runtime metrics are also reported. Eight sampled validation
episodes are limited evidence, not a statistical or final benchmark claim.

The training loop reserves six minutes for final validation and stops admitting
groups before the deadline. Optimizer minibatches also check the deadline.
Checkpoints and metrics survive a bounded early stop. A partial run must be
reported with its actual training-case coverage and update count.

## Artifacts and future hindsight

`config.json` records the resolved device, feature version, parameter count,
failure floor, and dataset manifest hash. `training_metrics.jsonl` records each
update's reward dispersion, advantages, entropy, KL, clipping and gradient norm.
`validation_metrics.jsonl` records per-episode outcomes and aggregate comparisons.
`acceptance.json` records the actual acceptance result, including non-improvement.

`checkpoints/{initial,last,best}.pt` contain model/optimizer state, update, config,
seeds, schema version and RNG state. Compressed `rollouts/group*.pt.gz` retain
each pre-action feature tensor, graph, IDs, eligible mask, selected action,
behavior probabilities and trajectory ID. Terminal legality, delay, reward and
conflict ownership are stored separately as outcomes. They can support a later
hindsight method; current inference sees none of them. SDPO is not implemented.

The 12-case training shard tests the pipeline and is not claimed sufficient for
a final scheduler. Actual experiment evidence is in `build/block4/acceptance.json`.
