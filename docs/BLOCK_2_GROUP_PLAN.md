# Block 2 V2: RL-controlled coordinated repair plan

RL is the intended controller for Block 2 V2. Bbox and deterministic seed/group
selection are optional baseline comparisons only. Neither 9/9 legality with
bbox nor improvement over the previous 4/9 bbox result is a prerequisite for
RL integration, training, evaluation, or V2 acceptance.

## Controller and environment responsibilities

- RL chooses routing and repair actions, including the neighborhood/group to
  reconsider. A heuristic controller is not the primary V2 evaluation path or
  an automatic fallback for an unavailable RL controller.
- Block 2 owns routing state, action validation, whole-group removal, bounded
  candidate reconstruction, joint scoring, rollback, budgets, and best legal
  snapshots. Interaction measurements are available as controller inputs.
- Block 1 constructs wires in the space supplied by Block 2. Its pathfinding
  algorithms remain unchanged. Deterministic candidate orders inside a chosen
  group's reconstruction are search mechanics, not a replacement controller.

## V2 implementation and evaluation plan

- [x] Implement and check the coordinated repair mechanics: all group routes
  removed together, outside routes fixed, complete candidates ranked jointly,
  rollback on failure, bounded search, and preservation of the legal incumbent.
- [ ] Expose the group decision through the RL action interface. The current
  runnable prototype takes a net/seed ID and chooses neighbors heuristically;
  that behavior belongs to the baseline and is not completed RL integration.
- [ ] Adapt the RL controller to the coordinated-repair action semantics.
  Controller integration and later training do not depend on bbox legality.
- [ ] Evaluate the RL-controlled V2 environment from empty routes with declared
  per-case budgets. Report controller/checkpoint, legality, missing nets,
  conflicts, legal delay, runtime, group counts/sizes, and Block 1 calls. If the
  RL controller is unavailable, report the RL evaluation as pending; do not
  substitute a heuristic run and label it V2 acceptance.
- [ ] Optionally run bbox as a separately labeled baseline comparison under
  comparable budgets. Record any budget or concurrency differences. This step
  is optional and never gates the RL path.

Environment readiness is established by the focused correctness checks. Primary
V2 routing performance is measured with RL controlling the environment. No
bbox score or legality threshold is used to decide whether RL work may proceed.
This planning correction does not itself train RL or claim RL readiness.

Block 1 algorithms, training data, and frozen Block 5 results remain outside
this change. No reference/submission warm starts, SAT/CBS/ILP, branches, or
worktrees are introduced.

## Historical prototype and optional baseline evidence

The initial prototype used deterministic bbox seed selection and heuristic
neighborhood growth to exercise the repair mechanism. Its five requested
checks plus existing lifecycle checks passed: 16 focused checks total.

An exclusive-only diagnostic stalled on the first two hard cases and was
stopped during case_03. Negotiated whole-group candidates were then added to
the same transaction; a planar check confirmed a 5-conflict -> 1-conflict
improvement when exclusive reconstruction was impossible. A candidate-call
counter that included undispatched work was also corrected and checked.

The subsequent nine-case bbox baseline produced 1/9 legal, compared with the
previous bbox result of 4/9. This remains a measured baseline regression; it
is neither an RL result nor a gate on the RL-controlled V2 plan.

That baseline used 180 seconds, 50 rounds, 5,000 calls, 200M expansions,
1M expansions/call, and 10 seconds/call, serially. The earlier bbox comparison
used 120 seconds, 20 passes, 1,500 calls, 20M expansions and eight concurrent
cases. These are historical baseline configurations, not prerequisites or
mandatory budgets for RL evaluation.

See [the baseline results](BLOCK_2_GROUP_RESULTS.md) and
[their JSON record](BLOCK_2_GROUP_RESULTS.json). RL-controlled V2 evaluation
has not yet been run.
