# Block 2 implementation plan

The user's Block 2 specification is the design authority. Work stays in this
checkout, without commits, worktrees, upstream suite runs, or changes to the
five-block roadmap. Implementation uses the planning, inline execution, TDD,
and verification workflows within those explicit constraints.

## Design

`RoutingEnv` owns an isolated instance, per-net trees, distinct-net ownership,
vertex/edge histories, pass eligibility, finite budgets, and a separately
copied checker-approved incumbent. A step makes exactly one Block 1 call after
ripping up only the selected net, restoring its tree on failure. Foreign pins
are always blocked; other wires are priced. The schedule starts at 0.5,
multiplies present pressure by 1.7 (capped at 1e6), and adds 0.5 per excess
owner to resource history between passes, matching `m3d/negotiated.py`.

The edge-only API receives `delay(e)*(history(e)+present*owners(e))` plus half
the vertex pressure at each endpoint. On a path, internal vertex pressure is
charged once. At a tree junction the surrogate charges degree/2 times the
vertex pressure; ownership itself still counts the net only once. Search
prices never enter reported routing delay.

An environment-owned worker reuses the unchanged `WireEngine` and provides a
wall timeout around each call. Expansion, call, pass, and elapsed budgets are
finite. Timed-out calls reserve their full expansion allowance conservatively.
The thin runner selects largest bounding-box span, then smallest net ID.

## Implementation steps

- [x] Add focused tests for transactional replacement, distinct occupancy,
  foreign pins, invalid actions, reset, snapshot isolation, finite termination,
  vertex-only contention, missing-net retries, and a real multi-net episode.
  Run `python3 -m unittest tests.test_routing_env -v`; expect missing module.
- [x] Implement `m3d/routing_env.py` and `m3d/_routing_worker.py` using the
  existing engine/checker, then run the focused file with a 60-second cap.
- [x] Implement `m3d/routing_runner.py`: external bbox selector, existing
  Submission export only on legal success, explicit failure report otherwise.
- [x] Run only official hard `case_05` from empty wires: 180 seconds,
  50 total passes, 5,000 calls, 200,000,000 total expansions; each call at most
  1,000,000 expansions and 10 seconds. Result: failure at 50 passes, 3 vertex
  conflicts, no missing nets; no legal export available for official checking.
- [x] Review timeout cleanup, observation/snapshot aliasing, sparse net IDs,
  pass boundaries, and failure reporting. Document actual evidence and usage.

No hard-case retries or budget increases without a concrete failure finding.
No training data, RL, comparative evaluation, submission, or reference reads.

## Execution record

- Initial focused file failed because the environment module did not exist;
  implementation then passed all six initial checks.
- Added a time-expiry observation regression: failed, then passed after making
  observations refresh terminal budget state.
- Independent focused review confirmed partial-frame IPC could overrun its
  timeout, and a rejected reset could mix old/new episode state. Both regression
  checks failed before fixes, then passed with deadline-bounded IPC exchange and
  Block 1 validation before publishing reset state.
- Added native worker recovery, mid-pass incumbent preservation, later-conflict
  budget termination, and CLI export/failure checks. Only this new file is run.
- Hard acceptance remained a documented failure. No second hard run or changed
  budget was used. See ROUTING_ENV.md and BLOCK_2_ACCEPTANCE.json for evidence.
