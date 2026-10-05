# Block 3: certified routing inputs and fixed splits

The roadmap stays unchanged: (1) wire engine implemented; (2) complete
environment implemented with hard-tier acceptance open; (3) training data,
this block; (4) RL scheduler later; (5) evaluation and submission later.

Block 2's `case_05` result remains a failure: 3 vertex conflicts after 50 passes.
Its acceptance record and routing algorithms are unchanged. Feasibility
certificates here prove that an input is solvable by the reference router;
they do not assert that our environment or selector can solve it. Admission
never runs or filters on that selector.

## Generate or resume

```sh
python3 -m m3d.data_generation \
  --config configs/routing_data_v1.json --out build/data/routing_v1
```

This command uses the existing `GenConfig` and `generate_feasible()` machinery.
`configs/routing_data_v1.json` declares the seed, counts, limits, and four
families. Parameters are fixed values or inclusive `[minimum, maximum]` ranges.
Integers are sampled uniformly; fractional parameters use a uniform continuous
distribution. Unspecified supported `GenConfig` fields retain upstream defaults.
There is no new chip-layout or routing algorithm.

The initial request is 12 training, 4 validation, and 4 test cases: respectively
3/1/1 cases per family. Split counts must be positive multiples of four.
This shard verifies the data pipeline; it is not enough evidence for effective
RL training. No policy trajectories, networks, rewards, or training are added.

| Family | Requested grid width/height | Nets | Max destinations | Two-pin probability | Local fraction | Cross-die fraction |
| --- | --- | --- | --- | --- | --- | --- |
| Sparse | 20–28 | 8–16 | 3–5 | .65–.85 | .50–.85 | .20–.50 |
| Mixed | 22–30 | 24–45 | 4–7 | .40–.70 | .25–.65 | .25–.65 |
| Dense | 24–30 | 62–78 | 5–7 | .50–.70 | .08–.18 | .35–.60 |
| Branching | 24–32 | 16–26 | 6–9 | .05–.25 | .25–.60 | .40–.75 |

Width and height vary independently. Dense settings follow the hard-tier
regime in `m3d/suite.py`, including 2×2 cells, two pins per cell, no cell gap,
low locality, and negotiated routing. Sparse certification uses the baseline;
the other families use the repository's negotiated router with its existing
defaults. All start at six layers with center delay 1, layer slope 1, and via
delay 3, producing the usual `[6,4,2,2,4,6]` layer profile.

The configuration declares a **300-second total generation cap**, **20 seconds
per case**, **15 seconds per outer attempt**, and **two outer attempts maximum
(retry indices 0 and 1)**. The upstream `max_attempts=2` means two attempts per grid size, with
its existing limit of seven sizes: at most 14 upstream candidates per outer
attempt. Both retry levels and reference solving are bounded; the worker is
killed if its time allowance expires. The checker runs inside the timed worker.
A timeout means **not certified within budget**, never proven infeasible.
Aggregate/resume audits also use the remaining total deadline. Small file
publication and process cleanup overhead can follow the wall-clock cutoff.

The upstream generator can add 6 to both grid dimensions after its configured
miss count. Every attempt records actual seeds/dimensions. Every accepted entry
records requested and effective parameters, grid growth, requested/effective net
density, observed destination counts, actual cross-die fraction, and mean
driver-to-sink Manhattan distance. `frac_local` is the generator's assignment
probability; it is not an observed geometric fraction. The report makes enlarged
cases and effective coverage visible instead of labeling them unchanged.

## Split and duplicate contract

Slots `(split, family, index)` are fixed before generation. SHA-256 derives
the parameter seed and each outer generation seed from the master seed and
that slot plus retry index and a purpose tag. Parameters stay fixed across
outer retries. Internal upstream seeds follow its deterministic formula
`base_seed + attempt*100003 + growth*1000003`, and the journal stores the actual
seed of every candidate. Python's randomized `hash()` is never used.

The plan processes test slots, then validation slots, then training slots.
This fixed collision priority means increasing training counts cannot displace
validation/test membership, including when building a fresh larger dataset.
Validation is for checkpoint selection. Test is reserved for final evaluation,
not tuning. Wall-clock cutoffs can change which slots become certified on
different machines; the requested problems/seeds and accepted files remain
reproducible. Failed or unfinished slots are visible rather than replaced with
an easier family.

Routing geometry is canonicalized as grid dimensions plus sorted nets of
`(driver coordinates, sorted sink coordinates)`. The geometry fingerprint ignores
arbitrary IDs, names, cell IDs, list order, delays, and generation metadata.
Cell footprints are not routing blockers in the official model. A separate
full routing-instance fingerprint includes layer/via delays. Exact file hashes
also cover the stored JSON, including IDs, cells, and metadata.

Identical geometry is rejected anywhere in the dataset, even within one split
or with different delays. The same check excludes all released `m3d-instance`
inputs in the repository's `benchmarks*` directories. No released input is copied
or augmented into this dataset. Those public benchmarks have already been
inspected and are **not an untouched held-out test set**.

## Files and provenance

```text
build/data/routing_v1/
  instances/{train,val,test}/<case-id>.json
  certificates/{train,val,test}/<case-id>.sol.json
  manifest.json
  generation_report.json
  .staging/                    # isolated worker requests, results, retry journals
```

Only manifest-accepted `instances/` files are policy inputs. Certificates and
worker artifacts are never read by the split loader or sampler.

Each manifest entry stores its fixed slot, requested/effective seeds and
parameters, both routing fingerprints, instance/certificate file hashes,
reference-router identity/configuration, source fingerprint, checker legality,
reference delay, attempt counts, and generation/certification timing. The
manifest records Python version, Git HEAD, per-file SHA-256 hashes of the
generator/routers/checker/model/pipeline, the split plan, and the released-input
exclusion catalog. Source-file hashes also capture uncommitted code.

The journal records placement failures, unsuccessful reference routing, checker
failures, duplicate rejection, timeouts, interrupted attempts, and shortfalls.
It includes events from failed internal upstream candidates as well as outer
attempts. The report distinguishes rejected outer attempts from these internal
events and includes effective coverage by split/family.

## Resume and integrity

Accepted instance and certificate files are durably written before the atomic
manifest update. Resume checks hashes, fingerprints, fixed split slots,
certificates, and reference delays before continuing. It does not rewrite
accepted files or reshuffle split assignments. Only a monotonic training-count
increase is compatible; validation/test counts, family parameters, seeds,
budgets, source hashes, or the released exclusion catalog cannot change in an
existing dataset. Use another output directory for incompatible configurations.

An interrupted attempt conservatively consumes its reserved time allowance.
Resume continues unattempted slots/retries; it does not grant exhausted cases
extra retries or a larger per-case allowance. Each explicit invocation receives
the configured total cap. A partial dataset remains marked `partial`, including
when all slots exist but aggregate audit is pending. Accepted, individually
checker-certified inputs remain loadable. Missing slots and pending audit are
reported explicitly. A file lock prevents simultaneous writers.

## Load inputs without reference wires

```python
from m3d.routing_data import TrainingSampler, load_split
from m3d.routing_env import RoutingBudget, RoutingEnv

instance, reward_metadata = TrainingSampler("build/data/routing_v1", seed=7).sample()
# reward_metadata.reference_total_delay is normalization metadata, not a route.

validation = load_split("build/data/routing_v1", split="val")  # explicit opt-in
test = load_split("build/data/routing_v1", split="test")       # final evaluation only

# In a script, guard native worker use with if __name__ == "__main__".
with RoutingEnv() as env:
    initial = env.reset(instance, seed=7, budget=RoutingBudget(max_seconds=10, max_calls=1))
    assert initial["routes"] == {}
    result = env.step(env.eligible_net_ids()[0])
```

The seeded sampler selects with replacement from training entries only and
opens only training input files. It returns `(Instance, ReferenceMetadata)`;
metadata contains scalar IDs/fingerprints and reference delay, with no certificate
path or wire tree. Explicit split loaders expose `len(split)`, `split.status`,
and `split.load(index)`. Inputs are freshly loaded and checked against their
stored hashes. The environment is unchanged and always resets to empty wires.

Runnable one-action integration example:

```sh
python3 examples/routing_data_example.py --data build/data/routing_v1
```

This action is a data-interface check, not a policy trajectory or another
Block 2 acceptance attempt.

## Focused checks

```sh
python3 -c 'import subprocess; subprocess.run(["python3", "-m", "unittest", "tests.test_routing_data", "-v"], timeout=60, check=True)'
python3 -m m3d.data_generation --out build/data/routing_v1 --audit-only
```

The compact tests cover deterministic inputs and stable held-out membership,
ID/order/metadata-invariant fingerprints, delay-independent geometry exclusion,
real checker-approved certificates, resume preservation and tamper detection,
cross-split/released duplicates, truthful growth metadata, bounded timeouts,
late-certificate rejection, partial status, and loaders operating with the
certificate directory unavailable. They do not rerun the upstream test suite.

## Actual initial-shard acceptance

The single real-data generation run completed in **40.8761 seconds** of the
declared 300-second cap. All 20 requested cases were accepted on their first
outer and first upstream attempt; no rejected attempts, reference failures,
timeouts, duplicate rejections, grid enlargement, or generation shortfalls
occurred. Aggregate audit completed inside that cap.

| Split | Sparse | Mixed | Dense | Branching | Accepted / rejected outer attempts |
| --- | --- | --- | --- | --- | --- |
| Training | 3 | 3 | 3 | 3 | 12 / 0 |
| Validation | 1 | 1 | 1 | 1 | 4 / 0 |
| Test | 1 | 1 | 1 | 1 | 4 / 0 |

Effective coverage across all three splits (all cases have six layers):

| Family | Width | Height | Nets | Destinations per net | Actual cross-die fraction | Actual multisink fraction |
| --- | --- | --- | --- | --- | --- | --- |
| Sparse | 22–28 | 20–27 | 9–16 | 1–5 | .154–.583 | .125–.308 |
| Mixed | 26–30 | 22–27 | 31–43 | 1–7 | .150–.774 | .175–.516 |
| Dense | 26–30 | 27–30 | 63–75 | 1–7 | .356–.640 | .294–.492 |
| Branching | 24–32 | 24–28 | 16–26 | 1–9 | .400–.737 | .731–.957 |

Fraction ranges in the configuration are probabilities used by the generator;
realized fractions fluctuate in a finite sample and are reported separately.
This small shard covers each family, not every value or endpoint in every
declared range. Per-split/family requested and effective ranges are in the
generation report.

All **20/20** certificates passed the unchanged official checker, with exactly
matching stored reference delays (range 538–15,037). Geometry audit found **0**
duplicates within/across splits or against the **45** released benchmark inputs.
The dataset is marked `complete`, with `audit_pending: false`.

All **8 focused data-contract tests passed** in 5.430 seconds. Tiny test fixtures
also verified resume and count extension without rewriting accepted files or
changing validation/test membership; the real shard was generated only once.

The loader/environment check selected `train-mixed-00000`: reset had no routes
or occupancy, net 0 was eligible, and one Block 1 call installed its route
successfully. Normalization metadata (reference delay 5,141) was returned
separately. No reference wires were loaded and no complete routing episode or
hard-tier acceptance claim was made.

Artifacts:

- [Manifest](../build/data/routing_v1/manifest.json)
- [Generation report and full retry journal](../build/data/routing_v1/generation_report.json)
- [Certificate directory](../build/data/routing_v1/certificates/)
- [Compact acceptance record](BLOCK_3_ACCEPTANCE.json)
- [One-action integration result](../build/block3-integration.json)

Source hashes confirm the existing generator, suite, checker, wire engine,
environment, and Block 2 acceptance record were not edited by this block.
**Block 2's 3-conflict hard-tier acceptance gap remains open.**
