# CLAUDE.md -- working notes for AI assistants in this repository

Read this before changing code. It records the decisions that are easy to undo
by accident and the invariants that tests enforce for a reason.

## What this package is

Loopwise chooses which experiments to run next. A lab has a model, a candidate
pool of thousands, and room for 96 wells. Greedy selection (test the 96 highest
predictions) maximizes this round's hit rate and minimizes what the model
learns. Loopwise implements policies along that trade-off and makes the choice
explicit, auditable, and measurable.

Everything in the repo serves one of three jobs:

1. **Choose a batch** (`policies.py`, `campaign.py`)
2. **Make the batch physically real** (`plate.py`, `cost.py`, `batch.py`)
3. **Prove any of it helps** (`replay.py`, `datasets.py`, the cookbook)

If a change does not serve one of those, it probably belongs elsewhere.

## THE RESOURCE GUARD -- read this before writing any allocation

This package is meant to run on a laptop in a wet lab. The contract:

**Every code path whose memory use depends on the pool size calls
`check_resources()` before allocating.** No exceptions.

```python
from loopwise.utils import check_resources, guard_pairwise_matrix, chunk_slices

report = check_resources(
    (n_rows, n_cols),      # shape about to be allocated
    dtype=np.float32,      # float32 is the default everywhere
    copies=2,              # how many arrays of this size are held at once
    task="k-DPP kernel",   # appears in the message
    n_jobs=n_jobs,         # capped at nproc-1 under pressure
)
report.raise_if_refused()
for block in chunk_slices(n_rows, report.recommended_chunk_rows):
    ...
```

Decisions and thresholds, measured against **available** RAM (not total --
a browser and a Slack client are already holding several GB):

| Fraction of available RAM | Decision | Behaviour |
| --- | --- | --- |
| < 50% | `ok` / `downcast` | allocate; float64 requests come back recommending float32 |
| >= 50% | `chunk` | stream `recommended_chunk_rows` at a time, `n_jobs` capped at `nproc - 1` |
| >= 90% | `refuse` | raise `ResourceRefusal` with an actionable message |

**The dangerous paths are k-DPP and facility-location selection.** Both want an
`n x n` similarity matrix over the candidate pool. At n = 50,000 that is 9.3 GiB
in float32. The hard rule, enforced by `guard_pairwise_matrix()`:

> A dense `n x n` matrix is **never** materialized for `n > 20_000` unless the
> caller passes `chunked=True`. This fires regardless of available RAM -- on a
> machine with a terabyte free it still refuses, because at that size the dense
> path is a design error.

With `chunked=True`:

- **k-DPP** streams kernel columns (`O(n k)` memory). `tests/test_resources.py`
  asserts the chunked path gives **bit-identical** selections, which is what
  makes the claim in the refusal message true. If you change the k-DPP inner
  loop, that test is the one that matters.
- **Facility location** subsamples the reference set and uses stochastic greedy.
  This *is* an approximation and reports `"exact": False` in its diagnostics.
  Do not quietly make it the default for small pools.

Rules for refusal messages: say what was requested, what the machine has, and
**what to do instead**. A refusal that does not name a remedy is a bug.

`float32` is the default dtype everywhere (`utils.DEFAULT_DTYPE`). Screening
data has two or three significant figures.

## Invariants the tests enforce

Do not "fix" a test that fails after your change until you have checked which
of these you broke.

1. **Every policy returns exactly `budget` unique items.** The batch-aware
   policies build selections incrementally and can terminate early; k-DPP
   explicitly backfills by quality when residuals collapse. See
   `tests/test_policies.py`.
2. **A sample is never placed in a reserved control well.** Checked inline in
   `PlateLayout.assign` and in `tests/test_plate.py` across templates, seeds and
   replicate counts. Every plate gets a full set of controls, not just plate 1.
3. **Replay is deterministic under a fixed seed**, and independent of the global
   NumPy random state. This is why Loopwise threads one `np.random.Generator`
   through everything and never calls `np.random.*`. If you add a random draw,
   take it from `ctx.rng` or `campaign.rng`.
4. **The resource guard refuses an oversized DPP** with an actionable message.
5. **The pool DataFrame is never mutated.** Labels live in `Campaign.labels`,
   keyed by compound id.
6. **Measured compounds leave the candidate set** unless `allow_retest=True`.

## Design decisions worth not re-litigating

**One posterior, every policy.** `models.Posterior` holds `(n_samples, n_items)`
draws. Mean gives greedy, variance gives uncertainty sampling, `H[E p] - E H[p]`
gives BALD, one row gives Thompson. Adding a model family means writing one
adapter in `contrib/` and touching no policy. Resist adding a policy that reaches
past this abstraction for model internals.

**A member's positive-class column comes from the MEMBER's `classes_`.**
Bagging subsamples rows, so in a low-hit-rate screen many ensemble members see
no actives and their `predict_proba` returns a single column. Indexing that
with the *ensemble's* class order reads P(the one class it knows) as
P(active) -- asserting every compound is certainly active, and corrupting the
posterior mean, the variance and the BALD score at once. `_proba_positive`
handles every case; use `strict=True` for members (absent positive class means
probability zero) and `strict=False` for a top-level model. The tell is
`posterior().mean` disagreeing with `predict_mean()` on the same fitted model,
and `tests/test_contrib.py` asserts they agree.

**Every hit count uses `hit_threshold`.** It is carried onto `RoundRecord`
rather than defaulted to 0.5, because `_calibration_metrics` and the oracle
comparison both use the configured threshold. Hard-coding 0.5 anywhere makes a
potency campaign (pIC50, threshold 6.0) report a 100% hit rate, negative
regret, and "fraction of achievable" above 1.0.

**A flipped label crosses `hit_threshold`, it does not become 0.0/1.0.**
Hit-ness is re-derived as `value > hit_threshold`, so writing a literal 1.0 is
still an inactive whenever the threshold sits outside (0, 1). Noise would then
only destroy actives and never create false ones, and `label_noise` would be
measuring label deletion rather than label noise.

**Degenerate posteriors raise, never degrade.** A single fitted classifier gives
identical samples, so every information score is exactly zero and the "selected"
batch is whatever the tie-break produced. `Posterior.require_informative()`
raises `DegeneratePosteriorError` naming the fix. This silent failure -- an
active-learning pipeline quietly becoming random sampling -- is the single most
common way this kind of code is wrong in the wild. Keep it loud.

**Rank-normalize before mixing scores.** A predicted probability (0-1) and a
BALD score (nats) are not commensurable, so `alpha * exploit + (1-alpha) *
explore` on raw values is meaningless. `utils.normalize_scores` does this.

**The hybrid annealing direction is a funding decision, not a statistical one.**
Default `alpha_start=0.4 -> alpha_end=0.9` (explore early, harvest late) because
information acquired early compounds over remaining rounds. A lab that needs
hits in round 1 to keep its funding should invert it. Both are supported; the
alpha actually used is recorded in every batch's reason columns. Do not remove
the inverted option because it is "wrong" -- it is right for its constraint.

**PR-AUC, not ROC-AUC, and on a fixed holdout.** With a 4% hit rate, ROC-AUC
can look excellent while the top 96 predictions are nearly all inactive. And
model quality must be scored on compounds no policy could select: scoring on
the *unselected remainder* punishes a policy for its own success. Measured on
Tox21 NR-AR, greedy scored PR-AUC 0.037 against random's 0.485 while finding
eight times as many actives -- that number described the difficulty of the
leftovers, not the model. `replay.py` defaults to `evaluate_on="holdout"`.

**Plate assignment uses proportional interleaving, not round-robin.** Dealing
one compound per (plate, edge, quadrant) cell per turn drains the small edge
cells first -- a 96-well plate with control columns has 20 edge sample wells
against 60 interior -- so the last stratum gets *no* edge wells. That is still a
score/position correlation, just inverted. See `PlateLayout._interleave`.

## Honest-reporting rules

This repo's value is its evidence. Protect it.

- **Never report a policy comparison from a single seed.** The spread across
  seeds is often as large as the gap between policies. `compare_policies` takes
  `seeds` for this reason, and `plot_comparison` shades the min-max envelope so
  a gap smaller than the bands is visibly not a result.
- **Replay measures selection only.** No assay noise unless `label_noise` is
  set, no synthesis delay, no compound attrition, fixed featurization. The
  `replay.py` module docstring lists the limits; keep it current.
- **Report results that go against the thesis.** On Tox21 NR-AR with a large
  budget, greedy ties the best policy on hits and nothing beats it
  significantly on final model quality. On BACE, greedy finds 83% more hits
  *and* has the worst model. Both are in the README. If a change makes a policy
  look better, check whether it made the measurement weaker.
- The `SyntheticFoldRescueModel` and `make_glb1_like_pool` are **stand-ins**,
  clearly labelled. Their numbers must never be presented as GLB1 predictions.

## Conventions

- Docstrings explain *why*, including the biology or statistics, because the
  intended reader is a bench scientist who did not write this. Keep that level.
- Public API is re-exported in `loopwise/__init__.py`; keep `__all__` current.
- Optional dependencies (LightGBM, PyTorch, RDKit) are imported lazily inside
  functions. `import loopwise.contrib` must work without them, and CI checks it.
- `pytest -m "not network"` must pass with core dependencies only.
- Lint with `ruff check loopwise tests && ruff format loopwise tests`.

## Commands

```bash
pip install -e ".[dev]"            # core + pytest
pip install -e ".[all]"            # plus lightgbm, torch, rdkit, notebooks
pytest -m "not network"            # the gate
pytest -m network                  # downloads Tox21 and BACE
loopwise check                     # what this machine can handle
loopwise replay tox21 --n-seeds 5  # reproduce the headline comparison
```
