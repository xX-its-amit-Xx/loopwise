# Running a campaign

The `Campaign` object is the thing a lab drives: `select()` -> run the plate ->
`observe()` -> `report()`, repeat.

## Constructing one

```python
from loopwise import Campaign
from sklearn.ensemble import RandomForestClassifier

c = Campaign(
    pool=candidates_df,                 # id column + numeric feature columns
    model=RandomForestClassifier(n_estimators=200),
    budget=96,                          # WELLS per round, not compounds
    policy="hybrid",
    feature_cols=[f"fp_{i}" for i in range(512)],
    id_col="compound_id",
    n_rounds_planned=6,                 # the hybrid schedule needs this
    seed=0,
)
```

**Pass `feature_cols` explicitly.** The default -- every numeric column except
the id, label and cost columns -- is convenient and changes meaning the moment
somebody adds a column to the CSV. Silently training on an extra column is a
quiet, expensive bug.

**`budget` is wells.** With `n_replicates=3` a budget of 96 selects 32
compounds. See [cost and budgets](#cost-and-budgets).

**`n_rounds_planned` is not optional for `hybrid`.** "Anneal toward
exploitation" is meaningless without knowing when the end is; without it,
`alpha` stays at `alpha_start` forever and you have silently chosen a
fixed-alpha policy.

## Three design choices

**The pool is never mutated.** Labels live in `Campaign.labels`, keyed by
compound id. If Loopwise wrote measurements into your DataFrame, a later
`select()` on a stale copy would be training on different data than you think.

**Measured compounds leave the candidate set permanently.** Pass
`allow_retest=True` if your assay is noisy enough that re-measurement is
genuinely informative -- it sometimes is, and then it should be a deliberate
choice rather than a default.

**Calibration is tracked, not assumed.** Round after round, the model is
retrained on a sample that the model itself chose -- the opposite of a random
sample. That bends calibration in a predictable direction, and `report()`
measures it rather than letting you discover it when a predicted 90% hit rate
delivers 30%.

## Round 1

With no labels there is nothing to train on, so there is no model to exploit
and no uncertainty to explore. Loopwise falls back to a **seeding policy** --
space-filling diversity (`quality_weight=0`) -- and says so:

```python
batch = c.select()
batch.diagnostics["used_seed_policy"]   # True
batch.policy                            # 'diversity (seeding)'
```

The same fallback fires, with a warning, when every label so far is the same
class. In a 3%-hit-rate screen that is the *normal* situation for the first few
rounds: no classifier can be fitted on one class, so keep covering the space
until you find a hit.

Seed with data you already have:

```python
c = Campaign(..., initial_labels={"CPD001": 1, "CPD002": 0, ...})
```

## What `select()` returns

A `SelectedBatch` whose `.frame` is the selected pool rows plus the policy's
reason columns:

| Column | Meaning |
| --- | --- |
| `rank` | position in the policy's ordering, 1 = first chosen |
| `posterior_mean` | the model's point prediction |
| `posterior_std` | spread across posterior samples |
| `reason` | a human-readable sentence |
| `alpha` | hybrid only: the exploit weight used this round |
| `mutual_information` | BALD only |
| `marginal_gain`, `max_similarity_to_earlier_pick` | diversity policies |
| `posterior_draw_id`, `drawn_score` | Thompson only |

```python
print(batch.explain())
batch.to_csv("round1_batch.csv")
batch.save_plate_map("plate_map.csv", "plate_map.png")
```

One CSV answers "what did we run, and why each one?" six months later. That is
the point of the reason columns.

**Selecting twice without observing raises.** The second batch would be chosen
by the same model as the first, so it is one batch of 192, not two rounds of
96. Call `observe()`, or `discard_pending()` to abandon the batch.

## Feeding results back

```python
c.observe(results_df)                    # needs id and label columns
c.observe({"CPD001": 1, "CPD002": 0})    # or a plain mapping
c.observe(results_df, label_col="pct_rescue")
```

Replicate rows for the same compound are averaged, which is the only sensible
default; pre-aggregate yourself for a median or a curve fit.

Two warnings worth reading rather than suppressing:

- *"N measured id(s) are not in the pool"* -- almost always an id-format
  mismatch. The classic version is ids coming back from the plate reader with a
  plate prefix attached (`PLATE1_CPD00001`).
- *"N of M selected compound(s) have no measurement"* -- failed wells. Those
  compounds stay in the candidate pool and may be selected again.

## The report

```python
report = c.report()                       # live campaign
report = c.report(oracle_labels=truth)    # replay only -- you know every answer
print(report.summary())
report.frame                              # one row per round
report.plot("report.png")
```

| Column | What it tells you |
| --- | --- |
| `hit_rate`, `cumulative_hits` | the number a project leader asks for |
| `cost`, `cumulative_cost`, `cost_per_hit` | what it took |
| `mean_predicted` vs `observed_hit_rate` | over-confidence, visible |
| `brier`, `ece` | calibration; see below |
| `batch_diversity` | mean pairwise distance in pool units; 1.0 = two random compounds |
| `alpha` | the exploit weight actually used |
| `cumulative_regret` | see the warning below |

**Calibration drift.** The Brier score is the mean squared error of a
probability forecast -- predict 0.9 for something that happens and you pay
`(1 - 0.9)^2`. Expected calibration error (ECE) bins compounds by predicted
probability and asks whether the observed hit rate in each bin matches the
predicted one: a model that says "0.8" for compounds of which 30% are actives
contributes 0.5 from that bin. ECE is the number that moves under selection
bias, while accuracy often does not.

**Regret, honestly.** With `oracle_labels` (a replay, where you know every
answer) regret is the real thing: hits a perfectly informed oracle would have
found minus hits you found. Without it, Loopwise reports *realized* regret
against the best hit rate any round of this campaign actually achieved, and
says so in `report.notes`. It is a lower bound and cannot be compared across
campaigns. Loopwise does not invent an oracle it does not have.

## Cost and budgets

```python
from loopwise.cost import CostModel, ReplicatePolicy

c = Campaign(
    ...,
    cost_col="acquisition_cost_usd",
    cost_model=CostModel(
        per_item_cost="acquisition_cost_usd",
        plate_fixed_cost=450.0,
        budget_currency=8000.0,
        replicates=ReplicatePolicy(n_replicates=2, count_against_budget=True),
    ),
)
```

Budget resolution happens *before* the policy runs, so the policy is never
asked for a batch that will then be truncated -- truncating afterwards breaks
the batch-aware policies, whose k-th pick was chosen in light of the other
k-1. The batch is re-priced after selection and trimmed from the bottom of the
ranking if a cost-aware policy overshot.

`batch.plan.explain()` tells you which constraint actually bound: the well
budget, the currency budget, or the candidate pool running out.

## Choosing a model

Any scikit-learn estimator, anything with `predict_proba`, or anything with
`sample()`. The one thing to get right: **information-seeking policies need a
model that can disagree with itself.**

| Model | What to pass |
| --- | --- |
| RandomForest / ExtraTrees | nothing -- per-tree predictions are free |
| LogisticRegression, SVM, GradientBoosting | `n_bootstrap=20` |
| LightGBM | `contrib.LightGBMAdapter` |
| PyTorch | `contrib.TorchEnsembleAdapter` |
| Gaussian process | `contrib.GaussianProcessAdapter` |

Without one of these, `bald`, `uncertainty`, `thompson` and `hybrid` raise
`DegeneratePosteriorError` naming the fix. That refusal is deliberate: a single
fitted classifier gives every information score exactly zero, so the "selected"
batch would be whatever the tie-break produced, and your active-learning
pipeline would have quietly become random sampling.

## Inspecting the model between rounds

```python
predictions = c.predict()          # whole pool: predicted, std, mutual_information
c.adapter.name
c.labeled_arrays()                 # (X, y) measured so far
len(c.labels), c.round_index
```
