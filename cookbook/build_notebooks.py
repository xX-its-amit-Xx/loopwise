"""Generate the cookbook notebooks from source, then execute them.

The notebooks are generated rather than hand-edited so that the prose and code
stay in one reviewable place and the committed .ipynb files are always the
executed output of exactly this source. Run:

    python cookbook/build_notebooks.py            # build and execute all
    python cookbook/build_notebooks.py 01 03      # only those
    python cookbook/build_notebooks.py --no-exec  # build without executing
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

COOKBOOK = Path(__file__).parent
FIGURES = COOKBOOK.parent / "results" / "figures"
TABLES = COOKBOOK.parent / "results" / "tables"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


# ===========================================================================
# 01 - quickstart
# ===========================================================================

NB01 = [
    md("""
# 01 - Quickstart: three policies on a toy pool

**The situation.** You have a model, a pool of 2,000 candidate compounds, and
room for 96 wells per round. You can run six rounds. Which 96 do you pick?

The default answer is *greedy*: take the 96 highest predictions. This notebook
shows what greedy costs you, using a synthetic pool where we know every answer
in advance so we can score the choices.

**What you will see:** greedy wins the first few rounds decisively, then its
productivity collapses, because it has spent every well confirming what the
model already believed.
"""),
    code("""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from loopwise import Campaign, available_policies
from loopwise.datasets import make_synthetic_pool
from loopwise.replay import compare_policies, plot_comparison

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
print("policies available:", available_policies())
"""),
    md("""
## The pool

Two properties of this synthetic pool are deliberate, because they are what
make real screening hard:

1. **Compounds come in clusters.** Real libraries are built from analogue
   series -- dozens of variations on one scaffold. This is why a diversity
   constraint can matter: the top 96 by any score are often near-duplicates,
   so you have bought one measurement 96 times.
2. **The hit rate is 5%.** Activity depends on a few features plus noise. At
   this rate, a greedy batch that finds 20 hits looks like a triumph while
   teaching the model almost nothing.
"""),
    code("""
pool, info = make_synthetic_pool(
    n_items=2000, n_features=20, hit_rate=0.05, n_clusters=12, seed=0
)
feature_cols = info["feature_cols"]
truth = dict(zip(pool["item_id"], pool["label"], strict=True))

print(f"{info['n_compounds']} compounds, {info['n_actives']} actives "
      f"({info['hit_rate']:.1%} hit rate), {info['n_clusters']} structural clusters")
print(f"activity depends on features {info['informative_features']} of 20")
pool[["item_id", "cluster", "label"] + feature_cols[:4]].head()
"""),
    md("""
## Driving one campaign by hand

This is the whole API. `select()` returns the rows to run plus *why each was
chosen*; `observe()` feeds measurements back; `report()` summarizes.

Note what the `pool` we hand to `Campaign` does **not** contain: the `label`
column. The campaign is blind, exactly as it would be in a lab, and we play the
role of the assay by looking answers up in `truth`.
"""),
    code("""
blind_pool = pool.drop(columns=["label", "latent_activity"])

campaign = Campaign(
    pool=blind_pool,
    model=RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
    budget=96,
    policy="hybrid",
    feature_cols=feature_cols,
    id_col="item_id",
    n_rounds_planned=6,
    seed=0,
)

batch = campaign.select()
print(batch.explain())
"""),
    md("""
Round 1 used the **seeding policy**, not the hybrid policy we asked for. With
no measurements there is nothing to train on, so there is no model to exploit
and no uncertainty to explore -- the only sensible objective is to cover the
feature space. Loopwise does that automatically and tells you it did, rather
than fitting a model to nothing.

### Why each compound was chosen

The reason columns are the point. Six months from now, "why was this compound
on my plate?" is answerable from this CSV alone.
"""),
    code("""
cols = ["item_id", "rank", "posterior_mean", "posterior_std", "reason"]
batch.frame[cols].head(5).style.set_properties(subset=["reason"], **{"width": "520px"})
"""),
    code("""
# Play the assay: look up the true labels for the selected compounds.
results = pd.DataFrame({
    "item_id": batch.frame["item_id"],
    "label": [truth[i] for i in batch.frame["item_id"]],
})
campaign.observe(results)
print(f"round 1: {int(results['label'].sum())} hits in {len(results)} wells "
      f"({results['label'].mean():.1%})")
"""),
    code("""
# Rounds 2-6.
for _ in range(5):
    batch = campaign.select()
    results = pd.DataFrame({
        "item_id": batch.frame["item_id"],
        "label": [truth[i] for i in batch.frame["item_id"]],
    })
    campaign.observe(results)
    alpha = batch.diagnostics.get("alpha")
    print(f"round {batch.round_index + 1}: {int(results['label'].sum()):>3} hits "
          f"({results['label'].mean():>5.1%})"
          + (f", alpha={alpha:.2f}" if alpha else ""))
"""),
    md("""
## The report

`report()` gives the three things worth tracking. Passing `oracle_labels` is
only possible here because this is a simulation -- in a live campaign you do
not know the answers, and Loopwise says so instead of inventing a regret curve.

**Regret** is how many hits a perfectly informed oracle would have found by now
minus how many you did find.

**Calibration drift** is the one people forget. Every round, the model is
retrained on a sample *it chose itself* -- the opposite of a random sample. That
bends calibration predictably: a model fed its own confident picks becomes
over-confident. Expected calibration error (ECE) catches this while accuracy
does not.

**Batch diversity** is mean pairwise distance within the batch, in units of the
pool's own spread. 1.0 means "as spread out as two random compounds"; 0.4 means
you bought a lot of near-duplicates.
"""),
    code("""
report = campaign.report(oracle_labels=truth)
print(report.summary())
"""),
    code("""
show = ["round", "n_hits", "hit_rate", "cumulative_hits", "oracle_cumulative_hits",
        "cumulative_regret", "mean_predicted", "observed_hit_rate", "ece",
        "batch_diversity", "alpha"]
report.frame[show].round(3)
"""),
    md("""
Look at `mean_predicted` against `observed_hit_rate`. Where predictions run
above the observed rate, the model is over-confident about the batch it chose
-- the selection-bias effect, visible in the numbers.
"""),
    code("""
fig = report.plot()
FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGDIR / "01_hybrid_campaign_report.png", dpi=150, bbox_inches="tight")
"""),
    md("""
## Three policies, head to head

One campaign tells you nothing about whether the policy was good -- you have no
counterfactual. The replay harness runs the same pool and the same model under
different policies, hiding all labels and revealing only selected batches.

Three seeds, because the spread between seeds is often as large as the gap
between policies. The shaded bands are the min-max envelope: **a gap smaller
than the bands is not a result.**
"""),
    code("""
long_frame, results = compare_policies(
    pool.drop(columns=["latent_activity"]),
    label_col="label",
    model_factory=lambda: RandomForestClassifier(
        n_estimators=200, random_state=0, n_jobs=-1
    ),
    policies=["random", "greedy", "bald", "hybrid"],
    seeds=(0, 1, 2),
    feature_cols=feature_cols,
    id_col="item_id",
    budget=96,
    n_rounds=6,
    n_initial_labels=96,
    verbose=True,
)
"""),
    code("""
n_actives = results["greedy"][0].n_actives
fig = plot_comparison(
    long_frame,
    n_actives=n_actives,
    title="Synthetic pool: 2,000 compounds, 5% hit rate, 96 wells x 6 rounds (3 seeds)",
)
fig.savefig(FIGDIR / "01_policy_comparison.png", dpi=150, bbox_inches="tight")
"""),
    code("""
final = long_frame[long_frame["round"] == long_frame["round"].max()]
summary = final.groupby("policy").agg(
    cumulative_hits=("cumulative_hits", "mean"),
    hits_sd=("cumulative_hits", "std"),
    fraction_found=("fraction_of_achievable", "mean"),
    final_pr_auc=("pr_auc", "mean"),
    pr_auc_sd=("pr_auc", "std"),
).round(3).sort_values("cumulative_hits", ascending=False)
summary
"""),
    md("""
## The regret curve, and the thing it hides

Cumulative hits is the number a project leader asks for. It also hides the
mechanism, so look at the *per-round* hit rate: that is where the plateau lives.
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for policy, group in long_frame.groupby("policy"):
    per_round = group.groupby("round")
    axes[0].plot(per_round["hit_rate"].mean().index,
                 per_round["hit_rate"].mean() * 100, "o-", label=policy, ms=4)
axes[0].set_xlabel("round"); axes[0].set_ylabel("hit rate this round (%)")
axes[0].set_title("Per-round productivity"); axes[0].legend(fontsize=8)

for policy, group in long_frame.groupby("policy"):
    per_round = group.groupby("round")["cumulative_hits"].mean()
    oracle = np.minimum(np.arange(1, len(per_round) + 1) * 96, n_actives)
    axes[1].plot(per_round.index, oracle - per_round.to_numpy(), "o-", label=policy, ms=4)
axes[1].set_xlabel("round"); axes[1].set_ylabel("hits an oracle would have found, minus yours")
axes[1].set_title("Cumulative regret (lower is better)"); axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "01_regret_curves.png", dpi=150, bbox_inches="tight")
"""),
    md("""
## What to take away

1. **Greedy is strong early.** It is not a straw man, and on an inexhaustible
   pool relative to your budget it can be very hard to beat on hits alone.
2. **Watch per-round productivity, not just the cumulative curve.** The
   cumulative curve is monotone and flatters whoever led early; the per-round
   rate is where a plateau shows up.
3. **Always include `random`.** It is the only line that tells you whether the
   features and model carry signal at all.
4. **Report seed spread.** With three seeds you can already see that some gaps
   here are not real.

Next: `02_replay_public_screen.ipynb` runs this comparison on real measured
screening data, where the answer is less flattering to the premise.
"""),
]


# ===========================================================================
# 02 - replay on public data (the headline)
# ===========================================================================

NB02 = [
    md("""
# 02 - Retrospective replay on real public screens

This is the evidence. Everything else in Loopwise is machinery; this notebook
asks whether the machinery pays for itself on data somebody actually measured.

**The method.** Take a public dataset where every compound has been measured.
Hide all the labels. A policy selects a batch; we reveal only that batch's
labels; the model retrains; repeat. Because the full answer key exists, we can
compute what no live campaign can: what each policy *would have* found, on the
same data, with the same model and budget.

**Two datasets, chosen because they disagree:**

| Dataset | What it is | Hit rate | Why it is here |
| --- | --- | --- | --- |
| **Tox21 NR-AR** | 7,258 compounds screened against the androgen receptor | 4.2% | A realistically scarce primary screen |
| **BACE-1** | 1,513 beta-secretase inhibitors with measured IC50 | 44.6% at pIC50 >= 7 | A dose-response set, where potency is continuous |

**Spoiler, stated up front:** on one of these datasets greedy wins on hits, and
on the other greedy wins on hits *and* produces the worst model of any policy.
Neither result is the tidy one. Both are reported.
"""),
    code("""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from loopwise.datasets import load_bace, load_tox21
from loopwise.replay import compare_policies, plot_comparison

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

POLICIES = ["random", "greedy", "uncertainty", "bald", "thompson", "hybrid", "kdpp"]
SEEDS = (0, 1, 2, 3, 4)
def forest():
    return RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2)
"""),
    md("""
## Dataset 1: Tox21 NR-AR

The Tox21 Data Challenge (2014) screened ~7,800 compounds against 12
nuclear-receptor and stress-response assays. We use the androgen receptor
(NR-AR) assay as a binary screen: 4.2% of the 7,258 measured compounds are
active. That scarcity is the point -- it is the regime where a greedy policy
can look excellent while learning nothing.

Compounds arrive as SMILES strings (text, e.g. `CC(=O)Oc1ccccc1C(=O)O` for
aspirin), which a model cannot consume. We convert them to **Morgan
fingerprints**: for every atom, describe the neighbourhood within 2 bonds, hash
each description into one of 512 buckets, set that bit. Molecules sharing
substructures share bits. This is the standard representation for ligand-based
screening (Rogers & Hahn, 2010).
"""),
    code("""
tox21, tox21_info = load_tox21("NR-AR", n_bits=512, radius=2)
print({k: v for k, v in tox21_info.items() if k != "feature_cols"})
tox21[["mol_id", "smiles", "label"]].head()
"""),
    md("""
### A note on how model quality is measured

PR-AUC (area under the precision-recall curve) rather than ROC-AUC: with a 4%
hit rate, a model can have a fine ROC-AUC while its top 96 predictions are
nearly all inactive, because ROC-AUC averages over thresholds nobody will act
on.

And it is computed on a **fixed 20% holdout that no policy can select.** This
is not fussiness. If you score the model on whatever is left unselected, a
successful policy is punished for its own success: greedy harvests the easy
actives, so the leftovers are the hard ones. We measured it -- scored that way,
greedy gets PR-AUC 0.037 and random gets 0.485, while greedy finds eight times
as many actives. That number describes the difficulty of the leftovers, not the
quality of the model. A fixed holdout keeps the measuring stick still.
"""),
    code("""
tox21_frame, tox21_results = compare_policies(
    tox21,
    label_col="label",
    model_factory=forest,
    policies=POLICIES,
    seeds=SEEDS,
    feature_cols=tox21_info["feature_cols"],
    id_col="mol_id",
    budget=96,
    n_rounds=8,
    n_initial_labels=96,
    holdout_fraction=0.2,
    verbose=True,
)
TABDIR.mkdir(parents=True, exist_ok=True)
tox21_frame.to_csv(TABDIR / "02_tox21_nr_ar_rounds.csv", index=False)
"""),
    code("""
n_actives_tox21 = tox21_results["greedy"][0].n_actives
fig = plot_comparison(
    tox21_frame,
    n_actives=n_actives_tox21,
    title=("Tox21 NR-AR: 7,258 compounds, 4.2% hit rate, "
           "96 wells x 8 rounds, 5 seeds (band = min-max)"),
)
FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGDIR / "02_headline_tox21_cumulative_hits.png", dpi=150, bbox_inches="tight")
"""),
    code("""
def final_table(frame):
    last = frame[frame["round"] == frame["round"].max()]
    return last.groupby("policy").agg(
        hits=("cumulative_hits", "mean"),
        hits_sd=("cumulative_hits", "std"),
        found=("fraction_of_achievable", "mean"),
        pr_auc=("pr_auc", "mean"),
        pr_auc_sd=("pr_auc", "std"),
    ).round(3).sort_values("hits", ascending=False)

final_table(tox21_frame)
"""),
    md("""
### Read that table honestly

Greedy is at or near the top on hits, and the final PR-AUC column is flat
across every model-driven policy -- the differences are smaller than the
seed-to-seed standard deviation. **On this dataset, at this budget, the
information-seeking policies do not buy a better model, and they do not buy
more hits.**

The only policy clearly separated is `random`, which is worse at both. That is
worth stating plainly: the single largest effect here is "use the model at
all", and everything after that is a second-order choice.

So where does exploration earn anything? Look at the *marginal* productivity
rather than the cumulative total.
"""),
    code("""
per_round = tox21_frame.pivot_table(index="round", columns="policy", values="hit_rate")
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for policy in per_round.columns:
    ax.plot(per_round.index, per_round[policy] * 100, "o-", label=policy, ms=4)
ax.set_xlabel("round"); ax.set_ylabel("hit rate this round (%)")
ax.set_title("Tox21 NR-AR: per-round productivity (mean of 5 seeds)")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(FIGDIR / "02_tox21_per_round_hit_rate.png", dpi=150, bbox_inches="tight")
(per_round * 100).round(1)
"""),
    md("""
Greedy's hit rate rises, peaks, and falls. It is consuming a chemotype: the
model knows one region of chemical space, that region has a finite number of
members, and every well spent there confirms what the model already believed.
The diversity-constrained policy's rate is flat or rising over the same rounds.

With 8 rounds of 96 (768 of 7,258 compounds) greedy never runs out. Extend the
horizon and the picture changes -- that experiment is in the README.
"""),
    md("""
## Dataset 2: BACE-1, a dose-response screen

1,513 inhibitors of human beta-secretase 1, each with a measured IC50
(Subramanian et al., 2016). BACE-1 is the enzyme that makes the first cut in
amyloid precursor protein, so it has been an Alzheimer's target for two
decades.

This dataset differs from Tox21 in three ways that matter:

1. **Potency is continuous.** `pIC50` is the negative log10 of the IC50
   concentration: 7.0 means 100 nM, and each unit is a tenfold improvement. We
   threshold at 7.0 to define a "hit", which gives a 44.6% hit rate -- and that
   number is a *choice*. Where you put the threshold decides your hit rate, so
   a campaign's apparent success is partly a decision about bookkeeping.
2. **It is pre-enriched.** These are published inhibitors of one target, not a
   diverse library, so the chemistry is narrow.
3. **It ships with 590 computed physicochemical descriptors**, so no chemistry
   toolkit is needed.

The budget here is 24 wells per round over 10 rounds -- a small-n campaign, the
case Loopwise is named for.
"""),
    code("""
bace, bace_info = load_bace(potency_threshold=7.0, use_descriptors=True)
print({k: v for k, v in bace_info.items() if k != "feature_cols"})
bace[["compound_id", "pIC50", "label"]].describe().round(2)
"""),
    code("""
bace_frame, bace_results = compare_policies(
    bace,
    label_col="label",
    model_factory=forest,
    policies=POLICIES,
    seeds=SEEDS,
    feature_cols=bace_info["feature_cols"],
    id_col="compound_id",
    budget=24,
    n_rounds=10,
    n_initial_labels=24,
    holdout_fraction=0.2,
    verbose=True,
)
bace_frame.to_csv(TABDIR / "02_bace_rounds.csv", index=False)
final_table(bace_frame)
"""),
    code("""
fig = plot_comparison(
    bace_frame,
    n_actives=bace_results["greedy"][0].n_actives,
    title=("BACE-1 dose-response: 1,513 compounds, 44.6% hit rate at pIC50>=7, "
           "24 wells x 10 rounds, 5 seeds"),
)
fig.savefig(FIGDIR / "02_bace_cumulative_hits.png", dpi=150, bbox_inches="tight")
"""),
    md("""
### The trade-off, finally visible

On BACE the two objectives come apart cleanly. Greedy finds far more hits than
BALD *and* ends with the worst model of any policy -- worse than `random`.

That is the trade-off this package is about, measured rather than asserted. It
also makes the choice concrete: if the campaign's deliverable is a shortlist of
actives, greedy. If the deliverable is a model you will use on the next target,
greedy is destroying the thing you are paying for.
"""),
    code("""
last = bace_frame[bace_frame["round"] == bace_frame["round"].max()]
agg = last.groupby("policy").agg(hits=("cumulative_hits", "mean"),
                                 pr_auc=("pr_auc", "mean"))
fig, ax = plt.subplots(figsize=(6.2, 4.6))
for policy, row in agg.iterrows():
    ax.scatter(row["hits"], row["pr_auc"], s=70)
    ax.annotate(policy, (row["hits"], row["pr_auc"]),
                textcoords="offset points", xytext=(7, 3), fontsize=9)
ax.set_xlabel("cumulative hits found (mean of 5 seeds)")
ax.set_ylabel("final PR-AUC on the never-selected holdout")
ax.set_title("BACE-1: hits now vs model quality later\\n(up and to the right is better; nothing is there)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGDIR / "02_bace_hits_vs_model_quality.png", dpi=150, bbox_inches="tight")
agg.round(3).sort_values("hits", ascending=False)
"""),
    md("""
## Rounds to reach half the achievable hits

The operational question is rarely "how many actives in total" -- it is "how
many plates until I have enough to move on". `rounds_to_fraction` answers it.
"""),
    code("""
rows = []
for name, dataset in (("Tox21 NR-AR", tox21_results), ("BACE-1", bace_results)):
    for policy, runs in dataset.items():
        reached = [r.rounds_to_fraction(0.5) for r in runs]
        got = [r for r in reached if r is not None]
        rows.append({
            "dataset": name,
            "policy": policy,
            "rounds_to_50pct": np.mean(got).round(1) if got else None,
            "seeds_reaching_50pct": f"{len(got)}/{len(runs)}",
            "final_pr_auc": round(float(np.mean([r.final_pr_auc for r in runs])), 3),
        })
pd.DataFrame(rows).pivot(index="policy", columns="dataset",
                         values=["rounds_to_50pct", "final_pr_auc"])
"""),
    md("""
## Does a noisy assay change the answer?

Everything above assumes the measurement is correct. Real assays are not. With
`label_noise=0.2`, each revealed label has a 20% chance of being flipped when
the model sees it -- while scoring still uses the true labels, so we measure how
much of a policy's advantage survives.
"""),
    code("""
noisy_frame, noisy_results = compare_policies(
    tox21,
    label_col="label",
    model_factory=forest,
    policies=["random", "greedy", "bald", "hybrid"],
    seeds=(0, 1, 2),
    feature_cols=tox21_info["feature_cols"],
    id_col="mol_id",
    budget=96,
    n_rounds=6,
    n_initial_labels=96,
    label_noise=0.2,
    verbose=False,
)
clean_frame, _ = compare_policies(
    tox21,
    label_col="label",
    model_factory=forest,
    policies=["random", "greedy", "bald", "hybrid"],
    seeds=(0, 1, 2),
    feature_cols=tox21_info["feature_cols"],
    id_col="mol_id",
    budget=96,
    n_rounds=6,
    n_initial_labels=96,
    verbose=False,
)
comparison = pd.DataFrame({
    "clean": final_table(clean_frame)["hits"],
    "20% label noise": final_table(noisy_frame)["hits"],
})
comparison["lost to noise"] = (comparison["clean"] - comparison["20% label noise"]).round(1)
comparison.round(1)
"""),
    md("""
## What replay cannot tell you

Worth being explicit, because this notebook is the repo's main claim:

1. **The pool is a measured library, not yours.** Tox21 compounds were selected
   for a screening programme; a vendor catalogue is less information-rich and
   its hit rate is lower.
2. **No synthesis delay, no attrition, no compound availability.** Round 4 here
   happens instantly; in a lab it happens in six weeks.
3. **The featurization is frozen.** Real campaigns change their features as
   they learn the chemistry, and that often matters more than the selection
   policy does.
4. **One dataset is one dataset.** We ran two, they disagreed, and the honest
   conclusion is the conditional one: *which* policy wins depends on the hit
   rate, the budget-to-pool ratio, the horizon, and which deliverable you are
   paying for.

Run `loopwise replay tox21 --n-seeds 5` to reproduce the Tox21 panel from the
command line.
"""),
]


# ===========================================================================
# 03 - plate layout
# ===========================================================================

NB03 = [
    md("""
# 03 - Plate layout: controls, edge effects, and why order matters

A policy hands you 96 compounds. Pipette them into the plate in the order the
policy returned them and you have introduced a bias that can be larger than
the effect you are measuring -- and that is *perfectly correlated with the
model's predictions*.

This notebook builds a real 96-well plate map, shows the confound, and shows
the fix.

## The biology

The outer ring of a microplate behaves differently from the interior:

- **Evaporation.** Edge wells have more exposed surface relative to volume and
  sit against the plate walls, so they lose solvent faster and their contents
  concentrate.
- **Temperature.** A stacked plate's interior equilibrates more slowly than its
  edges, so edge wells spend an incubation at a different temperature.
- **Gas exchange.** In cell-based assays, edge wells exchange gases differently
  and the cells grow differently.

The bias is typically 5-20% of signal and is systematic, so replicates in edge
wells do not average it away (Lundholt et al., 2003; Caraus et al., 2015).
"""),
    code("""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loopwise.plate import (
    PLATE_96,
    PLATE_384,
    PlateLayout,
    PlateTemplate,
    render_plate_map,
)

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
print(PLATE_96.describe())
print(PLATE_384.describe())
"""),
    md("""
## The template

The default 96-well layout reserves column 1 for negative (vehicle/DMSO)
controls and column 12 for positive controls. That is the bench convention, and
it has a useful side effect: both control columns are *themselves* on edges, so
the controls sample the edge effect and can be used to estimate it.

These wells are **reserved**: Loopwise will never place a sample in one. A
control silently overwritten by a sample gives you a plate you cannot
interpret and will not know you cannot interpret, so this is a hard invariant
with a test, not a convention.
"""),
    code("""
print("positive controls:", PLATE_96.positive_control_wells)
print("negative controls:", PLATE_96.negative_control_wells)
print("reserved wells:", len(PLATE_96.reserved_wells))
print("available for samples:", PLATE_96.usable_wells)
print("\\nedge wells among available sample wells:",
      sum(PLATE_96.is_edge(*PLATE_96.parse_well(w)) for w in PLATE_96.sample_wells),
      f"of {PLATE_96.usable_wells}")
"""),
    md("""
## A batch to lay out

80 compounds, with predicted scores running from 0.95 down to 0.02 -- i.e. a
batch in score order, which is what every selection policy returns.
"""),
    code("""
n = 80
compound_ids = [f"CPD{i:03d}" for i in range(n)]
predicted = np.linspace(0.95, 0.02, n)
print(f"{n} compounds, predicted scores {predicted.max():.2f} down to {predicted.min():.2f}")
"""),
    md("""
## The naive layout, and the confound

`randomize=False` fills wells in reading order (A1, A2, ... H12) with compounds
in the order given. This is what a liquid handler does by default.
"""),
    code("""
naive = PlateLayout.assign(
    compound_ids, template=PLATE_96, predicted_scores=predicted,
    seed=0, randomize=False, n_strata=4,
)
print(naive.summary())
print("\\nscore stratum by well (3 = highest-scoring quarter):")
naive.to_grid(0, "score_stratum")
"""),
    md("""
Every compound in the top-scoring quarter (stratum 3) is in rows A and B.
**Row A is an edge row.** The top stratum's edge enrichment is 2.0 -- twice
what chance would give -- so the highest predictions are systematically the
most edge-exposed wells on the plate.

If edge evaporation raises the signal, these compounds all read high, you
"confirm" the model, retrain on the confirmation, and chase an artifact. In an
active-learning loop that is worse than in a one-shot screen, because the
biased measurements become training data: the model learns the plate's
geometry.
"""),
    code("""
fig = render_plate_map(naive, title="NAIVE layout: score order into reading order")
FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGDIR / "03_plate_naive_confounded.png", dpi=150, bbox_inches="tight")
"""),
    md("""
## The stratified, randomized layout

The fix: bucket compounds by predicted score, then assign so each bucket is
spread across edge and interior wells and across all four quadrants *in
proportion to how many such wells exist*. Randomization within that constraint
keeps the assignment unbiased; the stratification stops chance from clustering
the predicted actives.

The obvious implementation is wrong, and the way it is wrong is instructive.
Dealing one compound per (plate, edge, quadrant) cell per turn -- plain
round-robin -- drains the small cells first: this plate has 20 edge sample
wells against 60 interior ones, so by the time the last stratum is dealt the
edge cells are empty and that stratum gets *no* edge wells. Still a
score/position correlation, just inverted. Loopwise instead spaces each cell's
wells evenly through a global sequence and does the same for the strata, so
zipping them balances both -- Bresenham's line algorithm, applied to wells.
"""),
    code("""
layout = PlateLayout.assign(
    compound_ids, template=PLATE_96, predicted_scores=predicted, seed=7, n_strata=4,
)
print(layout.summary())
layout.to_grid(0, "score_stratum")
"""),
    code("""
fig = render_plate_map(layout, title="STRATIFIED layout: each score stratum spread across the plate")
fig.savefig(FIGDIR / "03_plate_stratified.png", dpi=150, bbox_inches="tight")
"""),
    code("""
rows = []
for seed in range(8):
    lay = PlateLayout.assign(compound_ids, template=PLATE_96,
                             predicted_scores=predicted, seed=seed)
    naive_lay = PlateLayout.assign(compound_ids, template=PLATE_96,
                                   predicted_scores=predicted, seed=seed, randomize=False)
    rows.append({
        "seed": seed,
        "stratified: top-stratum edge enrichment":
            lay.diagnostics["top_stratum_edge_enrichment"],
        "naive: top-stratum edge enrichment":
            naive_lay.diagnostics["top_stratum_edge_enrichment"],
    })
enrichment = pd.DataFrame(rows).set_index("seed")
print("1.0 = exactly what chance would give; 2.0 = twice as edge-exposed as it should be")
enrichment
"""),
    md("""
The stratified layout hovers around 1.0 across seeds; the naive layout is
pinned at 2.0 every time, because it is not random -- it is deterministic and
wrong.

The ratio is coarse: 20 edge wells across 4 strata means chance gives each
stratum 5 wells, so one well moves the ratio by 0.2. Read it as "not
systematically clustered", not "uniform to three decimals".
"""),
    md("""
## Replicates

With `n_replicates=3`, 40 compounds need 120 wells -- two plates, since controls
leave 80 usable wells each. Replicates are placed in separate passes over the
global well sequence, so a compound's replicates land in different cells and
usually on different plates.

Why that matters: replicates sharing a position bias produce a reassuringly
tight standard deviation around a systematically wrong mean. That is worse than
no replicates, because it converts a known unknown into a false certainty.
"""),
    code("""
triplicate = PlateLayout.assign(
    compound_ids[:40], template=PLATE_96,
    predicted_scores=predicted[:40], n_replicates=3, seed=3,
)
frame = triplicate.to_frame()
samples = frame[frame["content_type"] == "sample"]

print(f"{len(samples)} sample wells across {triplicate.n_plates} plates")
print(f"controls intact on every plate: {triplicate.control_wells_intact()}")
print("\\nmean distinct plates per compound: "
      f"{samples.groupby('item_id')['plate'].nunique().mean():.2f} of 3 replicates")
print("mean distinct rows per compound:  "
      f"{samples.groupby('item_id')['row'].nunique().mean():.2f}")
samples[samples["item_id"] == "CPD000"][
    ["plate", "well", "row", "column", "replicate", "is_edge"]
]
"""),
    code("""
fig = render_plate_map(triplicate)
fig.savefig(FIGDIR / "03_plate_triplicate_two_plates.png", dpi=150, bbox_inches="tight")
"""),
    md("""
## 384-well plates

`exclude_edge_wells` defaults to **True** on 384-well plates and False on
96-well. The ring is 20% of a 384-well plate against 36% of a 96, so
sacrificing it is far cheaper -- and 384-well edge effects are *worse*, because
smaller wells have a higher surface-to-volume ratio and evaporate
proportionally faster.
"""),
    code("""
plate384 = PlateTemplate.plate384(exclude_edge_wells=True)
layout384 = PlateLayout.assign(
    [f"X{i:04d}" for i in range(280)], template=plate384,
    predicted_scores=np.linspace(0, 1, 280), seed=2,
)
print(layout384.summary())
fig = render_plate_map(layout384, annotate=False)
fig.savefig(FIGDIR / "03_plate_384_edge_excluded.png", dpi=150, bbox_inches="tight")
"""),
    md("""
Every sample well is in the interior; the ring is empty. Edge effects cannot
confound the samples at all, at the cost of 76 wells.

## The output you hand the lab

A CSV, one row per well.
"""),
    code("""
TABDIR.mkdir(parents=True, exist_ok=True)
path = layout.to_csv(str(TABDIR / "03_plate_map_96well.csv"))
written = pd.read_csv(path)
print(f"wrote {len(written)} wells to {path}")
written.head(14)
"""),
    md("""
Columns: `plate, well, row, column, content_type, item_id, replicate,
predicted_score, score_stratum, is_edge`.

`is_edge`, `row` and `column` are there so that if you want to *correct* for
spatial bias rather than just avoid confounding it, you can fit a
plate-position model to your control wells afterwards.

## What this does not handle

- It does not model the edge effect, only avoid confounding it.
- It does not know your liquid handler; translating the CSV into a worklist is
  instrument-specific.
- It does not handle compound-specific constraints -- DMSO solubility limits,
  compounds that must not be adjacent, dispense ordering for carryover. Generate
  the layout, then post-process the CSV.
"""),
]


# ===========================================================================
# 04 - FoldRescue end to end
# ===========================================================================

NB04 = [
    md("""
# 04 - End to end: FoldRescue predictions -> Loopwise batch -> plate map

> **This notebook uses a labelled stand-in, not FoldRescue.** The upstream
> FoldRescue model is not installed here, so `SyntheticFoldRescueModel` plays
> its part: a seeded simulator with a plausible structure-activity
> relationship. Everything about the *integration* is real -- the adapter, the
> budget arithmetic, the plate map. The *numbers* are simulated and are not
> predictions about GLB1. The last section shows the one-line change to point
> it at the real model.

## The biology

Some genetic diseases are not caused by a missing protein but by a
**misfolded** one. The gene is transcribed, the protein is made, and then the
cell's quality-control machinery sees a shape it does not recognize and
destroys it. The enzyme never reaches the compartment where it does its job.

**GM1 gangliosidosis** works this way. Mutations in *GLB1* destabilize
beta-galactosidase -- the lysosomal enzyme that breaks down GM1 ganglioside, a
lipid abundant in neuronal membranes. The misfolded enzyme is degraded before
it reaches the lysosome, GM1 accumulates, and neurons die. It is a
devastating, usually infantile disease.

A **pharmacological chaperone** is a small molecule that binds the unstable
protein and holds it in the right shape long enough to pass quality control and
reach the lysosome. Enough enzyme arrives to clear the substrate. The appeal is
that it rescues the patient's *own* enzyme -- no gene therapy, no enzyme
infusion -- and some chaperones cross the blood-brain barrier, which matters
when the disease is neurological.

## Why this is a Loopwise problem

- The candidate space is large (thousands of compounds).
- The readout is a cell-based enzyme-activity assay: days per round.
- The genuine rescue rate is low -- commonly under 1%.
- You can run a few hundred compounds a quarter.

A few thousand candidates, 96 wells, and a model that is wrong in ways you will
only discover by measuring the right things. That is exactly the regime where
*which* compounds you pick dominates how fast you learn.
"""),
    code("""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loopwise.contrib.foldrescue import (
    FOLDRESCUE_PROTOCOL,
    FoldRescueAdapter,
    SyntheticFoldRescueModel,
    foldrescue_campaign,
)
from loopwise.plate import render_plate_map

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

print("FoldRescue models must expose one of:", FOLDRESCUE_PROTOCOL["one_of"])
print("optionally:", FOLDRESCUE_PROTOCOL["optional"])
"""),
    md("""
## The candidate pool

Two features of this pool mirror a real chaperone screen:

1. **Scaffold clustering.** Compounds come from 15 chemical series, so the top
   predictions cluster -- which is what makes a diversity constraint earn its
   keep.
2. **A non-monotone optimum.** Rescue depends on binding affinity through an
   inverted U: bind too weakly and nothing happens; bind too tightly and you
   *inhibit* the enzyme you meant to rescue. This is a real and well-known
   trade-off for iminosugar chaperones such as those explored for GM1 and
   Fabry disease. A model that assumes "more binding is better" gets the top of
   its ranking exactly wrong -- a realistic failure for an active strategy to
   recover from.

Compounds also carry an **acquisition cost**: about 20% are bespoke syntheses
at $300-900, the rest library compounds at $15-60.
"""),
    code("""
pool, info = SyntheticFoldRescueModel.make_glb1_like_pool(
    n_items=3000, n_features=24, rescue_rate=0.03, n_scaffolds=15, seed=0
)
print({k: v for k, v in info.items() if k != "feature_cols"})
pool[["compound_id", "scaffold", "acquisition_cost_usd", "rescues_folding"]].head()
"""),
    code("""
# The inverted-U, visible: rescuers sit at intermediate binding affinity.
fig, ax = plt.subplots(figsize=(6.4, 3.6))
bins = np.linspace(pool["pharmacophore_0"].min(), pool["pharmacophore_0"].max(), 26)
ax.hist(pool.loc[pool["rescues_folding"] == 0, "pharmacophore_0"], bins=bins,
        alpha=0.55, label="no rescue", density=True)
ax.hist(pool.loc[pool["rescues_folding"] == 1, "pharmacophore_0"], bins=bins,
        alpha=0.8, label="rescues folding", density=True)
ax.set_xlabel("pharmacophore_0 (binding affinity proxy)")
ax.set_ylabel("density")
ax.set_title("Rescue peaks at intermediate affinity:\\ntoo weak does nothing, too strong inhibits")
ax.legend(fontsize=8)
fig.tight_layout()
FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGDIR / "04_inverted_u_affinity.png", dpi=150, bbox_inches="tight")
"""),
    md("""
## The adapter

FoldRescue supplies the **model**; Loopwise supplies the **selection**. The
adapter translates, and reports how it obtained uncertainty -- which is worth
printing, because the answer determines which policies are even available.
"""),
    code("""
model = SyntheticFoldRescueModel(random_state=0)
adapter = FoldRescueAdapter(model, refit=True)
print("uncertainty mode:", adapter.uncertainty_mode)
"""),
    md("""
`native-sampling(sample)` means the model exposes its own `sample()` method, so
Loopwise uses the model's own posterior rather than approximating one. The
fallbacks, in order:

| What the model exposes | How uncertainty is obtained |
| --- | --- |
| `sample(X, n)` | used directly -- the model's own posterior |
| `predict_proba` + `fit` | bootstrap ensemble (refit on resamples) |
| `predict_proba` only, frozen | none; information-seeking policies **raise** |

That last row is deliberate. A frozen point model cannot express uncertainty,
so BALD scores would all be exactly zero and the "selected" batch would be
whatever the tie-break produced. Loopwise refuses instead of quietly becoming
random sampling.
"""),
    md("""
## The campaign

The defaults in `foldrescue_campaign` encode what a cell-based rescue assay
actually costs, rather than what is convenient:

- **Duplicates.** A single well of a cell-based enzyme-activity assay is not a
  measurement you would act on. So a 96-well budget buys **48 compounds**, not
  96 -- the replicate policy does that arithmetic.
- **$450 per plate.** Cells, substrate, detection reagent and a technician's
  day. For library compounds this dominates per-compound cost, which is why the
  plate term is not optional.
- **Per-compound acquisition cost** from the pool, so a $600 bespoke synthesis
  and a $30 library compound do not compete equally for the same well.
- **A 96-well plate with controls**: wild-type enzyme (positive) and untreated
  mutant (negative) on every plate. A rescue assay without them is
  uninterpretable.

We seed with 150 already-measured compounds -- the pilot screen a project
typically has before anyone builds a model.
"""),
    code("""
truth = dict(zip(pool["compound_id"], pool["rescues_folding"], strict=True))
blind = pool.drop(columns=["rescues_folding"])
pilot_ids = list(blind["compound_id"][:150])
pilot = {i: truth[i] for i in pilot_ids}
print(f"pilot screen: {len(pilot)} compounds, {sum(pilot.values())} rescuers "
      f"({np.mean(list(pilot.values())):.1%})")

campaign = foldrescue_campaign(
    blind,
    model=model,
    budget=96,
    policy="hybrid",
    feature_cols=info["feature_cols"],
    cost_col="acquisition_cost_usd",
    n_replicates=2,
    plate_fixed_cost=450.0,
    n_rounds_planned=5,
    initial_labels=pilot,
    seed=0,
)
print()
print(campaign)
print(campaign.cost_model.describe())
"""),
    code("""
batch = campaign.select()
print(batch.explain())
"""),
    md("""
96 wells, 2 replicates each, 48 compounds -- and the cost breakdown separates
the acquisition spend from the plate overhead, so you can see which is actually
binding.

### Why each compound was chosen
"""),
    code("""
cols = ["compound_id", "rank", "posterior_mean", "posterior_std",
        "acquisition_cost_usd", "alpha", "reason"]
batch.frame[cols].head(6)
"""),
    md("""
## The plate map

This is the artifact that goes to the bench.
"""),
    code("""
layout = batch.plate_layout
print(layout.summary())
layout.to_grid(0, "content_type").iloc[:, :6]
"""),
    code("""
fig = render_plate_map(
    layout,
    title="GLB1 rescue assay, round 1: 48 compounds in duplicate\\n"
          "(+ = wild-type enzyme control, - = untreated mutant control)",
)
fig.savefig(FIGDIR / "04_glb1_plate_map.png", dpi=150, bbox_inches="tight")
"""),
    code("""
TABDIR.mkdir(parents=True, exist_ok=True)
batch.to_csv(str(TABDIR / "04_glb1_batch_round1.csv"))
layout.to_csv(str(TABDIR / "04_glb1_plate_map_round1.csv"))
print("wrote the batch and the plate map to results/tables/")

# Confirm the invariants that make the plate interpretable.
wells = layout.to_frame()
samples = wells[wells["content_type"] == "sample"]
print(f"\\ncontrols intact on every plate: {layout.control_wells_intact()}")
print(f"sample wells: {len(samples)}; replicates per compound: "
      f"{sorted(samples.groupby('item_id').size().unique())}")
print(f"compounds whose replicates span >1 plate: "
      f"{(samples.groupby('item_id')['plate'].nunique() > 1).sum()} of 48")
print("top-stratum edge enrichment vs chance: "
      f"{layout.diagnostics['top_stratum_edge_enrichment']}")
"""),
    md("""
## Five rounds

We play the assay from the answer key. In a real campaign this is where you
wait three weeks.
"""),
    code("""
for _ in range(5):
    if campaign.round_index > 0:
        batch = campaign.select()
    measured = {i: truth[i] for i in batch.frame["compound_id"]}
    campaign.observe(measured)
    rec = campaign.history[-1]
    print(f"round {rec.round_index + 1}: {rec.n_hits:>2} rescuers in "
          f"{rec.n_selected} compounds ({rec.n_hits / rec.n_selected:>5.1%}), "
          f"${rec.cost:,.0f}"
          + (f", alpha={rec.alpha:.2f}" if rec.alpha else ""))
"""),
    code("""
report = campaign.report(oracle_labels=truth)
print(report.summary())
"""),
    code("""
show = ["round", "n_selected", "n_hits", "hit_rate", "cumulative_hits",
        "cumulative_cost", "cost_per_hit", "ece", "batch_diversity", "alpha"]
report.frame[show].round(3)
"""),
    md("""
`cost_per_hit` is the number a project leader will actually ask about. It
combines everything -- selection quality, replicate policy, plate overhead, and
the acquisition cost of the compounds the policy chose.
"""),
    code("""
fig = report.plot()
fig.savefig(FIGDIR / "04_glb1_campaign_report.png", dpi=150, bbox_inches="tight")
"""),
    md("""
## Pointing this at the real FoldRescue model

One line changes:

```python
from foldrescue import FoldRescueModel          # the real package

model = FoldRescueModel.load("glb1-chaperone-v2")
campaign = foldrescue_campaign(
    pool,
    model=model,                                 # <- the only change
    feature_cols=[...],                          # whatever FoldRescue consumes
    budget=96,
    policy="hybrid",
)
```

`FoldRescueAdapter` discovers how to sample from whatever it is given, so no
Loopwise code changes. Check `adapter.uncertainty_mode` after construction: if
it prints `point-estimate-only`, the model cannot express uncertainty and the
information-seeking policies will raise rather than silently degrade. Fix that
by giving the model a `sample()` method or by passing `n_bootstrap=12` with a
re-fittable model.

### Honest limitations of what you just ran

- The model and the pool are **simulated**. Real GLB1 chaperone chemistry is not
  a 24-dimensional Gaussian mixture.
- The assay is noiseless here. A real cell-based rescue assay has well-to-well
  variation comparable to the effect size, which is why the duplicate policy is
  on by default and why `replay_campaign(..., label_noise=...)` exists.
- Rescue in a cell line is not rescue in a patient. Chaperones that work in
  fibroblasts routinely fail on brain penetration, and that is a question no
  selection policy can answer.
"""),
]


# ===========================================================================

PREAMBLE = """
import sys
from pathlib import Path

# Figures and tables land in results/ so the committed notebooks and the README
# reference the same files.
ROOT = Path.cwd()
if not (ROOT / "loopwise").exists():
    ROOT = ROOT.parent
FIGDIR = ROOT / "results" / "figures"
TABDIR = ROOT / "results" / "tables"
FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)
print("loopwise", __import__("loopwise").__version__, "| figures ->", FIGDIR)
""".strip()

NOTEBOOKS = {
    "01_quickstart.ipynb": NB01,
    "02_replay_public_screen.ipynb": NB02,
    "03_plate_layout.ipynb": NB03,
    "04_driving_foldrescue.ipynb": NB04,
}


def build(name: str, cells: list) -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [cells[0], code(PREAMBLE), *cells[1:]]
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    path = COOKBOOK / name
    nbf.write(nb, path)
    return path


def execute(path: Path, timeout: int = 3600) -> None:
    from nbclient import NotebookClient

    nb = nbf.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(COOKBOOK.parent)}},
    )
    client.execute()
    nbf.write(nb, path)


def main(argv: list[str]) -> int:
    no_exec = "--no-exec" in argv
    wanted = [a for a in argv if not a.startswith("-")]
    for name, cells in NOTEBOOKS.items():
        if wanted and not any(w in name for w in wanted):
            continue
        path = build(name, cells)
        print(f"built {path.name} ({len(cells) + 1} cells)")
        if not no_exec:
            print(f"  executing {path.name} ...", flush=True)
            execute(path)
            print(f"  done {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
