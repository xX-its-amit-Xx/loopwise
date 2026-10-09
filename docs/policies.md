# The policies, in detail

Each section says what the policy optimizes, where it came from, when it is the
right choice, and how it fails. The short version is in the README's decision
table; this is the reference.

A note on terms used throughout:

- **Exploitation** -- spending wells on compounds the model already thinks are
  good, to find hits now.
- **Exploration** -- spending wells on compounds that would teach the model the
  most, to find hits later.
- **Epistemic uncertainty** -- the model does not know. Reducible by measuring.
- **Aleatoric uncertainty** -- the assay is noisy. *Not* reducible by
  measuring. Confusing the two is the central mistake in this area, and it is
  what separates BALD from plain uncertainty sampling.

---

## `random`

Uniform random selection without replacement.

**Why it is here.** It is the floor. Any policy that cannot beat random is not
earning the complexity of running it, and a policy comparison that omits random
is not a comparison. In our Tox21 replay, random finds 16 actives where greedy
finds 127 -- but random also produces a *better-calibrated* model than greedy
on BACE, which is the kind of thing you only learn by measuring it.

**Use it.** Always, as a control line in any comparison. And genuinely for
round 1 if you have no model and no features you trust.

**Failure mode.** With a 4% hit rate and 96 wells you expect four actives. Two
rounds of that and the project is cancelled.

---

## `greedy`

Top-`k` by predicted score.

**What it optimizes.** This round's expected hit count, and nothing else.

**Why it is the default everywhere.** It is obviously right if this is your
only round. It also *looks* right for several rounds, because the first few
rounds of greedy on a clustered library are genuinely productive -- the model
knows a chemotype, and the chemotype has many members.

**How it fails, measured.** It exhausts what it knows. In our 30-round Tox21
replay (32 wells/round, 5 seeds), greedy's per-round hit rate runs 24% -> 71%
(round 4) -> 1.2% (round 12). It has consumed its chemotype and the model never
learned anything else, because every measurement confirmed what it already
believed. Meanwhile k-DPP's hit rate is still climbing at round 12. That
collapse is the plateau this package exists to address.

A second, quieter failure: greedy's *model* is the worst of any policy on BACE
(PR-AUC 0.864, against 0.946 for BALD and 0.932 for random). If the campaign's
output is a model you intend to use again -- for the next target, the next
series, the next grant -- greedy is actively destroying the asset.

**Use it.** One-shot selection. A final harvest round. As the reference line
every other policy is measured against.

---

## `uncertainty`

Top-`k` by predictive variance across posterior samples. Lewis & Gale (1994).

**What it optimizes.** Where the model's predictions disagree with themselves.

**Why it works.** The model is least reliable where it is least certain, so
measuring there corrects the most error per well. It is the simplest
exploration policy and is often within noise of the sophisticated ones -- our
Tox21 runs show `uncertainty` and `bald` statistically indistinguishable on
both hits and final PR-AUC.

**How it fails.** It cannot tell "I don't know" from "this is unknowable."
A compound whose posterior samples all say 0.5 has maximal predictive entropy
and zero information value: the model is *confident* the outcome is a coin
flip. Uncertainty sampling will buy a whole plate of those. In a cell-based
assay with genuinely noisy wells, this is not hypothetical.

`measure="margin"` (closeness to the decision boundary) and `"entropy"` are
variants of the same idea with the same blind spot.

**Use it.** When you want an exploration term and your assay is clean. Prefer
`bald` when it is not.

---

## `bald`

Maximize mutual information between a compound's label and the model
parameters: `H[E_theta p] - E_theta H[p]`. Houlsby et al. (2011).

**What it optimizes.** "How much would measuring this change my beliefs about
the model?" The first term is total uncertainty; the second is the part that is
irreducible noise. Subtracting makes the score zero both for compounds the
model is sure about *and* for compounds it is sure are unpredictable.

**Why that subtraction is the whole point.** It is the formal fix for
uncertainty sampling's blind spot. `tests/test_policies.py` encodes the
distinction directly: two compounds with identical mean prediction (0.5), one
with every sample at 0.5 and one with samples split 0.02/0.98. Predictive
entropy cannot separate them; BALD gives the first exactly 0.

**Requirement.** A posterior that can disagree with itself -- a forest, a
bootstrap ensemble, MC dropout, or a GP. A single fitted classifier gives all
scores 0, and Loopwise raises `DegeneratePosteriorError` rather than returning
an arbitrary batch.

**How it fails.** It is pointwise: the top 96 by BALD can still be 96 copies of
the same informative-but-redundant structure. Combine with diversity, or use
`hybrid(explore="diversity")`.

**Use it.** The default exploration term. Best when model quality is an output
you care about.

---

## `expected_model_change`

Two estimators of how much one measurement would move the model.

`mode="kl"` (default) treats the posterior samples as an equally weighted
ensemble, reweights them by each member's likelihood of each possible outcome,
and reports the expected KL divergence between the reweighted and original
weights. That is literally "how much did my belief about which model is right
move?", computed in `O(members)` per candidate with no refitting.

`mode="egl"` is Expected Gradient Length (Settles et al., 2008): the norm of
the gradient the new example would contribute. For logistic loss that is
`2 p (1-p) ||x||` in expectation -- it rewards uncertain compounds that also sit
far from the feature-space origin, i.e. structurally unusual ones.

**How it fails.** `kl` is a change in *ensemble weights*, which is a proxy for
a change in the model, not the thing itself; a bootstrap ensemble whose members
are all similar will report small changes even when a refit would move a lot.
`egl` is tied to a linear surrogate, not to your actual model.

**Use it.** When you want an exploration signal that weights *structural
novelty* alongside uncertainty. In practice it tracks BALD closely (both ~127
hits on our Tox21 large-budget run), so prefer BALD unless you have a reason.

---

## `diversity` (facility location) and `kdpp`

Batch-aware selection over feature space, with optional quality weighting by
predicted score.

**The problem they solve.** Every policy above scores compounds *in isolation*
and takes the top `k`. If compounds A and B are near-identical, measuring A
tells you B's answer -- so B's value after choosing A is near zero. A pointwise
score cannot see that. Pick the top 96 by any score from a vendor library and
you may get three chemical series: 96 measurements, perhaps five independent
ones.

**`objective="facility_location"`** maximizes `sum_pool max_batch sim(i, j)` --
"how well does my batch represent the pool?" The function is submodular, so
greedy is within `1 - 1/e` (~63%) of optimal (Nemhauser et al., 1978). Choose
this when you want *coverage*.

**`objective="kdpp"`** maximizes `log det L_S` for `L_ij = q_i q_j sim(i,j)`.
A determinantal point process makes a set's probability proportional to the
squared volume its vectors span, so near-duplicates -- which span no volume --
are exponentially unlikely. Solved with the fast greedy MAP of Chen et al.
(2018), which needs one kernel **column** per step rather than the full matrix.
Choose this when you want internal *non-redundancy*.

**`quality_weight`** is what keeps these from being chemistry-blind
space-filling designs. The predicted score is raised to this power and
multiplied into the objective, so the batch is diverse *among plausible
actives*. `quality_weight=0` is pure space-filling, which is the correct
objective for round 1 with no model -- and is exactly what `Campaign` uses as
its seeding policy.

**How they fail.** Diversity in the wrong space is worse than no diversity.
These policies are only as good as your features: "diverse" in a feature set
dominated by molecular weight means "a range of molecular weights". And pure
diversity is an expensive way to find hits -- on Tox21, `kdpp` found 54 actives
where greedy found 127. Its value showed up in the *late* rounds, where it was
the only policy whose hit rate was still rising.

**Memory.** These are the dangerous paths. See
[`resources.md`](resources.md) -- above 20,000 candidates they refuse without
`chunked=True`.

---

## `thompson`

Draw one plausible model from the posterior, take its best remaining compound,
repeat for each slot. Thompson (1933).

**What it optimizes.** Nothing explicitly -- and that is the appeal. A compound
enters the batch roughly in proportion to the probability that it genuinely is
the best one, so the exploration/exploitation balance is set by how uncertain
the model actually is rather than by a knob you tuned. A 1933 idea that is
still competitive is worth taking seriously.

**Why per-slot redrawing matters.** With one draw and its top `k`, the batch is
internally correlated (one hypothesis, its 96 favourites). Redrawing per slot
makes slots disagree *because plausible models disagree*, which is diversity
for the right reason. `per_slot_draw=True` is the default.

**Measured behaviour.** The most *stable* policy in our long-horizon Tox21 run:
its per-round hit rate neither spikes like greedy's nor collapses, sitting
around 20-26% from round 3 to round 12 while greedy fell from 71% to 1%. It is
also near the top on final model quality.

**Use it.** When you want one policy with no schedule to tune, and when you
care about performance over a long campaign rather than one impressive round.

---

## `hybrid`

`alpha * exploit + (1 - alpha) * explore`, with `alpha` annealed across rounds.
Scores are rank-normalized before mixing, because a predicted probability and a
BALD score in nats are not commensurable.

**Why it is the practical default.** Campaigns are funded by people who look at
hit rates. A policy that is optimal over eight rounds but returns nothing in
round 1 may not get a round 2.

**The schedule is a funding decision, not a statistical one.** Information
acquired early compounds over the remaining rounds, so the textbook schedule
explores early and harvests late -- Loopwise defaults to
`alpha_start=0.4 -> alpha_end=0.9`. A lab that must show actives immediately
should invert it (`alpha_start=0.8, alpha_end=0.3`), buying political survival
with a worse final model. Both are supported deliberately. What matters is that
the alpha used in each round is recorded in the batch's reason columns and in
`report()`, so the trade-off is auditable after the fact.

Schedules: `linear` (default), `cosine` (slow at both ends), `exponential`
(fast early shift), `step` (switch at the midpoint). `alpha=<float>` fixes it
and disables annealing.

**`explore=` accepts** `"bald"` (default), `"uncertainty"`, `"entropy"`,
`"expected_model_change"`, or a `DiversityGreedyPolicy` instance / `"diversity"`
/ `"diversity:kdpp"` -- the last of which routes the blended score through the
batch-aware greedy, giving you exploitation, information *and* non-redundancy in
one objective. That combination is the most defensible default for a real
multi-round campaign.

**How it fails.** It inherits its explore term's blind spots, and it needs
`n_rounds_planned` to anneal sensibly -- without it, alpha stays at
`alpha_start` forever and you have silently chosen a fixed-alpha policy.

---

## Choosing: the short version

| Situation | Policy |
| --- | --- |
| One round, then done | `greedy` |
| Many rounds, want hits throughout | `hybrid` or `thompson` |
| The model is the deliverable | `bald` or `uncertainty` |
| Round 1, no model yet | `diversity` with `quality_weight=0` (automatic) |
| Library is full of analogue series | add diversity: `hybrid(explore="diversity:kdpp")` |
| Assay is noisy | `bald` over `uncertainty`, and set `label_noise` in replay to check |
| No knobs, long campaign | `thompson` |
| Establishing whether any of this helps | `random` and `greedy` as the two reference lines |
