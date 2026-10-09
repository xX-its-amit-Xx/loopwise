# Loopwise documentation

Start with the [project README](../README.md) for the argument, the headline
evidence, and the policy decision table.

| Document | What is in it |
| --- | --- |
| [`campaign.md`](campaign.md) | Running a campaign: `select` / `observe` / `report`, seeding round 1, budgets, calibration drift, choosing a model |
| [`policies.md`](policies.md) | Each policy in detail: what it optimizes, where it came from, when to use it, and how it fails |
| [`plates.md`](plates.md) | Plate layout, reserved controls, edge effects, replicate placement, 384-well plates |
| [`resources.md`](resources.md) | The resource guard, the `n x n` hard cap, what to do when it refuses |

For contributors and AI assistants, [`../CLAUDE.md`](../CLAUDE.md) and
[`../AGENTS.md`](../AGENTS.md) carry the design rationale, the invariants the
tests enforce, and the honest-reporting rules.

## The one-page version

**The problem.** You can measure 96 of 5,000 compounds per round. Greedy
(test the top 96 predictions) maximizes this round's hits and teaches the model
nothing, so the campaign plateaus.

**The mechanism.** All policies read from one abstraction: posterior samples of
shape `(n_samples, n_items)`. Mean gives greedy; variance gives uncertainty
sampling; `H[E p] - E H[p]` gives BALD; one row gives Thompson; a
quality-weighted kernel over the rows gives the diversity policies.

**The practical default.** `policy="hybrid"` with `n_rounds_planned` set:
`alpha * exploit + (1 - alpha) * explore`, with `alpha` annealed across rounds.
Which direction you anneal is a funding decision, not a statistical one.

**The part that makes it runnable.** Plate layout with reserved controls and
edge-effect-aware randomization, a cost model with a per-plate step function,
and a replicate policy. A batch that confounds score with plate position is
worse than no batch.

**The part that makes it believable.** Retrospective replay on measured public
data, multi-seed, with model quality scored on a holdout no policy can select.
The README reports what we found, including the results that cut against the
premise.
