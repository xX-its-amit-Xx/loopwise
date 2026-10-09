"""Retrospective replay: the only honest evidence that any of this helps.

The idea
--------
Take a public dataset where *every* compound has already been measured. Hide
all the labels. Run a campaign: a policy selects a batch, we reveal only that
batch's labels, retrain, and repeat. Because the full answer key exists, we can
compute what no live campaign can -- what each policy would have found, on the
same data, against the same model, with the same budget.

This is the experiment the README's headline figure reports, and it is the
reason to trust (or dismiss) everything else in the package. A selection policy
that cannot beat ``random`` in replay is not worth the complexity of running
it, and replay is cheap enough that there is no excuse for not checking.

What replay can and cannot tell you
-----------------------------------
It is an honest simulation of *selection*, and that is all. Four limits, worth
stating because they bound how much weight the headline figure can carry:

1. **The pool is a measured library, not your library.** Tox21 compounds were
   chosen for a screening programme, which makes the pool more
   information-rich and the hit rate higher than a vendor catalogue's.
2. **No assay noise is simulated by default.** Replaying a measured value as
   ground truth assumes the measurement was right. Pass
   ``label_noise`` to add flip/Gaussian noise and watch every policy's
   advantage shrink -- that shrinkage is the honest picture for a noisy assay.
3. **No synthesis delay, no compound availability, no attrition.** Round 4 in
   replay happens immediately; in a lab it happens in six weeks, during which
   someone else published your series.
4. **The featurization is fixed.** Real campaigns change their features as they
   learn chemistry, which is a form of learning replay cannot capture and
   which often matters more than the selection policy.

Determinism
-----------
Under a fixed ``seed``, replay is exactly reproducible: identical selections,
identical metrics, bit-for-bit. Every random draw comes from one seeded
``Generator`` threaded through the campaign and the model, and the seeding
batch for round 1 is drawn from that same generator. This is tested.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .campaign import Campaign
from .cost import CostModel
from .plate import PlateTemplate
from .policies import Policy
from .utils import as_feature_matrix, make_rng

__all__ = [
    "ReplayResult",
    "compare_policies",
    "plot_comparison",
    "replay_campaign",
    "rounds_to_fraction",
]


def _pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve).

    Why PR-AUC and not ROC-AUC: in a screen with a 3% hit rate, a model can
    have an excellent ROC-AUC while its top 96 predictions are nearly all
    inactive, because ROC-AUC averages over thresholds nobody will ever use.
    PR-AUC weights the region you actually act on -- the top of the ranking --
    and collapses when a model's confident predictions are wrong.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == y_true.size:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    y = y_true[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    # Average precision: mean precision at each true-positive position.
    return float(np.sum(precision * y) / n_pos)


@dataclass
class ReplayResult:
    """Per-round metrics for one policy's replayed campaign."""

    policy: str
    frame: pd.DataFrame
    n_pool: int
    n_actives: int
    budget: int
    seed: int | None
    wall_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    calibration: list[dict] = field(default_factory=list)

    @property
    def cumulative_hits(self) -> np.ndarray:
        return self.frame["cumulative_hits"].to_numpy()

    @property
    def final_pr_auc(self) -> float:
        valid = self.frame["pr_auc"].dropna()
        return float(valid.iloc[-1]) if len(valid) else float("nan")

    def rounds_to_fraction(self, fraction: float = 0.5) -> int | None:
        """First round reaching ``fraction`` of all actives in the pool.

        The headline operational number: *how many plates until I have half the
        actives?* ``None`` means the campaign never got there within the rounds
        that were run.
        """
        return rounds_to_fraction(
            self.frame["cumulative_hits"].to_numpy(), self.n_actives, fraction
        )

    def summary(self) -> str:
        last = self.frame.iloc[-1]
        r50 = self.rounds_to_fraction(0.5)
        return (
            f"{self.policy}: {int(last['cumulative_hits'])}/{self.n_actives} actives "
            f"({last['fraction_of_achievable']:.1%}) after {len(self.frame)} rounds "
            f"of {self.budget}; final PR-AUC {self.final_pr_auc:.3f}; "
            f"rounds to 50% of actives: {r50 if r50 is not None else 'not reached'}; "
            f"{self.wall_seconds:.1f}s"
        )


def rounds_to_fraction(
    cumulative_hits: Sequence[float], n_actives: int, fraction: float = 0.5
) -> int | None:
    """First 1-based round index at which ``fraction`` of actives were found."""
    if n_actives <= 0:
        return None
    target = fraction * n_actives
    for i, value in enumerate(np.asarray(cumulative_hits, dtype=np.float64), start=1):
        if value >= target:
            return int(i)
    return None


def replay_campaign(
    pool: pd.DataFrame,
    *,
    label_col: str,
    model_factory: Callable[[], Any],
    policy: str | Policy,
    budget: int = 96,
    n_rounds: int = 8,
    feature_cols: Sequence[str] | None = None,
    id_col: str | None = None,
    task: str = "classification",
    hit_threshold: float = 0.5,
    seed: int | None = 0,
    n_initial_labels: int = 0,
    initial_strategy: str = "random",
    n_posterior_samples: int = 30,
    n_bootstrap: int | None = None,
    plate: PlateTemplate | None = None,
    cost_model: CostModel | None = None,
    label_noise: float = 0.0,
    policy_kwargs: dict | None = None,
    evaluate_on: str = "holdout",
    holdout_fraction: float = 0.2,
    chunked: bool = False,
    verbose: bool = False,
) -> ReplayResult:
    """Replay one policy against a fully measured dataset.

    Parameters
    ----------
    pool
        The full dataset, labels included. Labels are hidden from the campaign
        and revealed batch by batch.
    label_col
        Ground-truth column. For ``task="classification"`` values above
        ``hit_threshold`` count as hits.
    model_factory
        A zero-argument callable returning a *fresh* estimator. A factory
        rather than an instance so every policy starts from an identically
        untrained model -- reusing one instance across policies leaks the
        first policy's training into the second's results.
    n_rounds, budget
        Rounds to run, and wells per round.
    n_initial_labels
        Compounds revealed before round 1, the "we had some data already" case.
        0 means the campaign starts blind and round 1 uses the seeding policy.
    initial_strategy
        ``"random"`` (default) or ``"diverse"`` -- how those initial labels are
        chosen. ``"random"`` is the honest default: a lab's existing data is
        rarely a space-filling design.
    label_noise
        If > 0, labels are corrupted when revealed: for classification, the
        probability of flipping; for regression, the SD of added Gaussian
        noise (in units of the label's own SD). The *true* labels are still
        used for scoring, which is the point -- it measures how much a policy's
        advantage survives a noisy assay.
    evaluate_on
        Which compounds PR-AUC is computed on.

        ``"holdout"`` (default)
            A random ``holdout_fraction`` of the pool is removed up front and
            can never be selected by any policy. Model quality is measured
            there. This is the only option that compares policies fairly, for
            a reason worth spelling out: if you score the model on the
            *unselected remainder*, a successful policy is punished for its
            own success. Greedy harvests the easy actives in round 1, so the
            compounds left over are the hard ones, and its PR-AUC on that
            remainder collapses -- in a run on Tox21 NR-AR, greedy scored
            0.037 against random's 0.485 while finding *eight times* as many
            actives. That number measured the difficulty of the leftovers, not
            the quality of the model. A fixed holdout keeps the measuring
            stick still.
        ``"unselected"``
            Score on whatever has not been selected yet. Reported for
            comparison, but see above: it is not a model-quality metric.
        ``"all"``
            Score over the whole pool including training compounds. Flatters
            every policy.
    holdout_fraction
        Fraction reserved for evaluation when ``evaluate_on="holdout"``. The
        held-out compounds are excluded from the selectable pool, so
        ``n_actives`` and "fraction of achievable" refer to the selectable
        pool only.

    Returns
    -------
    ReplayResult
    """
    started = time.perf_counter()
    rng = make_rng(seed)
    frame = pool.reset_index(drop=True).copy()
    if label_col not in frame.columns:
        raise ValueError(f"label_col {label_col!r} not in pool columns.")
    if frame[label_col].isna().any():
        n_missing = int(frame[label_col].isna().sum())
        warnings.warn(
            f"Dropping {n_missing} row(s) with missing labels: replay needs a "
            "fully measured dataset, and an unmeasured compound cannot be "
            "scored as a hit or a miss.",
            RuntimeWarning,
            stacklevel=2,
        )
        frame = frame.dropna(subset=[label_col]).reset_index(drop=True)

    # Carve out a never-selectable holdout set for measuring model quality.
    holdout_frame = None
    if evaluate_on == "holdout" and holdout_fraction > 0:
        n_holdout = int(round(holdout_fraction * len(frame)))
        if n_holdout < 2:
            warnings.warn(
                f"holdout_fraction={holdout_fraction} leaves {n_holdout} "
                "compound(s), too few to score a model; evaluating on the "
                "unselected remainder instead (see evaluate_on docs for why "
                "that metric favours the policies that find the fewest hits).",
                RuntimeWarning,
                stacklevel=2,
            )
            evaluate_on = "unselected"
        else:
            holdout_idx = np.sort(rng.choice(len(frame), size=n_holdout, replace=False))
            mask = np.ones(len(frame), dtype=bool)
            mask[holdout_idx] = False
            holdout_frame = frame.iloc[holdout_idx].reset_index(drop=True)
            frame = frame.iloc[mask].reset_index(drop=True)

    truth = frame[label_col].to_numpy(dtype=np.float64)
    hit_flags = (truth > hit_threshold).astype(np.int64)
    n_actives = int(hit_flags.sum())
    if task == "classification" and n_actives == 0:
        raise ValueError(
            f"No actives in the pool at hit_threshold={hit_threshold}: every "
            "policy would score zero. Check the threshold and the label column."
        )

    # The campaign sees the pool *without* the label column.
    blind = frame.drop(columns=[label_col])
    campaign = Campaign(
        pool=blind,
        model=model_factory(),
        budget=budget,
        policy=policy,
        feature_cols=feature_cols,
        id_col=id_col,
        label_col=label_col,
        plate=plate,
        cost_model=cost_model,
        seed=seed,
        n_rounds_planned=n_rounds,
        n_posterior_samples=n_posterior_samples,
        task=task,
        hit_threshold=hit_threshold,
        n_bootstrap=n_bootstrap,
        policy_kwargs=policy_kwargs,
        chunked=chunked,
    )
    ids = campaign.pool[campaign.id_col].to_numpy()

    def reveal(indices: np.ndarray) -> dict:
        """Hand back (possibly noisy) measurements for these pool rows."""
        values = truth[indices].copy()
        if label_noise > 0:
            if task == "classification":
                flip = rng.random(values.size) < label_noise
                values = np.where(
                    flip,
                    np.where(values > hit_threshold, 0.0, 1.0),
                    values,
                )
            else:
                values = values + rng.normal(
                    0.0, label_noise * float(np.std(truth)), size=values.size
                )
        return dict(zip(ids[indices], values, strict=True))

    if n_initial_labels > 0:
        if initial_strategy == "diverse":
            X, _ = as_feature_matrix(blind, campaign.feature_cols)
            start = np.array([int(rng.integers(X.shape[0]))])
            chosen = [int(start[0])]
            # Farthest-point traversal: a cheap space-filling seed set.
            d = np.linalg.norm(X - X[chosen[0]], axis=1)
            while len(chosen) < min(n_initial_labels, X.shape[0]):
                nxt = int(np.argmax(d))
                chosen.append(nxt)
                d = np.minimum(d, np.linalg.norm(X - X[nxt], axis=1))
            seed_idx = np.array(chosen)
        elif initial_strategy == "random":
            seed_idx = rng.choice(
                len(frame), size=min(n_initial_labels, len(frame)), replace=False
            )
        else:
            raise ValueError(
                f"initial_strategy must be 'random' or 'diverse', got {initial_strategy!r}"
            )
        campaign.labels.update(reveal(seed_idx))

    rows = []
    cumulative_hits = 0
    for round_index in range(n_rounds):
        if campaign.candidate_mask.sum() == 0:
            break
        try:
            batch = campaign.select()
        except RuntimeError as exc:  # pool exhausted or budget zero
            if verbose:
                print(f"  stopping at round {round_index + 1}: {exc}")
            break

        selected_ids = batch.frame[campaign.id_col].to_numpy()
        selected_rows = np.array([campaign._id_to_row[i] for i in selected_ids])
        measured = reveal(selected_rows)
        campaign.observe(measured, label_col=label_col)

        true_hits = int(hit_flags[selected_rows].sum())
        cumulative_hits += true_hits

        # Model quality, measured on a fixed yardstick (see evaluate_on).
        pr_auc = float("nan")
        if task == "classification":
            Xl, yl = campaign.labeled_arrays()
            if np.unique(yl).size > 1:
                campaign.adapter.fit(Xl, yl)
                if holdout_frame is not None:
                    X_eval, _ = as_feature_matrix(holdout_frame, campaign.feature_cols)
                    y_eval = (
                        holdout_frame[label_col].to_numpy(dtype=np.float64) > hit_threshold
                    ).astype(np.int64)
                else:
                    labeled = set(campaign.labels)
                    eval_mask = (
                        np.array([i not in labeled for i in ids])
                        if evaluate_on == "unselected"
                        else np.ones(len(ids), dtype=bool)
                    )
                    X_eval = campaign.features[eval_mask]
                    y_eval = hit_flags[eval_mask]
                if y_eval.size > 1 and 0 < int(y_eval.sum()) < y_eval.size:
                    post = campaign.adapter.posterior(
                        X_eval, n_samples=n_posterior_samples, rng=campaign.rng
                    )
                    pr_auc = _pr_auc(y_eval, post.mean)

        calib = campaign._calibration[-1] if campaign._calibration else {}
        rows.append(
            {
                "round": round_index + 1,
                "policy": str(getattr(campaign.policy, "name", policy)),
                "n_selected": len(batch.frame),
                "n_hits": true_hits,
                "hit_rate": true_hits / max(len(batch.frame), 1),
                "cumulative_hits": cumulative_hits,
                "fraction_of_achievable": cumulative_hits / max(n_actives, 1),
                "pr_auc": pr_auc,
                "n_labeled_after": campaign.n_labeled,
                "cost": batch.cost.total,
                "alpha": batch.diagnostics.get("alpha"),
                "mean_predicted": calib.get("mean_predicted"),
                "ece": calib.get("ece"),
                "brier": calib.get("brier"),
                "batch_diversity": None,
                "used_seed_policy": batch.diagnostics.get("used_seed_policy", False),
            }
        )
        if verbose:
            r = rows[-1]
            print(
                f"  round {r['round']:>2}: {r['n_hits']:>3} hits "
                f"({r['hit_rate']:>5.1%}), cumulative {r['cumulative_hits']:>4} "
                f"({r['fraction_of_achievable']:>5.1%} of achievable), "
                f"PR-AUC {r['pr_auc']:.3f}"
            )

    report = campaign.report()
    result_frame = pd.DataFrame(rows)
    if not result_frame.empty and not report.diversity.empty:
        result_frame["batch_diversity"] = report.diversity["batch_diversity"].to_numpy()

    notes = []
    if holdout_frame is not None:
        notes.append(
            f"PR-AUC is measured on a fixed holdout of {len(holdout_frame)} "
            f"compounds ({holdout_fraction:.0%} of the dataset) that no policy "
            "could select, so model quality is comparable across policies and "
            "rounds. The selectable pool held "
            f"{n_actives} of the actives."
        )
    if label_noise > 0:
        notes.append(
            f"Labels were corrupted with noise={label_noise} when revealed; "
            "scoring still uses the true labels."
        )
    if evaluate_on == "all":
        notes.append(
            "PR-AUC was computed over the whole pool including training "
            "compounds, which flatters every policy."
        )
    if evaluate_on == "unselected":
        notes.append(
            "PR-AUC was computed on the unselected remainder, which penalizes "
            "policies that already harvested the easy actives; do not read it "
            "as model quality."
        )
    return ReplayResult(
        policy=str(policy if isinstance(policy, str) else policy.name),
        frame=result_frame,
        n_pool=len(frame),
        n_actives=n_actives,
        budget=budget,
        seed=seed,
        wall_seconds=time.perf_counter() - started,
        notes=notes,
        calibration=list(campaign._calibration),
    )


def compare_policies(
    pool: pd.DataFrame,
    *,
    label_col: str,
    model_factory: Callable[[], Any],
    policies: Sequence[str | Policy] = (
        "random",
        "greedy",
        "uncertainty",
        "bald",
        "hybrid",
    ),
    seeds: Sequence[int] = (0,),
    verbose: bool = True,
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, list[ReplayResult]]]:
    """Replay several policies (optionally over several seeds) and tabulate.

    Returns ``(long_frame, results_by_policy)``. The long frame has one row per
    (policy, seed, round), ready for groupby-and-plot.

    Running more than one seed is not optional if you intend to claim a policy
    is better. A single replay compares two sequences of coin flips; the
    spread across seeds is usually comparable to the gap between policies, and
    a mean over 5+ seeds with the spread shown is the minimum honest report.
    """
    frames, by_policy = [], {}
    for policy in policies:
        name = policy if isinstance(policy, str) else policy.name
        by_policy[name] = []
        for seed in seeds:
            if verbose:
                print(f"[replay] policy={name} seed={seed}")
            result = replay_campaign(
                pool,
                label_col=label_col,
                model_factory=model_factory,
                policy=policy,
                seed=seed,
                verbose=False,
                **kwargs,
            )
            by_policy[name].append(result)
            sub = result.frame.copy()
            sub["policy"] = name
            sub["seed"] = seed
            frames.append(sub)
            if verbose:
                print("   " + result.summary())
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        by_policy,
    )


def plot_comparison(
    long_frame: pd.DataFrame,
    *,
    n_actives: int | None = None,
    path: str | None = None,
    title: str = "Retrospective replay",
):
    """The headline figure: cumulative hits vs round, PR-AUC vs round.

    Lines are the mean over seeds; the shaded band is the min-max envelope, so
    a gap that is smaller than the bands is not a result.
    """
    import matplotlib.pyplot as plt

    policies = list(dict.fromkeys(long_frame["policy"]))
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    for i, policy in enumerate(policies):
        sub = long_frame[long_frame["policy"] == policy]
        grouped = sub.groupby("round")
        color = colors(i % 10)

        for ax, column, label in (
            (axes[0], "cumulative_hits", "cumulative hits found"),
            (axes[1], "pr_auc", "PR-AUC on held-out pool"),
        ):
            mean = grouped[column].mean()
            lo, hi = grouped[column].min(), grouped[column].max()
            ax.plot(mean.index, mean.to_numpy(), "o-", color=color, label=policy, ms=4)
            if sub["seed"].nunique() > 1:
                ax.fill_between(
                    mean.index, lo.to_numpy(), hi.to_numpy(), color=color, alpha=0.15
                )
            ax.set_xlabel("round")
            ax.set_ylabel(label)

        hr = grouped["hit_rate"].mean()
        axes[2].plot(hr.index, hr.to_numpy() * 100, "o-", color=color, label=policy, ms=4)

    if n_actives:
        axes[0].axhline(
            n_actives, ls=":", color="#444444", lw=1, label=f"all {n_actives} actives"
        )
    axes[0].set_title("Cumulative hits vs round")
    axes[1].set_title("Model quality vs round")
    axes[2].set_title("Per-round hit rate")
    axes[2].set_xlabel("round")
    axes[2].set_ylabel("hit rate (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
