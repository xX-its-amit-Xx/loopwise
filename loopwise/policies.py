"""Selection policies: eight ways to spend 96 wells.

The trade-off, stated plainly
-----------------------------
You have a model, thousands of candidates, and room for 96. Two extremes:

**Greedy** tests the 96 highest-scoring compounds. This maximizes *this
round's* hit rate, and it is what almost every lab does by default. It is also
the worst possible way to learn anything, because the compounds the model
scores highest are the ones it is already most confident about. Measuring them
confirms what you knew. The model after round 3 is the model after round 1,
and the campaign plateaus: you keep re-finding the same chemical series.

**Pure exploration** tests the 96 compounds the model is least sure about.
The model improves fastest, and your hit rate this round is terrible. In a real
lab that means a quarterly review where you report zero actives, and the
campaign is cancelled before the better model ever gets used.

Every policy here is a point on the line between those two. The honest
position, which the hybrid policy encodes, is that the right point depends on
how many rounds you have left and how much political capital you have now --
not on which method has the nicest convergence theorem.

The policies
------------
``random``
    The floor. Any policy that cannot beat random is not earning its
    complexity, and reporting it keeps everyone honest.
``greedy``
    Exploit-only reference. Top-k by predicted score.
``uncertainty``
    Max predictive variance (Lewis & Gale, 1994). Simple and often effective,
    but it cannot tell "I don't know" from "this is inherently noisy", so it
    will happily spend a whole plate on the assay's noisiest compounds.
``bald``
    Bayesian Active Learning by Disagreement (Houlsby et al., 2011). Maximizes
    mutual information between the label and the model parameters: it asks
    which measurement would most change the model. Ignores irreducible noise,
    which is exactly where uncertainty sampling fails.
``expected_model_change``
    Which measurement would most move the model's parameters (Settles et al.,
    2008). Related to BALD but measured in parameter space rather than in
    label space.
``diversity``
    Diversity-constrained greedy: facility location or k-DPP over feature
    space. Fixes the failure mode unique to *batch* selection -- the top 96 by
    any pointwise score are often 96 near-identical analogues, so you have
    bought one measurement 96 times.
``thompson``
    Thompson sampling (Thompson, 1933). Draw a plausible model from the
    posterior, act optimally under it, repeat. Explores in proportion to the
    probability a compound is genuinely best, which is a remarkably good
    automatic answer to "how much should I explore?".
``hybrid``
    ``alpha * exploit + (1 - alpha) * explore`` with ``alpha`` annealed across
    rounds. The practical default.

Batch selection is not k independent choices
--------------------------------------------
Every pointwise policy above scores each compound in isolation and takes the
top 96. That is wrong in a specific way: if compounds A and B are structurally
identical, measuring A tells you B's answer, so B's information value after
choosing A is near zero -- but a pointwise score cannot see that. This is why
``diversity`` exists and why the hybrid policy can route its exploration term
through a diversity objective. The submodular (facility location) and
determinantal (k-DPP) objectives both handle it, by making the value of adding
a compound depend on what is already in the batch.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .models import Posterior
from .utils import (
    DEFAULT_DTYPE,
    check_resources,
    chunk_slices,
    guard_pairwise_matrix,
    normalize_scores,
)

__all__ = [
    "POLICY_REGISTRY",
    "BALDPolicy",
    "DiversityGreedyPolicy",
    "ExpectedModelChangePolicy",
    "GreedyPolicy",
    "HybridPolicy",
    "Policy",
    "PolicyContext",
    "RandomPolicy",
    "SelectionResult",
    "ThompsonPolicy",
    "UncertaintyPolicy",
    "available_policies",
    "get_policy",
]

_EPS = 1e-12


@dataclass
class PolicyContext:
    """Everything a policy is allowed to look at.

    Keeping this explicit (rather than handing policies the whole Campaign)
    is what makes policies independently testable and keeps a policy from
    reaching for the labels of unmeasured compounds -- the bug that makes a
    retrospective replay look miraculous.
    """

    features: np.ndarray
    posterior: Posterior
    budget: int
    rng: np.random.Generator
    round_index: int = 0
    n_rounds_planned: int | None = None
    labeled_features: np.ndarray | None = None
    labeled_targets: np.ndarray | None = None
    item_costs: np.ndarray | None = None
    chunked: bool = False
    extras: dict = field(default_factory=dict)

    @property
    def n_candidates(self) -> int:
        return int(self.features.shape[0])

    def effective_budget(self) -> int:
        return int(max(0, min(self.budget, self.n_candidates)))


@dataclass
class SelectionResult:
    """Which compounds were chosen, and why.

    ``reasons`` maps column name -> array aligned with ``indices``. Campaign
    turns it into the DataFrame returned by ``select()``, so "why was this
    compound on my plate?" is answerable six months later from the CSV alone.
    """

    indices: np.ndarray
    reasons: dict[str, np.ndarray]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=np.int64).ravel()
        if self.indices.size != np.unique(self.indices).size:
            raise AssertionError(
                "Policy returned duplicate indices -- this is a bug in the "
                "policy, not in your data."
            )
        for key, value in self.reasons.items():
            arr = np.asarray(value)
            if arr.shape[0] != self.indices.size:
                raise AssertionError(
                    f"Reason column {key!r} has length {arr.shape[0]} but "
                    f"{self.indices.size} items were selected."
                )


class Policy:
    """Base class. Subclasses implement :meth:`select`."""

    name: str = "policy"
    #: Does this policy need a posterior that can disagree with itself?
    needs_uncertainty: bool = False

    def select(self, ctx: PolicyContext) -> SelectionResult:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> str:  # pragma: no cover - cosmetic
        return self.__doc__.strip().splitlines()[0] if self.__doc__ else self.name

    # ------------------------------------------------------------- utilities

    def _check_uncertainty(self, ctx: PolicyContext) -> None:
        if self.needs_uncertainty:
            ctx.posterior.require_informative(self.name)

    @staticmethod
    def _top_k(scores: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        """Indices of the ``k`` largest scores, ties broken randomly.

        Random tie-breaking matters more than it sounds: with a tree model on
        a sparse fingerprint, hundreds of compounds often share an identical
        score, and ``argsort`` would return them in pool order -- which is
        usually vendor-catalogue order, i.e. correlated with chemical series.
        """
        scores = np.asarray(scores, dtype=np.float64).ravel()
        k = int(min(max(k, 0), scores.size))
        if k == 0:
            return np.empty(0, dtype=np.int64)
        jitter = rng.random(scores.size) * 1e-12
        order = np.argsort(-(scores + jitter), kind="stable")
        return order[:k].astype(np.int64)


# ------------------------------------------------------------------ baselines


class RandomPolicy(Policy):
    """Uniform random selection -- the floor every other policy must beat."""

    name = "random"

    def select(self, ctx: PolicyContext) -> SelectionResult:
        k = ctx.effective_budget()
        idx = ctx.rng.choice(ctx.n_candidates, size=k, replace=False).astype(np.int64)
        mean = ctx.posterior.mean[idx]
        return SelectionResult(
            indices=idx,
            reasons={
                "policy": np.full(k, self.name, dtype=object),
                "rank": np.arange(1, k + 1),
                "predicted_score": mean,
                "reason": np.array(
                    [f"random draw {i + 1}/{k} (unbiased floor)" for i in range(k)],
                    dtype=object,
                ),
            },
            diagnostics={"policy": self.name},
        )


class GreedyPolicy(Policy):
    """Top-k by predicted score: maximum exploitation, minimum learning."""

    name = "greedy"

    def select(self, ctx: PolicyContext) -> SelectionResult:
        k = ctx.effective_budget()
        mean = ctx.posterior.mean
        idx = self._top_k(mean, k, ctx.rng)
        return SelectionResult(
            indices=idx,
            reasons={
                "policy": np.full(k, self.name, dtype=object),
                "rank": np.arange(1, k + 1),
                "predicted_score": mean[idx],
                "reason": np.array(
                    [
                        f"rank {r}/{k} by predicted score ({mean[i]:.4f}); "
                        "chosen to maximize this round's hit rate"
                        for r, i in enumerate(idx, start=1)
                    ],
                    dtype=object,
                ),
            },
            diagnostics={
                "policy": self.name,
                "score_at_cutoff": float(mean[idx[-1]]) if k else None,
            },
        )


# ---------------------------------------------------------------- information


class UncertaintyPolicy(Policy):
    """Max predictive variance: test what the model is least sure about."""

    name = "uncertainty"
    needs_uncertainty = True

    def __init__(self, measure: str = "variance"):
        if measure not in ("variance", "std", "entropy", "margin"):
            raise ValueError(
                f"measure must be one of variance/std/entropy/margin, got {measure!r}"
            )
        self.measure = measure

    def select(self, ctx: PolicyContext) -> SelectionResult:
        self._check_uncertainty(ctx)
        post = ctx.posterior
        if self.measure in ("variance", "std"):
            score = post.variance if self.measure == "variance" else post.std
        elif self.measure == "entropy":
            score = post.predictive_entropy()
        else:  # margin: closeness of the mean prediction to the decision boundary
            score = -np.abs(post.mean - 0.5)
        k = ctx.effective_budget()
        idx = self._top_k(score, k, ctx.rng)
        return SelectionResult(
            indices=idx,
            reasons={
                "policy": np.full(k, self.name, dtype=object),
                "rank": np.arange(1, k + 1),
                "predicted_score": post.mean[idx],
                "uncertainty": score[idx],
                "posterior_std": post.std[idx],
                "reason": np.array(
                    [
                        f"rank {r}/{k} by predictive {self.measure} "
                        f"({score[i]:.4g}); model's prediction here spans "
                        f"+/-{post.std[i]:.3f} across posterior samples"
                        for r, i in enumerate(idx, start=1)
                    ],
                    dtype=object,
                ),
            },
            diagnostics={"policy": self.name, "measure": self.measure},
        )


class BALDPolicy(Policy):
    """BALD: maximize mutual information between the label and the model."""

    name = "bald"
    needs_uncertainty = True

    def select(self, ctx: PolicyContext) -> SelectionResult:
        self._check_uncertainty(ctx)
        post = ctx.posterior
        mi = post.mutual_information()
        total = post.predictive_entropy()
        aleatoric = post.expected_entropy()
        k = ctx.effective_budget()
        idx = self._top_k(mi, k, ctx.rng)
        return SelectionResult(
            indices=idx,
            reasons={
                "policy": np.full(k, self.name, dtype=object),
                "rank": np.arange(1, k + 1),
                "predicted_score": post.mean[idx],
                "mutual_information": mi[idx],
                "total_entropy": total[idx],
                "aleatoric_entropy": aleatoric[idx],
                "reason": np.array(
                    [
                        f"rank {r}/{k} by mutual information ({mi[i]:.4f} nats "
                        f"= {total[i]:.4f} total - {aleatoric[i]:.4f} noise); "
                        "measuring this would most update the model"
                        for r, i in enumerate(idx, start=1)
                    ],
                    dtype=object,
                ),
            },
            diagnostics={
                "policy": self.name,
                "mean_mi_selected": float(mi[idx].mean()) if k else None,
                "mean_mi_pool": float(mi.mean()),
            },
        )


class ExpectedModelChangePolicy(Policy):
    """How much would this one measurement move the model?

    Two estimators, both cheap and both honest about being approximations:

    ``mode="kl"`` (default)
        Treat the posterior samples as a weighted ensemble of models with equal
        prior weights. Observing label ``y`` for candidate ``i`` reweights the
        ensemble by each member's likelihood of ``y`` -- members that predicted
        the observed outcome gain weight. The KL divergence between the
        reweighted and original weights is exactly "how much did my belief
        about which model is right move?", and we average it over the possible
        outcomes under the current predictive distribution. This is a genuine
        Bayesian model change in ensemble-weight space, computable in O(M) per
        candidate with no refitting.

    ``mode="egl"``
        Expected Gradient Length (Settles et al., 2008): the norm of the
        gradient the new example would contribute to training. For logistic
        loss the gradient is ``(y - p) * x``, so its expectation over
        ``y ~ Bernoulli(p)`` has magnitude ``2 p (1 - p) ||x||``. Classic,
        instant, and tied to a linear surrogate rather than to your actual
        model -- it rewards uncertain compounds that also sit far from the
        feature-space origin, i.e. structurally unusual ones.
    """

    name = "expected_model_change"
    needs_uncertainty = True

    def __init__(self, mode: str = "kl", n_outcome_draws: int = 8):
        if mode not in ("kl", "egl"):
            raise ValueError(f"mode must be 'kl' or 'egl', got {mode!r}")
        self.mode = mode
        self.n_outcome_draws = int(n_outcome_draws)

    def select(self, ctx: PolicyContext) -> SelectionResult:
        self._check_uncertainty(ctx)
        post = ctx.posterior
        if self.mode == "egl":
            score, extra_name = self._egl(ctx), "expected_gradient_length"
        else:
            score, extra_name = self._ensemble_kl(ctx), "expected_weight_kl"
        k = ctx.effective_budget()
        idx = self._top_k(score, k, ctx.rng)
        return SelectionResult(
            indices=idx,
            reasons={
                "policy": np.full(k, self.name, dtype=object),
                "rank": np.arange(1, k + 1),
                "predicted_score": post.mean[idx],
                extra_name: score[idx],
                "posterior_std": post.std[idx],
                "reason": np.array(
                    [
                        f"rank {r}/{k} by expected model change "
                        f"({self.mode}: {score[i]:.4g}); measuring this would "
                        "most shift the fitted model"
                        for r, i in enumerate(idx, start=1)
                    ],
                    dtype=object,
                ),
            },
            diagnostics={"policy": self.name, "mode": self.mode},
        )

    def _ensemble_kl(self, ctx: PolicyContext) -> np.ndarray:
        """Expected KL between reweighted and prior ensemble weights."""
        post = ctx.posterior
        samples = np.clip(post.samples.astype(np.float64), 1e-6, 1 - 1e-6)
        n_members, n_items = samples.shape
        check_resources(
            (n_members, n_items),
            dtype=np.float64,
            copies=3,
            task="expected-model-change likelihood reweighting",
        ).raise_if_refused()
        prior = 1.0 / n_members

        if post.task == "classification":
            outcomes = [
                (samples, post.mean.astype(np.float64)),  # y = 1, weight p
                (1.0 - samples, 1.0 - post.mean.astype(np.float64)),  # y = 0
            ]
        else:
            # Regression: draw plausible outcomes from the predictive spread and
            # score members by a Gaussian likelihood around their prediction.
            sigma = float(np.mean(post.std)) + 1e-6
            draws = samples[
                ctx.rng.integers(0, n_members, size=self.n_outcome_draws),
                np.arange(n_items)[None, :],
            ]
            outcomes = []
            weight = np.full(n_items, 1.0 / self.n_outcome_draws)
            for d in range(self.n_outcome_draws):
                like = np.exp(-0.5 * ((draws[d][None, :] - samples) / sigma) ** 2)
                outcomes.append((like, weight))

        total = np.zeros(n_items, dtype=np.float64)
        for likelihood, outcome_prob in outcomes:
            # Posterior weights over ensemble members, per candidate.
            evidence = likelihood.sum(axis=0) * prior
            w = (likelihood * prior) / np.maximum(evidence, 1e-300)
            kl = np.sum(w * np.log(np.maximum(w, 1e-300) / prior), axis=0)
            total += outcome_prob * kl
        return total.astype(DEFAULT_DTYPE)

    def _egl(self, ctx: PolicyContext) -> np.ndarray:
        """``2 p (1 - p) ||x||`` -- logistic-surrogate expected gradient length."""
        post = ctx.posterior
        norms = np.linalg.norm(ctx.features.astype(np.float64), axis=1)
        if post.task == "classification":
            p = np.clip(post.mean.astype(np.float64), 1e-6, 1 - 1e-6)
            sensitivity = 2.0 * p * (1.0 - p)
        else:
            sensitivity = post.std.astype(np.float64)
        return (sensitivity * norms).astype(DEFAULT_DTYPE)


class ThompsonPolicy(Policy):
    """Thompson sampling: act optimally under a model drawn from the posterior.

    For each of the ``k`` slots, draw one plausible model from the posterior and
    take its best remaining compound. A compound enters the batch roughly in
    proportion to the probability that it really is the best one -- so
    exploration is calibrated to genuine uncertainty rather than to a
    hand-tuned knob. That self-tuning property is why a 1933 idea is still
    competitive with everything invented since.

    ``per_slot_draw=True`` (default) redraws the model for every slot, which
    makes the batch diverse in the right way: slots disagree because plausible
    models disagree. With ``False``, one model is drawn and its top ``k`` taken,
    which is cheaper but gives a batch that is internally correlated.
    """

    name = "thompson"
    needs_uncertainty = True

    def __init__(self, per_slot_draw: bool = True):
        self.per_slot_draw = bool(per_slot_draw)

    def select(self, ctx: PolicyContext) -> SelectionResult:
        self._check_uncertainty(ctx)
        post = ctx.posterior
        k = ctx.effective_budget()
        chosen: list[int] = []
        draw_ids: list[int] = []
        drawn_values: list[float] = []

        if not self.per_slot_draw:
            draw_id = int(ctx.rng.integers(post.n_samples))
            sample = post.samples[draw_id]
            idx = self._top_k(sample, k, ctx.rng)
            chosen = idx.tolist()
            draw_ids = [draw_id] * k
            drawn_values = sample[idx].tolist()
        else:
            taken = np.zeros(ctx.n_candidates, dtype=bool)
            for _ in range(k):
                draw_id = int(ctx.rng.integers(post.n_samples))
                sample = post.samples[draw_id].astype(np.float64).copy()
                sample[taken] = -np.inf
                pick = int(np.argmax(sample + ctx.rng.random(sample.size) * 1e-12))
                taken[pick] = True
                chosen.append(pick)
                draw_ids.append(draw_id)
                drawn_values.append(float(post.samples[draw_id, pick]))

        idx = np.asarray(chosen, dtype=np.int64)
        return SelectionResult(
            indices=idx,
            reasons={
                "policy": np.full(k, self.name, dtype=object),
                "rank": np.arange(1, k + 1),
                "predicted_score": post.mean[idx],
                "posterior_draw_id": np.asarray(draw_ids, dtype=np.int64),
                "drawn_score": np.asarray(drawn_values, dtype=DEFAULT_DTYPE),
                "posterior_std": post.std[idx],
                "reason": np.array(
                    [
                        f"slot {r}/{k}: best under posterior draw #{d} "
                        f"(drawn score {v:.4f} vs posterior mean "
                        f"{post.mean[i]:.4f})"
                        for r, (i, d, v) in enumerate(
                            zip(idx, draw_ids, drawn_values, strict=True), start=1
                        )
                    ],
                    dtype=object,
                ),
            },
            diagnostics={
                "policy": self.name,
                "distinct_draws": len(set(draw_ids)),
                "per_slot_draw": self.per_slot_draw,
            },
        )


# ------------------------------------------------------------------ diversity


def _standardize(X: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance columns, in float32.

    Without this, a descriptor measured in Daltons (hundreds) dominates one
    measured in log units (single digits), and "diversity" silently means
    "diverse in molecular weight".
    """
    Xf = np.asarray(X, dtype=DEFAULT_DTYPE)
    mu = Xf.mean(axis=0, keepdims=True)
    sd = Xf.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (Xf - mu) / sd


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return (X / norms).astype(DEFAULT_DTYPE)


def _similarity_block(
    A: np.ndarray, B: np.ndarray, kernel: str, gamma: float | None = None
) -> np.ndarray:
    """Similarity between every row of ``A`` and every row of ``B``.

    ``"cosine"`` on L2-normalized rows is a dot product -- one BLAS call, and
    the natural choice for fingerprint-like features. ``"rbf"`` is a squared
    exponential, which gives a sharper notion of "nearby" and is what a
    Gaussian process would use.
    """
    if kernel == "cosine":
        return np.clip(A @ B.T, -1.0, 1.0).astype(DEFAULT_DTYPE)
    if kernel == "rbf":
        d2 = (
            np.sum(A * A, axis=1)[:, None]
            + np.sum(B * B, axis=1)[None, :]
            - 2.0 * (A @ B.T)
        )
        np.maximum(d2, 0.0, out=d2)
        g = gamma if gamma is not None else 1.0 / max(A.shape[1], 1)
        return np.exp(-g * d2).astype(DEFAULT_DTYPE)
    raise ValueError(f"kernel must be 'cosine' or 'rbf', got {kernel!r}")


class DiversityGreedyPolicy(Policy):
    """Diversity-constrained greedy: facility location or k-DPP.

    Why a *batch* needs its own machinery
    -------------------------------------
    Pointwise scores treat the 96 wells as 96 separate decisions. They are not.
    Pick the top 96 by any score on a vendor library and you will often get
    three chemical series, because similar molecules get similar scores. You
    have paid for 96 measurements and bought perhaps 5 independent ones.

    Two objectives fix this, both chosen so that a plain greedy algorithm is
    provably near-optimal:

    ``objective="facility_location"``
        Maximize ``sum over pool i of max over batch j of sim(i, j)`` -- "how
        well does my batch *represent* the pool?" This function is submodular
        (each addition helps less than the last), and for submodular maximization
        greedy is within ``1 - 1/e`` (about 63%) of the best possible subset
        (Nemhauser et al., 1978). Good when you want coverage of the whole pool.

    ``objective="kdpp"``
        Maximize ``log det L_S`` for the quality-weighted kernel
        ``L_ij = q_i q_j sim(i, j)``. A determinantal point process makes the
        probability of a set proportional to the squared volume its vectors
        span, so sets of near-duplicates are exponentially unlikely: duplicate
        vectors span zero volume. Solved with the fast greedy MAP algorithm of
        Chen et al. (2018), which needs one kernel *column* per step -- never
        the full matrix. Good when you want internal non-redundancy.

    Quality weighting is what keeps this from being a chemistry-blind
    space-filling design: ``quality_weight`` raises the predicted score to a
    power and multiplies it into the objective, so the batch is diverse *among
    plausible actives* rather than diverse across the whole pool. With
    ``quality_weight=0`` you get pure space-filling, which is occasionally what
    you want for a first round with no model worth trusting.

    Memory
    ------
    These are the dangerous paths in Loopwise. Both call
    :func:`~loopwise.utils.guard_pairwise_matrix`, which refuses a pool above
    20,000 candidates unless the caller passes ``chunked=True``. With
    ``chunked=True``:

    * k-DPP streams kernel columns (memory ``O(n k)``, exact -- same selection).
    * facility location subsamples the reference set that the coverage sum runs
      over and uses stochastic greedy (Mirzasoleiman et al., 2015), which is an
      approximation and says so in its diagnostics.
    """

    name = "diversity"

    def __init__(
        self,
        objective: str = "facility_location",
        *,
        kernel: str = "cosine",
        gamma: float | None = None,
        quality_weight: float = 1.0,
        max_reference_set: int = 4000,
        stochastic_epsilon: float = 0.01,
        standardize: bool = True,
    ):
        if objective not in ("facility_location", "kdpp"):
            raise ValueError(
                f"objective must be 'facility_location' or 'kdpp', got {objective!r}"
            )
        self.objective = objective
        self.kernel = kernel
        self.gamma = gamma
        self.quality_weight = float(quality_weight)
        self.max_reference_set = int(max_reference_set)
        self.stochastic_epsilon = float(stochastic_epsilon)
        self.standardize = bool(standardize)

    # ----------------------------------------------------------------- public

    def select(self, ctx: PolicyContext) -> SelectionResult:
        k = ctx.effective_budget()
        n = ctx.n_candidates
        report = guard_pairwise_matrix(
            n,
            chunked=ctx.chunked,
            task=f"{self.objective} kernel over {n:,} candidates",
        )
        X = (
            _standardize(ctx.features)
            if self.standardize
            else np.asarray(ctx.features, dtype=DEFAULT_DTYPE)
        )
        if self.kernel == "cosine":
            X = _l2_normalize(X)
        quality = self._quality(ctx)

        if self.objective == "kdpp":
            idx, gains, notes = self._kdpp_greedy(X, quality, k, report, ctx)
        else:
            idx, gains, notes = self._facility_location(X, quality, k, report, ctx)

        post = ctx.posterior
        nearest = self._nearest_selected_similarity(X, idx)
        return SelectionResult(
            indices=idx,
            reasons={
                "policy": np.full(idx.size, f"{self.name}:{self.objective}", dtype=object),
                "rank": np.arange(1, idx.size + 1),
                "predicted_score": post.mean[idx],
                "quality_weight": quality[idx],
                "marginal_gain": gains,
                "max_similarity_to_earlier_pick": nearest,
                "reason": np.array(
                    [
                        f"pick {r}/{idx.size} by {self.objective}: marginal gain "
                        f"{g:.4g}, predicted score {post.mean[i]:.4f}, "
                        f"max similarity to an earlier pick {s:.3f}"
                        for r, (i, g, s) in enumerate(
                            zip(idx, gains, nearest, strict=True), start=1
                        )
                    ],
                    dtype=object,
                ),
            },
            diagnostics={
                "policy": self.name,
                "objective": self.objective,
                "kernel": self.kernel,
                "quality_weight_exponent": self.quality_weight,
                "resource_decision": report.decision,
                "mean_intra_batch_similarity": float(np.mean(nearest))
                if idx.size > 1
                else 0.0,
                **notes,
            },
        )

    # ---------------------------------------------------------------- helpers

    def _quality(self, ctx: PolicyContext) -> np.ndarray:
        """Per-candidate quality multiplier from the predicted score."""
        if self.quality_weight == 0.0:
            return np.ones(ctx.n_candidates, dtype=DEFAULT_DTYPE)
        base = normalize_scores(ctx.posterior.mean, "rank")
        # Keep strictly positive: a zero-quality item would make log det -inf.
        return np.power(np.clip(base, 1e-3, 1.0), self.quality_weight).astype(DEFAULT_DTYPE)

    def _kdpp_greedy(
        self,
        X: np.ndarray,
        quality: np.ndarray,
        k: int,
        report,
        ctx: PolicyContext,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Fast greedy MAP for a k-DPP (Chen et al., 2018).

        Maintains, for every candidate ``i``, a vector ``c_i`` and a residual
        ``d_i^2 = L_ii - ||c_i||^2``. Each step picks the largest residual --
        the candidate whose contribution is least explained by what is already
        selected -- then updates every residual using **one column** of the
        kernel. That is why this scales: ``k`` columns, not ``n^2`` entries.
        """
        n = X.shape[0]
        k = int(min(k, n))
        check_resources(
            (n, max(k, 1)),
            task="k-DPP Cholesky work array",
            copies=2,
            n_jobs=report.recommended_n_jobs,
        ).raise_if_refused()

        c = np.zeros((n, k), dtype=np.float64)
        # Diagonal of L is q_i^2 * sim(i, i), and sim(i, i) == 1 for cosine on
        # L2-normalized rows and for the RBF kernel, so the initial residual is
        # just the squared quality weight.
        d2 = quality.astype(np.float64) ** 2
        selected: list[int] = []
        gains: list[float] = []

        for step in range(k):
            d2_masked = d2.copy()
            if selected:
                d2_masked[np.asarray(selected)] = -np.inf
            j = int(np.argmax(d2_masked))
            if not np.isfinite(d2_masked[j]) or d2_masked[j] <= 1e-18:
                # Remaining candidates are linearly dependent on the batch:
                # the determinant cannot grow. Fill by quality and say so.
                remaining = np.setdiff1d(np.arange(n), np.asarray(selected, dtype=np.int64))
                order = remaining[np.argsort(-quality[remaining])]
                for extra in order[: k - len(selected)]:
                    selected.append(int(extra))
                    gains.append(0.0)
                break
            selected.append(j)
            gains.append(float(max(d2[j], 0.0)))

            # One kernel column: L[:, j] = q * q_j * sim(:, j), computed in
            # row blocks so the column itself is the only n-sized allocation.
            col = np.empty(n, dtype=np.float64)
            for sl in chunk_slices(n, report.recommended_chunk_rows):
                sim = _similarity_block(X[sl], X[j : j + 1], self.kernel, self.gamma)[:, 0]
                col[sl] = quality[sl].astype(np.float64) * float(quality[j]) * sim

            dj = math.sqrt(max(d2[j], 1e-18))
            e = (col - c[:, :step] @ c[j, :step]) / dj if step else col / dj
            c[:, step] = e
            d2 = np.maximum(d2 - e**2, 0.0)
            d2[j] = -np.inf

        return (
            np.asarray(selected, dtype=np.int64),
            np.asarray(gains, dtype=DEFAULT_DTYPE),
            {
                "exact": True,
                "note": "greedy MAP for log det L_S; one kernel column per step",
            },
        )

    def _facility_location(
        self,
        X: np.ndarray,
        quality: np.ndarray,
        k: int,
        report,
        ctx: PolicyContext,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Greedy maximization of quality-weighted facility-location coverage."""
        n = X.shape[0]
        k = int(min(k, n))

        # Reference set: the items whose coverage we sum over. Subsampling it
        # is what keeps the streaming path affordable.
        if n <= self.max_reference_set:
            ref_idx = np.arange(n)
            subsampled = False
        else:
            ref_idx = np.sort(ctx.rng.choice(n, size=self.max_reference_set, replace=False))
            subsampled = True
        ref = X[ref_idx]
        n_ref = ref.shape[0]

        # Candidate shortlist per step: full sweep when affordable, otherwise
        # stochastic greedy (Mirzasoleiman et al., 2015).
        sweep_cost_gb = (n_ref * n * np.dtype(DEFAULT_DTYPE).itemsize) / float(1 << 30)
        stochastic = subsampled or sweep_cost_gb > 0.25 * report.ram_available_gb
        subset_size = n
        if stochastic and k > 0:
            subset_size = int(
                min(
                    n,
                    max(
                        k,
                        math.ceil(
                            (n / k) * math.log(1.0 / max(self.stochastic_epsilon, 1e-6))
                        ),
                    ),
                )
            )

        covered = np.zeros(n_ref, dtype=np.float64)  # max sim to the batch so far
        selected: list[int] = []
        gains: list[float] = []
        taken = np.zeros(n, dtype=bool)

        for _ in range(k):
            if stochastic:
                pool = np.flatnonzero(~taken)
                if pool.size == 0:
                    break
                cand = ctx.rng.choice(
                    pool, size=int(min(subset_size, pool.size)), replace=False
                )
            else:
                cand = np.flatnonzero(~taken)
                if cand.size == 0:
                    break

            best_j, best_gain, best_newcov = -1, -np.inf, None
            chunk = report.recommended_chunk_rows or max(1, min(cand.size, 512))
            for sl in chunk_slices(cand.size, chunk):
                block = cand[sl]
                sim = _similarity_block(ref, X[block], self.kernel, self.gamma)
                sim = sim.astype(np.float64) * quality[block].astype(np.float64)[None, :]
                newcov = np.maximum(sim, covered[:, None])
                block_gains = newcov.sum(axis=0) - covered.sum()
                local = int(np.argmax(block_gains))
                if block_gains[local] > best_gain:
                    best_gain = float(block_gains[local])
                    best_j = int(block[local])
                    best_newcov = newcov[:, local].copy()
            if best_j < 0:
                break
            selected.append(best_j)
            gains.append(best_gain)
            taken[best_j] = True
            covered = best_newcov

        return (
            np.asarray(selected, dtype=np.int64),
            np.asarray(gains, dtype=DEFAULT_DTYPE),
            {
                "exact": not stochastic,
                "reference_set_size": int(n_ref),
                "reference_set_subsampled": bool(subsampled),
                "stochastic_greedy": bool(stochastic),
                "stochastic_subset_size": int(subset_size) if stochastic else None,
                "note": (
                    "stochastic greedy with a subsampled reference set "
                    "(approximate; selections vary with seed)"
                    if stochastic
                    else "exact lazy-free greedy; (1 - 1/e) guarantee"
                ),
            },
        )

    def _nearest_selected_similarity(self, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """For each pick, its max similarity to the picks before it.

        This is the auditable number that answers "is my batch actually
        diverse?" -- values near 1.0 mean near-duplicates got through.
        """
        if idx.size == 0:
            return np.empty(0, dtype=DEFAULT_DTYPE)
        sub = X[idx]
        sim = _similarity_block(sub, sub, self.kernel, self.gamma).astype(np.float64)
        out = np.zeros(idx.size, dtype=DEFAULT_DTYPE)
        for pos in range(1, idx.size):
            out[pos] = float(np.max(sim[pos, :pos]))
        return out


# -------------------------------------------------------------------- hybrid


def _anneal(
    start: float, end: float, round_index: int, n_rounds: int | None, schedule: str
) -> float:
    """Interpolate ``start`` -> ``end`` across the planned rounds."""
    if n_rounds is None or n_rounds <= 1:
        return float(start)
    t = min(max(round_index / float(n_rounds - 1), 0.0), 1.0)
    if schedule == "linear":
        f = t
    elif schedule == "cosine":
        f = 0.5 * (1.0 - math.cos(math.pi * t))
    elif schedule == "exponential":
        f = 1.0 - math.exp(-3.0 * t)
    elif schedule == "step":
        f = 0.0 if t < 0.5 else 1.0
    else:
        raise ValueError(
            f"schedule must be linear/cosine/exponential/step, got {schedule!r}"
        )
    return float(start + (end - start) * f)


class HybridPolicy(Policy):
    """``alpha * exploit + (1 - alpha) * explore``, with ``alpha`` annealed.

    The practical default, and the only policy here that takes seriously the
    fact that campaigns are funded by people who look at hit rates.

    The tension
    -----------
    Information is most valuable *early*: a measurement taken in round 1 can
    inform rounds 2 through 8, while one taken in the final round informs
    nothing. The textbook schedule therefore explores early and harvests late,
    and that is Loopwise's default (``alpha_start=0.4`` rising to
    ``alpha_end=0.9``).

    Real campaigns often cannot afford it. If round 1 returns no actives, there
    may be no round 2 -- the project loses its slot, or the collaborator loses
    interest. A lab in that position should set ``alpha_start=0.8,
    alpha_end=0.3`` and front-load hits, buying political survival with a worse
    final model. Loopwise supports both because the choice is a funding
    decision, not a statistical one, and pretending otherwise is how good
    methods go unused. What matters is that the choice is *recorded*: the alpha
    used in every round lands in the batch's reason columns and in
    ``Campaign.report()``.

    Tie-breaking by mixed rank, not by mixed raw score: predicted probabilities
    and BALD scores in nats are not commensurable, so both are rank-normalized
    to [0, 1] before mixing (see :func:`~loopwise.utils.normalize_scores`).
    """

    name = "hybrid"
    needs_uncertainty = True

    def __init__(
        self,
        *,
        alpha_start: float = 0.4,
        alpha_end: float = 0.9,
        schedule: str = "linear",
        explore: str | Policy = "bald",
        exploit: str | Policy = "greedy",
        normalization: str = "rank",
        alpha: float | None = None,
    ):
        if alpha is not None:  # fixed alpha, no annealing
            alpha_start = alpha_end = float(alpha)
        for value, label in ((alpha_start, "alpha_start"), (alpha_end, "alpha_end")):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1], got {value}")
        self.alpha_start = float(alpha_start)
        self.alpha_end = float(alpha_end)
        self.schedule = schedule
        self.normalization = normalization
        self._explore_spec = explore
        self._exploit_spec = exploit

    def alpha_for_round(self, round_index: int, n_rounds: int | None) -> float:
        return _anneal(
            self.alpha_start, self.alpha_end, round_index, n_rounds, self.schedule
        )

    def select(self, ctx: PolicyContext) -> SelectionResult:
        alpha = self.alpha_for_round(ctx.round_index, ctx.n_rounds_planned)
        post = ctx.posterior
        k = ctx.effective_budget()

        exploit_raw = post.mean
        explore_raw, explore_label = self._explore_scores(ctx)
        if alpha < 1.0:
            post.require_informative(f"hybrid(explore={explore_label})")

        exploit_n = normalize_scores(exploit_raw, self.normalization)
        explore_n = normalize_scores(explore_raw, self.normalization)
        blended = alpha * exploit_n + (1.0 - alpha) * explore_n

        if isinstance(self._explore_spec, DiversityGreedyPolicy) or (
            isinstance(self._explore_spec, str)
            and self._explore_spec.startswith("diversity")
        ):
            # Sequential diversity needs the batch-aware greedy, so hand the
            # blended score to the diversity policy as its quality signal.
            idx, gains = self._diversity_blend(ctx, blended, k)
        else:
            idx = self._top_k(blended, k, ctx.rng)
            gains = blended[idx]

        return SelectionResult(
            indices=idx,
            reasons={
                "policy": np.full(idx.size, f"{self.name}:{explore_label}", dtype=object),
                "rank": np.arange(1, idx.size + 1),
                "predicted_score": exploit_raw[idx],
                "exploit_rank_norm": exploit_n[idx],
                "explore_score": explore_raw[idx],
                "explore_rank_norm": explore_n[idx],
                "alpha": np.full(idx.size, alpha, dtype=DEFAULT_DTYPE),
                "blended_score": gains,
                "reason": np.array(
                    [
                        f"rank {r}/{idx.size} by {alpha:.2f}*exploit + "
                        f"{1 - alpha:.2f}*{explore_label} "
                        f"(exploit {exploit_n[i]:.3f}, explore {explore_n[i]:.3f} "
                        f"-> {b:.3f}); round {ctx.round_index + 1}"
                        f"{f' of {ctx.n_rounds_planned}' if ctx.n_rounds_planned else ''}"
                        for r, (i, b) in enumerate(zip(idx, gains, strict=True), start=1)
                    ],
                    dtype=object,
                ),
            },
            diagnostics={
                "policy": self.name,
                "alpha": alpha,
                "alpha_start": self.alpha_start,
                "alpha_end": self.alpha_end,
                "schedule": self.schedule,
                "explore": explore_label,
                "round_index": ctx.round_index,
            },
        )

    def _explore_scores(self, ctx: PolicyContext) -> tuple[np.ndarray, str]:
        spec = self._explore_spec
        post = ctx.posterior
        if isinstance(spec, str):
            if spec == "bald":
                return post.mutual_information(), "bald"
            if spec == "uncertainty":
                return post.variance, "uncertainty"
            if spec == "entropy":
                return post.predictive_entropy(), "entropy"
            if spec.startswith("diversity"):
                return post.mutual_information(), spec
            if spec == "expected_model_change":
                emc = ExpectedModelChangePolicy()
                return emc._ensemble_kl(ctx), "expected_model_change"
            raise ValueError(f"Unknown explore term {spec!r}")
        if isinstance(spec, DiversityGreedyPolicy):
            return post.mutual_information(), f"diversity:{spec.objective}"
        raise TypeError(f"explore must be a str or DiversityGreedyPolicy, got {type(spec)}")

    def _diversity_blend(
        self, ctx: PolicyContext, blended: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the diversity greedy using the blended score as quality."""
        spec = self._explore_spec
        policy = (
            spec
            if isinstance(spec, DiversityGreedyPolicy)
            else DiversityGreedyPolicy(
                objective="kdpp" if "kdpp" in str(spec) else "facility_location"
            )
        )
        sub_ctx = PolicyContext(
            features=ctx.features,
            posterior=Posterior(blended[None, :], task="classification")
            if np.all((blended >= 0) & (blended <= 1))
            else ctx.posterior,
            budget=k,
            rng=ctx.rng,
            round_index=ctx.round_index,
            n_rounds_planned=ctx.n_rounds_planned,
            chunked=ctx.chunked,
        )
        result = policy.select(sub_ctx)
        return result.indices, np.asarray(
            result.reasons["marginal_gain"], dtype=DEFAULT_DTYPE
        )


# ------------------------------------------------------------------ registry


POLICY_REGISTRY: dict[str, Callable[..., Policy]] = {
    "random": RandomPolicy,
    "greedy": GreedyPolicy,
    "uncertainty": UncertaintyPolicy,
    "bald": BALDPolicy,
    "expected_model_change": ExpectedModelChangePolicy,
    "emc": ExpectedModelChangePolicy,
    "diversity": DiversityGreedyPolicy,
    "facility_location": lambda **kw: DiversityGreedyPolicy(
        objective="facility_location", **kw
    ),
    "kdpp": lambda **kw: DiversityGreedyPolicy(objective="kdpp", **kw),
    "thompson": ThompsonPolicy,
    "hybrid": HybridPolicy,
}


def get_policy(policy: str | Policy, **kwargs: Any) -> Policy:
    """Resolve a policy name (or pass through a Policy instance).

    >>> get_policy("bald").name
    'bald'
    >>> get_policy("hybrid", alpha_start=0.9, alpha_end=0.2).alpha_start
    0.9
    """
    if isinstance(policy, Policy):
        if kwargs:
            raise ValueError(
                "Pass policy kwargs when constructing the Policy, or give a "
                "policy name string -- not both."
            )
        return policy
    key = str(policy).strip().lower()
    if key not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy {policy!r}. Available: {sorted(set(POLICY_REGISTRY))}"
        )
    return POLICY_REGISTRY[key](**kwargs)


def available_policies() -> list[str]:
    """Canonical policy names, in the order the docs discuss them."""
    return [
        "random",
        "greedy",
        "uncertainty",
        "bald",
        "expected_model_change",
        "diversity",
        "kdpp",
        "thompson",
        "hybrid",
    ]
