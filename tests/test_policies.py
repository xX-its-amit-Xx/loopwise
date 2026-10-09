"""Policy contract tests.

The headline invariant: **every policy returns exactly ``budget`` unique
items.** It sounds trivial. It is the single most common bug in this kind of
code, because the batch-aware policies (facility location, k-DPP) build their
selection incrementally and can quietly terminate early -- a k-DPP whose
residuals all collapse, a facility-location greedy whose candidate subset is
exhausted. A policy that returns 94 of 96 wastes two wells silently, and a
policy that returns a duplicate wastes one and corrupts the label ledger.
"""

from __future__ import annotations

import numpy as np
import pytest

from loopwise.models import DegeneratePosteriorError, Posterior, auto_adapter
from loopwise.policies import (
    DiversityGreedyPolicy,
    HybridPolicy,
    PolicyContext,
    available_policies,
    get_policy,
)
from loopwise.utils import make_rng

BUDGETS = [1, 7, 24, 96]


@pytest.fixture
def context_factory():
    """Builds a PolicyContext over a clustered pool with a real posterior."""

    def build(n_items: int = 400, n_features: int = 10, n_samples: int = 24, seed: int = 0):
        rng = make_rng(seed)
        centers = rng.normal(scale=2.0, size=(6, n_features))
        assign = rng.integers(0, 6, size=n_items)
        X = (centers[assign] + rng.normal(scale=0.7, size=(n_items, n_features))).astype(
            np.float32
        )
        # A posterior with genuine disagreement between samples.
        base = 1.0 / (1.0 + np.exp(-(X[:, 0] - 0.5)))
        samples = np.clip(
            base[None, :] + rng.normal(scale=0.12, size=(n_samples, n_items)), 0.01, 0.99
        ).astype(np.float32)
        return PolicyContext(
            features=X,
            posterior=Posterior(samples),
            budget=24,
            rng=make_rng(seed + 1),
            round_index=1,
            n_rounds_planned=6,
        )

    return build


@pytest.mark.parametrize("name", available_policies())
@pytest.mark.parametrize("budget", BUDGETS)
def test_policy_returns_exactly_budget_unique_items(context_factory, name, budget):
    ctx = context_factory()
    ctx.budget = budget
    result = get_policy(name).select(ctx)

    assert result.indices.size == budget, (
        f"{name} returned {result.indices.size} items for budget={budget}"
    )
    assert np.unique(result.indices).size == budget, f"{name} returned duplicates"
    assert result.indices.min() >= 0 and result.indices.max() < ctx.n_candidates


@pytest.mark.parametrize("name", available_policies())
def test_policy_emits_a_reason_per_item(context_factory, name):
    ctx = context_factory()
    result = get_policy(name).select(ctx)
    assert "reason" in result.reasons, f"{name} emitted no human-readable reason"
    for column, values in result.reasons.items():
        assert len(values) == result.indices.size, f"{name}: ragged reason {column!r}"
    assert all(isinstance(r, str) and r for r in result.reasons["reason"])


@pytest.mark.parametrize("name", available_policies())
def test_policy_handles_budget_larger_than_pool(context_factory, name):
    """Asking for more than exists returns everything, not an exception."""
    ctx = context_factory(n_items=15)
    ctx.budget = 50
    result = get_policy(name).select(ctx)
    assert result.indices.size == 15
    assert np.unique(result.indices).size == 15


@pytest.mark.parametrize("name", available_policies())
def test_policy_is_deterministic_under_a_fixed_seed(context_factory, name):
    first = get_policy(name).select(context_factory(seed=3))
    second = get_policy(name).select(context_factory(seed=3))
    np.testing.assert_array_equal(first.indices, second.indices)


def test_greedy_takes_the_highest_scores(context_factory):
    ctx = context_factory()
    result = get_policy("greedy").select(ctx)
    expected = np.sort(ctx.posterior.mean)[-ctx.budget :]
    np.testing.assert_allclose(
        np.sort(ctx.posterior.mean[result.indices]), expected, rtol=1e-6
    )


def test_bald_ignores_purely_aleatoric_uncertainty():
    """The distinction BALD exists for.

    Item 0: every posterior sample says 0.5 -- the model is *certain* the
    outcome is a coin flip. Maximal predictive entropy, zero information.
    Item 1: samples split between 0.02 and 0.98 -- the model does not know.
    Same mean prediction (0.5). Uncertainty sampling cannot tell them apart;
    BALD must prefer item 1.
    """
    samples = np.array(
        [
            [0.5, 0.02, 0.5],
            [0.5, 0.98, 0.5],
            [0.5, 0.02, 0.5],
            [0.5, 0.98, 0.5],
        ],
        dtype=np.float32,
    )
    posterior = Posterior(samples)
    mi = posterior.mutual_information()

    assert mi[1] > mi[0] + 0.2, "BALD failed to prefer epistemic over aleatoric"
    np.testing.assert_allclose(mi[0], 0.0, atol=1e-5)
    # Predictive entropy, by contrast, cannot distinguish them.
    entropy = posterior.predictive_entropy()
    np.testing.assert_allclose(entropy[0], entropy[1], atol=1e-3)


def test_information_policies_refuse_a_degenerate_posterior(context_factory):
    """A model that cannot disagree with itself must raise, not return junk.

    This is the silent-failure guard: with a single fitted classifier every
    information score is identically zero, so the "chosen" batch is whatever
    the tie-break produced. Loopwise refuses instead.
    """
    ctx = context_factory()
    ctx.posterior = Posterior(ctx.posterior.mean[None, :])  # one sample only
    assert ctx.posterior.is_degenerate

    for name in ("uncertainty", "bald", "thompson", "expected_model_change", "hybrid"):
        with pytest.raises(DegeneratePosteriorError):
            get_policy(name).select(ctx)

    # Greedy, random and diversity need no uncertainty and must still work.
    for name in ("greedy", "random", "diversity", "kdpp"):
        assert get_policy(name).select(ctx).indices.size == ctx.budget


def test_diversity_beats_greedy_on_intra_batch_similarity(context_factory):
    """The reason diversity policies exist: greedy buys the same compound twice."""
    ctx = context_factory(n_items=600)
    ctx.budget = 30

    greedy = get_policy("greedy").select(ctx)
    diverse = get_policy("diversity").select(ctx)
    kdpp = get_policy("kdpp").select(ctx)

    def mean_pairwise_distance(indices):
        X = ctx.features[indices].astype(np.float64)
        d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        iu = np.triu_indices(len(indices), k=1)
        return float(d[iu].mean())

    assert mean_pairwise_distance(diverse.indices) > mean_pairwise_distance(greedy.indices)
    assert mean_pairwise_distance(kdpp.indices) > mean_pairwise_distance(greedy.indices)


def test_hybrid_alpha_anneals_across_rounds():
    policy = HybridPolicy(alpha_start=0.2, alpha_end=0.9, schedule="linear")
    alphas = [policy.alpha_for_round(r, 5) for r in range(5)]

    assert alphas[0] == pytest.approx(0.2)
    assert alphas[-1] == pytest.approx(0.9)
    assert all(b > a for a, b in zip(alphas, alphas[1:], strict=False)), (
        "alpha is not monotone"
    )

    # Front-loading hits is the opposite schedule, and must be supported.
    reverse = HybridPolicy(alpha_start=0.9, alpha_end=0.2)
    assert reverse.alpha_for_round(0, 5) > reverse.alpha_for_round(4, 5)

    # A fixed alpha disables annealing entirely.
    fixed = HybridPolicy(alpha=0.5)
    assert fixed.alpha_for_round(0, 5) == fixed.alpha_for_round(4, 5) == 0.5


def test_hybrid_alpha_extremes_match_pure_policies(context_factory):
    """alpha=1 must be greedy; alpha=0 must be the explore term alone."""
    ctx = context_factory()

    exploit_only = HybridPolicy(alpha=1.0).select(ctx)
    greedy = get_policy("greedy").select(ctx)
    assert set(exploit_only.indices.tolist()) == set(greedy.indices.tolist())

    explore_only = HybridPolicy(alpha=0.0, explore="bald").select(ctx)
    bald = get_policy("bald").select(ctx)
    overlap = len(set(explore_only.indices.tolist()) & set(bald.indices.tolist()))
    assert overlap >= int(0.9 * ctx.budget), f"only {overlap}/{ctx.budget} overlap"


def test_hybrid_records_the_alpha_it_used(context_factory):
    ctx = context_factory()
    result = HybridPolicy(alpha_start=0.3, alpha_end=0.8).select(ctx)
    assert "alpha" in result.reasons
    assert result.diagnostics["alpha"] == pytest.approx(
        result.reasons["alpha"][0], rel=1e-5
    )


def test_thompson_uses_several_posterior_draws(context_factory):
    ctx = context_factory()
    result = get_policy("thompson").select(ctx)
    assert result.diagnostics["distinct_draws"] > 1, (
        "per-slot Thompson sampling collapsed to a single draw"
    )


def test_quality_weight_zero_gives_pure_space_filling(context_factory):
    """quality_weight=0 must ignore the model entirely -- the round-1 case."""
    ctx = context_factory()
    space_filling = DiversityGreedyPolicy(quality_weight=0.0).select(ctx)
    quality_weighted = DiversityGreedyPolicy(quality_weight=2.0).select(ctx)

    assert (
        ctx.posterior.mean[quality_weighted.indices].mean()
        > ctx.posterior.mean[space_filling.indices].mean()
    )


def test_tree_adapter_posterior_has_real_disagreement():
    from sklearn.ensemble import RandomForestClassifier

    rng = make_rng(0)
    X = rng.normal(size=(200, 8)).astype(np.float32)
    y = (X[:, 0] > 0.3).astype(int)
    adapter = auto_adapter(RandomForestClassifier(n_estimators=30, random_state=0))
    adapter.fit(X[:60], y[:60])

    posterior = adapter.posterior(X[60:], n_samples=30, rng=rng)
    assert posterior.n_samples > 1
    assert not posterior.is_degenerate
    assert posterior.mutual_information().max() > 0.0


def test_unknown_policy_name_is_a_clear_error():
    with pytest.raises(ValueError, match="Unknown policy"):
        get_policy("definitely_not_a_policy")
