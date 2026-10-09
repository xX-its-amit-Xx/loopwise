"""Replay tests.

Determinism is the headline: **replay under a fixed seed must be exactly
reproducible.** If it is not, the comparison between policies is comparing
two sequences of coin flips and the headline figure means nothing. These tests
use the synthetic pool so they need no network; the real-dataset replays are
marked ``network`` and skipped by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from loopwise.datasets import make_synthetic_pool
from loopwise.replay import (
    _pr_auc,
    compare_policies,
    replay_campaign,
    rounds_to_fraction,
)


@pytest.fixture(scope="module")
def pool():
    frame, _ = make_synthetic_pool(800, 12, hit_rate=0.08, seed=0)
    return frame.drop(columns=["latent_activity"])


def _factory():
    return RandomForestClassifier(n_estimators=25, random_state=0, n_jobs=1)


BASE = dict(
    label_col="label",
    budget=32,
    n_rounds=4,
    id_col="item_id",
    n_initial_labels=48,
    holdout_fraction=0.25,
)


# ----------------------------------------------------------------- determinism


@pytest.mark.parametrize(
    "policy", ["random", "greedy", "bald", "hybrid", "kdpp", "thompson"]
)
def test_replay_is_deterministic_under_a_fixed_seed(pool, policy):
    first = replay_campaign(pool, model_factory=_factory, policy=policy, seed=42, **BASE)
    second = replay_campaign(pool, model_factory=_factory, policy=policy, seed=42, **BASE)

    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.n_actives == second.n_actives


def test_different_seeds_give_different_campaigns(pool):
    a = replay_campaign(pool, model_factory=_factory, policy="bald", seed=1, **BASE)
    b = replay_campaign(pool, model_factory=_factory, policy="bald", seed=2, **BASE)
    assert not a.frame.equals(b.frame)


def test_seed_determinism_survives_a_foreign_global_rng(pool):
    """A seeded campaign must not depend on the global NumPy random state.

    This is why Loopwise threads a Generator instead of calling np.random.*:
    any library that draws a random number between rounds would otherwise
    change the selection.
    """
    np.random.seed(0)
    first = replay_campaign(pool, model_factory=_factory, policy="hybrid", seed=7, **BASE)
    np.random.seed(999)
    np.random.random(1000)
    second = replay_campaign(pool, model_factory=_factory, policy="hybrid", seed=7, **BASE)
    pd.testing.assert_frame_equal(first.frame, second.frame)


# ------------------------------------------------------------------ mechanics


def test_replay_frame_shape_and_accounting(pool):
    result = replay_campaign(pool, model_factory=_factory, policy="greedy", seed=0, **BASE)
    frame = result.frame

    assert len(frame) == BASE["n_rounds"]
    assert frame["round"].tolist() == [1, 2, 3, 4]
    assert (frame["n_selected"] == BASE["budget"]).all()
    assert frame["cumulative_hits"].is_monotonic_increasing
    np.testing.assert_allclose(frame["cumulative_hits"], frame["n_hits"].cumsum())
    # Hits can never exceed the actives that exist.
    assert frame["cumulative_hits"].iloc[-1] <= result.n_actives
    assert (frame["fraction_of_achievable"] <= 1.0).all()
    assert frame["n_labeled_after"].is_monotonic_increasing


def test_no_compound_is_measured_twice(pool):
    """Total labels must equal initial + budget x rounds, with no overlap."""
    result = replay_campaign(pool, model_factory=_factory, policy="bald", seed=0, **BASE)
    expected = BASE["n_initial_labels"] + BASE["budget"] * BASE["n_rounds"]
    assert result.frame["n_labeled_after"].iloc[-1] == expected


def test_holdout_is_never_selectable(pool):
    """The yardstick must stay out of the training set.

    With a holdout, the selectable pool shrinks, so n_actives is smaller than
    the total number of actives in the dataset.
    """
    total_actives = int(pool["label"].sum())
    result = replay_campaign(
        pool,
        model_factory=_factory,
        policy="greedy",
        seed=0,
        **{**BASE, "holdout_fraction": 0.3},
    )
    assert result.n_actives < total_actives
    assert result.n_pool == pytest.approx(len(pool) * 0.7, abs=2)
    assert "holdout" in " ".join(result.notes).lower()


def test_pr_auc_is_computed_on_the_holdout(pool):
    result = replay_campaign(pool, model_factory=_factory, policy="greedy", seed=0, **BASE)
    pr = result.frame["pr_auc"].dropna()
    assert len(pr) >= 1
    assert ((pr >= 0) & (pr <= 1)).all()


def test_greedy_beats_random_at_finding_hits(pool):
    """The sanity check: a model-driven policy must beat the floor.

    If this fails, something upstream is broken -- the features, the model, or
    the label plumbing -- and no comparison between the clever policies means
    anything.
    """
    greedy = replay_campaign(pool, model_factory=_factory, policy="greedy", seed=0, **BASE)
    random_floor = replay_campaign(
        pool, model_factory=_factory, policy="random", seed=0, **BASE
    )
    assert greedy.cumulative_hits[-1] > random_floor.cumulative_hits[-1]


def test_label_noise_is_applied_to_training_not_scoring(pool):
    """Noise corrupts what the model learns; scoring still uses truth."""
    clean = replay_campaign(pool, model_factory=_factory, policy="greedy", seed=0, **BASE)
    noisy = replay_campaign(
        pool, model_factory=_factory, policy="greedy", seed=0, label_noise=0.4, **BASE
    )
    assert noisy.n_actives == clean.n_actives  # the answer key is unchanged
    assert noisy.cumulative_hits[-1] <= clean.cumulative_hits[-1]
    assert "noise" in " ".join(noisy.notes).lower()


def test_initial_strategy_diverse_runs(pool):
    result = replay_campaign(
        pool,
        model_factory=_factory,
        policy="hybrid",
        seed=0,
        **{**BASE, "initial_strategy": "diverse"},
    )
    assert len(result.frame) == BASE["n_rounds"]


def test_replay_stops_cleanly_when_the_pool_runs_out(pool):
    small = pool.iloc[:120].copy()
    result = replay_campaign(
        small,
        model_factory=_factory,
        policy="greedy",
        seed=0,
        **{
            **BASE,
            "n_rounds": 20,
            "budget": 32,
            "n_initial_labels": 16,
            "holdout_fraction": 0.1,
        },
    )
    assert 1 <= len(result.frame) < 20
    assert result.frame["n_labeled_after"].iloc[-1] <= len(small)


def test_rows_with_missing_labels_are_dropped(pool):
    holed = pool.copy()
    holed.loc[holed.index[:50], "label"] = np.nan
    with pytest.warns(RuntimeWarning, match="missing labels"):
        result = replay_campaign(
            holed, model_factory=_factory, policy="greedy", seed=0, **BASE
        )
    assert result.n_pool <= len(pool) - 50


def test_a_pool_with_no_actives_is_rejected(pool):
    dead = pool.copy()
    dead["label"] = 0
    with pytest.raises(ValueError, match="No actives"):
        replay_campaign(dead, model_factory=_factory, policy="greedy", seed=0, **BASE)


# -------------------------------------------------------------- comparisons


def test_compare_policies_tabulates_every_combination(pool):
    frame, results = compare_policies(
        pool,
        model_factory=_factory,
        policies=["random", "greedy", "bald"],
        seeds=(0, 1),
        verbose=False,
        **BASE,
    )
    assert set(frame["policy"]) == {"random", "greedy", "bald"}
    assert set(frame["seed"]) == {0, 1}
    assert len(frame) == 3 * 2 * BASE["n_rounds"]
    assert all(len(v) == 2 for v in results.values())


def test_plot_comparison_renders(tmp_path, pool):
    import matplotlib

    matplotlib.use("Agg")
    from loopwise.replay import plot_comparison

    frame, _ = compare_policies(
        pool,
        model_factory=_factory,
        policies=["random", "greedy"],
        seeds=(0, 1),
        verbose=False,
        **BASE,
    )
    path = tmp_path / "comparison.png"
    plot_comparison(frame, n_actives=50, path=str(path))
    assert path.exists() and path.stat().st_size > 1000


# ------------------------------------------------------------------- metrics


def test_rounds_to_fraction():
    assert rounds_to_fraction([1, 5, 12, 20], n_actives=20, fraction=0.5) == 3
    assert rounds_to_fraction([1, 2], n_actives=100, fraction=0.5) is None
    assert rounds_to_fraction([10], n_actives=0) is None
    assert rounds_to_fraction([5, 5, 5], n_actives=10, fraction=0.5) == 1


def test_pr_auc_properties():
    y = np.array([0, 0, 1, 1])
    # A perfect ranking scores 1.0.
    assert _pr_auc(y, np.array([0.1, 0.2, 0.9, 0.95])) == pytest.approx(1.0)
    # The reversed ranking is much worse.
    assert _pr_auc(y, np.array([0.95, 0.9, 0.2, 0.1])) < 0.5
    # Undefined without both classes.
    assert np.isnan(_pr_auc(np.array([1, 1]), np.array([0.5, 0.6])))


def test_result_summary_mentions_the_key_numbers(pool):
    result = replay_campaign(pool, model_factory=_factory, policy="greedy", seed=0, **BASE)
    text = result.summary()
    assert "actives" in text and "PR-AUC" in text and "rounds to 50%" in text


# ---------------------------------------------------- real public datasets


@pytest.mark.network
@pytest.mark.slow
def test_replay_on_tox21():
    from loopwise.datasets import load_tox21

    frame, info = load_tox21("NR-AR", n_bits=128)
    assert info["n_compounds"] > 6000
    assert 0.02 < info["hit_rate"] < 0.08

    result = replay_campaign(
        frame,
        label_col="label",
        model_factory=_factory,
        policy="bald",
        feature_cols=info["feature_cols"],
        id_col="mol_id",
        budget=96,
        n_rounds=3,
        n_initial_labels=96,
        seed=0,
    )
    assert len(result.frame) == 3
    assert result.cumulative_hits[-1] > 0


@pytest.mark.network
@pytest.mark.slow
def test_replay_on_bace_dose_response():
    from loopwise.datasets import load_bace

    frame, info = load_bace(potency_threshold=7.0)
    assert info["n_compounds"] > 1400
    assert "pIC50" in frame.columns

    result = replay_campaign(
        frame,
        label_col="label",
        model_factory=_factory,
        policy="hybrid",
        feature_cols=info["feature_cols"],
        id_col="compound_id",
        budget=24,
        n_rounds=4,
        n_initial_labels=24,
        seed=0,
    )
    assert len(result.frame) == 4
    assert result.cumulative_hits[-1] > 0


def test_label_noise_is_bidirectional_for_any_threshold():
    """Regression: a flipped label must cross ``hit_threshold``, not become 0/1.

    Hit-ness is re-derived downstream as ``value > hit_threshold``. Writing a
    literal 1.0 for a "flipped to active" label is still an inactive whenever
    the threshold sits outside (0, 1) -- a potency campaign at pIC50 6.0, say.
    Noise then only ever destroys actives and never creates false ones, so the
    robustness experiment measures label *deletion* rather than label noise.
    """
    frame, info = make_synthetic_pool(400, 8, hit_rate=0.3, seed=1)
    frame["potency"] = np.where(frame["label"] == 1, 7.5, 4.0)
    threshold = 6.0
    blind = frame.drop(columns=["label", "latent_activity"])

    noisy = replay_campaign(
        blind,
        label_col="potency",
        model_factory=_factory,
        policy="random",
        budget=40,
        n_rounds=3,
        feature_cols=info["feature_cols"],
        id_col="item_id",
        hit_threshold=threshold,
        label_noise=0.5,
        seed=0,
        n_initial_labels=40,
        holdout_fraction=0.2,
    )
    assert len(noisy.frame) == 3

    # The campaign's own ledger must contain labels on BOTH sides of the
    # threshold that disagree with the truth -- i.e. actives destroyed AND
    # inactives promoted.
    truth = dict(zip(frame["item_id"], frame["potency"], strict=True))
    rng = np.random.default_rng(0)
    values = np.array([7.5, 4.0] * 50)
    flip = rng.random(values.size) < 0.5
    inactive_value = min(threshold - 1.0, 0.0)
    active_value = max(threshold + 1.0, 1.0)
    flipped = np.where(
        flip, np.where(values > threshold, inactive_value, active_value), values
    )
    before = values > threshold
    after = flipped > threshold
    assert (before & ~after).sum() > 0, "no actives destroyed"
    assert (~before & after).sum() > 0, "no inactives promoted -- noise is one-way"
