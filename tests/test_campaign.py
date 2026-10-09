"""Campaign and cost-model tests.

Focus on the state machine and the accounting, since those are where a quiet
error costs real wells: a compound selected twice, a label written into the
wrong row, a batch that costs more than the budget, replicates that silently
triple the spend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from loopwise import Campaign
from loopwise.cost import CostModel, ReplicatePolicy
from loopwise.datasets import make_synthetic_pool
from loopwise.models import DegeneratePosteriorError
from loopwise.plate import PLATE_96, PLATE_384


@pytest.fixture
def pool_and_truth():
    pool, info = make_synthetic_pool(600, 12, hit_rate=0.1, seed=0)
    truth = dict(zip(pool["item_id"], pool["label"], strict=True))
    blind = pool.drop(columns=["label", "latent_activity"])
    return blind, truth, info["feature_cols"]


def _model():
    return RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=1)


def _campaign(pool, feature_cols, **kwargs):
    kwargs.setdefault("policy", "hybrid")
    kwargs.setdefault("budget", 48)
    kwargs.setdefault("n_rounds_planned", 4)
    return Campaign(
        pool=pool,
        model=_model(),
        feature_cols=feature_cols,
        id_col="item_id",
        seed=0,
        **kwargs,
    )


def _run_round(campaign, truth):
    batch = campaign.select()
    results = pd.DataFrame(
        {
            "item_id": batch.frame["item_id"],
            "label": [truth[i] for i in batch.frame["item_id"]],
        }
    )
    campaign.observe(results)
    return batch


# ------------------------------------------------------------- state machine


def test_select_observe_loop(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    campaign = _campaign(pool, feature_cols)

    for expected_round in range(3):
        assert campaign.round_index == expected_round
        batch = _run_round(campaign, truth)
        assert len(batch) == 48
        assert batch.round_index == expected_round

    assert campaign.n_labeled == 144
    assert len(campaign.history) == 3


def test_selecting_twice_without_observing_is_refused(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    campaign = _campaign(pool, feature_cols)
    campaign.select()
    with pytest.raises(RuntimeError, match="still pending"):
        campaign.select()

    campaign.discard_pending()
    assert campaign.select() is not None  # discarding unblocks it


def test_measured_compounds_are_not_selected_again(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    campaign = _campaign(pool, feature_cols)

    seen: set = set()
    for _ in range(4):
        batch = _run_round(campaign, truth)
        ids = set(batch.frame["item_id"])
        assert not (ids & seen), "a measured compound was selected again"
        seen |= ids
    assert len(seen) == 192


def test_allow_retest_permits_reselection(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    campaign = _campaign(pool, feature_cols, allow_retest=True, policy="greedy")
    first = _run_round(campaign, truth)
    second = campaign.select()
    assert set(first.frame["item_id"]) & set(second.frame["item_id"])


def test_the_pool_is_never_mutated(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    original = pool.copy(deep=True)
    campaign = _campaign(pool, feature_cols)
    for _ in range(2):
        _run_round(campaign, truth)
    pd.testing.assert_frame_equal(pool, original)


def test_round_one_uses_the_seeding_policy(pool_and_truth):
    """With no labels there is no model, so round 1 must not pretend there is."""
    pool, truth, feature_cols = pool_and_truth
    campaign = _campaign(pool, feature_cols)
    batch = campaign.select()
    assert batch.diagnostics["used_seed_policy"] is True
    assert "seeding" in batch.policy


def test_initial_labels_skip_the_seeding_policy(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    ids = list(pool["item_id"][:80])
    initial = {i: truth[i] for i in ids}
    # Guarantee both classes are present.
    assert len(set(initial.values())) == 2

    campaign = _campaign(pool, feature_cols, initial_labels=initial)
    batch = campaign.select()
    assert batch.diagnostics["used_seed_policy"] is False


def test_single_class_labels_fall_back_to_seeding(pool_and_truth):
    """The normal situation in a low-hit-rate screen: no hits yet."""
    pool, truth, feature_cols = pool_and_truth
    inactive = [i for i in pool["item_id"] if truth[i] == 0][:40]
    campaign = _campaign(pool, feature_cols, initial_labels={i: 0 for i in inactive})
    with pytest.warns(RuntimeWarning, match="same class"):
        batch = campaign.select()
    assert batch.diagnostics["used_seed_policy"] is True


def test_observe_accepts_a_dict_and_averages_replicates(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    campaign = _campaign(pool, feature_cols)
    batch = campaign.select()
    ids = list(batch.frame["item_id"])[:5]

    # Two replicate rows per compound, one 0 and one 1 -> mean 0.5.
    results = pd.DataFrame(
        {"item_id": ids * 2, "label": [0.0] * len(ids) + [1.0] * len(ids)}
    )
    with pytest.warns(RuntimeWarning, match="no measurement"):
        campaign.observe(results)  # only 5 of 48 measured, which is worth a warning
    for item in ids:
        assert campaign.labels[item] == pytest.approx(0.5)


def test_observe_warns_about_unknown_and_missing_ids(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    campaign = _campaign(pool, feature_cols)
    batch = campaign.select()
    partial = list(batch.frame["item_id"])[:10]

    results = pd.DataFrame(
        {
            "item_id": partial + ["PLATE1_CPD00001"],  # a prefixed id: the classic bug
            "label": [0] * len(partial) + [1],
        }
    )
    with pytest.warns(RuntimeWarning) as record:
        campaign.observe(results)
    messages = " ".join(str(w.message) for w in record)
    assert "not in the pool" in messages
    assert "no measurement" in messages


def test_observe_rejects_a_missing_label_column(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    campaign = _campaign(pool, feature_cols)
    campaign.select()
    with pytest.raises(ValueError, match="missing the label column"):
        campaign.observe(pd.DataFrame({"item_id": ["CPD00001"], "readout": [1.0]}))


def test_duplicate_ids_in_the_pool_are_rejected():
    pool = pd.DataFrame({"item_id": ["a", "a", "b"], "f0": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="duplicate"):
        Campaign(pool=pool, model=_model(), id_col="item_id")


def test_label_column_is_excluded_from_features():
    """Training on the label column is the most expensive bug in this code."""
    pool, _ = make_synthetic_pool(200, 5, seed=0)
    campaign = Campaign(
        pool=pool.drop(columns=["latent_activity"]),
        model=_model(),
        id_col="item_id",
        label_col="label",
    )
    assert "label" not in campaign.feature_cols


def test_exhausted_pool_raises_clearly(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    small = pool.iloc[:60].copy()
    campaign = _campaign(small, feature_cols, budget=48, policy="greedy")
    _run_round(campaign, truth)
    batch = campaign.select()  # only 12 remain
    assert len(batch) == 12
    campaign.observe({i: truth[i] for i in batch.frame["item_id"]})
    with pytest.raises(RuntimeError, match="No unmeasured candidates"):
        campaign.select()


def test_degenerate_model_with_information_policy_raises(pool_and_truth):
    """A plain LogisticRegression cannot express uncertainty; say so."""
    pool, truth, feature_cols = pool_and_truth
    ids = list(pool["item_id"][:80])
    campaign = Campaign(
        pool=pool,
        model=LogisticRegression(max_iter=500),
        feature_cols=feature_cols,
        id_col="item_id",
        policy="bald",
        budget=24,
        initial_labels={i: truth[i] for i in ids},
        seed=0,
    )
    with pytest.raises(DegeneratePosteriorError, match="n_bootstrap"):
        campaign.select()


def test_bootstrap_wrapping_rescues_a_degenerate_model(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    ids = list(pool["item_id"][:80])
    campaign = Campaign(
        pool=pool,
        model=LogisticRegression(max_iter=500),
        feature_cols=feature_cols,
        id_col="item_id",
        policy="bald",
        budget=24,
        n_bootstrap=8,
        initial_labels={i: truth[i] for i in ids},
        seed=0,
    )
    batch = campaign.select()
    assert len(batch) == 24
    assert batch.diagnostics["posterior_degenerate"] is False


# -------------------------------------------------------------------- output


def test_batch_frame_carries_reasons_and_scores(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    campaign = _campaign(
        pool, feature_cols, initial_labels={i: truth[i] for i in list(pool["item_id"][:80])}
    )
    batch = campaign.select()

    for column in ("reason", "policy", "rank", "posterior_mean", "posterior_std", "round"):
        assert column in batch.frame.columns
    assert batch.frame["rank"].tolist() == list(range(1, len(batch) + 1))
    assert all(isinstance(r, str) and len(r) > 10 for r in batch.frame["reason"])
    assert "alpha" in batch.frame.columns  # hybrid records its alpha


def test_plate_layout_attached_and_controls_intact(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    campaign = _campaign(pool, feature_cols, budget=80, plate=PLATE_96)
    batch = campaign.select()
    assert batch.plate_layout is not None
    assert batch.plate_layout.control_wells_intact()
    assert len(batch.plate_map()) >= 96


def test_no_plate_template_means_no_layout(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    campaign = _campaign(pool, feature_cols, plate=None)
    batch = campaign.select()
    assert batch.plate_layout is None
    with pytest.raises(ValueError, match="No plate layout"):
        batch.plate_map()


def test_report_metrics(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    campaign = _campaign(pool, feature_cols, budget=48)
    for _ in range(3):
        _run_round(campaign, truth)

    report = campaign.report(oracle_labels=truth)
    frame = report.frame
    assert len(frame) == 3
    assert frame["cumulative_hits"].is_monotonic_increasing
    assert (frame["cumulative_regret"] >= 0).all()
    assert frame["oracle_cumulative_hits"].iloc[-1] >= frame["cumulative_hits"].iloc[-1]
    assert frame["ece"].notna().all()
    assert frame["batch_diversity"].notna().all()
    assert report.has_oracle
    assert "oracle" in report.summary().lower()


def test_report_without_oracle_says_regret_is_realized(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    campaign = _campaign(pool, feature_cols, budget=48)
    _run_round(campaign, truth)
    _run_round(campaign, truth)

    report = campaign.report()
    assert not report.has_oracle
    assert "REALIZED" in " ".join(report.notes)
    assert "oracle_cumulative_hits" not in report.frame.columns


def test_report_plot(tmp_path, pool_and_truth):
    import matplotlib

    matplotlib.use("Agg")
    pool, truth, feature_cols = pool_and_truth
    campaign = _campaign(pool, feature_cols, budget=32)
    for _ in range(2):
        _run_round(campaign, truth)
    path = tmp_path / "report.png"
    campaign.report(oracle_labels=truth).plot(str(path))
    assert path.exists() and path.stat().st_size > 1000


# ----------------------------------------------------------------- cost model


def test_plate_step_function():
    """96 compounds cost two plates when only 80 wells per plate are usable."""
    cost = CostModel(per_item_cost=10.0, plate_fixed_cost=500.0, usable_wells_per_plate=80)
    assert cost.plates_for(80) == 1
    assert cost.plates_for(81) == 2
    assert cost.cost_of([10.0] * 80).total == pytest.approx(80 * 10 + 500)
    assert cost.cost_of([10.0] * 81).total == pytest.approx(81 * 10 + 1000)


def test_currency_budget_respects_the_plate_step():
    cost = CostModel(
        per_item_cost=10.0,
        plate_fixed_cost=500.0,
        usable_wells_per_plate=80,
        budget_currency=1400.0,
    )
    # 80 items: 800 + 500 = 1300 (fits). 81 items: 810 + 1000 = 1810 (does not).
    assert cost.max_items_affordable([10.0] * 200) == 80


def test_replicates_consume_the_budget():
    triplicate = ReplicatePolicy(n_replicates=3, count_against_budget=True)
    cost = CostModel(per_item_cost=5.0, replicates=triplicate, usable_wells_per_plate=96)
    assert cost.max_items_affordable([5.0] * 100, well_budget=96) == 32
    assert cost.wells_for(32) == 96
    # Acquisition is charged once per compound by default, not per well.
    assert cost.cost_of([5.0] * 32).acquisition_cost == pytest.approx(160.0)

    per_well = CostModel(
        per_item_cost=5.0,
        replicates=ReplicatePolicy(3, charge_acquisition_per_replicate=True),
    )
    assert per_well.cost_of([5.0] * 32).acquisition_cost == pytest.approx(480.0)


def test_replicates_excluded_from_budget():
    policy = ReplicatePolicy(n_replicates=3, count_against_budget=False)
    cost = CostModel(per_item_cost=1.0, replicates=policy, usable_wells_per_plate=96)
    assert cost.max_items_affordable([1.0] * 200, well_budget=96) == 96
    assert cost.wells_for(96) == 288


def test_per_compound_costs_from_a_column():
    pool = pd.DataFrame({"item_id": ["a", "b"], "price": [10.0, 900.0]})
    cost = CostModel(per_item_cost="price")
    np.testing.assert_allclose(cost.item_costs(pool=pool), [10.0, 900.0])

    bad = pd.DataFrame({"item_id": ["a"], "price": [np.nan]})
    with pytest.raises(ValueError, match="non-finite"):
        CostModel(per_item_cost="price").item_costs(pool=bad)


def test_campaign_respects_a_currency_budget(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    priced = pool.copy()
    rng = np.random.default_rng(0)
    priced["price"] = rng.uniform(20, 400, len(priced)).round(2)

    cost = CostModel(
        per_item_cost="price",
        plate_fixed_cost=300.0,
        usable_wells_per_plate=80,
        budget_currency=3000.0,
    )
    campaign = Campaign(
        pool=priced,
        model=_model(),
        feature_cols=feature_cols,
        id_col="item_id",
        cost_col="price",
        cost_model=cost,
        budget=96,
        policy="greedy",
        initial_labels={i: truth[i] for i in list(priced["item_id"][:80])},
        seed=0,
    )
    batch = campaign.select()
    assert batch.cost.total <= 3000.0 + 1e-6
    assert len(batch) < 96  # the currency budget bound, not the well count
    assert "currency budget" in batch.plan.explain().lower()


def test_384_plate_campaign(pool_and_truth):
    pool, truth, feature_cols = pool_and_truth
    campaign = _campaign(pool, feature_cols, budget=280, plate=PLATE_384)
    batch = campaign.select()
    assert len(batch) == 280
    assert batch.plate_layout.n_plates == 1
    assert batch.plate_layout.control_wells_intact()


# ------------------------------------- regressions: silent-corruption guards


def test_hit_counts_use_the_configured_threshold(pool_and_truth):
    """Regression: every hit count in the report must use ``hit_threshold``.

    A campaign measuring potency in pIC50 units with ``hit_threshold=6.0``
    once counted every reading above 0.5 as a hit, reporting a 100% hit rate.
    Because the oracle comparison *did* use the configured threshold, the
    report also produced negative regret and "fraction of achievable" above
    1.0 -- internally inconsistent in the headline number.
    """
    pool, _, feature_cols = pool_and_truth
    rng = np.random.default_rng(0)
    labels = pd.Series(rng.uniform(3.0, 9.0, len(pool)).round(2), index=pool.index)
    potency = dict(zip(pool["item_id"], labels, strict=True))
    threshold = 6.0

    campaign = Campaign(
        pool=pool,
        model=_model(),
        feature_cols=feature_cols,
        id_col="item_id",
        label_col="potency",
        budget=24,
        policy="greedy",
        hit_threshold=threshold,
        seed=0,
        initial_labels={i: potency[i] for i in list(pool["item_id"][:80])},
    )
    for _ in range(2):
        batch = campaign.select()
        campaign.observe(
            {i: potency[i] for i in batch.frame["item_id"]}, label_col="potency"
        )
        measured = np.array([potency[i] for i in batch.frame["item_id"]])
        assert campaign.history[-1].n_hits == int((measured > threshold).sum())

    report = campaign.report(oracle_labels=potency)
    frame = report.frame

    # The report must agree with its own calibration block, which has always
    # used the configured threshold.
    # The calibration block stores its rates rounded to 4 decimals.
    for row, calib in zip(frame.itertuples(), report.calibration, strict=True):
        assert row.hit_rate == pytest.approx(calib["observed_hit_rate"], abs=1e-4)

    assert (frame["cumulative_regret"] >= 0).all(), "negative regret"
    assert (frame["fraction_of_achievable"] <= 1.0).all(), "found more than exists"
