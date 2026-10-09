"""Adapter tests, including the FoldRescue integration.

Every adapter must satisfy the same contract -- fit, then produce posterior
samples with real disagreement -- so a policy cannot tell them apart. The
LightGBM and PyTorch tests skip when their library is missing, which is how CI
stays fast without pretending the adapters are untested.
"""

from __future__ import annotations

import numpy as np
import pytest

from loopwise.models import DegeneratePosteriorError, Posterior
from loopwise.policies import PolicyContext, get_policy
from loopwise.utils import make_rng


@pytest.fixture
def training_data():
    rng = make_rng(0)
    X = rng.normal(size=(260, 10)).astype(np.float32)
    y = (X[:, 0] + 0.6 * X[:, 1] + rng.normal(scale=0.4, size=260) > 0.5).astype(int)
    return X[:160], y[:160], X[160:]


def _assert_usable_posterior(posterior: Posterior, n_items: int) -> None:
    """The contract every adapter must meet."""
    assert posterior.n_items == n_items
    assert posterior.n_samples >= 2
    assert not posterior.is_degenerate, "adapter produced no disagreement"
    assert np.isfinite(posterior.mean).all()
    assert np.isfinite(posterior.variance).all()
    assert (posterior.variance >= 0).all()
    mi = posterior.mutual_information()
    assert np.isfinite(mi).all() and (mi >= 0).all()
    if posterior.task == "classification":
        assert ((posterior.mean >= 0) & (posterior.mean <= 1)).all()


def _assert_drives_every_policy(adapter, X_train, y_train, X_pool) -> None:
    adapter.fit(X_train, y_train)
    posterior = adapter.posterior(X_pool, n_samples=16, rng=make_rng(1))
    _assert_usable_posterior(posterior, X_pool.shape[0])

    for name in ("greedy", "uncertainty", "bald", "thompson", "hybrid", "kdpp"):
        ctx = PolicyContext(
            features=X_pool,
            posterior=posterior,
            budget=12,
            rng=make_rng(2),
            round_index=0,
            n_rounds_planned=4,
        )
        result = get_policy(name).select(ctx)
        assert result.indices.size == 12
        assert np.unique(result.indices).size == 12


# ---------------------------------------------------------------- LightGBM


@pytest.mark.parametrize("uncertainty", ["bootstrap"])
def test_lightgbm_classification_adapter(training_data, uncertainty):
    pytest.importorskip("lightgbm")
    from loopwise.contrib import LightGBMAdapter

    X_train, y_train, X_pool = training_data
    adapter = LightGBMAdapter(task="classification", uncertainty=uncertainty, n_members=6)
    _assert_drives_every_policy(adapter, X_train, y_train, X_pool)
    assert adapter.predict_mean(X_pool).shape == (X_pool.shape[0],)


def test_lightgbm_quantile_regression(training_data):
    pytest.importorskip("lightgbm")
    from loopwise.contrib import LightGBMAdapter

    X_train, y_train, X_pool = training_data
    y_continuous = X_train[:, 0] * 2.0 + 1.0
    adapter = LightGBMAdapter(task="regression", uncertainty="quantile")
    adapter.fit(X_train, y_continuous)
    posterior = adapter.posterior(X_pool, n_samples=20, rng=make_rng(0))
    _assert_usable_posterior(posterior, X_pool.shape[0])
    assert posterior.task == "regression"


def test_lightgbm_rejects_quantile_for_classification():
    pytest.importorskip("lightgbm")
    from loopwise.contrib import LightGBMAdapter

    with pytest.raises(ValueError, match="regression-only"):
        LightGBMAdapter(task="classification", uncertainty="quantile")


# -------------------------------------------------------------------- Torch


def test_torch_ensemble_adapter(training_data):
    pytest.importorskip("torch")
    from loopwise.contrib import TorchEnsembleAdapter

    X_train, y_train, X_pool = training_data
    adapter = TorchEnsembleAdapter(
        task="classification",
        n_models=2,
        hidden=(16, 16),
        dropout=0.2,
        mc_dropout_samples=5,
        epochs=25,
    )
    _assert_drives_every_policy(adapter, X_train, y_train, X_pool)
    # 2 models x 5 dropout passes
    assert adapter.posterior(X_pool, n_samples=10).n_samples == 10


def test_torch_adapter_rejects_a_deterministic_configuration():
    pytest.importorskip("torch")
    from loopwise.contrib import TorchEnsembleAdapter

    with pytest.raises(ValueError, match="deterministic"):
        TorchEnsembleAdapter(n_models=1, mc_dropout_samples=1)
    with pytest.raises(ValueError, match="requires dropout"):
        TorchEnsembleAdapter(n_models=1, dropout=0.0, mc_dropout_samples=5)


def test_torch_adapter_works_through_auto_adapter(training_data):
    """It exposes sample(), so auto_adapter must find it."""
    pytest.importorskip("torch")
    from loopwise.contrib import TorchEnsembleAdapter
    from loopwise.models import auto_adapter

    X_train, y_train, X_pool = training_data
    inner = TorchEnsembleAdapter(n_models=2, hidden=(8,), mc_dropout_samples=4, epochs=10)
    inner.fit(X_train, y_train)
    wrapped = auto_adapter(inner)
    assert wrapped.posterior(X_pool, n_samples=8).n_samples >= 2


# ----------------------------------------------------------- Gaussian process


def test_gaussian_process_adapter(training_data):
    from loopwise.contrib import GaussianProcessAdapter

    X_train, y_train, X_pool = training_data
    adapter = GaussianProcessAdapter(task="classification", use_regressor=True)
    _assert_drives_every_policy(adapter, X_train[:80], y_train[:80], X_pool[:60])
    assert "Matern" in adapter.kernel_summary() or "kernel" in adapter.kernel_summary()


def test_gaussian_process_refuses_an_oversized_training_set():
    """A GP in a loop grows its training set every round; it must refuse early."""
    from loopwise.contrib import GaussianProcessAdapter

    adapter = GaussianProcessAdapter(max_train=100)
    X = np.zeros((101, 3), dtype=np.float32)
    y = np.zeros(101)
    with pytest.raises(MemoryError, match="O\\(n\\^3\\)"):
        adapter.fit(X, y)


def test_gaussian_process_posterior_is_exact_for_regression():
    """A GP fitted on noiseless data must reproduce it and be confident there."""
    from loopwise.contrib import GaussianProcessAdapter

    rng = make_rng(0)
    X = rng.uniform(-2, 2, size=(40, 1)).astype(np.float32)
    y = np.sin(X[:, 0]) * 2.0

    adapter = GaussianProcessAdapter(task="regression")
    adapter.fit(X, y)
    posterior = adapter.posterior(X, n_samples=30, rng=make_rng(1))
    np.testing.assert_allclose(posterior.mean, y, atol=0.35)

    # Uncertainty must be larger away from the training data than on it.
    far = np.array([[9.0]], dtype=np.float32)
    assert adapter.posterior(far, n_samples=30, rng=make_rng(1)).std[0] > float(
        posterior.std.mean()
    )


# ---------------------------------------------------------------- FoldRescue


@pytest.fixture
def glb1_pool():
    from loopwise.contrib.foldrescue import SyntheticFoldRescueModel

    return SyntheticFoldRescueModel.make_glb1_like_pool(
        n_items=700, n_features=12, rescue_rate=0.05, seed=0
    )


def test_foldrescue_adapter_uses_native_sampling(glb1_pool):
    from loopwise.contrib.foldrescue import (
        FoldRescueAdapter,
        SyntheticFoldRescueModel,
    )

    pool, info = glb1_pool
    X = pool[info["feature_cols"]].to_numpy(dtype=np.float32)
    y = pool["rescues_folding"].to_numpy()

    adapter = FoldRescueAdapter(SyntheticFoldRescueModel(random_state=0))
    assert "native-sampling" in adapter.uncertainty_mode
    _assert_drives_every_policy(adapter, X[:200], y[:200], X[200:400])


def test_foldrescue_adapter_bootstraps_a_point_model(glb1_pool):
    """A model with only predict_proba + fit gets a bootstrap posterior."""
    from sklearn.linear_model import LogisticRegression

    from loopwise.contrib.foldrescue import FoldRescueAdapter

    pool, info = glb1_pool
    X = pool[info["feature_cols"]].to_numpy(dtype=np.float32)
    y = pool["rescues_folding"].to_numpy()

    adapter = FoldRescueAdapter(LogisticRegression(max_iter=400), n_bootstrap=8)
    assert adapter.uncertainty_mode.startswith("bootstrap")
    _assert_drives_every_policy(adapter, X[:200], y[:200], X[200:400])


def test_foldrescue_adapter_warns_about_a_frozen_point_model():
    """No sample(), no fit() -> no uncertainty, and it must say so loudly."""
    from loopwise.contrib.foldrescue import FoldRescueAdapter

    class FrozenModel:
        def predict_rescue(self, X):
            return np.full(len(X), 0.3, dtype=np.float32)

    with pytest.warns(RuntimeWarning, match="cannot express uncertainty"):
        adapter = FoldRescueAdapter(FrozenModel(), n_bootstrap=None)
    assert adapter.uncertainty_mode == "point-estimate-only"

    posterior = adapter.posterior(np.zeros((5, 3), dtype=np.float32))
    assert posterior.is_degenerate
    # Greedy still works; BALD must refuse rather than return an arbitrary batch.
    ctx = PolicyContext(
        features=np.zeros((5, 3), dtype=np.float32),
        posterior=posterior,
        budget=2,
        rng=make_rng(0),
    )
    assert get_policy("greedy").select(ctx).indices.size == 2
    with pytest.raises(DegeneratePosteriorError):
        get_policy("bald").select(ctx)


def test_foldrescue_rejects_a_model_with_no_usable_method():
    from loopwise.contrib.foldrescue import FOLDRESCUE_PROTOCOL, FoldRescueAdapter

    class Useless:
        pass

    with pytest.raises(TypeError, match="protocol"):
        FoldRescueAdapter(Useless())
    assert "sample" in FOLDRESCUE_PROTOCOL["one_of"]


def test_foldrescue_campaign_end_to_end(glb1_pool):
    """FoldRescue predictions -> Loopwise batch -> plate map, with costs."""
    from loopwise.contrib.foldrescue import foldrescue_campaign

    pool, info = glb1_pool
    truth = dict(zip(pool["compound_id"], pool["rescues_folding"], strict=True))
    blind = pool.drop(columns=["rescues_folding"])
    seeds = list(blind["compound_id"][:120])

    with pytest.warns(RuntimeWarning, match="stand-in"):
        campaign = foldrescue_campaign(
            blind,
            model=None,
            budget=96,
            policy="hybrid",
            feature_cols=info["feature_cols"],
            n_replicates=2,
            n_rounds_planned=3,
            initial_labels={i: truth[i] for i in seeds},
            seed=0,
        )

    batch = campaign.select()
    # 96 wells at 2 replicates each buys 48 compounds.
    assert len(batch) == 48
    assert batch.plate_layout is not None
    assert batch.plate_layout.control_wells_intact()

    frame = batch.plate_layout.to_frame()
    samples = frame[frame["content_type"] == "sample"]
    assert len(samples) == 96
    assert samples.groupby("item_id").size().unique().tolist() == [2]

    # Cost accounting: per-compound acquisition plus the plate fixed cost.
    assert batch.cost.plate_cost == pytest.approx(450.0 * batch.cost.n_plates)
    assert batch.cost.acquisition_cost > 0

    campaign.observe({i: truth[i] for i in batch.frame["compound_id"]})
    assert len(campaign.history) == 1
    report = campaign.report(oracle_labels=truth)
    assert len(report.frame) == 1


def test_glb1_pool_has_the_structure_the_docstring_claims(glb1_pool):
    pool, info = glb1_pool
    assert info["n_compounds"] == 700
    assert 0.02 <= info["rescue_rate"] <= 0.09
    assert "NOT real data" in info["dataset"]
    assert pool["acquisition_cost_usd"].min() > 0
    # Bespoke compounds cost an order of magnitude more than library ones.
    assert pool["acquisition_cost_usd"].max() / pool["acquisition_cost_usd"].min() > 5

    # The inverted-U: rescuers cluster at intermediate binding affinity, so the
    # mean affinity of rescuers must sit inside the pool's range, not at an end.
    affinity = pool["pharmacophore_0"]
    rescuer_mean = affinity[pool["rescues_folding"] == 1].mean()
    assert affinity.quantile(0.2) < rescuer_mean < affinity.quantile(0.95)
