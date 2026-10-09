"""LightGBM adapter.

LightGBM is a gradient-boosting library: it fits a sequence of small decision
trees where each tree corrects the errors of the ones before it. That makes it
excellent at ranking compounds and useless at saying how sure it is -- the
trees are not independent samples of anything, they are successive corrections,
so their spread is not a posterior.

Two ways to get uncertainty out of it:

``uncertainty="bootstrap"`` (default)
    Fit ``n_members`` boosters on bootstrap resamples of the training data.
    Spread across boosters estimates how much the *data* pins the answer
    down. Costs ``n_members`` fits per round, which for a few hundred labelled
    compounds is a second or two.

``uncertainty="quantile"``
    Fit three regression objectives -- lower quantile, median, upper quantile
    -- and treat the resulting interval as a predictive distribution.
    One-third the cost of a 3-member bootstrap and genuinely calibrated for
    *aleatoric* noise, but it says nothing about epistemic uncertainty: a
    quantile model is equally confident in a region with 500 training examples
    and a region with none. Since epistemic uncertainty is the thing active
    learning exploits, this mode is for regression campaigns where you care
    about the spread of the readout, not for BALD. Regression only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models import ModelAdapter, Posterior, _proba_positive
from ..utils import DEFAULT_DTYPE, check_resources, make_rng

__all__ = ["LightGBMAdapter"]


def _require_lightgbm():
    try:
        import lightgbm as lgb

        return lgb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "LightGBMAdapter needs LightGBM: pip install 'loopwise[lightgbm]' "
            "(or pip install lightgbm)."
        ) from exc


class LightGBMAdapter(ModelAdapter):
    """Wrap LightGBM so Loopwise can sample from it.

    Parameters
    ----------
    task
        ``"classification"`` or ``"regression"``.
    uncertainty
        ``"bootstrap"`` or ``"quantile"`` (regression only).
    n_members
        Bootstrap members. 10-20 is the usual sweet spot; below ~8 the
        variance estimate is itself too noisy to rank compounds by.
    params
        Passed to LightGBM. The defaults are tuned for the small-n case
        Loopwise targets (a few hundred labelled compounds), where LightGBM's
        own defaults overfit badly: shallow trees, strong leaf minimums,
        subsampling on.
    quantiles
        Lower/median/upper quantiles for ``uncertainty="quantile"``.
    """

    def __init__(
        self,
        task: str = "classification",
        *,
        uncertainty: str = "bootstrap",
        n_members: int = 12,
        params: dict | None = None,
        quantiles: tuple[float, float, float] = (0.159, 0.5, 0.841),
        random_state: int | None = 0,
    ):
        if uncertainty not in ("bootstrap", "quantile"):
            raise ValueError(
                f"uncertainty must be 'bootstrap' or 'quantile', got {uncertainty!r}"
            )
        if uncertainty == "quantile" and task != "regression":
            raise ValueError(
                "uncertainty='quantile' is regression-only: a quantile of a "
                "binary label is not a probability. Use 'bootstrap' for "
                "classification."
            )
        self.task = task
        self.uncertainty = uncertainty
        self.n_members = int(n_members)
        self.quantiles = quantiles
        self.random_state = random_state
        self.name = f"LightGBM[{uncertainty}]"
        # Small-n defaults: 31 leaves on 200 compounds memorizes the training set.
        self.params = {
            "verbosity": -1,
            "num_leaves": 15,
            "min_child_samples": 5,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "n_jobs": 1,
            **(params or {}),
        }
        self.members_: list[Any] = []

    # ------------------------------------------------------------------ fit

    def fit(self, X: np.ndarray, y: np.ndarray) -> LightGBMAdapter:
        lgb = _require_lightgbm()
        rng = make_rng(self.random_state)
        X = np.asarray(X, dtype=DEFAULT_DTYPE)
        y = np.asarray(y)
        self.members_ = []

        if self.uncertainty == "quantile":
            for q in self.quantiles:
                model = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.params)
                model.fit(X, y)
                self.members_.append(model)
            return self

        classes = np.unique(y)
        size = X.shape[0]
        for _ in range(self.n_members):
            idx = rng.integers(0, size, size=size)
            if self.task == "classification" and classes.size > 1:
                # Guarantee both classes: a resample of a 3%-hit-rate screen
                # often contains no actives, and a booster trained on one class
                # predicts a constant, inflating apparent disagreement.
                for cls in classes:
                    if cls not in np.unique(y[idx]):
                        pool = np.flatnonzero(y == cls)
                        idx[int(rng.integers(idx.size))] = int(rng.choice(pool))
            maker = (
                lgb.LGBMClassifier if self.task == "classification" else lgb.LGBMRegressor
            )
            model = maker(random_state=int(rng.integers(0, 2**31 - 1)), **self.params)
            model.fit(X[idx], y[idx])
            self.members_.append(model)
        return self

    # ------------------------------------------------------------ posterior

    def posterior(
        self, X: np.ndarray, n_samples: int = 20, rng: np.random.Generator | None = None
    ) -> Posterior:
        if not self.members_:
            raise RuntimeError("Call fit() before posterior().")
        X = np.asarray(X, dtype=DEFAULT_DTYPE)
        check_resources(
            (max(len(self.members_), n_samples), X.shape[0]),
            task="LightGBM posterior samples",
            copies=2,
        ).raise_if_refused()

        if self.uncertainty == "quantile":
            lower, median, upper = (
                np.asarray(m.predict(X), dtype=np.float64) for m in self.members_
            )
            # The quantiles were chosen one SD out, so (upper - lower) / 2 is
            # a direct SD estimate; draw Gaussian samples from it.
            sigma = np.maximum((upper - lower) / 2.0, 1e-9)
            gen = make_rng(rng if rng is not None else self.random_state)
            draws = median[None, :] + sigma[None, :] * gen.normal(
                size=(n_samples, X.shape[0])
            )
            return Posterior(draws.astype(DEFAULT_DTYPE), task="regression")

        rows = []
        for model in self.members_:
            if self.task == "classification":
                rows.append(_proba_positive(model, X))
            else:
                rows.append(np.asarray(model.predict(X), dtype=DEFAULT_DTYPE))
        return Posterior(np.vstack(rows), task=self.task)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        return self.posterior(X, n_samples=max(len(self.members_), 2)).mean
