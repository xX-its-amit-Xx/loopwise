"""Driving a FoldRescue model with Loopwise.

What FoldRescue is for
----------------------
Some genetic diseases are not caused by a missing protein but by a *misfolded*
one. The gene is transcribed, the protein is made, and then the cell's quality
control sees a shape it does not recognize and destroys it. The enzyme never
reaches the compartment where it does its job. GM1 gangliosidosis works this
way: mutations in *GLB1* destabilize beta-galactosidase, the enzyme that
breaks down GM1 ganglioside in the lysosome, and the substrate accumulates
instead.

A **pharmacological chaperone** is a small molecule that binds the unstable
protein and holds it in the right shape long enough to get past quality
control and reach the lysosome. Enough enzyme arrives to clear the substrate.
The appeal is that it rescues the patient's *own* enzyme -- no gene therapy,
no enzyme infusion, and some chaperones cross the blood-brain barrier, which
matters because GM1 is largely a neurological disease.

The screening problem is brutal. The chemical space of candidate chaperones is
large, the readout is a cell-based enzyme-activity assay that takes days, and
the hit rate for genuine rescue is low -- below 1% is normal. You can run
perhaps a few hundred compounds a quarter. That is precisely the regime where
*which* compounds you run dominates how fast you learn: a few thousand
candidates, a budget of 96, and a model that is wrong in ways you will only
discover by measuring the right things.

How the adapter works
---------------------
FoldRescue supplies the **model**; Loopwise supplies the **selection**. The
adapter translates between them, and is deliberately written against a small
duck-typed protocol rather than a hard import, so it keeps working as
FoldRescue's internals change:

A FoldRescue-like model must expose **one** of

``sample(X, n_samples) -> (n_samples, n_items)``
    Best case: the model expresses its own uncertainty (an ensemble, an
    MC-dropout network, a posterior over rescue probability). Used directly.

``predict_proba(X) -> (n_items, 2)`` or ``predict_rescue(X) -> (n_items,)``
    A point prediction of rescue probability. Uncertainty then has to come
    from somewhere else: the adapter wraps it in a bootstrap ensemble if it is
    re-fittable, and otherwise refuses the information-seeking policies rather
    than pretending to a confidence it does not have.

and may expose ``fit(X, y)`` for retraining between rounds. A frozen,
non-refittable model is supported (``refit=False``) -- then Loopwise is doing
pure batch design under a fixed model, which is still worth doing but cannot
improve across rounds, and the adapter says so.

.. warning::

   The ``SyntheticFoldRescueModel`` in this module is a **stand-in**, not
   FoldRescue. It is a seeded simulator with a plausible structure-activity
   relationship, included so the cookbook notebook and the tests run
   end-to-end without the upstream package. Its numbers are not predictions
   about GLB1 and must not be reported as such. Point
   :func:`foldrescue_campaign` at the real model for real work.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..campaign import Campaign
from ..cost import CostModel, ReplicatePolicy
from ..models import (
    BootstrapEnsembleAdapter,
    ModelAdapter,
    Posterior,
    SamplingModelAdapter,
)
from ..plate import PLATE_96, PlateTemplate
from ..utils import DEFAULT_DTYPE, make_rng

__all__ = [
    "FOLDRESCUE_PROTOCOL",
    "FoldRescueAdapter",
    "SyntheticFoldRescueModel",
    "foldrescue_campaign",
]

#: The duck-typed contract this adapter expects. Documented as data so a
#: FoldRescue maintainer can check compliance programmatically.
FOLDRESCUE_PROTOCOL = {
    "one_of": ("sample", "predict_proba", "predict_rescue", "predict"),
    "optional": ("fit", "feature_names", "n_bootstrap_members"),
    "sample_signature": "sample(X: array (n, d), n_samples: int) -> array (n_samples, n)",
    "semantics": "values are rescue probabilities in [0, 1], higher = more likely to rescue folding",
}


class FoldRescueAdapter(ModelAdapter):
    """Adapt a FoldRescue-like model to the Loopwise posterior interface.

    Parameters
    ----------
    model
        The FoldRescue model object (see :data:`FOLDRESCUE_PROTOCOL`).
    refit
        Retrain the model on accumulated measurements each round. Requires
        ``model.fit``. False keeps the model frozen, which makes Loopwise a
        pure batch-design layer.
    n_bootstrap
        If the model has no ``sample()`` but is re-fittable, wrap it in a
        bootstrap ensemble of this size to obtain uncertainty.
    predict_attr
        Override the prediction method name.
    """

    def __init__(
        self,
        model: Any,
        *,
        refit: bool = True,
        n_bootstrap: int | None = 12,
        predict_attr: str | None = None,
        task: str = "classification",
    ):
        self.model = model
        self.refit = bool(refit)
        self.task = task
        self.name = f"FoldRescue[{type(model).__name__}]"
        self._inner: ModelAdapter | None = None
        self._mode = "unknown"

        sampler = predict_attr or next(
            (
                a
                for a in ("sample", "posterior_samples")
                if callable(getattr(model, a, None))
            ),
            None,
        )
        if sampler is not None:
            self._inner = SamplingModelAdapter(
                model, task=task, sample_attr=sampler, name=self.name
            )
            self._mode = f"native-sampling({sampler})"
        elif callable(getattr(model, "fit", None)) and n_bootstrap:
            if not hasattr(model, "predict_proba") and not hasattr(model, "predict"):
                raise TypeError(
                    f"{type(model).__name__} has fit() but no predict_proba(), "
                    "predict() or sample(); Loopwise cannot read predictions "
                    f"from it. Expected protocol: {FOLDRESCUE_PROTOCOL}"
                )
            self._inner = BootstrapEnsembleAdapter(
                model, n_members=int(n_bootstrap), task=task
            )
            self._mode = f"bootstrap(n={n_bootstrap})"
        else:
            self._mode = "point-estimate-only"
            if not any(
                callable(getattr(model, a, None))
                for a in ("predict_proba", "predict_rescue", "predict")
            ):
                raise TypeError(
                    f"{type(model).__name__} satisfies none of the FoldRescue "
                    f"protocol. Expected: {FOLDRESCUE_PROTOCOL}"
                )
            warnings.warn(
                f"{self.name} exposes only a point prediction and is not "
                "re-fittable, so it cannot express uncertainty. Policies "
                "'greedy', 'random' and the diversity policies will work; "
                "'bald', 'uncertainty', 'thompson' and 'hybrid' will raise. "
                "Either give the model a sample() method or pass "
                "n_bootstrap=... with a fit()-able model.",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def uncertainty_mode(self) -> str:
        """How uncertainty is obtained -- worth printing in a notebook."""
        return self._mode

    def fit(self, X: np.ndarray, y: np.ndarray) -> FoldRescueAdapter:
        if not self.refit:
            return self
        if self._inner is not None:
            self._inner.fit(X, y)
        elif callable(getattr(self.model, "fit", None)):
            self.model.fit(X, y)
        return self

    def posterior(
        self, X: np.ndarray, n_samples: int = 20, rng: np.random.Generator | None = None
    ) -> Posterior:
        if self._inner is not None:
            return self._inner.posterior(X, n_samples=n_samples, rng=rng)
        return Posterior(self._point(X)[None, :], task=self.task)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        if self._inner is not None:
            return self._inner.predict_mean(X)
        return self._point(X)

    def _point(self, X: np.ndarray) -> np.ndarray:
        for attr in ("predict_rescue", "predict_proba", "predict"):
            fn = getattr(self.model, attr, None)
            if callable(fn):
                out = np.asarray(fn(X), dtype=DEFAULT_DTYPE)
                if out.ndim == 2 and out.shape[1] == 2:
                    return out[:, 1]
                return out.ravel()
        raise TypeError(f"{self.name} exposes no usable prediction method.")


class SyntheticFoldRescueModel:
    """A seeded stand-in for FoldRescue, for notebooks and tests only.

    Simulates the kind of structure-activity relationship a chaperone screen
    shows: rescue probability depends on a few pharmacophore-like features
    with a sharp optimum (a chaperone that binds too weakly does nothing, one
    that binds too tightly inhibits the enzyme it was meant to rescue -- a real
    and well-known trade-off for iminosugar chaperones). Fits a small random
    forest internally so it is re-fittable and can be sampled.

    This is **not** FoldRescue and its outputs are not GLB1 predictions.
    """

    def __init__(self, n_estimators: int = 60, random_state: int = 0):
        from sklearn.ensemble import RandomForestClassifier

        self._forest = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            min_samples_leaf=2,
            n_jobs=1,
        )
        self.random_state = random_state
        self._fitted = False
        self.is_synthetic = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> SyntheticFoldRescueModel:
        self._forest.fit(np.asarray(X, dtype=DEFAULT_DTYPE), np.asarray(y))
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SyntheticFoldRescueModel.fit() has not been called.")
        return self._forest.predict_proba(np.asarray(X, dtype=DEFAULT_DTYPE))

    def predict_rescue(self, X: np.ndarray) -> np.ndarray:
        """Rescue probability per compound, the FoldRescue-style entry point."""
        proba = self.predict_proba(X)
        return proba[:, -1].astype(DEFAULT_DTYPE)

    def sample(self, X: np.ndarray, n_samples: int = 20) -> np.ndarray:
        """Per-tree predictions as posterior samples."""
        if not self._fitted:
            raise RuntimeError("SyntheticFoldRescueModel.fit() has not been called.")
        Xf = np.asarray(X, dtype=DEFAULT_DTYPE)
        trees = self._forest.estimators_
        rng = make_rng(self.random_state)
        if len(trees) > n_samples:
            pick = rng.choice(len(trees), size=n_samples, replace=False)
            trees = [trees[i] for i in pick]
        rows = []
        for tree in trees:
            proba = tree.predict_proba(Xf)
            rows.append(
                proba[:, -1].astype(DEFAULT_DTYPE)
                if proba.ndim == 2 and proba.shape[1] > 1
                else np.ravel(proba).astype(DEFAULT_DTYPE)
            )
        return np.vstack(rows)

    @staticmethod
    def make_glb1_like_pool(
        n_items: int = 3000,
        n_features: int = 24,
        *,
        rescue_rate: float = 0.03,
        n_scaffolds: int = 15,
        seed: int = 0,
    ) -> tuple[pd.DataFrame, dict]:
        """A GLB1-chaperone-*shaped* candidate pool. Synthetic, not real data.

        Two properties deliberately mirror a chaperone screen:

        * **Scaffold clustering.** Compounds come from ``n_scaffolds``
          chemical series, so the top predictions cluster -- which is what
          makes diversity constraints earn their keep.
        * **A non-monotone optimum.** Rescue depends on binding affinity
          through an inverted-U: too weak does nothing, too strong inhibits the
          enzyme. A model that assumes "more binding is better" gets the top of
          the ranking exactly wrong, which is the realistic failure an active
          strategy has to recover from.
        """
        rng = np.random.default_rng(seed)
        centers = rng.normal(scale=2.0, size=(n_scaffolds, n_features))
        scaffold = rng.integers(0, n_scaffolds, size=n_items)
        X = centers[scaffold] + rng.normal(scale=0.9, size=(n_items, n_features))

        # Feature 0 stands in for binding affinity; the optimum is interior.
        affinity = X[:, 0]
        binding_term = -((affinity - 1.2) ** 2)
        # A few other features act like permeability / solubility requirements.
        aux = X[:, 1:4] @ np.array([0.6, -0.45, 0.35])
        latent = 1.4 * binding_term + aux + rng.normal(scale=0.7, size=n_items)
        threshold = float(np.quantile(latent, 1.0 - rescue_rate))

        pool = pd.DataFrame(
            X.astype(DEFAULT_DTYPE),
            columns=[f"pharmacophore_{i}" for i in range(n_features)],
        )
        pool.insert(0, "compound_id", [f"GLB1-{i:05d}" for i in range(n_items)])
        pool["scaffold"] = scaffold
        pool["rescues_folding"] = (latent > threshold).astype(int)
        # Acquisition cost: library compounds cheap, bespoke synthesis dear.
        bespoke = rng.random(n_items) < 0.2
        pool["acquisition_cost_usd"] = np.where(
            bespoke, rng.uniform(300, 900, n_items), rng.uniform(15, 60, n_items)
        ).round(2)
        info = {
            "dataset": "synthetic GLB1-like chaperone pool (NOT real data)",
            "n_compounds": n_items,
            "n_rescuers": int(pool["rescues_folding"].sum()),
            "rescue_rate": round(float(pool["rescues_folding"].mean()), 4),
            "n_scaffolds": n_scaffolds,
            "feature_cols": [f"pharmacophore_{i}" for i in range(n_features)],
            "structure": (
                "rescue depends on an inverted-U in pharmacophore_0 (binding "
                "affinity: too weak does nothing, too strong inhibits) plus "
                "linear terms in pharmacophore_1..3"
            ),
        }
        return pool, info


def foldrescue_campaign(
    pool: pd.DataFrame,
    model: Any = None,
    *,
    budget: int = 96,
    policy: str = "hybrid",
    feature_cols: Sequence[str] | None = None,
    id_col: str = "compound_id",
    cost_col: str | None = "acquisition_cost_usd",
    plate: PlateTemplate | None = None,
    n_replicates: int = 2,
    budget_currency: float | None = None,
    plate_fixed_cost: float = 450.0,
    n_rounds_planned: int = 6,
    seed: int = 0,
    refit: bool = True,
    initial_labels: dict | None = None,
) -> Campaign:
    """Build a Campaign driven by a FoldRescue model.

    The defaults encode what a cell-based rescue assay actually looks like,
    rather than what is convenient:

    * **Duplicates** (``n_replicates=2``). A single well of a cell-based
      enzyme-activity assay is not a measurement you would act on.
    * **A real plate cost** (``plate_fixed_cost=450``). Substrate, cells,
      and a technician's day dominate the per-compound cost for a library
      compound, which is why the cost model's plate term is not optional.
    * **Per-compound costs** from ``acquisition_cost_usd``, because a bespoke
      synthesis at $600 and a library compound at $30 should not compete on
      equal terms for the same well.
    * **A 96-well plate with controls**: a rescue assay is meaningless without
      wild-type enzyme (positive) and untreated mutant (negative) on every
      plate.

    Parameters
    ----------
    model
        A FoldRescue-like model. If None, a :class:`SyntheticFoldRescueModel`
        is used and a warning is issued -- convenient for the cookbook, never
        correct for real work.
    """
    if model is None:
        warnings.warn(
            "No FoldRescue model given, so a SyntheticFoldRescueModel "
            "stand-in is being used. Its outputs are simulated and are not "
            "GLB1 predictions. Pass the real model for real work.",
            RuntimeWarning,
            stacklevel=2,
        )
        model = SyntheticFoldRescueModel(random_state=seed)

    adapter = FoldRescueAdapter(model, refit=refit, task="classification")
    template = plate if plate is not None else PLATE_96
    cost_model = CostModel(
        per_item_cost=cost_col if (cost_col and cost_col in pool.columns) else 0.0,
        plate_fixed_cost=plate_fixed_cost,
        usable_wells_per_plate=template.usable_wells,
        budget_currency=budget_currency,
        replicates=ReplicatePolicy(
            n_replicates=n_replicates,
            count_against_budget=True,
            charge_acquisition_per_replicate=False,
        ),
        currency="USD",
    )
    campaign = Campaign(
        pool=pool,
        model=adapter,
        budget=budget,
        policy=policy,
        feature_cols=feature_cols,
        id_col=id_col,
        cost_col=cost_col if (cost_col and cost_col in pool.columns) else None,
        cost_model=cost_model,
        plate=template,
        seed=seed,
        n_rounds_planned=n_rounds_planned,
        task="classification",
        initial_labels=initial_labels,
    )
    return campaign
