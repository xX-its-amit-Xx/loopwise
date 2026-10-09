"""The posterior abstraction: one interface every policy reads from.

The problem
-----------
Eight policies, four kinds of model. Greedy wants a point prediction. BALD
wants a *distribution over* predictions. Thompson wants a single draw from
that distribution. Written naively that is eight policies times four model
types = thirty-two code paths.

Loopwise collapses this to one. Every model is wrapped in an adapter that can
produce **posterior samples**: an array of shape ``(n_samples, n_items)`` where
each row is one plausible version of the model's predictions. From that single
object every policy's quantity falls out:

    mean over rows          -> the point prediction        (greedy)
    variance over rows      -> epistemic uncertainty       (uncertainty sampling)
    entropy of mean minus
      mean of entropies     -> mutual information          (BALD)
    one row, picked at random -> a posterior draw           (Thompson)

The deep point: **disagreement between samples is what "the model does not
know" means.** If every sample says 0.5, the model is confident the answer is
a coin flip -- that is *aleatoric* uncertainty, irreducible noise in the assay,
and testing that compound teaches you nothing. If half the samples say 0.1 and
half say 0.9, the model genuinely does not know -- *epistemic* uncertainty --
and that compound is informative. The mean prediction is 0.5 in both cases.
This distinction, due in this form to Houlsby et al. (2011) for BALD, is the
whole reason uncertainty sampling and BALD disagree, and why BALD is usually
the better explorer: plain uncertainty sampling happily spends your wells on
compounds that are simply noisy.

Where samples come from
-----------------------
=========================  ======================================================
Model                      Sampling mechanism
=========================  ======================================================
Random forest / extra      One sample per tree. Trees differ because each saw a
trees                      different bootstrap of the data, so their spread is a
                           genuine (if crude) posterior.
Gradient boosting, SVM,    Bootstrap ensemble: refit the estimator on K bootstrap
logistic regression        resamples. Expensive but model-agnostic and honest.
PyTorch network            MC dropout (Gal & Ghahramani, 2016): keep dropout on
                           at prediction time, predict K times. Equivalent to
                           sampling from an approximate posterior over weights.
Gaussian process           Draw from the exact analytic posterior. The only case
                           where "posterior sample" is not an approximation.
=========================  ======================================================

A bare classifier with ``predict_proba`` and no sampling mechanism gives a
*degenerate* posterior: all samples identical, so variance and mutual
information are exactly zero. Loopwise refuses to run an information-seeking
policy on such a model instead of silently returning an arbitrary batch --
that silent failure is the single most common way an active-learning pipeline
quietly becomes random sampling.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .utils import DEFAULT_DTYPE, check_resources, make_rng

__all__ = [
    "BootstrapEnsembleAdapter",
    "DegeneratePosteriorError",
    "ModelAdapter",
    "Posterior",
    "SamplingModelAdapter",
    "SklearnAdapter",
    "TreeEnsembleAdapter",
    "auto_adapter",
]

# Entropy guards are evaluated in float64 with a float32-safe epsilon: in
# float32, 1 - 1e-9 rounds to exactly 1.0, which would make log(1 - p) = -inf
# and turn every BALD score into NaN. This exact bug silently reduces BALD to
# random tie-breaking, so the epsilon is deliberately coarse and the
# computations below promote to float64 first.
_EPS = 1e-7


class DegeneratePosteriorError(ValueError):
    """Raised when an information-seeking policy is given a model that cannot
    express uncertainty, so every "information gain" would be exactly zero."""


@dataclass
class Posterior:
    """Posterior predictive samples over a candidate pool.

    Attributes
    ----------
    samples
        ``(n_samples, n_items)`` float32. For classification these are
        probabilities of the positive class; for regression, predicted values.
    task
        ``"classification"`` or ``"regression"``.
    """

    samples: np.ndarray
    task: str = "classification"

    def __post_init__(self) -> None:
        self.samples = np.atleast_2d(np.asarray(self.samples, dtype=DEFAULT_DTYPE))
        if self.task not in ("classification", "regression"):
            raise ValueError(
                f"task must be 'classification' or 'regression', got {self.task!r}"
            )
        if self.task == "classification":
            lo, hi = (
                float(self.samples.min(initial=0.0)),
                float(self.samples.max(initial=0.0)),
            )
            if lo < -1e-4 or hi > 1 + 1e-4:
                raise ValueError(
                    f"Classification posterior samples must be probabilities in "
                    f"[0, 1]; got range [{lo:.3f}, {hi:.3f}]. If this is a "
                    "regression target, pass task='regression'."
                )
            np.clip(self.samples, 0.0, 1.0, out=self.samples)

    # ------------------------------------------------------------- properties

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.samples.shape[1])

    @property
    def mean(self) -> np.ndarray:
        """Point prediction: the posterior mean. What greedy maximizes."""
        return self.samples.mean(axis=0)

    @property
    def variance(self) -> np.ndarray:
        """Spread across samples: total epistemic uncertainty.

        What uncertainty sampling maximizes. Note this is variance *of the
        model's predictions*, not of the assay readout.
        """
        if self.n_samples < 2:
            return np.zeros(self.n_items, dtype=DEFAULT_DTYPE)
        return self.samples.var(axis=0, ddof=1).astype(DEFAULT_DTYPE)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance, dtype=DEFAULT_DTYPE)

    @property
    def is_degenerate(self) -> bool:
        """True when samples carry no disagreement, so info gain is all zero."""
        return self.n_samples < 2 or bool(np.all(self.variance <= 1e-12))

    # -------------------------------------------------------------- quantities

    def predictive_entropy(self) -> np.ndarray:
        """``H[E_theta p(y|x,theta)]`` -- entropy of the averaged prediction.

        Total uncertainty: aleatoric (assay noise) plus epistemic (ignorance).
        Measured in nats. Maximal (0.693) at p = 0.5.
        """
        if self.task == "regression":
            # Gaussian differential entropy, up to an additive constant.
            return 0.5 * np.log(self.variance.astype(np.float64) + _EPS).astype(
                DEFAULT_DTYPE
            )
        p = np.clip(self.mean.astype(np.float64), _EPS, 1.0 - _EPS)
        return (-(p * np.log(p) + (1 - p) * np.log(1 - p))).astype(DEFAULT_DTYPE)

    def expected_entropy(self) -> np.ndarray:
        """``E_theta H[p(y|x,theta)]`` -- mean entropy of individual samples.

        Aleatoric only: the noise that remains even for a model that knows
        the true parameters. Subtracting this is what makes BALD ignore
        compounds that are merely noisy.
        """
        if self.task == "regression":
            return np.zeros(self.n_items, dtype=DEFAULT_DTYPE)
        p = np.clip(self.samples.astype(np.float64), _EPS, 1.0 - _EPS)
        per_sample = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        return per_sample.mean(axis=0).astype(DEFAULT_DTYPE)

    def mutual_information(self) -> np.ndarray:
        """BALD score: ``H[E p] - E H[p]`` (Houlsby et al., 2011).

        The mutual information between the label of this compound and the
        model's parameters. Reads as: *how much would measuring this compound
        reduce my uncertainty about the model itself?* Zero for a compound the
        model is sure about, and -- crucially -- also zero for one it is sure
        is unpredictable.
        """
        if self.task == "regression":
            return 0.5 * np.log1p(self.variance.astype(np.float64) / _EPS).astype(
                DEFAULT_DTYPE
            )
        return np.maximum(self.predictive_entropy() - self.expected_entropy(), 0.0)

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        """One posterior sample, for Thompson sampling."""
        return self.samples[int(rng.integers(self.n_samples))]

    def require_informative(self, policy_name: str) -> None:
        """Raise if this posterior cannot support an information-based policy."""
        if self.is_degenerate:
            raise DegeneratePosteriorError(
                f"Policy {policy_name!r} needs a model that can express "
                f"uncertainty, but the posterior has {self.n_samples} "
                "sample(s) with no disagreement, so every information score "
                "would be exactly 0 and the batch would be arbitrary. Wrap "
                "the model so it can be sampled: pass "
                "n_bootstrap=20 to get a bootstrap ensemble, use a "
                "RandomForest/ExtraTrees (one sample per tree), an MC-dropout "
                "network, or a Gaussian process. Alternatively use "
                "policy='greedy' or 'random', which need no uncertainty."
            )


# --------------------------------------------------------------------- adapters


@runtime_checkable
class SupportsFitPredictProba(Protocol):  # pragma: no cover - typing only
    def fit(self, X: Any, y: Any) -> Any: ...
    def predict_proba(self, X: Any) -> Any: ...


class ModelAdapter:
    """Base class: wraps a model so Loopwise can fit it and sample from it.

    Subclasses implement :meth:`fit` and :meth:`posterior`. Everything else in
    Loopwise talks only to this interface, which is why adding a new model
    family means writing one adapter and touching no policy.
    """

    task: str = "classification"
    name: str = "adapter"

    def fit(self, X: np.ndarray, y: np.ndarray) -> ModelAdapter:
        raise NotImplementedError

    def posterior(
        self, X: np.ndarray, n_samples: int = 20, rng: np.random.Generator | None = None
    ) -> Posterior:
        raise NotImplementedError

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Point prediction. Default: the mean of a small posterior."""
        return self.posterior(X, n_samples=8).mean

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.name})"


def _proba_positive(model: Any, X: np.ndarray) -> np.ndarray:
    """Positive-class probability from a sklearn-style classifier.

    Handles the degenerate-but-real case of a fold in which only one class was
    observed, where ``predict_proba`` returns a single column.
    """
    proba = np.asarray(model.predict_proba(X), dtype=DEFAULT_DTYPE)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 1:
        only = getattr(model, "classes_", np.array([0]))[0]
        return np.full(X.shape[0], 1.0 if only == 1 else 0.0, dtype=DEFAULT_DTYPE)
    classes = list(getattr(model, "classes_", [0, 1]))
    pos_index = classes.index(1) if 1 in classes else proba.shape[1] - 1
    return proba[:, pos_index]


class SklearnAdapter(ModelAdapter):
    """A single scikit-learn-compatible estimator, no sampling mechanism.

    Supports greedy and random selection. Information-seeking policies raise
    :class:`DegeneratePosteriorError` rather than silently degrading, because a
    single fitted model has no way to disagree with itself.
    """

    def __init__(self, estimator: Any, task: str = "classification"):
        self.estimator = estimator
        self.task = task
        self.name = type(estimator).__name__

    def fit(self, X: np.ndarray, y: np.ndarray) -> SklearnAdapter:
        self.estimator.fit(X, y)
        return self

    def posterior(
        self, X: np.ndarray, n_samples: int = 20, rng: np.random.Generator | None = None
    ) -> Posterior:
        mean = self._point(X)
        return Posterior(mean[None, :], task=self.task)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        return self._point(X)

    def _point(self, X: np.ndarray) -> np.ndarray:
        if self.task == "classification" and hasattr(self.estimator, "predict_proba"):
            return _proba_positive(self.estimator, X)
        if hasattr(self.estimator, "predict"):
            return np.asarray(self.estimator.predict(X), dtype=DEFAULT_DTYPE)
        raise TypeError(
            f"{self.name} exposes neither predict_proba nor predict; Loopwise "
            "needs one of them."
        )


class TreeEnsembleAdapter(ModelAdapter):
    """Forest-style estimator: one posterior sample per tree.

    Each tree in a random forest was fitted on a different bootstrap resample
    and a random subset of features, so the trees' predictions on the same
    compound form an empirical distribution. It is not a Bayesian posterior,
    but it captures the thing that matters -- disagreement where the data is
    thin -- and it is free, since the trees are already fitted.
    """

    def __init__(self, estimator: Any, task: str = "classification"):
        self.estimator = estimator
        self.task = task
        self.name = type(estimator).__name__

    def fit(self, X: np.ndarray, y: np.ndarray) -> TreeEnsembleAdapter:
        self.estimator.fit(X, y)
        return self

    def posterior(
        self, X: np.ndarray, n_samples: int = 20, rng: np.random.Generator | None = None
    ) -> Posterior:
        trees = list(getattr(self.estimator, "estimators_", []))
        if not trees:
            raise RuntimeError(f"{self.name} has no estimators_; call fit() first.")
        if len(trees) > n_samples and rng is not None:
            idx = rng.choice(len(trees), size=n_samples, replace=False)
            trees = [trees[i] for i in idx]
        check_resources(
            (len(trees), X.shape[0]),
            task=f"{self.name} per-tree posterior samples",
            copies=2,
        ).raise_if_refused()
        rows = []
        for tree in trees:
            if self.task == "classification" and hasattr(tree, "predict_proba"):
                # Individual trees' classes_ can be a subset of the forest's.
                proba = np.asarray(tree.predict_proba(X), dtype=DEFAULT_DTYPE)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    forest_classes = list(getattr(self.estimator, "classes_", [0, 1]))
                    pos = forest_classes.index(1) if 1 in forest_classes else -1
                    rows.append(proba[:, pos])
                else:
                    rows.append(np.ravel(proba).astype(DEFAULT_DTYPE))
            else:
                rows.append(np.asarray(tree.predict(X), dtype=DEFAULT_DTYPE))
        return Posterior(np.vstack(rows), task=self.task)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            return _proba_positive(self.estimator, X)
        return np.asarray(self.estimator.predict(X), dtype=DEFAULT_DTYPE)


class BootstrapEnsembleAdapter(ModelAdapter):
    """Model-agnostic posterior by refitting on bootstrap resamples.

    The bootstrap (Efron, 1979) answers "how much would my model change if I
    had collected a slightly different dataset?" by sampling the data it *did*
    collect, with replacement, and refitting. Spread across the refits is an
    estimate of how much the data -- rather than the model class -- pins the
    answer down.

    This is the fallback that makes *any* estimator usable with BALD and
    Thompson sampling. The cost is ``n_members`` fits per round, which is why
    it is a fallback: with 20 members and a 30-second fit, each round spends
    ten minutes training. For small-n assays (the case Loopwise targets) fits
    are seconds and this is the right trade.
    """

    def __init__(
        self,
        estimator: Any,
        n_members: int = 20,
        task: str = "classification",
        *,
        sample_fraction: float = 1.0,
        random_state: int | None = None,
    ):
        from sklearn.base import clone  # local import: keeps import time down

        self._clone = clone
        self.estimator = estimator
        self.n_members = int(n_members)
        self.task = task
        self.sample_fraction = float(sample_fraction)
        self.random_state = random_state
        self.name = f"Bootstrap[{type(estimator).__name__}]x{n_members}"
        self.members_: list[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> BootstrapEnsembleAdapter:
        rng = make_rng(self.random_state)
        n = X.shape[0]
        size = max(2, int(round(self.sample_fraction * n)))
        classes = np.unique(y)
        self.members_ = []
        for member in range(self.n_members):
            idx = self._stratified_bootstrap(y, classes, size, rng)
            est = self._clone(self.estimator)
            if hasattr(est, "random_state"):
                try:
                    est.set_params(random_state=int(rng.integers(0, 2**31 - 1)))
                except (ValueError, TypeError):  # pragma: no cover
                    pass
            try:
                est.fit(X[idx], y[idx])
            except ValueError as exc:  # pragma: no cover - pathological resample
                warnings.warn(
                    f"Bootstrap member {member} failed to fit ({exc}); "
                    "falling back to the full sample for this member.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                est.fit(X, y)
            self.members_.append(est)
        return self

    @staticmethod
    def _stratified_bootstrap(
        y: np.ndarray, classes: np.ndarray, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Resample with replacement, keeping at least one of each class.

        An unstratified bootstrap of a 2%-hit-rate screen quite often draws
        zero positives, producing a member that predicts "nothing is active"
        and inflating apparent disagreement for the wrong reason.
        """
        if len(classes) < 2 or y.size == 0:
            return rng.integers(0, max(y.size, 1), size=size)
        idx = rng.integers(0, y.size, size=size)
        present = np.unique(y[idx])
        for cls in classes:
            if cls not in present:
                pool = np.flatnonzero(y == cls)
                if pool.size:
                    idx[int(rng.integers(idx.size))] = int(rng.choice(pool))
        return idx

    def posterior(
        self, X: np.ndarray, n_samples: int = 20, rng: np.random.Generator | None = None
    ) -> Posterior:
        if not self.members_:
            raise RuntimeError("Call fit() before posterior().")
        check_resources(
            (len(self.members_), X.shape[0]),
            task=f"{self.name} posterior samples",
            copies=2,
        ).raise_if_refused()
        rows = []
        for est in self.members_:
            if self.task == "classification" and hasattr(est, "predict_proba"):
                rows.append(_proba_positive(est, X))
            else:
                rows.append(np.asarray(est.predict(X), dtype=DEFAULT_DTYPE))
        return Posterior(np.vstack(rows), task=self.task)


class SamplingModelAdapter(ModelAdapter):
    """Wraps any object exposing a ``sample(X, n_samples)`` method.

    This is the escape hatch for models that already know how to express their
    own uncertainty -- a Gaussian process, a variational network, an MC-dropout
    ensemble, a Bayesian neural net, or a domain model such as FoldRescue. The
    contract is deliberately tiny: return an array broadcastable to
    ``(n_samples, n_items)``.
    """

    def __init__(
        self,
        model: Any,
        task: str = "classification",
        *,
        fit_fn: Callable[[Any, np.ndarray, np.ndarray], Any] | None = None,
        sample_attr: str = "sample",
        name: str | None = None,
    ):
        self.model = model
        self.task = task
        self._fit_fn = fit_fn
        self._sample_attr = sample_attr
        self.name = name or type(model).__name__
        if not hasattr(model, sample_attr):
            raise TypeError(
                f"{self.name} has no .{sample_attr}() method. Pass "
                f"sample_attr= the right name, or use auto_adapter()."
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> SamplingModelAdapter:
        if self._fit_fn is not None:
            self._fit_fn(self.model, X, y)
        elif hasattr(self.model, "fit"):
            self.model.fit(X, y)
        return self

    def posterior(
        self, X: np.ndarray, n_samples: int = 20, rng: np.random.Generator | None = None
    ) -> Posterior:
        sampler = getattr(self.model, self._sample_attr)
        try:
            raw = sampler(X, n_samples)
        except TypeError:
            raw = sampler(X, n_samples=n_samples)
        samples = np.atleast_2d(np.asarray(raw, dtype=DEFAULT_DTYPE))
        if samples.shape[0] != n_samples and samples.shape[-1] == n_samples:
            samples = samples.T  # tolerate (n_items, n_samples) orientation
        if samples.shape[-1] != X.shape[0]:
            raise ValueError(
                f"{self.name}.{self._sample_attr}() returned shape "
                f"{samples.shape}; expected ({n_samples}, {X.shape[0]})."
            )
        return Posterior(samples, task=self.task)


def auto_adapter(
    model: Any,
    *,
    task: str = "classification",
    n_bootstrap: int | None = None,
    random_state: int | None = None,
) -> ModelAdapter:
    """Pick the right adapter for ``model``, preferring free uncertainty.

    Resolution order, cheapest honest posterior first:

    1. Already a :class:`ModelAdapter` -> returned unchanged.
    2. Has ``sample()`` / ``sample_y()`` -> :class:`SamplingModelAdapter`
       (the model knows its own posterior; trust it).
    3. Has ``estimators_`` (forest) -> :class:`TreeEnsembleAdapter`
       (free: the trees are already fitted).
    4. ``n_bootstrap`` given -> :class:`BootstrapEnsembleAdapter`
       (works for anything, costs ``n_bootstrap`` fits per round).
    5. Otherwise -> :class:`SklearnAdapter`, which supports greedy/random and
       raises a clear error for information-seeking policies.

    Pass ``n_bootstrap`` explicitly to opt into the expensive-but-general
    option; Loopwise will not silently multiply your training time by 20.
    """
    if isinstance(model, ModelAdapter):
        return model
    for attr in ("sample", "sample_y", "posterior_samples"):
        if callable(getattr(model, attr, None)):
            return SamplingModelAdapter(model, task=task, sample_attr=attr)
    if n_bootstrap:
        return BootstrapEnsembleAdapter(
            model, n_members=int(n_bootstrap), task=task, random_state=random_state
        )
    if hasattr(model, "estimators_") or hasattr(model, "n_estimators"):
        if type(model).__name__ in {
            "RandomForestClassifier",
            "RandomForestRegressor",
            "ExtraTreesClassifier",
            "ExtraTreesRegressor",
            "BaggingClassifier",
            "BaggingRegressor",
        }:
            return TreeEnsembleAdapter(model, task=task)
    if not (hasattr(model, "predict_proba") or hasattr(model, "predict")):
        raise TypeError(
            f"{type(model).__name__} exposes none of predict_proba, predict, "
            "or sample; Loopwise cannot use it. See loopwise.contrib for "
            "adapter examples."
        )
    return SklearnAdapter(model, task=task)
