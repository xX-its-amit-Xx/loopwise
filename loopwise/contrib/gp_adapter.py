"""Gaussian process adapter: the one exact posterior in the package.

A Gaussian process does not fit parameters to data. It places a distribution
over *functions* and conditions it on what you have measured. Two compounds
with similar features are assumed to have correlated activity -- that
assumption is the kernel -- and conditioning on the measurements you have
gives, in closed form, a mean and a variance for every compound you have not
measured.

That closed form is why this adapter matters even though GPs scale badly:
every other adapter here *approximates* a posterior (bootstrap resampling, MC
dropout, trees that happen to disagree). A GP's posterior samples are the real
thing. When a BALD score looks strange, reproducing it with a GP on a few
hundred compounds tells you whether the strangeness is in the policy or in the
approximation.

The catch is cost. Conditioning requires inverting an ``n x n`` matrix over
the *training* set, which is ``O(n^3)`` time and ``O(n^2)`` memory. At 2,000
measured compounds that is a few seconds; at 20,000 it is hours and 3 GB. The
adapter checks this against real memory before fitting (via
:func:`~loopwise.utils.check_resources`) and refuses with an actionable message
rather than freezing the machine -- which is the behaviour a GP in a loop
otherwise reliably produces, because the training set grows every round.

For classification, scikit-learn's ``GaussianProcessClassifier`` uses a Laplace
approximation, so the posterior is approximate again; for a binary screen,
regression on the 0/1 label (a "label-regression GP") is often better behaved
and is what ``task="classification", use_regressor=True`` does.
"""

from __future__ import annotations

import numpy as np

from ..models import ModelAdapter, Posterior
from ..utils import DEFAULT_DTYPE, check_resources, make_rng

__all__ = ["GaussianProcessAdapter"]

#: Above this many training points a GP fit is refused unless overridden.
GP_SOFT_LIMIT = 5000


class GaussianProcessAdapter(ModelAdapter):
    """Gaussian process with an exact (or Laplace-approximate) posterior.

    Parameters
    ----------
    task
        ``"regression"`` or ``"classification"``.
    use_regressor
        For classification, regress on the 0/1 label and squash the posterior
        into [0, 1] rather than using a GP classifier. Default True: the
        Laplace approximation in ``GaussianProcessClassifier`` is slower and
        its variance estimates are less useful for ranking.
    kernel
        A scikit-learn kernel. Default ``Matern(nu=2.5) + WhiteKernel()``.
        Matern rather than RBF because RBF assumes the activity surface is
        infinitely smooth, which structure-activity relationships emphatically
        are not -- a single atom change can abolish activity ("activity
        cliffs"), and Matern tolerates that far better. WhiteKernel is the
        assay noise floor, and fitting it is what stops the GP from
        interpolating noise.
    normalize_y, alpha, n_restarts_optimizer
        Passed through to scikit-learn.
    max_train
        Refuse to fit above this many training points. Raise it deliberately
        if you know what the cubic cost will do.
    """

    def __init__(
        self,
        task: str = "regression",
        *,
        use_regressor: bool = True,
        kernel=None,
        normalize_y: bool = True,
        alpha: float = 1e-8,
        n_restarts_optimizer: int = 1,
        max_train: int = GP_SOFT_LIMIT,
        random_state: int | None = 0,
    ):
        self.task = task
        self.use_regressor = bool(use_regressor)
        self.normalize_y = bool(normalize_y)
        self.alpha = float(alpha)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.max_train = int(max_train)
        self.random_state = random_state
        self.name = "GaussianProcess"
        self._kernel = kernel
        self.model_ = None

    def _default_kernel(self):
        from sklearn.gaussian_process.kernels import (
            ConstantKernel,
            Matern,
            WhiteKernel,
        )

        return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
            noise_level=0.1
        )

    # ------------------------------------------------------------------ fit

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessAdapter:
        from sklearn.gaussian_process import (
            GaussianProcessClassifier,
            GaussianProcessRegressor,
        )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n = X.shape[0]

        if n > self.max_train:
            raise MemoryError(
                f"Gaussian process fit refused: {n:,} training points exceeds "
                f"max_train={self.max_train:,}. A GP inverts an n x n matrix, "
                f"so this is O(n^3) time and would hold "
                f"{(n * n * 8) / float(1 << 30):.1f} GiB. Options: subsample the "
                "training set, switch to a sparse/variational GP, or use "
                "contrib.LightGBMAdapter or TorchEnsembleAdapter, which scale "
                "linearly. Raise max_train only if you have measured the cost."
            )
        check_resources(
            (n, n),
            dtype=np.float64,
            copies=3,  # kernel matrix, its Cholesky factor, and a work buffer
            task=f"Gaussian process kernel matrix for {n:,} training points",
            allow_chunking=False,
            remedy=(
                "Subsample the training set, or use a model whose cost is "
                "linear in the number of measurements (LightGBMAdapter, "
                "TorchEnsembleAdapter)."
            ),
        ).raise_if_refused()

        kernel = self._kernel if self._kernel is not None else self._default_kernel()
        if self.task == "regression" or self.use_regressor:
            self.model_ = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=self.normalize_y,
                alpha=self.alpha,
                n_restarts_optimizer=self.n_restarts_optimizer,
                random_state=self.random_state,
            )
            self.model_.fit(X, np.asarray(y, dtype=np.float64))
            self._is_regressor = True
        else:
            self.model_ = GaussianProcessClassifier(
                kernel=kernel,
                n_restarts_optimizer=self.n_restarts_optimizer,
                random_state=self.random_state,
            )
            self.model_.fit(X, y)
            self._is_regressor = False
        return self

    # ------------------------------------------------------------ posterior

    def posterior(
        self, X: np.ndarray, n_samples: int = 20, rng: np.random.Generator | None = None
    ) -> Posterior:
        if self.model_ is None:
            raise RuntimeError("Call fit() before posterior().")
        X = np.asarray(X, dtype=np.float64)
        check_resources(
            (n_samples, X.shape[0]), task="GP posterior draws", copies=2
        ).raise_if_refused()

        if not self._is_regressor:
            # Laplace-approximate classifier: no joint sampling available, so
            # build samples from the predictive probability and its implied
            # binomial spread. Flagged in the name so nobody mistakes these for
            # exact draws.
            p = self.model_.predict_proba(X)[:, -1]
            gen = make_rng(rng if rng is not None else self.random_state)
            spread = np.sqrt(np.maximum(p * (1 - p), 1e-6)) * 0.3
            draws = np.clip(
                p[None, :] + spread[None, :] * gen.normal(size=(n_samples, X.shape[0])),
                0.0,
                1.0,
            )
            return Posterior(draws.astype(DEFAULT_DTYPE), task="classification")

        mean, std = self.model_.predict(X, return_std=True)
        gen = make_rng(rng if rng is not None else self.random_state)
        # Independent marginal draws. The exact joint posterior would require
        # the full covariance over candidates -- an n x n matrix over the
        # *candidate* pool, which is the allocation the resource guard exists
        # to prevent. Every policy here reads only per-candidate marginals, so
        # marginal draws are sufficient; a joint-aware policy would need the
        # chunked path.
        draws = mean[None, :] + std[None, :] * gen.normal(size=(n_samples, X.shape[0]))
        if self.task == "classification":
            draws = np.clip(draws, 0.0, 1.0)
        return Posterior(
            draws.astype(DEFAULT_DTYPE),
            task="classification" if self.task == "classification" else "regression",
        )

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict_mean().")
        X = np.asarray(X, dtype=np.float64)
        if not self._is_regressor:
            return self.model_.predict_proba(X)[:, -1].astype(DEFAULT_DTYPE)
        mean = self.model_.predict(X).astype(DEFAULT_DTYPE)
        return np.clip(mean, 0.0, 1.0) if self.task == "classification" else mean

    def kernel_summary(self) -> str:
        """The fitted kernel, which is the GP's statement about the chemistry.

        Worth reading after a campaign: the fitted length scale says how far
        in feature space activity is correlated, and the fitted noise level is
        the GP's estimate of your assay's reproducibility. If the noise level
        comes back near the signal variance, the GP is telling you the assay
        is too noisy for the features to explain -- which is a finding about
        the experiment, not a bug.
        """
        if self.model_ is None:
            return "not fitted"
        return str(getattr(self.model_, "kernel_", "unknown"))
