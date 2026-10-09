"""PyTorch adapter: deep ensembles and MC dropout.

Two mechanisms, both standard, and they measure different things:

**MC dropout** (Gal & Ghahramani, 2016). Dropout randomly zeroes a fraction of
hidden units during training; it is normally switched off at prediction time.
Leave it *on* and predict K times, and each prediction comes from a slightly
different sub-network. Gal and Ghahramani showed that this is approximate
Bayesian inference: the dropout masks are samples from an approximate
posterior over the weights. The appeal is that it is nearly free -- one trained
network, K forward passes.

**Deep ensembles** (Lakshminarayanan et al., 2017). Train K networks from
different random initializations. More expensive by a factor of K, and
reliably better at estimating uncertainty: independently initialized networks
find genuinely different solutions, while dropout masks explore one solution's
neighbourhood. If you can afford it, use it.

Both at once (``n_models>1, dropout>0``) gives K x S samples, which is the
best estimate here and the default for ``n_models=3``.

The network itself is deliberately small -- two hidden layers, default width
64. With 200 labelled compounds, a bigger network memorizes and reports
serene confidence everywhere, which is the worst possible failure for active
learning: the policy then has no signal to explore toward.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..models import ModelAdapter, Posterior
from ..utils import DEFAULT_DTYPE, check_resources, make_rng

__all__ = ["MLP", "TorchEnsembleAdapter"]


def _require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TorchEnsembleAdapter needs PyTorch: pip install 'loopwise[torch]' "
            "(or see pytorch.org for a CUDA build)."
        ) from exc


def MLP(n_features: int, hidden: Sequence[int] = (64, 64), dropout: float = 0.2):
    """A small dropout MLP. Returns an ``nn.Sequential``.

    Dropout sits after every hidden activation, which is what makes MC dropout
    available at prediction time.
    """
    _require_torch()  # raise a helpful ImportError before touching torch.nn
    from torch import nn

    layers: list = []
    in_dim = int(n_features)
    for width in hidden:
        layers += [nn.Linear(in_dim, int(width)), nn.ReLU(), nn.Dropout(float(dropout))]
        in_dim = int(width)
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


class TorchEnsembleAdapter(ModelAdapter):
    """Deep ensemble + MC-dropout adapter.

    Parameters
    ----------
    n_models
        Networks in the ensemble. 1 means MC dropout only.
    hidden, dropout
        Architecture of the default MLP. Ignored if ``model_factory`` is given.
    model_factory
        ``callable(n_features) -> nn.Module`` for a custom architecture. Must
        output a single logit (classification) or value (regression) per row.
    mc_dropout_samples
        Forward passes per network with dropout active. 1 disables MC dropout.
    epochs, lr, batch_size, weight_decay
        Training loop. ``weight_decay`` defaults to 1e-4 because small-n
        training without regularization is the fastest route to an
        over-confident model.
    device
        ``"cpu"``, ``"cuda"``, or None to pick automatically.
    """

    def __init__(
        self,
        task: str = "classification",
        *,
        n_models: int = 3,
        hidden: Sequence[int] = (64, 64),
        dropout: float = 0.2,
        model_factory=None,
        mc_dropout_samples: int = 10,
        epochs: int = 150,
        lr: float = 1e-3,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
        device: str | None = None,
        random_state: int | None = 0,
        verbose: bool = False,
    ):
        self.task = task
        self.n_models = max(1, int(n_models))
        self.hidden = tuple(hidden)
        self.dropout = float(dropout)
        self.model_factory = model_factory
        self.mc_dropout_samples = max(1, int(mc_dropout_samples))
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.device = device
        self.random_state = random_state
        self.verbose = bool(verbose)
        self.name = f"TorchEnsemble[n={self.n_models},mc={self.mc_dropout_samples}]"
        self.models_: list = []
        self._n_features: int | None = None

        if self.n_models == 1 and self.mc_dropout_samples == 1:
            raise ValueError(
                "n_models=1 with mc_dropout_samples=1 gives a single "
                "deterministic prediction, so every information-seeking policy "
                "would score zero. Raise one of them."
            )
        if self.mc_dropout_samples > 1 and self.dropout <= 0 and model_factory is None:
            raise ValueError(
                "mc_dropout_samples > 1 requires dropout > 0 -- without "
                "dropout, repeated forward passes are identical."
            )

    # ------------------------------------------------------------------ fit

    def fit(self, X: np.ndarray, y: np.ndarray) -> TorchEnsembleAdapter:
        torch = _require_torch()
        from torch import nn

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        self._n_features = X.shape[1]
        device = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        rng = make_rng(self.random_state)

        # Standardize inputs: an un-normalized fingerprint column scale makes
        # the first layer's gradients wildly uneven.
        self._mu = X.mean(axis=0, keepdims=True)
        self._sd = X.std(axis=0, keepdims=True)
        self._sd[self._sd < 1e-8] = 1.0
        Xs = (X - self._mu) / self._sd

        X_t = torch.from_numpy(Xs).to(device)
        y_t = torch.from_numpy(y).to(device).unsqueeze(1)
        loss_fn = nn.BCEWithLogitsLoss() if self.task == "classification" else nn.MSELoss()

        self.models_ = []
        for member in range(self.n_models):
            torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
            model = (
                self.model_factory(self._n_features)
                if self.model_factory is not None
                else MLP(self._n_features, self.hidden, self.dropout)
            ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            model.train()
            n = X_t.shape[0]
            for epoch in range(self.epochs):
                perm = torch.randperm(n, device=device)
                total = 0.0
                for start in range(0, n, self.batch_size):
                    idx = perm[start : start + self.batch_size]
                    optimizer.zero_grad()
                    loss = loss_fn(model(X_t[idx]), y_t[idx])
                    loss.backward()
                    optimizer.step()
                    total += float(loss.item())
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(
                        f"  member {member} epoch {epoch + 1}: "
                        f"loss {total / max(1, n // self.batch_size):.4f}"
                    )
            self.models_.append(model)
        self._device = device
        return self

    # ------------------------------------------------------------ posterior

    def posterior(
        self, X: np.ndarray, n_samples: int = 20, rng: np.random.Generator | None = None
    ) -> Posterior:
        torch = _require_torch()
        if not self.models_:
            raise RuntimeError("Call fit() before posterior().")
        X = np.asarray(X, dtype=np.float32)
        Xs = (X - self._mu) / self._sd
        total_samples = self.n_models * self.mc_dropout_samples
        check_resources(
            (total_samples, X.shape[0]),
            task="Torch ensemble posterior samples",
            copies=2,
        ).raise_if_refused()

        X_t = torch.from_numpy(Xs).to(self._device)
        rows = []
        with torch.no_grad():
            for model in self.models_:
                # Dropout stays ON: that is what makes these posterior samples
                # rather than K copies of one prediction.
                model.train(self.mc_dropout_samples > 1 and self.dropout > 0)
                for _ in range(self.mc_dropout_samples):
                    out = model(X_t).squeeze(-1)
                    if self.task == "classification":
                        out = torch.sigmoid(out)
                    rows.append(out.cpu().numpy().astype(DEFAULT_DTYPE))
                model.eval()
        return Posterior(np.vstack(rows), task=self.task)

    def sample(self, X: np.ndarray, n_samples: int = 20) -> np.ndarray:
        """``sample()`` so the model also works through ``auto_adapter``."""
        return self.posterior(X, n_samples=n_samples).samples
