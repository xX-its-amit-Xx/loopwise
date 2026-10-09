"""Adapters for specific model families.

Everything here produces a :class:`~loopwise.models.ModelAdapter`, which is the
only interface the policies know about. Each adapter answers one question --
*where do this model family's posterior samples come from?* -- and nothing else.

The imports are lazy: ``loopwise.contrib`` is importable without LightGBM or
PyTorch installed, and asking for an adapter whose library is missing raises a
message naming the ``pip install`` that fixes it.

=================================  ========================================
Adapter                            Uncertainty mechanism
=================================  ========================================
:class:`LightGBMAdapter`           Bootstrap ensemble of boosters, or
                                   quantile regression for intervals
:class:`TorchEnsembleAdapter`      Deep ensemble and/or MC dropout
:class:`GaussianProcessAdapter`    Exact analytic posterior
:mod:`loopwise.contrib.foldrescue` Domain adapter: FoldRescue stability or
                                   rescue predictions driving selection
=================================  ========================================
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FoldRescueAdapter",
    "GaussianProcessAdapter",
    "LightGBMAdapter",
    "TorchEnsembleAdapter",
    "foldrescue_campaign",
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    if name == "LightGBMAdapter":
        from .lightgbm_adapter import LightGBMAdapter

        return LightGBMAdapter
    if name == "TorchEnsembleAdapter":
        from .torch_adapter import TorchEnsembleAdapter

        return TorchEnsembleAdapter
    if name == "GaussianProcessAdapter":
        from .gp_adapter import GaussianProcessAdapter

        return GaussianProcessAdapter
    if name in ("FoldRescueAdapter", "foldrescue_campaign"):
        from . import foldrescue

        return getattr(foldrescue, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
