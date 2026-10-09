"""Loopwise -- active-learning experiment selection for small-n assays.

A lab has a model, a candidate pool of thousands, and room for 96 wells. The
default move is greedy: test the 96 highest predictions. Greedy maximizes this
round's hit rate and minimizes what the model learns, so the campaign
plateaus. Loopwise implements and compares selection policies that trade
exploitation against information gain, and makes the trade-off explicit and
auditable.

    from loopwise import Campaign

    c = Campaign(pool=df, model=estimator, budget=96, policy="bald")
    batch = c.select()        # the chosen rows + why each was chosen
    c.observe(results_df)     # feed measurements back
    c.report()                # regret curve, calibration drift, diversity

See the README for the policy decision table and for the section on when
active learning does *not* help.
"""

from __future__ import annotations

__version__ = "0.1.0"
__license__ = "GPL-3.0-or-later"

from .campaign import Campaign, CampaignReport, SelectedBatch
from .cost import CostBreakdown, CostModel, ReplicatePolicy
from .models import (
    BootstrapEnsembleAdapter,
    DegeneratePosteriorError,
    ModelAdapter,
    Posterior,
    SamplingModelAdapter,
    SklearnAdapter,
    TreeEnsembleAdapter,
    auto_adapter,
)
from .plate import PlateLayout, PlateTemplate, WellAssignment
from .policies import (
    BALDPolicy,
    DiversityGreedyPolicy,
    ExpectedModelChangePolicy,
    GreedyPolicy,
    HybridPolicy,
    Policy,
    PolicyContext,
    RandomPolicy,
    SelectionResult,
    ThompsonPolicy,
    UncertaintyPolicy,
    available_policies,
    get_policy,
)
from .replay import ReplayResult, compare_policies, replay_campaign
from .utils import ResourceRefusal, ResourceReport, check_resources

__all__ = [
    "BALDPolicy",
    "BootstrapEnsembleAdapter",
    "Campaign",
    "CampaignReport",
    "CostBreakdown",
    "CostModel",
    "DegeneratePosteriorError",
    "DiversityGreedyPolicy",
    "ExpectedModelChangePolicy",
    "GreedyPolicy",
    "HybridPolicy",
    "ModelAdapter",
    "PlateLayout",
    "PlateTemplate",
    "Policy",
    "PolicyContext",
    "Posterior",
    "RandomPolicy",
    "ReplayResult",
    "ReplicatePolicy",
    "ResourceRefusal",
    "ResourceReport",
    "SamplingModelAdapter",
    "SelectedBatch",
    "SelectionResult",
    "SklearnAdapter",
    "ThompsonPolicy",
    "TreeEnsembleAdapter",
    "UncertaintyPolicy",
    "WellAssignment",
    "__version__",
    "auto_adapter",
    "available_policies",
    "check_resources",
    "compare_policies",
    "get_policy",
    "replay_campaign",
]
