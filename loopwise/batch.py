"""Turning a policy's ranking into a batch you can actually run.

A policy returns an ordered list of candidates. Between that and a plate there
are three constraints the policy never saw:

1. **Money.** The well budget says 96; the currency budget may say 61, because
   half the picks are custom syntheses. The cost model owns the plate step
   function (see :mod:`loopwise.cost`), so the batch is trimmed on the real
   cost curve rather than on an average price.
2. **Replicates.** With ``n_replicates=3`` and a 96-well budget you are
   choosing 32 compounds, not 96 -- unless replicates are declared not to
   count against the budget.
3. **Plates.** The number of compounds that fit depends on how many wells the
   control layout leaves free, which is a property of the plate template.

:func:`resolve_batch_size` answers "how many compounds may I actually pick?"
*before* the policy runs, so the policy is never asked for a batch that will
then be truncated -- truncating after the fact silently breaks the batch-aware
policies, whose k-th pick is chosen in light of the other k-1.
:func:`finalize_batch` then prices the chosen batch and trims it if a
cost-aware policy over-shot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cost import CostBreakdown, CostModel
from .plate import PlateTemplate

__all__ = ["BatchPlan", "finalize_batch", "resolve_batch_size"]


@dataclass
class BatchPlan:
    """How many compounds to select, and why that number."""

    n_items: int
    requested_budget: int
    well_budget: int
    usable_wells_per_plate: int
    binding_constraint: str
    estimated_cost: CostBreakdown | None = None
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []

    def explain(self) -> str:
        lines = [
            f"Selecting {self.n_items} compound(s); binding constraint: "
            f"{self.binding_constraint}.",
            f"Requested budget {self.requested_budget}; well budget "
            f"{self.well_budget}; {self.usable_wells_per_plate} usable wells per plate.",
        ]
        if self.estimated_cost is not None:
            lines.append(f"Estimated cost: {self.estimated_cost}")
        lines.extend(self.notes)
        return "\n".join(lines)


def resolve_batch_size(
    *,
    budget: int,
    n_candidates: int,
    cost_model: CostModel,
    template: PlateTemplate,
    candidate_costs: np.ndarray | None = None,
    budget_is_wells: bool = True,
) -> BatchPlan:
    """Largest batch that satisfies every constraint at once.

    ``budget`` is read as a well count when ``budget_is_wells`` (the default,
    matching ``Campaign(budget=96)``), or as a compound count otherwise.

    When a currency budget is set, the per-compound costs are used in the order
    given -- which is the *pool* order, not the selection order, since the
    policy has not run yet. The result is therefore an upper bound based on the
    cheapest feasible prefix; :func:`finalize_batch` re-prices the real batch
    afterwards and trims if necessary. Two-stage, because selection and pricing
    are genuinely circular: you cannot price a batch you have not chosen, and
    you should not choose a batch you cannot price.
    """
    notes: list[str] = []
    usable = template.usable_wells
    reps = cost_model.replicates

    well_budget = int(budget)
    if budget_is_wells:
        n_items = well_budget // max(reps.budget_cost_per_item(), 1)
        if reps.n_replicates > 1 and reps.count_against_budget:
            notes.append(
                f"{reps.n_replicates} replicates per compound count against the "
                f"budget, so a {well_budget}-well budget buys {n_items} compounds "
                f"({n_items * reps.n_replicates} wells)."
            )
    else:
        n_items = well_budget
        well_budget = n_items * reps.wells_per_item

    binding = "well budget"

    if n_items > n_candidates:
        n_items = n_candidates
        binding = "candidate pool exhausted"
        notes.append(
            f"Only {n_candidates} unmeasured candidates remain, fewer than the "
            "budget allows."
        )

    if cost_model.budget_currency is not None:
        costs = (
            np.asarray(candidate_costs, dtype=np.float64)
            if candidate_costs is not None
            else cost_model.item_costs(n_items=n_candidates)
        )
        # Cheapest-first prefix: the loosest honest upper bound on batch size.
        affordable = cost_model.max_items_affordable(
            np.sort(costs)[:n_items],
            well_budget=None,
            usable_wells_per_plate=usable,
        )
        if affordable < n_items:
            notes.append(
                f"Currency budget "
                f"{cost_model.budget_currency:,.2f} {cost_model.currency} caps the "
                f"batch at {affordable} compound(s), down from {n_items}; this "
                "assumes the cheapest candidates and is re-checked after selection."
            )
            n_items = affordable
            binding = "currency budget"
        if n_items == 0:
            notes.append(
                "WARNING: the currency budget does not cover a single compound "
                f"plus one plate ({cost_model.plate_fixed_cost:,.2f} "
                f"{cost_model.currency} per plate). Nothing will be selected."
            )

    plan = BatchPlan(
        n_items=int(max(n_items, 0)),
        requested_budget=int(budget),
        well_budget=int(well_budget),
        usable_wells_per_plate=int(usable),
        binding_constraint=binding,
        notes=notes,
    )
    if plan.n_items and cost_model.budget_currency is not None:
        plan.estimated_cost = cost_model.cost_of(
            np.sort(np.asarray(candidate_costs, dtype=np.float64))[: plan.n_items]
            if candidate_costs is not None
            else cost_model.item_costs(n_items=plan.n_items),
            usable_wells_per_plate=usable,
        )
    return plan


def finalize_batch(
    frame: pd.DataFrame,
    *,
    cost_model: CostModel,
    template: PlateTemplate,
    cost_column: str = "_loopwise_item_cost",
) -> tuple[pd.DataFrame, CostBreakdown, list[str]]:
    """Price the selected batch and trim it if it breaks the currency budget.

    Trimming drops the *lowest-ranked* picks, because a policy's ranking is
    its own statement about which picks it would give up first.
    """
    notes: list[str] = []
    costs = (
        frame[cost_column].to_numpy(dtype=np.float64)
        if cost_column in frame.columns
        else cost_model.item_costs(n_items=len(frame)).astype(np.float64)
    )
    breakdown = cost_model.cost_of(costs, usable_wells_per_plate=template.usable_wells)

    if (
        cost_model.budget_currency is not None
        and breakdown.total > cost_model.budget_currency + 1e-9
    ):
        keep = cost_model.max_items_affordable(
            costs, usable_wells_per_plate=template.usable_wells
        )
        notes.append(
            f"Selected batch costs {breakdown.total:,.2f} {cost_model.currency}, "
            f"over the {cost_model.budget_currency:,.2f} budget; dropping the "
            f"{len(frame) - keep} lowest-ranked pick(s)."
        )
        frame = frame.iloc[:keep].copy()
        costs = costs[:keep]
        breakdown = cost_model.cost_of(costs, usable_wells_per_plate=template.usable_wells)
    return frame, breakdown, notes
