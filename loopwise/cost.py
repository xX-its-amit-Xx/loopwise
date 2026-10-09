"""The cost model: budgets in currency, not just in wells.

Why this exists
---------------
Academic active learning assumes a budget of "k queries". A real screening
budget is money, and money does not map linearly onto wells:

* Each compound has an **acquisition cost** -- buying it from a vendor,
  synthesizing it, or thawing an aliquot. These differ by orders of magnitude
  between a commercial fragment and a bespoke synthesis.
* Each plate has a **fixed cost** -- reagents, the detection antibody, the
  plate reader's time, the technician's afternoon. This is often the dominant
  term, which is why running 97 compounds can cost nearly twice as much as
  running 96: you have opened a second plate.
* **Replicates** multiply the well count but not the acquisition cost: the
  same compound stock fills three wells. Whether replicates "count against
  the budget" depends on what is scarce -- wells or money -- so Loopwise makes
  it a flag rather than an assumption.

The step function at plate boundaries is the interesting part. A naive cost
model (cost = n * per_item) says the 97th compound is cheap. The real curve
says it triggers a whole extra plate. :meth:`CostModel.max_items_affordable`
walks the real curve, so selection stops at 96 rather than at 97.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import DEFAULT_DTYPE

__all__ = ["CostBreakdown", "CostModel", "ReplicatePolicy"]


@dataclass(frozen=True)
class ReplicatePolicy:
    """How many wells each selected compound occupies, and who pays.

    Parameters
    ----------
    n_replicates
        Wells per selected compound. 1 means singlicate -- common in a primary
        screen, where the plan is to confirm hits later. 3 is typical for a
        confirmation or dose-response run.
    count_against_budget
        If True (default), replicates consume the well budget: ``budget=96``
        with ``n_replicates=3`` selects 32 compounds. If False, the budget is
        read as "96 distinct compounds" and the plate count grows to hold
        their replicates -- appropriate when your constraint is compound
        availability or model learning rate rather than plate throughput.
    charge_acquisition_per_replicate
        If False (default), a compound is acquired once however many wells it
        fills -- the usual case, since one vial fills many wells. Set True
        when each replicate genuinely consumes a separate purchase, e.g. a
        single-use assay cartridge.
    """

    n_replicates: int = 1
    count_against_budget: bool = True
    charge_acquisition_per_replicate: bool = False

    def __post_init__(self) -> None:
        if self.n_replicates < 1:
            raise ValueError(f"n_replicates must be >= 1, got {self.n_replicates}")

    @property
    def wells_per_item(self) -> int:
        return int(self.n_replicates)

    def budget_cost_per_item(self) -> int:
        """Well-budget units consumed per distinct compound."""
        return int(self.n_replicates) if self.count_against_budget else 1


@dataclass(frozen=True)
class CostBreakdown:
    """Itemized cost of a concrete batch."""

    n_items: int
    n_wells: int
    n_plates: int
    acquisition_cost: float
    plate_cost: float
    currency: str = "USD"

    @property
    def total(self) -> float:
        return float(self.acquisition_cost + self.plate_cost)

    def as_dict(self) -> dict:
        return {
            "n_items": self.n_items,
            "n_wells": self.n_wells,
            "n_plates": self.n_plates,
            "acquisition_cost": round(self.acquisition_cost, 4),
            "plate_cost": round(self.plate_cost, 4),
            "total_cost": round(self.total, 4),
            "currency": self.currency,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.n_items} compounds in {self.n_wells} wells on "
            f"{self.n_plates} plate(s): {self.acquisition_cost:,.2f} acquisition "
            f"+ {self.plate_cost:,.2f} plate = {self.total:,.2f} {self.currency}"
        )


@dataclass
class CostModel:
    """Per-compound and per-plate costs, with a currency budget.

    Parameters
    ----------
    per_item_cost
        Cost to acquire one compound. Either a scalar applied to all
        candidates, or the name of a pool column holding per-compound costs
        (so a commercial fragment and a custom synthesis are priced apart).
    plate_fixed_cost
        Cost incurred per plate started, regardless of how full it is. This is
        what creates the step function.
    usable_wells_per_plate
        Wells available for samples after controls are reserved. Default 0
        means "ask the plate template" -- :class:`~loopwise.plate.PlateTemplate`
        knows how many wells its control layout leaves free.
    budget_currency
        Hard spending cap, or None for "no currency limit, well count rules".
    replicates
        A :class:`ReplicatePolicy`.
    currency
        Label used in reports. Loopwise never converts between currencies.
    """

    per_item_cost: float | str = 0.0
    plate_fixed_cost: float = 0.0
    usable_wells_per_plate: int = 0
    budget_currency: float | None = None
    replicates: ReplicatePolicy = ReplicatePolicy()
    currency: str = "USD"

    def __post_init__(self) -> None:
        if isinstance(self.per_item_cost, (int, float)) and self.per_item_cost < 0:
            raise ValueError("per_item_cost must be non-negative")
        if self.plate_fixed_cost < 0:
            raise ValueError("plate_fixed_cost must be non-negative")
        if self.budget_currency is not None and self.budget_currency < 0:
            raise ValueError("budget_currency must be non-negative")

    # ------------------------------------------------------------------ costs

    def item_costs(
        self, pool: pd.DataFrame | None = None, n_items: int | None = None
    ) -> np.ndarray:
        """Per-compound acquisition costs as an array.

        Resolves a column name against ``pool`` if ``per_item_cost`` is a
        string; otherwise broadcasts the scalar.
        """
        if isinstance(self.per_item_cost, str):
            if pool is None:
                raise ValueError(
                    f"per_item_cost={self.per_item_cost!r} names a pool column, "
                    "so a pool DataFrame is required to resolve it."
                )
            if self.per_item_cost not in pool.columns:
                raise ValueError(
                    f"per_item_cost column {self.per_item_cost!r} not in pool "
                    f"(columns: {list(pool.columns)[:12]}...)"
                )
            costs = pool[self.per_item_cost].to_numpy(dtype=DEFAULT_DTYPE)
            if not np.isfinite(costs).all():
                raise ValueError(
                    f"Column {self.per_item_cost!r} contains non-finite costs; "
                    "a NaN cost would silently make a compound look free."
                )
            if (costs < 0).any():
                raise ValueError(f"Column {self.per_item_cost!r} contains negative costs.")
            return costs
        n = int(n_items if n_items is not None else (0 if pool is None else len(pool)))
        return np.full(n, float(self.per_item_cost), dtype=DEFAULT_DTYPE)

    def wells_for(self, n_items: int) -> int:
        return int(n_items) * self.replicates.wells_per_item

    def plates_for(self, n_items: int, usable_wells_per_plate: int | None = None) -> int:
        """Plates needed to run ``n_items`` compounds with their replicates."""
        capacity = int(usable_wells_per_plate or self.usable_wells_per_plate)
        if capacity <= 0:
            return 1 if n_items > 0 else 0
        return int(math.ceil(self.wells_for(n_items) / capacity)) if n_items > 0 else 0

    def cost_of(
        self,
        selected_costs: Sequence[float] | np.ndarray,
        *,
        usable_wells_per_plate: int | None = None,
    ) -> CostBreakdown:
        """Full itemized cost of a batch, given its per-compound costs."""
        costs = np.asarray(selected_costs, dtype=np.float64).ravel()
        n_items = int(costs.size)
        multiplier = (
            self.replicates.wells_per_item
            if self.replicates.charge_acquisition_per_replicate
            else 1
        )
        acquisition = float(costs.sum()) * multiplier
        n_plates = self.plates_for(n_items, usable_wells_per_plate)
        return CostBreakdown(
            n_items=n_items,
            n_wells=self.wells_for(n_items),
            n_plates=n_plates,
            acquisition_cost=acquisition,
            plate_cost=float(self.plate_fixed_cost) * n_plates,
            currency=self.currency,
        )

    # -------------------------------------------------------------- budgeting

    def max_items_affordable(
        self,
        candidate_costs: Sequence[float] | np.ndarray,
        *,
        well_budget: int | None = None,
        usable_wells_per_plate: int | None = None,
    ) -> int:
        """Largest number of compounds affordable under both budgets.

        ``candidate_costs`` must be ordered the way the policy intends to
        select, cheapest-selected-first is *not* assumed: the k-th prefix is
        priced as the batch the policy would actually run if it stopped at k.

        Walks the real cost curve one compound at a time so the plate step
        function is respected -- it will return 96 rather than 97 when the
        97th compound would open a second plate you cannot afford.
        """
        costs = np.asarray(candidate_costs, dtype=np.float64).ravel()
        n_max = int(costs.size)

        if well_budget is not None:
            per_item = self.replicates.budget_cost_per_item()
            n_max = min(n_max, int(well_budget) // max(per_item, 1))

        if self.budget_currency is None:
            return n_max

        cumulative = np.cumsum(costs[:n_max]) * (
            self.replicates.wells_per_item
            if self.replicates.charge_acquisition_per_replicate
            else 1
        )
        best = 0
        for k in range(1, n_max + 1):
            plates = self.plates_for(k, usable_wells_per_plate)
            total = cumulative[k - 1] + self.plate_fixed_cost * plates
            if total <= self.budget_currency + 1e-9:
                best = k
            else:
                break  # cost is monotone in k, so the first failure is the ceiling
        return best

    def describe(self) -> str:
        """One-paragraph plain-language summary for reports and notebooks."""
        item = (
            f"column {self.per_item_cost!r}"
            if isinstance(self.per_item_cost, str)
            else f"{self.per_item_cost:,.2f} {self.currency}"
        )
        budget = (
            "no currency cap"
            if self.budget_currency is None
            else f"{self.budget_currency:,.2f} {self.currency} cap"
        )
        rep = self.replicates
        return (
            f"Acquisition {item} per compound; {self.plate_fixed_cost:,.2f} "
            f"{self.currency} per plate started; {budget}; "
            f"{rep.n_replicates} replicate(s) per compound "
            f"({'counted against' if rep.count_against_budget else 'excluded from'} "
            f"the well budget)."
        )
