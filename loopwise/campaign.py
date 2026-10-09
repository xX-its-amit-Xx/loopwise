"""The Campaign: the object a lab actually drives.

    c = Campaign(pool=df, model=estimator, budget=96, policy="hybrid")
    batch = c.select()          # the rows to run, plus why each was chosen
    c.observe(results_df)       # feed measurements back
    c.report()                  # regret, calibration drift, diversity

Three design choices worth stating, because each one prevents a specific way
these pipelines go wrong:

**The pool is never mutated.** Loopwise keeps labels in its own ledger keyed by
compound id. If it wrote measurements into your DataFrame, a later ``select()``
on a stale copy would be silently training on different data than you think.

**Measured compounds leave the candidate set permanently.** The default is that
you do not re-test what you have measured. (Pass ``allow_retest=True`` if your
assay is noisy enough that re-measurement is genuinely informative -- it
sometimes is, and then it should be a deliberate choice.)

**Calibration is tracked, not assumed.** Round after round, the model is
retrained on a sample that the model itself chose. That is the opposite of a
random sample, and it bends calibration in a predictable direction: a model
fed its own confident picks becomes over-confident. ``report()`` measures this
drift (Brier score, expected calibration error) so you can see it happen
rather than discovering it when a predicted 90% hit rate delivers 30%.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .batch import BatchPlan, finalize_batch, resolve_batch_size
from .cost import CostBreakdown, CostModel
from .models import ModelAdapter, Posterior, auto_adapter
from .plate import PLATE_96, PlateLayout, PlateTemplate, render_plate_map
from .policies import Policy, PolicyContext, SelectionResult, get_policy
from .utils import DEFAULT_DTYPE, as_feature_matrix, make_rng

__all__ = ["Campaign", "CampaignReport", "RoundRecord", "SelectedBatch"]

_COST_COL = "_loopwise_item_cost"


@dataclass
class SelectedBatch:
    """What :meth:`Campaign.select` returns.

    ``frame`` is the selected pool rows with the policy's reason columns
    appended, so one CSV answers "what did we run, and why each one?".
    """

    frame: pd.DataFrame
    round_index: int
    policy: str
    plan: BatchPlan
    cost: CostBreakdown
    plate_layout: PlateLayout | None = None
    diagnostics: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def item_ids(self) -> list:
        return self.frame[self.diagnostics["id_column"]].tolist()

    def plate_map(self) -> pd.DataFrame:
        if self.plate_layout is None:
            raise ValueError(
                "No plate layout for this batch; construct the Campaign with "
                "plate=PlateTemplate... (it defaults to a 96-well plate, so "
                "this means plate=None was passed explicitly)."
            )
        return self.plate_layout.to_frame()

    def to_csv(self, path: str) -> str:
        self.frame.to_csv(path, index=False)
        return path

    def save_plate_map(self, csv_path: str | None = None, figure_path: str | None = None):
        """Write the plate map as CSV and/or a rendered figure."""
        if self.plate_layout is None:
            raise ValueError("No plate layout for this batch.")
        if csv_path:
            self.plate_layout.to_csv(csv_path)
        fig = None
        if figure_path:
            fig = render_plate_map(self.plate_layout, figure_path)
        return fig

    def explain(self) -> str:
        lines = [
            f"Round {self.round_index + 1}: {len(self.frame)} compound(s) "
            f"selected by policy '{self.policy}'.",
            self.plan.explain(),
            f"Cost: {self.cost}",
        ]
        if self.plate_layout is not None:
            lines.append(self.plate_layout.summary())
        return "\n".join(lines)

    def _repr_html_(self):  # pragma: no cover - notebook convenience
        return f"<pre>{self.explain()}</pre>" + self.frame.head(10).to_html()


@dataclass
class RoundRecord:
    """One round's audit trail."""

    round_index: int
    policy: str
    n_selected: int
    n_labeled_before: int
    n_candidates_before: int
    cost: float
    alpha: float | None
    selected_ids: list
    predicted_scores: np.ndarray
    observed_values: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)
    #: The campaign's hit threshold, carried here so that every hit count in
    #: the report uses the same definition. Hard-coding 0.5 instead means a
    #: campaign measuring potency in pIC50 units counts every positive reading
    #: as a hit, reports a 100% hit rate, and -- because the oracle comparison
    #: *does* use the configured threshold -- produces negative regret and
    #: "fraction of achievable" above 1.0.
    hit_threshold: float = 0.5

    @property
    def n_hits(self) -> int | None:
        if self.observed_values is None:
            return None
        return int(np.nansum(self.observed_values > self.hit_threshold))


class Campaign:
    """An active-learning campaign over a fixed candidate pool.

    Parameters
    ----------
    pool
        Candidate compounds, one row each. Must contain an id column (named by
        ``id_col``, or a unique index is used) and numeric feature columns.
    model
        Any scikit-learn-compatible estimator, anything exposing
        ``predict_proba``, or anything exposing ``sample()``. Wrapped by
        :func:`~loopwise.models.auto_adapter`; pass a
        :class:`~loopwise.models.ModelAdapter` directly for full control.
    budget
        Wells available per round (96, 384, ...). Interpreted through the
        replicate policy: see :mod:`loopwise.batch`.
    policy
        Policy name or instance. ``"hybrid"`` is the practical default;
        ``"greedy"`` reproduces what most labs do today.
    feature_cols
        Feature columns. Defaults to every numeric column except the id, label
        and cost columns -- an explicit list is strongly preferred, because
        "every numeric column" changes meaning when someone adds a column.
    label_col
        Column in ``results_df`` carrying the measurement. Also read from
        ``pool`` at construction if present, which is how you seed a campaign
        with the compounds you have already run.
    initial_labels
        Optional ``{item_id: value}`` of measurements already in hand. A
        campaign with no labels at all cannot fit a model, so round 1 falls
        back to a seeding strategy (see ``seed_policy``).
    seed_policy
        Policy used for round 1 when there is nothing to train on. Defaults to
        ``"diversity"`` with pure space-filling (``quality_weight=0``), which
        is the right answer: with no model, the only sensible objective is to
        cover the feature space.
    n_rounds_planned
        How many rounds you intend to run. Only the hybrid policy's annealing
        schedule needs it -- but it needs it badly, since "anneal toward
        exploitation" is meaningless without knowing when the end is.
    n_posterior_samples
        Posterior draws per round. More is better and costs prediction time,
        not training time; 20-50 is usually plenty.
    task
        ``"classification"`` (hit / not hit) or ``"regression"`` (potency).
    """

    def __init__(
        self,
        pool: pd.DataFrame,
        model: Any,
        budget: int = 96,
        policy: str | Policy = "hybrid",
        *,
        feature_cols: Sequence[str] | None = None,
        id_col: str | None = None,
        label_col: str = "label",
        cost_col: str | None = None,
        initial_labels: dict | pd.Series | None = None,
        cost_model: CostModel | None = None,
        plate: PlateTemplate | None = PLATE_96,
        seed: int | None = 0,
        n_rounds_planned: int | None = None,
        n_posterior_samples: int = 30,
        task: str = "classification",
        hit_threshold: float = 0.5,
        seed_policy: str | Policy | None = None,
        allow_retest: bool = False,
        n_bootstrap: int | None = None,
        policy_kwargs: dict | None = None,
        chunked: bool = False,
    ):
        if not isinstance(pool, pd.DataFrame):
            raise TypeError(f"pool must be a pandas DataFrame, got {type(pool)}")
        if len(pool) == 0:
            raise ValueError("pool is empty.")
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")

        self.pool = pool.reset_index(drop=True).copy()
        self.id_col = self._resolve_id_column(id_col)
        self.label_col = label_col
        self.cost_col = cost_col
        self.task = task
        self.hit_threshold = float(hit_threshold)
        self.budget = int(budget)
        self.plate_template = plate
        self.seed = seed
        self.rng = make_rng(seed)
        self.n_rounds_planned = n_rounds_planned
        self.n_posterior_samples = int(n_posterior_samples)
        self.allow_retest = bool(allow_retest)
        self.chunked = bool(chunked)

        self.cost_model = cost_model or CostModel(
            per_item_cost=cost_col if cost_col else 0.0,
            usable_wells_per_plate=plate.usable_wells if plate else 0,
        )
        if plate is not None and self.cost_model.usable_wells_per_plate == 0:
            self.cost_model.usable_wells_per_plate = plate.usable_wells

        excluded = {self.id_col, label_col}
        if cost_col:
            excluded.add(cost_col)
        self.features, self.feature_cols = as_feature_matrix(
            self.pool, feature_cols, exclude=tuple(excluded)
        )

        self.adapter: ModelAdapter = auto_adapter(
            model, task=task, n_bootstrap=n_bootstrap, random_state=seed
        )
        self.policy = get_policy(policy, **(policy_kwargs or {}))
        self.seed_policy = (
            get_policy(seed_policy)
            if seed_policy is not None
            else get_policy("diversity", quality_weight=0.0)
        )

        # The label ledger: item_id -> measured value. The pool is never touched.
        self.labels: dict = {}
        if label_col in self.pool.columns:
            seeded = self.pool[[self.id_col, label_col]].dropna(subset=[label_col])
            self.labels.update(
                dict(zip(seeded[self.id_col], seeded[label_col], strict=True))
            )
        if initial_labels is not None:
            self.labels.update(dict(initial_labels))

        self._id_to_row = {item: i for i, item in enumerate(self.pool[self.id_col])}
        self.history: list[RoundRecord] = []
        self._pending: SelectedBatch | None = None
        self._last_posterior: Posterior | None = None
        self._calibration: list[dict] = []

    # ------------------------------------------------------------------ setup

    def _resolve_id_column(self, id_col: str | None) -> str:
        if id_col is not None:
            if id_col not in self.pool.columns:
                raise ValueError(f"id_col {id_col!r} not in pool columns.")
            if self.pool[id_col].duplicated().any():
                dupes = self.pool[id_col][self.pool[id_col].duplicated()].unique()[:5]
                raise ValueError(
                    f"id_col {id_col!r} has duplicate values (e.g. {list(dupes)}); "
                    "Loopwise keys its label ledger on this column, so ids must "
                    "be unique."
                )
            return id_col
        for candidate in ("item_id", "id", "compound_id", "mol_id", "smiles"):
            if (
                candidate in self.pool.columns
                and not self.pool[candidate].duplicated().any()
            ):
                return candidate
        self.pool["item_id"] = np.arange(len(self.pool))
        return "item_id"

    # -------------------------------------------------------------- accessors

    @property
    def labeled_ids(self) -> list:
        return [i for i in self.pool[self.id_col] if i in self.labels]

    @property
    def candidate_mask(self) -> np.ndarray:
        """True for rows still eligible for selection."""
        if self.allow_retest:
            return np.ones(len(self.pool), dtype=bool)
        measured = set(self.labels)
        return np.array([i not in measured for i in self.pool[self.id_col]], dtype=bool)

    @property
    def n_labeled(self) -> int:
        return len(self.labels)

    @property
    def round_index(self) -> int:
        return len(self.history)

    def labeled_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(X, y)`` for the compounds measured so far."""
        rows, values = [], []
        for item, value in self.labels.items():
            idx = self._id_to_row.get(item)
            if idx is None:
                continue
            rows.append(idx)
            values.append(value)
        if not rows:
            return (
                np.empty((0, self.features.shape[1]), dtype=DEFAULT_DTYPE),
                np.empty(0, dtype=DEFAULT_DTYPE),
            )
        y = np.asarray(values, dtype=DEFAULT_DTYPE)
        if self.task == "classification":
            y = (y > self.hit_threshold).astype(np.int64)
        return self.features[np.asarray(rows)], y

    # ------------------------------------------------------------------ select

    def select(
        self,
        budget: int | None = None,
        *,
        policy: str | Policy | None = None,
        randomize_plate: bool = True,
    ) -> SelectedBatch:
        """Choose the next batch.

        Returns a :class:`SelectedBatch` whose ``frame`` holds the chosen pool
        rows plus the policy's reason columns, and whose ``plate_layout`` holds
        the randomized plate assignment.

        Raises if a batch is already pending -- call :meth:`observe` (or
        :meth:`discard_pending`) first. Selecting twice without observing is
        almost always a mistake: the second batch is chosen by the same model
        as the first, so it is one batch of 192, not two rounds of 96.
        """
        if self._pending is not None:
            raise RuntimeError(
                f"Round {self._pending.round_index + 1} is still pending "
                f"({len(self._pending)} compounds selected but not observed). "
                "Call observe(results_df) with the measurements, or "
                "discard_pending() to abandon that batch."
            )

        active_policy = get_policy(policy) if policy is not None else self.policy
        mask = self.candidate_mask
        candidate_rows = np.flatnonzero(mask)
        if candidate_rows.size == 0:
            raise RuntimeError(
                "No unmeasured candidates left in the pool. The campaign has "
                "exhausted its pool; add candidates or pass allow_retest=True."
            )

        Xl, yl = self.labeled_arrays()
        use_seed_policy = self._must_seed(yl)
        if use_seed_policy:
            active_policy = self.seed_policy
            posterior = Posterior(
                np.zeros((1, candidate_rows.size), dtype=DEFAULT_DTYPE), task=self.task
            )
        else:
            self.adapter.fit(Xl, yl)
            posterior = self.adapter.posterior(
                self.features[candidate_rows],
                n_samples=self.n_posterior_samples,
                rng=self.rng,
            )
        self._last_posterior = posterior

        candidate_costs = self.cost_model.item_costs(
            pool=self.pool.iloc[candidate_rows], n_items=candidate_rows.size
        )
        plan = resolve_batch_size(
            budget=int(budget if budget is not None else self.budget),
            n_candidates=candidate_rows.size,
            cost_model=self.cost_model,
            template=self.plate_template or PLATE_96,
            candidate_costs=candidate_costs,
        )
        if plan.n_items == 0:
            raise RuntimeError(
                "Budget allows zero compounds this round.\n" + plan.explain()
            )

        ctx = PolicyContext(
            features=self.features[candidate_rows],
            posterior=posterior,
            budget=plan.n_items,
            rng=self.rng,
            round_index=self.round_index,
            n_rounds_planned=self.n_rounds_planned,
            labeled_features=Xl,
            labeled_targets=yl,
            item_costs=candidate_costs,
            chunked=self.chunked,
        )
        result: SelectionResult = active_policy.select(ctx)

        chosen_rows = candidate_rows[result.indices]
        frame = self.pool.iloc[chosen_rows].copy()
        for key, values in result.reasons.items():
            frame[key] = np.asarray(values)
        frame[_COST_COL] = candidate_costs[result.indices]
        frame["round"] = self.round_index
        frame["posterior_mean"] = posterior.mean[result.indices]
        frame["posterior_std"] = posterior.std[result.indices]

        frame, cost, notes = finalize_batch(
            frame,
            cost_model=self.cost_model,
            template=self.plate_template or PLATE_96,
        )
        plan.notes.extend(notes)
        plan.n_items = len(frame)
        plan.estimated_cost = cost

        layout = None
        if self.plate_template is not None and len(frame):
            layout = PlateLayout.assign(
                frame[self.id_col].tolist(),
                template=self.plate_template,
                predicted_scores=frame["posterior_mean"].to_numpy(),
                n_replicates=self.cost_model.replicates.n_replicates,
                seed=int(self.rng.integers(0, 2**31 - 1)),
                randomize=randomize_plate,
            )

        diagnostics = dict(result.diagnostics)
        diagnostics.update(
            {
                "id_column": self.id_col,
                "n_labeled_before": self.n_labeled,
                "n_candidates_before": int(candidate_rows.size),
                "used_seed_policy": use_seed_policy,
                "posterior_n_samples": posterior.n_samples,
                "posterior_degenerate": posterior.is_degenerate,
                "model": self.adapter.name,
            }
        )
        if use_seed_policy:
            diagnostics["note"] = (
                "Round 1 with no usable labels: selected by the seeding policy "
                "(space-filling diversity), since there is no model to exploit."
            )

        batch = SelectedBatch(
            frame=frame.drop(columns=[_COST_COL]),
            round_index=self.round_index,
            policy=f"{active_policy.name}{' (seeding)' if use_seed_policy else ''}",
            plan=plan,
            cost=cost,
            plate_layout=layout,
            diagnostics=diagnostics,
        )
        self._pending = batch
        return batch

    def _must_seed(self, y: np.ndarray) -> bool:
        """True when there is nothing a model could be fitted on."""
        if y.size == 0:
            return True
        if self.task == "classification" and np.unique(y).size < 2:
            warnings.warn(
                f"All {y.size} labels so far are the same class, so no "
                "classifier can be fitted; using the seeding policy "
                "(space-filling) for this round. This is the normal situation "
                "in a low-hit-rate screen: keep going until you find a hit.",
                RuntimeWarning,
                stacklevel=3,
            )
            return True
        return y.size < 2

    def discard_pending(self) -> None:
        """Abandon the pending batch without recording measurements."""
        self._pending = None

    # ----------------------------------------------------------------- observe

    def observe(
        self,
        results: pd.DataFrame | dict | pd.Series,
        *,
        label_col: str | None = None,
        id_col: str | None = None,
    ) -> Campaign:
        """Record measurements and close the round.

        ``results`` may be a DataFrame with id and label columns, or a
        ``{item_id: value}`` mapping. Replicates are averaged, which is the
        only sensible default; pre-aggregate yourself if you want a median or
        a curve fit.

        Ids not in the pending batch are still recorded (you may have measured
        extra compounds) but are reported, because a mismatch usually means an
        id-format problem -- the most common one being compound ids that came
        back from the plate reader with a plate prefix attached.
        """
        label_col = label_col or self.label_col
        id_col = id_col or self.id_col

        if isinstance(results, pd.DataFrame):
            if id_col not in results.columns:
                raise ValueError(
                    f"results is missing the id column {id_col!r}; columns are "
                    f"{list(results.columns)}."
                )
            if label_col not in results.columns:
                raise ValueError(
                    f"results is missing the label column {label_col!r}; columns "
                    f"are {list(results.columns)}. Pass label_col=..."
                )
            observed = (
                results[[id_col, label_col]]
                .dropna(subset=[label_col])
                .groupby(id_col)[label_col]
                .mean()
                .to_dict()
            )
        else:
            observed = {k: v for k, v in dict(results).items() if pd.notna(v)}

        if not observed:
            raise ValueError("No non-null measurements found in results.")

        unknown = [k for k in observed if k not in self._id_to_row]
        if unknown:
            warnings.warn(
                f"{len(unknown)} measured id(s) are not in the pool and will be "
                f"ignored for model fitting (e.g. {unknown[:3]}). Check that the "
                "ids coming back from the assay match the pool's "
                f"{self.id_col!r} column exactly.",
                RuntimeWarning,
                stacklevel=2,
            )

        pending = self._pending
        if pending is not None:
            expected = set(pending.frame[self.id_col])
            missing = expected - set(observed)
            if missing:
                warnings.warn(
                    f"{len(missing)} of {len(expected)} selected compound(s) have "
                    f"no measurement (e.g. {list(missing)[:3]}). They stay in the "
                    "candidate pool and may be selected again.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self.labels.update(observed)
        self._record_round(observed, pending)
        self._pending = None
        return self

    def _record_round(self, observed: dict, pending: SelectedBatch | None) -> None:
        if pending is None:
            return
        ids = pending.frame[self.id_col].tolist()
        values = np.array([observed.get(i, np.nan) for i in ids], dtype=np.float64)
        predicted = pending.frame["posterior_mean"].to_numpy(dtype=np.float64)
        alpha = pending.diagnostics.get("alpha")

        self._calibration.append(
            self._calibration_metrics(predicted, values, pending.round_index)
        )
        self.history.append(
            RoundRecord(
                round_index=pending.round_index,
                policy=pending.policy,
                n_selected=len(ids),
                n_labeled_before=pending.diagnostics["n_labeled_before"],
                n_candidates_before=pending.diagnostics["n_candidates_before"],
                cost=pending.cost.total,
                alpha=float(alpha) if alpha is not None else None,
                selected_ids=ids,
                predicted_scores=predicted,
                observed_values=values,
                diagnostics=pending.diagnostics,
                hit_threshold=self.hit_threshold,
            )
        )

    def _calibration_metrics(
        self, predicted: np.ndarray, observed: np.ndarray, round_index: int
    ) -> dict:
        """Brier score and expected calibration error for one round.

        **Brier score** is the mean squared error of a probability forecast:
        predict 0.9 for something that happens and you pay ``(1 - 0.9)^2``.
        Lower is better; 0.25 is what you get by always predicting 0.5.

        **Expected calibration error (ECE)** bins compounds by predicted
        probability and asks, in each bin, whether the observed hit rate
        matches the predicted one. A model that says "0.8" for a group of
        compounds of which 30% are actives has an ECE contribution of 0.5 from
        that bin. This is the number that moves when selection bias bites:
        training on your own confident picks makes the model confident about
        the wrong things, and ECE catches it while accuracy does not.
        """
        mask = np.isfinite(observed) & np.isfinite(predicted)
        if mask.sum() == 0:
            return {"round": round_index, "n": 0}
        p = predicted[mask]
        if self.task == "classification":
            y = (observed[mask] > self.hit_threshold).astype(np.float64)
            brier = float(np.mean((p - y) ** 2))
            n_bins = 5
            edges = np.linspace(0.0, 1.0, n_bins + 1)
            ece, bins = 0.0, []
            for b in range(n_bins):
                lo, hi = edges[b], edges[b + 1]
                in_bin = (p >= lo) & (p < hi if b < n_bins - 1 else p <= hi)
                if not in_bin.any():
                    continue
                conf, acc = float(p[in_bin].mean()), float(y[in_bin].mean())
                ece += (in_bin.sum() / p.size) * abs(conf - acc)
                bins.append(
                    {
                        "bin": f"[{lo:.1f},{hi:.1f}]",
                        "n": int(in_bin.sum()),
                        "mean_predicted": round(conf, 4),
                        "observed_rate": round(acc, 4),
                    }
                )
            return {
                "round": round_index,
                "n": int(p.size),
                "brier": round(brier, 5),
                "ece": round(float(ece), 5),
                "mean_predicted": round(float(p.mean()), 4),
                "observed_hit_rate": round(float(y.mean()), 4),
                "bins": bins,
            }
        resid = observed[mask] - p
        return {
            "round": round_index,
            "n": int(mask.sum()),
            "rmse": round(float(np.sqrt(np.mean(resid**2))), 5),
            "bias": round(float(np.mean(resid)), 5),
            "mean_predicted": round(float(p.mean()), 4),
            "mean_observed": round(float(observed[mask].mean()), 4),
        }

    # ------------------------------------------------------------------ report

    def report(self, oracle_labels: pd.Series | dict | None = None) -> CampaignReport:
        """Summarize the campaign: regret, calibration drift, diversity.

        Parameters
        ----------
        oracle_labels
            Only available in a retrospective replay, where the true label of
            every compound is known. Given it, regret is the real thing:
            cumulative hits you *could* have found with perfect knowledge minus
            what you did find. Without it, Loopwise reports *realized* regret
            against the best you have actually seen so far, and says so --
            because pretending to know the oracle is how an active-learning
            report becomes marketing.
        """
        return CampaignReport.build(self, oracle_labels=oracle_labels)

    # ------------------------------------------------------------------- misc

    def predict(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        """Current model predictions for ``frame`` (default: the whole pool)."""
        target = self.pool if frame is None else frame
        X, _ = as_feature_matrix(target, self.feature_cols)
        Xl, yl = self.labeled_arrays()
        if self._must_seed(yl):
            raise RuntimeError(
                "Cannot predict: no usable labels yet (need at least two, and "
                "for classification at least one of each class)."
            )
        self.adapter.fit(Xl, yl)
        post = self.adapter.posterior(X, n_samples=self.n_posterior_samples, rng=self.rng)
        out = target[[self.id_col]].copy()
        out["predicted"] = post.mean
        out["posterior_std"] = post.std
        out["mutual_information"] = post.mutual_information()
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Campaign(pool={len(self.pool)}, labeled={self.n_labeled}, "
            f"rounds_done={len(self.history)}, policy={self.policy.name!r}, "
            f"budget={self.budget}, model={self.adapter.name!r})"
        )


@dataclass
class CampaignReport:
    """Round-by-round metrics, plus the plotting helpers.

    ``frame`` has one row per completed round with the columns:

    ``n_selected``, ``n_hits``, ``hit_rate``, ``cumulative_hits``,
    ``cumulative_cost``, ``cost_per_hit``, ``alpha``, ``brier``, ``ece``,
    ``mean_predicted``, ``observed_hit_rate``, ``batch_diversity``,
    ``cumulative_regret`` (and ``oracle_cumulative_hits`` when an oracle was
    supplied).
    """

    frame: pd.DataFrame
    calibration: list[dict]
    diversity: pd.DataFrame
    has_oracle: bool
    notes: list[str] = field(default_factory=list)

    @classmethod
    def build(
        cls, campaign: Campaign, oracle_labels: pd.Series | dict | None = None
    ) -> CampaignReport:
        rows, notes = [], []
        oracle = dict(oracle_labels) if oracle_labels is not None else None

        cumulative_hits = 0
        cumulative_cost = 0.0
        for rec, calib in zip(campaign.history, campaign._calibration, strict=True):
            hits = rec.n_hits or 0
            cumulative_hits += hits
            cumulative_cost += rec.cost
            row = {
                "round": rec.round_index + 1,
                "policy": rec.policy,
                "n_selected": rec.n_selected,
                "n_labeled_before": rec.n_labeled_before,
                "n_hits": hits,
                "hit_rate": hits / rec.n_selected if rec.n_selected else np.nan,
                "cumulative_hits": cumulative_hits,
                "cost": rec.cost,
                "cumulative_cost": cumulative_cost,
                "cost_per_hit": (cumulative_cost / cumulative_hits)
                if cumulative_hits
                else np.nan,
                "alpha": rec.alpha,
                "mean_predicted": calib.get("mean_predicted"),
                "observed_hit_rate": calib.get("observed_hit_rate"),
                "brier": calib.get("brier"),
                "ece": calib.get("ece"),
                "rmse": calib.get("rmse"),
                "batch_diversity": cls._batch_diversity(campaign, rec),
            }
            rows.append(row)

        frame = pd.DataFrame(rows)

        if oracle is not None and len(frame):
            # True regret: the best achievable cumulative hits by this round.
            values = np.array(
                [v for k, v in oracle.items() if k in campaign._id_to_row],
                dtype=np.float64,
            )
            hit_flags = np.sort((values > campaign.hit_threshold).astype(int))[::-1]
            tested = np.cumsum(frame["n_selected"].to_numpy())
            best = np.array(
                [int(hit_flags[:t].sum()) for t in np.minimum(tested, hit_flags.size)]
            )
            frame["oracle_cumulative_hits"] = best
            frame["cumulative_regret"] = best - frame["cumulative_hits"]
            frame["fraction_of_achievable"] = frame["cumulative_hits"] / np.maximum(
                int(hit_flags.sum()), 1
            )
            notes.append(
                "Regret is measured against an oracle that knows every label: "
                f"{int(hit_flags.sum())} actives exist in the pool. This is only "
                "possible in a retrospective replay."
            )
        elif len(frame):
            running_best = frame["hit_rate"].cummax()
            frame["cumulative_regret"] = (
                (running_best - frame["hit_rate"]) * frame["n_selected"]
            ).cumsum()
            notes.append(
                "No oracle labels supplied, so 'cumulative_regret' is REALIZED "
                "regret: hits missed relative to the best hit rate any round of "
                "this campaign actually achieved. It is a lower bound on true "
                "regret and cannot be compared across campaigns."
            )

        diversity = pd.DataFrame(
            [
                {
                    "round": rec.round_index + 1,
                    "batch_diversity": cls._batch_diversity(campaign, rec),
                    "n_selected": rec.n_selected,
                }
                for rec in campaign.history
            ]
        )
        return cls(
            frame=frame,
            calibration=list(campaign._calibration),
            diversity=diversity,
            has_oracle=oracle is not None,
            notes=notes,
        )

    @staticmethod
    def _batch_diversity(campaign: Campaign, rec: RoundRecord) -> float:
        """Mean pairwise Euclidean distance within the batch, standardized.

        Reported in units of the pool's own spread: 1.0 means "as spread out as
        two random compounds from the pool". Below ~0.7 the batch is noticeably
        redundant -- you are paying for measurements that duplicate each other.
        """
        rows = [
            campaign._id_to_row[i] for i in rec.selected_ids if i in campaign._id_to_row
        ]
        if len(rows) < 2:
            return float("nan")
        X = campaign.features[np.asarray(rows)].astype(np.float64)
        sd = campaign.features.std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xs = X / sd
        # Mean pairwise distance via the identity E||a-b||^2 = 2*trace(cov).
        pool_scale = float(np.sqrt(2.0 * Xs.shape[1]))
        d2 = (
            np.sum(Xs**2, axis=1)[:, None]
            + np.sum(Xs**2, axis=1)[None, :]
            - 2.0 * (Xs @ Xs.T)
        )
        iu = np.triu_indices(len(rows), k=1)
        mean_dist = float(np.sqrt(np.maximum(d2[iu], 0)).mean())
        return round(mean_dist / pool_scale, 4) if pool_scale else float("nan")

    # ------------------------------------------------------------------ output

    def summary(self) -> str:
        if self.frame.empty:
            return "No completed rounds yet: call select() then observe(results)."
        f = self.frame
        lines = [
            f"{len(f)} round(s), {int(f['n_selected'].sum())} compounds measured, "
            f"{int(f['n_hits'].sum())} hit(s).",
            "Hit rate by round: "
            + ", ".join(f"R{int(r.round)}={r.hit_rate:.1%}" for r in f.itertuples()),
            f"Cumulative hits: {list(f['cumulative_hits'])}",
        ]
        if "fraction_of_achievable" in f:
            lines.append(
                f"Fraction of achievable hits found: "
                f"{f['fraction_of_achievable'].iloc[-1]:.1%}"
            )
        if f["ece"].notna().any():
            first, last = f["ece"].dropna().iloc[0], f["ece"].dropna().iloc[-1]
            drift = last - first
            lines.append(
                f"Calibration error (ECE) {first:.3f} -> {last:.3f} "
                f"({'worsening' if drift > 0.02 else 'stable' if abs(drift) <= 0.02 else 'improving'}"
                ")."
            )
        if f["batch_diversity"].notna().any():
            lines.append(
                "Batch diversity (1.0 = two random pool members): "
                + ", ".join(
                    f"R{int(r.round)}={r.batch_diversity:.2f}"
                    for r in f.itertuples()
                    if pd.notna(r.batch_diversity)
                )
            )
        if f["alpha"].notna().any():
            lines.append(
                "Hybrid alpha by round: "
                + ", ".join(
                    f"R{int(r.round)}={r.alpha:.2f}"
                    for r in f.itertuples()
                    if pd.notna(r.alpha)
                )
            )
        lines.extend(self.notes)
        return "\n".join(lines)

    def plot(self, path: str | None = None):
        """Three-panel figure: hits, calibration drift, diversity."""
        import matplotlib.pyplot as plt

        f = self.frame
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        ax = axes[0]
        ax.plot(f["round"], f["cumulative_hits"], "o-", label="found", color="#1f77b4")
        if "oracle_cumulative_hits" in f:
            ax.plot(
                f["round"],
                f["oracle_cumulative_hits"],
                "s--",
                label="oracle (perfect knowledge)",
                color="#999999",
            )
        ax.set_xlabel("round")
        ax.set_ylabel("cumulative hits")
        ax.set_title("Hits found")
        ax.legend(fontsize=8)

        ax = axes[1]
        if f["ece"].notna().any():
            ax.plot(f["round"], f["ece"], "o-", color="#d62728", label="ECE")
            ax.plot(
                f["round"],
                f["mean_predicted"],
                "^--",
                color="#ff7f0e",
                label="mean predicted",
            )
            ax.plot(
                f["round"],
                f["observed_hit_rate"],
                "v--",
                color="#2ca02c",
                label="observed rate",
            )
        elif f["rmse"].notna().any():
            ax.plot(f["round"], f["rmse"], "o-", color="#d62728", label="RMSE")
        ax.set_xlabel("round")
        ax.set_title("Calibration drift")
        ax.legend(fontsize=8)

        ax = axes[2]
        ax.plot(f["round"], f["batch_diversity"], "o-", color="#9467bd")
        ax.axhline(1.0, ls=":", color="#666666", lw=1)
        ax.set_xlabel("round")
        ax.set_ylabel("mean pairwise distance (pool units)")
        ax.set_title("Batch diversity")

        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
        return fig

    def _repr_html_(self):  # pragma: no cover - notebook convenience
        return f"<pre>{self.summary()}</pre>" + self.frame.to_html(index=False)
