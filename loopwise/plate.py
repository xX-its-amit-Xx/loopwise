"""Plate layout: where on the physical plate each compound goes.

Why selection alone is not enough
---------------------------------
A policy hands you 96 compounds. If you pipette them into the plate in the
order the policy returned them, you have introduced a confound that can be
larger than the effect you are measuring.

**Edge effects.** The outer ring of a microplate behaves differently from the
interior. Evaporation is faster at the edges, so wells concentrate; thermal
equilibration differs; in cell-based assays the edge wells see different
humidity and gas exchange. The resulting bias is typically 5-20% of signal and
is well documented in HTS quality-control literature (Lundholt et al., 2003;
Caraus et al., 2015). Now suppose your policy returned its picks in
score order and you filled the plate row-wise: the highest-scoring compounds
all land in row A -- an edge row. Every one of them gets the same systematic
bias, in the same direction, correlated perfectly with the model's prediction.
If the bias is positive you "confirm" your model and chase an artifact for six
months. If negative, you discard your best series.

**Positional confounding more generally.** Any pattern where a compound's
plate position correlates with a property of the compound creates this
problem: pipetting order, row-wise concentration gradients, plate-reader scan
direction, time-to-read.

**Controls.** You cannot normalize any of this away without controls, and
controls must be in known, reserved positions that nothing else can overwrite.
A control silently overwritten by a sample is a plate you cannot interpret and
will not know you cannot interpret.

What this module does
---------------------
1. Reserves wells for positive and negative controls, and refuses to place a
   sample in one. (This is a hard invariant with a test.)
2. **Stratified randomization.** Compounds are bucketed by predicted score,
   then assigned so each bucket is spread evenly across edge and interior
   wells and across the four plate quadrants. Randomization within the
   constraint keeps the assignment unbiased; the stratification keeps the
   randomness from accidentally clustering the predicted actives.
3. Reports an **edge-enrichment diagnostic**: what fraction of the top-score
   bucket landed on the edge, against the fraction expected by chance. A plate
   map is only auditable if that number is in the record.
4. Emits the plate map as a CSV and as a rendered figure.

Randomization is seeded and recorded, so a plate map can be regenerated
exactly from the campaign's seed and round number.
"""

from __future__ import annotations

import math
import string
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .utils import DEFAULT_DTYPE, make_rng

__all__ = [
    "PLATE_96",
    "PLATE_384",
    "PlateLayout",
    "PlateTemplate",
    "WellAssignment",
    "render_plate_map",
]

_ROW_LETTERS = string.ascii_uppercase


@dataclass(frozen=True)
class PlateTemplate:
    """Geometry and control layout of one plate format.

    Parameters
    ----------
    n_rows, n_cols
        8 x 12 for a 96-well plate, 16 x 24 for a 384.
    positive_control_wells, negative_control_wells
        Explicit well IDs (``"A1"``, ``"H12"``, ...). These are *reserved*:
        :class:`PlateLayout` will never put a sample in them.
    exclude_edge_wells
        If True, the entire outer ring is left empty rather than used for
        samples. This is the most conservative answer to edge effects and
        costs 34 of 96 wells (36%) or 76 of 384 (20%) -- which is why it is
        off by default on 96 and worth considering on 384, where the ring is a
        smaller fraction of the plate.
    control_columns
        Convenience: reserve whole columns for controls, the common bench
        practice of "column 1 is low control, column 12 is high control".
    name
        Label used in reports and figures.
    """

    n_rows: int = 8
    n_cols: int = 12
    positive_control_wells: tuple[str, ...] = ()
    negative_control_wells: tuple[str, ...] = ()
    exclude_edge_wells: bool = False
    name: str = "96-well"

    def __post_init__(self) -> None:
        if self.n_rows < 2 or self.n_cols < 2:
            raise ValueError("A plate needs at least 2 rows and 2 columns.")
        if self.n_rows > len(_ROW_LETTERS):
            raise ValueError(
                f"{self.n_rows} rows exceeds the A-Z row labelling this module uses."
            )
        overlap = set(self.positive_control_wells) & set(self.negative_control_wells)
        if overlap:
            raise ValueError(
                f"Wells assigned to both positive and negative controls: {sorted(overlap)}"
            )
        for well in self.positive_control_wells + self.negative_control_wells:
            self.validate_well(well)

    # ------------------------------------------------------------- geometry

    @property
    def n_wells(self) -> int:
        return self.n_rows * self.n_cols

    def row_label(self, row: int) -> str:
        return _ROW_LETTERS[row]

    def well_id(self, row: int, col: int) -> str:
        """``(0, 0) -> "A1"``. Columns are 1-based, as printed on the plate."""
        return f"{self.row_label(row)}{col + 1}"

    def parse_well(self, well: str) -> tuple[int, int]:
        """``"A1" -> (0, 0)``."""
        text = str(well).strip().upper()
        letter, digits = text[:1], text[1:]
        if letter not in _ROW_LETTERS or not digits.isdigit():
            raise ValueError(f"Malformed well id {well!r}; expected e.g. 'A1' or 'H12'.")
        return _ROW_LETTERS.index(letter), int(digits) - 1

    def validate_well(self, well: str) -> tuple[int, int]:
        row, col = self.parse_well(well)
        if not (0 <= row < self.n_rows and 0 <= col < self.n_cols):
            raise ValueError(
                f"Well {well!r} is outside a {self.n_rows}x{self.n_cols} "
                f"({self.n_wells}-well) plate."
            )
        return row, col

    def is_edge(self, row: int, col: int) -> bool:
        """True for the outer ring -- the wells with anomalous evaporation."""
        return row in (0, self.n_rows - 1) or col in (0, self.n_cols - 1)

    def quadrant(self, row: int, col: int) -> int:
        """0-3, used to spread each score stratum across the plate."""
        top = row < self.n_rows / 2
        left = col < self.n_cols / 2
        return (0 if top else 2) + (0 if left else 1)

    # ------------------------------------------------------------- reserved

    @property
    def control_wells(self) -> dict[str, str]:
        """Map well id -> ``"positive_control"`` / ``"negative_control"``."""
        out = {w: "positive_control" for w in self.positive_control_wells}
        out.update({w: "negative_control" for w in self.negative_control_wells})
        return out

    @property
    def reserved_wells(self) -> set[str]:
        """Every well a sample may not occupy (controls, plus the edge ring if
        ``exclude_edge_wells``)."""
        reserved = set(self.control_wells)
        if self.exclude_edge_wells:
            reserved |= {
                self.well_id(r, c)
                for r in range(self.n_rows)
                for c in range(self.n_cols)
                if self.is_edge(r, c)
            }
        return reserved

    @property
    def sample_wells(self) -> list[str]:
        """Wells available for compounds, in row-major order."""
        reserved = self.reserved_wells
        return [
            self.well_id(r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if self.well_id(r, c) not in reserved
        ]

    @property
    def usable_wells(self) -> int:
        return len(self.sample_wells)

    # ------------------------------------------------------------ factories

    @classmethod
    def plate96(
        cls,
        *,
        control_columns: bool = True,
        exclude_edge_wells: bool = False,
        n_positive: int = 8,
        n_negative: int = 8,
    ) -> PlateTemplate:
        """A 96-well plate (8 x 12).

        Default control layout: column 1 as negative (vehicle/DMSO) and column
        12 as positive, 8 wells each. That is the standard bench convention,
        and it has a useful side effect -- both control columns sit on the
        left/right edges, so the controls themselves sample the edge effect
        and can be used to estimate it.

        With ``control_columns=False``, controls are scattered in a fixed
        checkerboard of interior wells instead, which gives a better estimate
        of *within-plate* spatial drift at the cost of breaking convention.
        """
        if control_columns:
            neg = tuple(f"{_ROW_LETTERS[r]}1" for r in range(8))[:n_negative]
            pos = tuple(f"{_ROW_LETTERS[r]}12" for r in range(8))[:n_positive]
        else:
            interior = [
                f"{_ROW_LETTERS[r]}{c + 1}" for r in range(1, 7) for c in range(1, 11)
            ]
            neg = tuple(interior[:: max(1, len(interior) // max(n_negative, 1))])[
                :n_negative
            ]
            pos = tuple(w for w in interior[::-1] if w not in neg)[:n_positive]
        return cls(
            n_rows=8,
            n_cols=12,
            negative_control_wells=neg,
            positive_control_wells=pos,
            exclude_edge_wells=exclude_edge_wells,
            name="96-well",
        )

    @classmethod
    def plate384(
        cls,
        *,
        control_columns: bool = True,
        exclude_edge_wells: bool = True,
        n_positive: int = 16,
        n_negative: int = 16,
    ) -> PlateTemplate:
        """A 384-well plate (16 x 24).

        ``exclude_edge_wells`` defaults to True here: the ring is only 20% of
        a 384 plate (against 36% of a 96), and 384-well edge effects are worse
        because the wells are smaller, so the surface-to-volume ratio -- which
        is what drives evaporation -- is higher.
        """
        if control_columns:
            # Columns 2 and 23 when the outer ring is excluded, else 1 and 24.
            neg_col, pos_col = (2, 23) if exclude_edge_wells else (1, 24)
            rows = range(1, 15) if exclude_edge_wells else range(16)
            neg = tuple(f"{_ROW_LETTERS[r]}{neg_col}" for r in rows)[:n_negative]
            pos = tuple(f"{_ROW_LETTERS[r]}{pos_col}" for r in rows)[:n_positive]
        else:
            interior = [
                f"{_ROW_LETTERS[r]}{c + 1}" for r in range(2, 14) for c in range(2, 22)
            ]
            step = max(1, len(interior) // max(n_negative, 1))
            neg = tuple(interior[::step])[:n_negative]
            pos = tuple(w for w in interior[::-1] if w not in neg)[:n_positive]
        return cls(
            n_rows=16,
            n_cols=24,
            negative_control_wells=neg,
            positive_control_wells=pos,
            exclude_edge_wells=exclude_edge_wells,
            name="384-well",
        )

    def describe(self) -> str:
        return (
            f"{self.name} ({self.n_rows}x{self.n_cols} = {self.n_wells} wells): "
            f"{len(self.positive_control_wells)} positive + "
            f"{len(self.negative_control_wells)} negative control wells, "
            f"edge ring {'excluded' if self.exclude_edge_wells else 'in use'}, "
            f"{self.usable_wells} wells available for samples."
        )


#: A conventional 96-well plate: column 1 negative controls, column 12 positive.
PLATE_96 = PlateTemplate.plate96()

#: A 384-well plate with the outer ring left empty.
PLATE_384 = PlateTemplate.plate384()


@dataclass(frozen=True)
class WellAssignment:
    """One well's contents."""

    plate: int
    well: str
    row: str
    column: int
    content_type: str  # "sample" | "positive_control" | "negative_control" | "empty"
    item_id: object = None
    replicate: int = 1
    predicted_score: float = float("nan")
    score_stratum: int = -1
    is_edge: bool = False


@dataclass
class PlateLayout:
    """A concrete assignment of a selected batch to physical wells.

    Build one with :meth:`assign`. The result carries the per-well table plus
    the diagnostics that make the layout auditable.
    """

    template: PlateTemplate
    assignments: list[WellAssignment]
    diagnostics: dict = field(default_factory=dict)
    seed: int | None = None

    # ------------------------------------------------------------ construction

    @classmethod
    def assign(
        cls,
        item_ids: Sequence,
        *,
        template: PlateTemplate = PLATE_96,
        predicted_scores: Sequence[float] | None = None,
        n_replicates: int = 1,
        n_strata: int = 4,
        seed: int | np.random.Generator | None = None,
        randomize: bool = True,
    ) -> PlateLayout:
        """Lay out ``item_ids`` across as many plates as needed.

        The algorithm, in order:

        1. Bucket compounds into ``n_strata`` strata by predicted score
           (quantile bins). With no scores given, everything is one stratum.
        2. Partition the available sample wells -- across all plates, controls
           already excluded -- into *cells* keyed by (plate, edge/interior,
           quadrant), and shuffle within each cell.
        3. Build a **proportionally interleaved well sequence**. Each cell's
           wells are spaced evenly through the sequence (the j-th of a cell's
           ``c`` wells gets position ``(j + 0.5) / c``, then everything is
           sorted by position). The effect: any contiguous stretch of the
           sequence contains cells in proportion to their capacity.
        4. Build a **stratum-interleaved task sequence** the same way, so the
           score strata alternate rather than appearing in blocks.
        5. Zip the two. Because both sequences are evenly mixed, every stratum
           lands on edge and interior wells, and in each quadrant, in
           proportion to how many such wells exist -- while the choice of
           *which* edge well is still random.

        Step 3 is what a naive round-robin gets wrong. Dealing one well per
        cell per turn drains the small edge cells first (a 96-well plate with
        control columns has 20 edge sample wells against 60 interior ones), so
        whichever stratum is dealt last gets no edge wells at all. That is
        still a correlation between score and position -- just the opposite
        one from the naive layout.

        Replicates are placed in separate passes over the same global well
        sequence, so a compound's replicates land in different cells and
        usually on different plates. Replicates sharing a position bias would
        make the replicate SD look reassuringly tight while being
        systematically wrong, which is worse than no replicates.

        Parameters
        ----------
        randomize
            If False, compounds are placed in the order given, into wells in
            reading order (A1, A2, ... H12). This is the naive layout, kept
            only so a notebook can show side by side what it does to a
            score-ordered batch: every top pick lands in row A, an edge row.
            Do not use it for a real plate.
        """
        rng = make_rng(seed)
        ids = list(item_ids)
        n_items = len(ids)
        n_replicates = max(1, int(n_replicates))

        scores = (
            np.asarray(predicted_scores, dtype=DEFAULT_DTYPE)
            if predicted_scores is not None
            else np.zeros(n_items, dtype=DEFAULT_DTYPE)
        )
        if scores.size != n_items:
            raise ValueError(
                f"predicted_scores has length {scores.size} but {n_items} items were given."
            )

        strata = cls._stratify(scores, n_strata)
        wells_needed = n_items * n_replicates
        per_plate = template.usable_wells
        if per_plate == 0:
            raise ValueError(
                f"{template.describe()} leaves no wells for samples; relax "
                "exclude_edge_wells or reserve fewer control wells."
            )
        n_plates = int(math.ceil(wells_needed / per_plate))

        well_sequence = cls._well_sequence(template, n_plates, rng, randomize)
        tasks = cls._task_sequence(strata, n_replicates, rng, randomize)
        if len(tasks) > len(well_sequence):  # pragma: no cover - arithmetic guard
            raise RuntimeError(
                f"{len(tasks)} wells needed but only {len(well_sequence)} "
                "available; plate-count arithmetic is wrong."
            )

        assignments: list[WellAssignment] = []
        used: set[tuple[int, str]] = set()
        # Deliberately truncating: well_sequence is at least as long as tasks
        # (the plate count was computed from len(tasks)), and the surplus wells
        # are filled in as 'empty' below. strict=True would be wrong here.
        for (item_index, replicate), (plate, well) in zip(tasks, well_sequence):  # noqa: B905
            if well in template.reserved_wells:  # pragma: no cover - invariant
                raise AssertionError(
                    f"INVARIANT VIOLATED: sample would occupy reserved well "
                    f"{well!r}. This is a bug in loopwise.plate, not in your input."
                )
            row, col = template.parse_well(well)
            used.add((plate, well))
            assignments.append(
                WellAssignment(
                    plate=plate,
                    well=well,
                    row=template.row_label(row),
                    column=col + 1,
                    content_type="sample",
                    item_id=ids[item_index],
                    replicate=replicate,
                    predicted_score=float(scores[item_index]),
                    score_stratum=int(strata[item_index]),
                    is_edge=template.is_edge(row, col),
                )
            )

        # Controls and leftover empties, for every plate.
        for plate in range(n_plates):
            for well, kind in template.control_wells.items():
                row, col = template.parse_well(well)
                assignments.append(
                    WellAssignment(
                        plate=plate,
                        well=well,
                        row=template.row_label(row),
                        column=col + 1,
                        content_type=kind,
                        item_id=kind,
                        is_edge=template.is_edge(row, col),
                    )
                )
            for well in template.sample_wells:
                if (plate, well) not in used:
                    row, col = template.parse_well(well)
                    assignments.append(
                        WellAssignment(
                            plate=plate,
                            well=well,
                            row=template.row_label(row),
                            column=col + 1,
                            content_type="empty",
                            is_edge=template.is_edge(row, col),
                        )
                    )
            if template.exclude_edge_wells:
                for r in range(template.n_rows):
                    for c in range(template.n_cols):
                        well = template.well_id(r, c)
                        if template.is_edge(r, c) and well not in template.control_wells:
                            assignments.append(
                                WellAssignment(
                                    plate=plate,
                                    well=well,
                                    row=template.row_label(r),
                                    column=c + 1,
                                    content_type="empty",
                                    item_id="edge_excluded",
                                    is_edge=True,
                                )
                            )

        layout = cls(
            template=template,
            assignments=assignments,
            seed=seed if isinstance(seed, int) else None,
        )
        layout.diagnostics = layout._compute_diagnostics(n_plates, n_strata, randomize)
        return layout

    @staticmethod
    def _interleave(groups: list[list], rng: np.random.Generator) -> list:
        """Merge groups so each is spread evenly through the output.

        The j-th of a group's ``c`` members is given the fractional position
        ``(j + 0.5) / c``; everything is then sorted by position. A group with
        many members appears often, a group with few appears rarely, and both
        are spaced uniformly -- so any contiguous slice of the result holds the
        groups roughly in proportion to their sizes.

        This is the same idea as Bresenham's line algorithm, or as the largest-
        remainder method for apportioning seats: spread ``c`` items over a unit
        interval as evenly as integers allow. Ties are broken randomly so that
        equal-sized groups do not always resolve in the same order.
        """
        decorated = []
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            for j, member in enumerate(group):
                decorated.append(((j + 0.5) / size, float(rng.random()), member))
        decorated.sort(key=lambda t: (t[0], t[1]))
        return [member for _, _, member in decorated]

    @classmethod
    def _well_sequence(
        cls,
        template: PlateTemplate,
        n_plates: int,
        rng: np.random.Generator,
        randomize: bool,
    ) -> list[tuple[int, str]]:
        """Global ``(plate, well)`` order to fill, proportionally interleaved."""
        if not randomize:
            # Naive: reading order, plate by plate.
            return [
                (plate, well) for plate in range(n_plates) for well in template.sample_wells
            ]

        cells: dict[tuple[int, bool, int], list[tuple[int, str]]] = {}
        for plate in range(n_plates):
            for well in template.sample_wells:
                row, col = template.parse_well(well)
                key = (plate, template.is_edge(row, col), template.quadrant(row, col))
                cells.setdefault(key, []).append((plate, well))
        groups = []
        for key in sorted(cells):
            members = cells[key]
            order = rng.permutation(len(members))
            groups.append([members[i] for i in order])
        return cls._interleave(groups, rng)

    @staticmethod
    def _task_sequence(
        strata: np.ndarray,
        n_replicates: int,
        rng: np.random.Generator,
        randomize: bool,
    ) -> list[tuple[int, int]]:
        """``(item_index, replicate)`` order, with score strata interleaved.

        Replicates are placed in separate passes so a compound's replicates are
        far apart in the well sequence, hence in different cells and usually on
        different plates.
        """
        if not randomize:
            return [
                (int(i), rep + 1) for rep in range(n_replicates) for i in range(strata.size)
            ]
        tasks: list[tuple[int, int]] = []
        for rep in range(n_replicates):
            groups = []
            for stratum in sorted(set(strata.tolist())):
                members = np.flatnonzero(strata == stratum)
                members = members[rng.permutation(members.size)]
                groups.append([(int(i), rep + 1) for i in members])
            tasks.extend(PlateLayout._interleave(groups, rng))
        return tasks

    @staticmethod
    def _stratify(scores: np.ndarray, n_strata: int) -> np.ndarray:
        """Quantile-bin scores into strata, highest score = highest stratum."""
        n_strata = max(1, int(n_strata))
        if scores.size == 0:
            return np.zeros(0, dtype=np.int64)
        if n_strata == 1 or np.allclose(scores, scores[0]):
            return np.zeros(scores.size, dtype=np.int64)
        ranks = np.argsort(np.argsort(scores, kind="stable"), kind="stable")
        return (ranks * n_strata // max(scores.size, 1)).astype(np.int64)

    # ------------------------------------------------------------- reporting

    def to_frame(self) -> pd.DataFrame:
        """Long-form well table: one row per well. This is the CSV."""
        frame = pd.DataFrame([a.__dict__ for a in self.assignments])
        if frame.empty:  # pragma: no cover
            return frame
        return frame.sort_values(["plate", "row", "column"]).reset_index(drop=True)

    def to_grid(self, plate: int = 0, value: str = "item_id") -> pd.DataFrame:
        """Wide plate-shaped view, the way the plate looks on the bench."""
        frame = self.to_frame()
        frame = frame[frame["plate"] == plate]
        grid = pd.DataFrame(
            index=[self.template.row_label(r) for r in range(self.template.n_rows)],
            columns=list(range(1, self.template.n_cols + 1)),
            dtype=object,
        )
        for _, row in frame.iterrows():
            grid.loc[row["row"], row["column"]] = row[value]
        return grid

    def to_csv(self, path: str) -> str:
        """Write the plate map CSV; returns the path."""
        self.to_frame().to_csv(path, index=False)
        return path

    @property
    def n_plates(self) -> int:
        return int(max((a.plate for a in self.assignments), default=-1) + 1)

    def control_wells_intact(self) -> bool:
        """True if every reserved control well holds a control.

        The invariant the tests enforce. If this is ever False the plate is
        uninterpretable, so :meth:`assign` also asserts it inline.
        """
        reserved = self.template.control_wells
        for a in self.assignments:
            if a.well in reserved and a.content_type != reserved[a.well]:
                return False
        holders = {
            (a.plate, a.well)
            for a in self.assignments
            if a.content_type.endswith("control")
        }
        return len(holders) == self.n_plates * len(reserved)

    def _compute_diagnostics(self, n_plates: int, n_strata: int, randomized: bool) -> dict:
        """Edge enrichment per stratum: the number that catches the confound.

        For each score stratum, the fraction of its wells that landed on the
        plate edge, against the fraction of *available* sample wells that are
        edge wells. A ratio near 1.0 means the stratum is spread as chance
        would spread it. A ratio near 2.0 means that stratum is twice as
        edge-exposed as it should be, and the plate will confound score with
        evaporation.
        """
        samples = [a for a in self.assignments if a.content_type == "sample"]
        if not samples:  # pragma: no cover
            return {"n_plates": n_plates, "n_samples": 0}

        edge_available = sum(
            1
            for well in self.template.sample_wells
            if self.template.is_edge(*self.template.parse_well(well))
        )
        expected = edge_available / max(len(self.template.sample_wells), 1)

        per_stratum = {}
        for stratum in sorted({a.score_stratum for a in samples}):
            members = [a for a in samples if a.score_stratum == stratum]
            observed = sum(1 for a in members if a.is_edge) / len(members)
            per_stratum[int(stratum)] = {
                "n": len(members),
                "edge_fraction": round(observed, 4),
                "enrichment_vs_chance": round(observed / expected, 3) if expected else None,
            }

        top = max(per_stratum) if per_stratum else None
        return {
            "n_plates": n_plates,
            "n_samples": len(samples),
            "n_strata": n_strata,
            "randomized": randomized,
            "expected_edge_fraction": round(expected, 4),
            "edge_enrichment_by_stratum": per_stratum,
            "top_stratum_edge_enrichment": (
                per_stratum[top]["enrichment_vs_chance"] if top is not None else None
            ),
            "controls_intact": self.control_wells_intact(),
        }

    def summary(self) -> str:
        """Plain-language summary, suitable for pasting into a lab notebook."""
        d = self.diagnostics
        enrich = d.get("top_stratum_edge_enrichment")
        expected = d.get("expected_edge_fraction", 0.0) or 0.0
        if expected <= 0:
            edge_line = (
                "No sample wells sit on the edge ring (it is excluded), so edge "
                "effects cannot confound the samples at all."
            )
        else:
            verdict = (
                "balanced"
                if enrich is not None and 0.6 <= enrich <= 1.4
                else "CHECK: uneven edge exposure, re-run with a different seed"
            )
            edge_line = (
                f"Edge wells are {expected * 100:.0f}% of available sample wells; "
                f"highest-score stratum edge enrichment vs chance {enrich} "
                f"({verdict}). With {d.get('n_samples', 0)} samples in "
                f"{d.get('n_strata', 1)} strata this ratio is coarse -- one well "
                "either way moves it noticeably."
            )
        return (
            f"{self.template.describe()}\n"
            f"{d.get('n_samples', 0)} sample wells across {d.get('n_plates', 0)} "
            f"plate(s); controls intact: {d.get('controls_intact')}.\n" + edge_line
        )


def render_plate_map(
    layout: PlateLayout,
    path: str | None = None,
    *,
    plate: int | None = None,
    annotate: bool = True,
    title: str | None = None,
):
    """Render plate map(s) as a matplotlib figure.

    Colour encodes content: controls in their own colours, samples shaded by
    predicted score, empties grey. Edge wells are outlined heavily, because the
    whole point of looking at this figure is to see whether the dark (high
    score) wells have clustered against the outline.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    frame = layout.to_frame()
    plates = [plate] if plate is not None else sorted(frame["plate"].unique())
    tpl = layout.template
    fig, axes = plt.subplots(
        len(plates),
        1,
        figsize=(1 + 0.62 * tpl.n_cols, (0.9 + 0.62 * tpl.n_rows) * len(plates)),
        squeeze=False,
    )

    scores = frame.loc[frame["content_type"] == "sample", "predicted_score"]
    vmin, vmax = (float(scores.min()), float(scores.max())) if len(scores) else (0.0, 1.0)
    if vmax - vmin < 1e-9:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    cmap = plt.get_cmap("viridis")

    for ax_row, plate_id in zip(axes, plates, strict=True):
        ax = ax_row[0]
        sub = frame[frame["plate"] == plate_id]
        ax.add_patch(
            Rectangle(
                (0.5, 0.5),
                tpl.n_cols,
                tpl.n_rows,
                facecolor="#f7f7f7",
                edgecolor="#333333",
                linewidth=1.5,
                zorder=0,
            )
        )
        for _, well in sub.iterrows():
            r = _ROW_LETTERS.index(well["row"])
            x, y = well["column"], r + 1
            kind = well["content_type"]
            if kind == "positive_control":
                color, label = "#d62728", "+"
            elif kind == "negative_control":
                color, label = "#1f77b4", "-"
            elif kind == "sample":
                frac = (well["predicted_score"] - vmin) / (vmax - vmin)
                color = cmap(float(np.clip(frac, 0, 1)))
                label = str(well["score_stratum"])
            else:
                color, label = "#e0e0e0", ""
            ax.add_patch(
                Circle(
                    (x, y),
                    0.42,
                    facecolor=color,
                    edgecolor="#000000" if well["is_edge"] else "#999999",
                    linewidth=2.0 if well["is_edge"] else 0.6,
                    zorder=2,
                )
            )
            if annotate and label:
                ax.text(
                    x,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if kind != "empty" else "#666666",
                    zorder=3,
                )
        ax.set_xlim(0.2, tpl.n_cols + 0.8)
        ax.set_ylim(tpl.n_rows + 0.8, 0.2)
        ax.set_xticks(range(1, tpl.n_cols + 1))
        ax.set_yticks(range(1, tpl.n_rows + 1))
        ax.set_yticklabels([tpl.row_label(i) for i in range(tpl.n_rows)])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        enrich = layout.diagnostics.get("top_stratum_edge_enrichment")
        ax.set_title(
            title
            or (
                f"{tpl.name} plate {plate_id + 1}/{layout.n_plates} - "
                f"shade = predicted score, digits = score stratum, "
                f"bold outline = edge well\n"
                f"top-stratum edge enrichment vs chance: {enrich}"
            ),
            fontsize=9,
        )
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
