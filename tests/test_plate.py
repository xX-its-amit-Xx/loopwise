"""Plate layout tests.

The hard invariant: **a sample never occupies a reserved control well.** A
control silently overwritten by a sample produces a plate that looks fine,
normalizes against a compound instead of a control, and yields numbers nobody
can interpret or audit. The rest of these tests cover the edge-effect
balancing, which is the reason the layout is randomized at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from loopwise.plate import (
    PLATE_96,
    PLATE_384,
    PlateLayout,
    PlateTemplate,
    render_plate_map,
)


@pytest.fixture
def ids_and_scores():
    n = 80
    return [f"CPD{i:03d}" for i in range(n)], np.linspace(0.95, 0.02, n)


# ----------------------------------------------------------- the hard invariant


@pytest.mark.parametrize(
    "template", [PLATE_96, PLATE_384, PlateTemplate.plate96(control_columns=False)]
)
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("n_replicates", [1, 3])
def test_controls_are_never_overwritten(template, seed, n_replicates, ids_and_scores):
    ids, scores = ids_and_scores
    layout = PlateLayout.assign(
        ids,
        template=template,
        predicted_scores=scores,
        n_replicates=n_replicates,
        seed=seed,
    )

    control_map = template.control_wells
    for assignment in layout.assignments:
        if assignment.well in control_map:
            assert assignment.content_type == control_map[assignment.well], (
                f"well {assignment.well} should be {control_map[assignment.well]} "
                f"but holds {assignment.content_type}"
            )
        if assignment.content_type == "sample":
            assert assignment.well not in template.reserved_wells

    assert layout.control_wells_intact()
    assert layout.diagnostics["controls_intact"] is True


def test_every_plate_carries_a_full_set_of_controls(ids_and_scores):
    """Controls on plate 1 only would make plate 2 uninterpretable."""
    ids, scores = ids_and_scores
    layout = PlateLayout.assign(
        ids, template=PLATE_96, predicted_scores=scores, n_replicates=3, seed=0
    )
    assert layout.n_plates > 1

    frame = layout.to_frame()
    for plate in range(layout.n_plates):
        sub = frame[frame["plate"] == plate]
        assert set(sub.loc[sub["content_type"] == "positive_control", "well"]) == set(
            PLATE_96.positive_control_wells
        )
        assert set(sub.loc[sub["content_type"] == "negative_control", "well"]) == set(
            PLATE_96.negative_control_wells
        )


def test_no_well_is_assigned_twice(ids_and_scores):
    ids, scores = ids_and_scores
    layout = PlateLayout.assign(
        ids, template=PLATE_96, predicted_scores=scores, n_replicates=2, seed=1
    )
    occupied = [(a.plate, a.well) for a in layout.assignments]
    assert len(occupied) == len(set(occupied)), "a well was assigned twice"


def test_all_requested_items_are_placed(ids_and_scores):
    ids, scores = ids_and_scores
    layout = PlateLayout.assign(
        ids, template=PLATE_96, predicted_scores=scores, n_replicates=2, seed=2
    )
    frame = layout.to_frame()
    samples = frame[frame["content_type"] == "sample"]
    assert len(samples) == len(ids) * 2
    assert set(samples["item_id"]) == set(ids)
    assert samples.groupby("item_id").size().unique().tolist() == [2]


# ------------------------------------------------------------- edge  effects


@pytest.mark.parametrize("seed", range(8))
def test_high_scorers_are_not_concentrated_on_the_edge(seed, ids_and_scores):
    """The confound this module exists to prevent.

    With 20 edge sample wells split across 4 strata, chance gives each stratum
    5. The tolerance below (0.25-1.75x) is wide because that granularity makes
    one well worth 0.2 -- it is testing "not systematically clustered", not
    "perfectly uniform".
    """
    ids, scores = ids_and_scores
    layout = PlateLayout.assign(ids, template=PLATE_96, predicted_scores=scores, seed=seed)
    by_stratum = layout.diagnostics["edge_enrichment_by_stratum"]
    for stratum, stats in by_stratum.items():
        assert 0.25 <= stats["enrichment_vs_chance"] <= 1.75, (
            f"seed {seed} stratum {stratum}: edge enrichment "
            f"{stats['enrichment_vs_chance']}"
        )


def test_the_naive_layout_demonstrates_the_confound(ids_and_scores):
    """randomize=False must reproduce the bug, so the docs can show it.

    Filling wells in reading order with a score-ordered batch puts the whole
    top stratum in rows A and B -- row A being an edge row. The stratified
    layout must not do that.
    """
    ids, scores = ids_and_scores
    naive = PlateLayout.assign(
        ids, template=PLATE_96, predicted_scores=scores, seed=0, randomize=False
    )
    stratified = PlateLayout.assign(
        ids, template=PLATE_96, predicted_scores=scores, seed=0, randomize=True
    )

    naive_top = naive.diagnostics["top_stratum_edge_enrichment"]
    strat_top = stratified.diagnostics["top_stratum_edge_enrichment"]
    assert naive_top > 1.5, f"naive layout was not confounded ({naive_top})"
    assert abs(strat_top - 1.0) < abs(naive_top - 1.0)

    # And the naive layout's top-scoring compounds really are in row A.
    frame = naive.to_frame()
    samples = frame[frame["content_type"] == "sample"]
    top_rows = set(samples.nlargest(10, "predicted_score")["row"])
    assert top_rows <= {"A", "B"}


def test_replicates_of_one_compound_are_spread_across_positions(ids_and_scores):
    """Replicates sharing a position bias give a falsely tight SD."""
    ids, scores = ids_and_scores
    layout = PlateLayout.assign(
        ids[:40], template=PLATE_96, predicted_scores=scores[:40], n_replicates=3, seed=4
    )
    frame = layout.to_frame()
    samples = frame[frame["content_type"] == "sample"]

    distinct_plates = samples.groupby("item_id")["plate"].nunique()
    assert distinct_plates.mean() > 1.2, "replicates are not being spread across plates"
    # No compound should have all of its replicates in one row.
    same_row = samples.groupby("item_id")["row"].nunique()
    assert (same_row == 1).mean() < 0.2


def test_exclude_edge_wells_leaves_the_ring_empty():
    template = PlateTemplate.plate96(exclude_edge_wells=True)
    assert template.usable_wells < PLATE_96.usable_wells

    layout = PlateLayout.assign(
        [f"C{i}" for i in range(30)],
        template=template,
        predicted_scores=np.linspace(0, 1, 30),
        seed=0,
    )
    frame = layout.to_frame()
    samples = frame[frame["content_type"] == "sample"]
    assert not samples["is_edge"].any(), "a sample was placed on an excluded edge"
    assert layout.diagnostics["expected_edge_fraction"] == 0.0


# ------------------------------------------------------------------ geometry


def test_well_id_round_trip():
    assert PLATE_96.well_id(0, 0) == "A1"
    assert PLATE_96.well_id(7, 11) == "H12"
    assert PLATE_96.parse_well("H12") == (7, 11)
    assert PLATE_384.well_id(15, 23) == "P24"

    with pytest.raises(ValueError):
        PLATE_96.validate_well("I1")  # row beyond an 8-row plate
    with pytest.raises(ValueError):
        PLATE_96.validate_well("A13")  # column beyond 12
    with pytest.raises(ValueError):
        PLATE_96.parse_well("hello")


def test_edge_detection():
    assert PLATE_96.is_edge(0, 5) and PLATE_96.is_edge(7, 5)
    assert PLATE_96.is_edge(3, 0) and PLATE_96.is_edge(3, 11)
    assert not PLATE_96.is_edge(3, 5)


def test_plate_sizes():
    assert PLATE_96.n_wells == 96
    assert PLATE_384.n_wells == 384
    assert PLATE_96.usable_wells == 96 - 16  # two control columns


def test_overlapping_control_assignment_is_rejected():
    with pytest.raises(ValueError, match="both positive and negative"):
        PlateTemplate(positive_control_wells=("A1",), negative_control_wells=("A1",))


def test_control_well_outside_the_plate_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        PlateTemplate(n_rows=8, n_cols=12, positive_control_wells=("A99",))


# ------------------------------------------------------------------- outputs


def test_plate_map_csv_and_grid(tmp_path, ids_and_scores):
    ids, scores = ids_and_scores
    layout = PlateLayout.assign(ids, template=PLATE_96, predicted_scores=scores, seed=0)

    path = layout.to_csv(str(tmp_path / "plate.csv"))
    import pandas as pd

    reloaded = pd.read_csv(path)
    assert len(reloaded) == 96
    assert set(reloaded["content_type"]) <= {
        "sample",
        "positive_control",
        "negative_control",
        "empty",
    }

    grid = layout.to_grid(0, "content_type")
    assert grid.shape == (8, 12)
    assert grid.loc["A", 1] == "negative_control"
    assert grid.loc["A", 12] == "positive_control"


def test_render_plate_map_produces_a_figure(tmp_path, ids_and_scores):
    import matplotlib

    matplotlib.use("Agg")
    ids, scores = ids_and_scores
    layout = PlateLayout.assign(ids, template=PLATE_96, predicted_scores=scores, seed=0)
    path = tmp_path / "plate.png"
    figure = render_plate_map(layout, str(path))
    assert path.exists() and path.stat().st_size > 1000
    assert figure is not None


def test_layout_is_deterministic_under_a_seed(ids_and_scores):
    ids, scores = ids_and_scores
    first = PlateLayout.assign(ids, template=PLATE_96, predicted_scores=scores, seed=11)
    second = PlateLayout.assign(ids, template=PLATE_96, predicted_scores=scores, seed=11)
    assert first.to_frame().equals(second.to_frame())

    different = PlateLayout.assign(ids, template=PLATE_96, predicted_scores=scores, seed=12)
    assert not first.to_frame().equals(different.to_frame())


def test_template_with_no_usable_wells_is_rejected():
    template = PlateTemplate(
        n_rows=2,
        n_cols=2,
        positive_control_wells=("A1", "A2"),
        negative_control_wells=("B1", "B2"),
    )
    assert template.usable_wells == 0
    with pytest.raises(ValueError, match="no wells for samples"):
        PlateLayout.assign(["x"], template=template)
