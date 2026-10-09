"""Shared plumbing for Loopwise: the resource guard, RNG discipline, and
feature-matrix coercion.

THE RESOURCE GUARD
------------------
Loopwise is meant to run on a laptop in a wet lab, not on a cluster. The
selection policies that are most useful (facility location, k-DPP) are also
the ones that want to build an ``n x n`` similarity matrix over the candidate
pool. For a pool of 50,000 compounds that matrix is

    50_000 * 50_000 * 4 bytes (float32) = 10 GB

which will either swap the machine to a standstill or get the process killed
by the OOM killer with no explanation. Neither failure mode tells the user
what went wrong or what to do about it.

``check_resources()`` is the single contract every allocating code path in
Loopwise calls *before* it allocates. It measures what the machine actually
has (via ``psutil``), estimates the dense footprint of what is about to be
built, and returns a decision:

    "ok"       -> allocate as planned
    "downcast" -> allocate, but in float32 instead of float64
    "chunk"    -> do not allocate the whole thing; stream it in blocks of
                  ``recommended_chunk_rows`` rows, with n_jobs capped
    "refuse"   -> raise ``ResourceRefusal`` with an actionable message

The thresholds: above 50% of *available* RAM we chunk and downcast and cap
parallelism at ``nproc - 1``; above 90% we refuse. "Available" rather than
"total" is deliberate -- a browser and a Slack client are usually already
holding a few GB, and the number that matters is what is left.

The hard rule, enforced separately by ``guard_pairwise_matrix()``: a dense
``n x n`` matrix is never materialized for ``n > 20_000`` without the caller
explicitly passing ``chunked=True``. Even if the machine has the RAM, a
400-million-entry matrix means the caller has almost certainly not thought
about scaling, and the chunked path is both correct and nearly as fast.

Default dtype is float32 throughout. Screening data has two or three
significant figures; float64 doubles the memory for precision nobody has.
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:  # psutil is a hard dependency, but degrade loudly rather than crash
    import psutil

    _HAVE_PSUTIL = True
except ImportError:  # pragma: no cover
    _HAVE_PSUTIL = False

__all__ = [
    "DEFAULT_DTYPE",
    "PAIRWISE_HARD_CAP",
    "ResourceRefusal",
    "ResourceReport",
    "as_feature_matrix",
    "check_resources",
    "chunk_slices",
    "guard_pairwise_matrix",
    "make_rng",
    "normalize_scores",
]

#: Loopwise computes in float32 unless the caller insists otherwise.
DEFAULT_DTYPE = np.float32

#: Above this pool size a dense n x n matrix requires ``chunked=True``.
PAIRWISE_HARD_CAP = 20_000

#: Fraction of *available* RAM above which we chunk/downcast.
CHUNK_THRESHOLD = 0.50

#: Fraction of *available* RAM above which we refuse outright.
REFUSE_THRESHOLD = 0.90

#: Target footprint for a single chunk when chunking (fraction of available).
CHUNK_TARGET_FRACTION = 0.10


class ResourceRefusal(MemoryError):
    """Raised when a requested allocation cannot be served safely.

    Subclasses ``MemoryError`` so that callers who catch ``MemoryError``
    still work, but carries a message that says what to do instead.
    """


@dataclass
class ResourceReport:
    """The verdict from :func:`check_resources`.

    Attributes
    ----------
    decision
        One of ``"ok"``, ``"downcast"``, ``"chunk"``, ``"refuse"``.
    estimated_gb
        Dense footprint of the requested allocation, in GiB.
    ram_available_gb
        RAM the OS reports as available right now, in GiB.
    ram_total_gb
        Total physical RAM, in GiB.
    fraction_of_available
        ``estimated_gb / ram_available_gb``.
    cpu_count
        Logical CPUs visible to this process.
    recommended_n_jobs
        Parallelism to use. Capped at ``cpu_count - 1`` once the allocation
        crosses :data:`CHUNK_THRESHOLD`, so the machine stays interactive.
    recommended_chunk_rows
        Rows per block if ``decision == "chunk"``. ``None`` otherwise.
    recommended_dtype
        ``np.float32`` whenever downcasting would help.
    message
        Human-readable explanation, including the remedy.
    """

    decision: str
    estimated_gb: float
    ram_available_gb: float
    ram_total_gb: float
    fraction_of_available: float
    cpu_count: int
    recommended_n_jobs: int
    recommended_chunk_rows: int | None
    recommended_dtype: type
    message: str
    task: str = ""
    details: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the allocation may proceed as requested."""
        return self.decision in ("ok", "downcast")

    @property
    def must_chunk(self) -> bool:
        return self.decision == "chunk"

    def raise_if_refused(self) -> ResourceReport:
        """Raise :class:`ResourceRefusal` if the verdict was ``"refuse"``."""
        if self.decision == "refuse":
            raise ResourceRefusal(self.message)
        return self

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.decision}] {self.message}"


def _memory_snapshot() -> tuple[float, float, int]:
    """Return ``(total_gb, available_gb, cpu_count)``.

    Falls back to conservative guesses (and warns) if psutil is missing, so a
    broken install degrades to caution rather than to an unguarded allocation.
    """
    gib = float(1 << 30)
    if _HAVE_PSUTIL:
        vm = psutil.virtual_memory()
        cpus = psutil.cpu_count(logical=True) or 1
        return vm.total / gib, vm.available / gib, int(cpus)

    warnings.warn(
        "psutil is not installed; Loopwise cannot measure real memory and is "
        "assuming a conservative 4 GiB available. Install psutil for accurate "
        "resource guarding (`pip install psutil`).",
        RuntimeWarning,
        stacklevel=3,
    )
    return 8.0, 4.0, int(os.cpu_count() or 1)


def estimate_dense_gb(
    shape: Sequence[int],
    dtype: type = DEFAULT_DTYPE,
    copies: int = 1,
) -> float:
    """Dense footprint of an array of ``shape`` in GiB, times ``copies``.

    ``copies`` exists because real code rarely holds one array: a pairwise
    distance computation typically holds the matrix plus a working buffer, and
    an argsort holds an index array of the same length. Passing ``copies=2``
    is the honest estimate, not padding.
    """
    n_elements = 1
    for dim in shape:
        n_elements *= int(dim)
    itemsize = np.dtype(dtype).itemsize
    return n_elements * itemsize * max(1, int(copies)) / float(1 << 30)


def check_resources(
    shape: Sequence[int],
    *,
    dtype: type = DEFAULT_DTYPE,
    copies: int = 1,
    task: str = "allocation",
    n_jobs: int | None = None,
    allow_chunking: bool = True,
    remedy: str | None = None,
) -> ResourceReport:
    """Decide whether ``shape`` can be allocated, and how.

    This is *the* resource contract in Loopwise. Every code path that is about
    to allocate something whose size depends on the pool size calls it first.

    Parameters
    ----------
    shape
        Shape of the array about to be allocated, e.g. ``(n, n)`` for a
        similarity matrix or ``(n_samples, n_items)`` for posterior draws.
    dtype
        Intended dtype. float64 requests may come back as a ``"downcast"``
        decision recommending float32.
    copies
        How many arrays of this size the algorithm holds at once.
    task
        Short label used in the message, e.g. ``"k-DPP similarity matrix"``.
    n_jobs
        Requested parallelism; ``None`` means "all cores". The returned
        ``recommended_n_jobs`` caps this at ``cpu_count - 1`` under pressure,
        because each worker typically holds its own chunk.
    allow_chunking
        If False, an allocation that would otherwise be chunked is refused
        instead. Use this when the caller has no streaming implementation.
    remedy
        Extra, caller-specific advice appended to a refusal message.

    Returns
    -------
    ResourceReport
        Never raises for a merely-large request; call
        :meth:`ResourceReport.raise_if_refused` (or :func:`guard_pairwise_matrix`)
        to turn a refusal into an exception.
    """
    total_gb, available_gb, cpus = _memory_snapshot()
    requested = estimate_dense_gb(shape, dtype=dtype, copies=copies)

    # If the caller asked for float64, price the float32 alternative too.
    downcast_dtype = np.dtype(dtype)
    downcast_gb = requested
    if downcast_dtype.itemsize > 4 and np.issubdtype(downcast_dtype, np.floating):
        downcast_dtype = np.dtype(DEFAULT_DTYPE)
        downcast_gb = estimate_dense_gb(shape, dtype=downcast_dtype, copies=copies)

    effective_gb = min(requested, downcast_gb)
    fraction = effective_gb / max(available_gb, 1e-9)

    requested_jobs = cpus if n_jobs in (None, -1) else max(1, int(n_jobs))
    safe_jobs = max(1, min(requested_jobs, cpus - 1)) if cpus > 1 else 1

    rows = int(shape[0]) if len(shape) else 1
    per_row_gb = effective_gb / max(rows, 1)
    chunk_rows = None
    if per_row_gb > 0:
        budget = available_gb * CHUNK_TARGET_FRACTION
        chunk_rows = int(max(1, min(rows, math.floor(budget / per_row_gb))))

    shape_str = " x ".join(str(int(d)) for d in shape)
    base = (
        f"{task}: dense {shape_str} {np.dtype(dtype).name} "
        f"({requested:.2f} GiB"
        + (f", {downcast_gb:.2f} GiB as float32" if downcast_gb < requested else "")
        + f") against {available_gb:.2f} GiB available "
        f"of {total_gb:.2f} GiB total, {cpus} CPUs"
    )

    if fraction >= REFUSE_THRESHOLD or (fraction >= CHUNK_THRESHOLD and not allow_chunking):
        advice = remedy or (
            "Reduce the candidate pool (pre-filter to the top ~20k by model "
            "score), lower the batch size, or switch to a policy that does not "
            "need pairwise similarity (e.g. policy='bald' or 'uncertainty')."
        )
        return ResourceReport(
            decision="refuse",
            estimated_gb=requested,
            ram_available_gb=available_gb,
            ram_total_gb=total_gb,
            fraction_of_available=fraction,
            cpu_count=cpus,
            recommended_n_jobs=safe_jobs,
            recommended_chunk_rows=chunk_rows,
            recommended_dtype=downcast_dtype.type,
            message=(
                f"REFUSED. {base}. That is {fraction * 100:.0f}% of available "
                f"RAM, past the {REFUSE_THRESHOLD * 100:.0f}% ceiling"
                + ("" if allow_chunking else " and this code path cannot stream")
                + f". {advice}"
            ),
            task=task,
            details={"threshold": REFUSE_THRESHOLD},
        )

    if fraction >= CHUNK_THRESHOLD:
        return ResourceReport(
            decision="chunk",
            estimated_gb=requested,
            ram_available_gb=available_gb,
            ram_total_gb=total_gb,
            fraction_of_available=fraction,
            cpu_count=cpus,
            recommended_n_jobs=safe_jobs,
            recommended_chunk_rows=chunk_rows,
            recommended_dtype=downcast_dtype.type,
            message=(
                f"CHUNKING. {base}. That is {fraction * 100:.0f}% of available "
                f"RAM, past the {CHUNK_THRESHOLD * 100:.0f}% chunking threshold: "
                f"streaming {chunk_rows} rows at a time in "
                f"{downcast_dtype.name}, n_jobs capped at {safe_jobs}."
            ),
            task=task,
            details={"threshold": CHUNK_THRESHOLD},
        )

    decision = "downcast" if downcast_gb < requested else "ok"
    return ResourceReport(
        decision=decision,
        estimated_gb=requested,
        ram_available_gb=available_gb,
        ram_total_gb=total_gb,
        fraction_of_available=fraction,
        cpu_count=cpus,
        recommended_n_jobs=requested_jobs,
        recommended_chunk_rows=None,
        recommended_dtype=downcast_dtype.type,
        message=(
            f"OK. {base}. That is {fraction * 100:.1f}% of available RAM"
            + (f"; using {downcast_dtype.name}." if decision == "downcast" else ".")
        ),
        task=task,
    )


def guard_pairwise_matrix(
    n_items: int,
    *,
    chunked: bool = False,
    task: str = "pairwise similarity matrix",
    dtype: type = DEFAULT_DTYPE,
    n_jobs: int | None = None,
) -> ResourceReport:
    """Guard the dangerous path: an ``n x n`` similarity or kernel matrix.

    Two gates, in order:

    1. **The hard cap.** For ``n > PAIRWISE_HARD_CAP`` (20,000) a dense matrix
       is refused outright unless the caller passes ``chunked=True`` to state
       that it streams the matrix rather than materializing it. This is not a
       memory check -- it fires even on a machine with 1 TB of RAM -- because
       at that size the dense path is a design error.
    2. **The memory check.** :func:`check_resources` on an ``(n, n)`` array,
       which may still return ``"chunk"`` or ``"refuse"``.

    Raises
    ------
    ResourceRefusal
        If either gate rejects the request.
    """
    n_items = int(n_items)
    if n_items > PAIRWISE_HARD_CAP and not chunked:
        raise ResourceRefusal(
            f"REFUSED. {task} for n={n_items:,} would need a dense "
            f"{n_items:,} x {n_items:,} matrix "
            f"({estimate_dense_gb((n_items, n_items), dtype=dtype):.1f} GiB in "
            f"{np.dtype(dtype).name}). Loopwise never materializes an n x n "
            f"matrix above n={PAIRWISE_HARD_CAP:,} without explicit chunking. "
            "Either call this path with chunked=True (the chunked greedy "
            "facility-location / k-DPP implementation gives identical "
            "selections), or pre-filter the pool to the top "
            f"{PAIRWISE_HARD_CAP:,} candidates by model score before selecting."
        )
    report = check_resources(
        (n_items, n_items),
        dtype=dtype,
        copies=2,  # the matrix plus a working buffer
        task=task,
        n_jobs=n_jobs,
        allow_chunking=chunked,
        remedy=(
            "Pre-filter the pool by model score, or use chunked=True so the "
            "similarity matrix is streamed in row blocks."
        ),
    )
    return report.raise_if_refused()


def chunk_slices(n_rows: int, chunk_rows: int | None) -> Iterator[slice]:
    """Yield ``slice`` objects covering ``n_rows`` in blocks.

    ``chunk_rows=None`` yields a single slice covering everything, so callers
    can use the same loop whether or not chunking was requested.
    """
    n_rows = int(n_rows)
    if not chunk_rows or chunk_rows >= n_rows:
        yield slice(0, n_rows)
        return
    for start in range(0, n_rows, int(chunk_rows)):
        yield slice(start, min(start + int(chunk_rows), n_rows))


def as_feature_matrix(
    frame: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    *,
    dtype: type = DEFAULT_DTYPE,
    exclude: Sequence[str] = (),
) -> tuple[np.ndarray, list[str]]:
    """Coerce a DataFrame to a contiguous ``float32`` design matrix.

    Returns ``(X, used_columns)``. If ``feature_cols`` is None, every numeric
    column not in ``exclude`` is used -- which is why Campaign always passes
    its id/label/cost columns as ``exclude``: silently training on the label
    column is the most expensive bug in this kind of code.

    Non-finite values are replaced with 0.0 after a warning. Screening data
    arrives with NaNs from failed descriptor calculations, and a policy that
    returns NaN scores selects an arbitrary batch without complaining.
    """
    if feature_cols is None:
        numeric = frame.select_dtypes(include=[np.number])
        cols = [c for c in numeric.columns if c not in set(exclude)]
        if not cols:
            raise ValueError(
                "No numeric feature columns found. Pass feature_cols=[...] "
                "explicitly, or add numeric descriptor columns to the pool."
            )
    else:
        cols = list(feature_cols)
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"feature_cols not present in pool: {missing}")

    X = np.ascontiguousarray(frame[cols].to_numpy(dtype=dtype, copy=True))
    bad = ~np.isfinite(X)
    if bad.any():
        warnings.warn(
            f"{int(bad.sum())} non-finite value(s) in features "
            f"({bad.any(axis=0).sum()} affected column(s)); replacing with 0.0. "
            "Impute deliberately upstream if that is not what you want.",
            RuntimeWarning,
            stacklevel=2,
        )
        X[bad] = 0.0
    return X, cols


def make_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a NumPy ``Generator``, accepting a seed, a Generator, or None.

    Loopwise threads one ``Generator`` through a whole campaign rather than
    calling the global ``np.random`` functions. That is what makes replay
    reproducible: the global state can be perturbed by any library that
    happens to draw a random number, a Generator cannot.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def normalize_scores(values: np.ndarray, method: str = "rank") -> np.ndarray:
    """Map arbitrary scores onto [0, 1] so they can be mixed.

    The hybrid policy adds an exploitation score to an exploration score. Those
    live on incomparable scales -- a predicted probability is in [0, 1] while
    a BALD score is in nats -- so ``alpha * exploit + (1 - alpha) * explore``
    is meaningless on raw values. Two ways to fix that:

    ``"rank"`` (default)
        Replace each value by its normalized rank. Scale-free and robust to the
        outliers that dominate min-max on screening data. The cost is that it
        discards magnitude: the top compound and the runner-up are one rank
        apart whether the gap is 0.001 or 0.5.
    ``"minmax"``
        Linear rescaling. Keeps magnitude, but one extreme value squashes
        everything else toward zero.
    """
    values = np.asarray(values, dtype=DEFAULT_DTYPE).ravel()
    if values.size == 0:
        return values
    if method == "minmax":
        lo, hi = float(np.min(values)), float(np.max(values))
        if hi - lo < 1e-12:
            return np.full_like(values, 0.5)
        return (values - lo) / (hi - lo)
    if method == "rank":
        if values.size == 1:
            return np.full_like(values, 0.5)
        order = np.argsort(values, kind="stable")
        ranks = np.empty(values.size, dtype=DEFAULT_DTYPE)
        ranks[order] = np.arange(values.size, dtype=DEFAULT_DTYPE)
        # Average ties so that identical scores get identical normalized ranks.
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        if counts.max() > 1:
            sums = np.zeros(counts.size, dtype=np.float64)
            np.add.at(sums, inverse, ranks.astype(np.float64))
            ranks = (sums / counts)[inverse].astype(DEFAULT_DTYPE)
        return ranks / DEFAULT_DTYPE(values.size - 1)
    raise ValueError(f"Unknown normalization method {method!r}; use 'rank' or 'minmax'.")
