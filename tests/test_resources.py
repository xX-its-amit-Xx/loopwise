"""Resource-guard tests.

The guard's job is to fail *before* allocating, with a message that says what
to do. The most important case: an oversized k-DPP or facility-location
selection over a large pool, which is the one code path in Loopwise that wants
an ``n x n`` matrix. At n = 50,000 that is 9.3 GiB in float32, and the
unguarded outcome is either a swapped-out machine or an OOM kill with no
explanation.
"""

from __future__ import annotations

import numpy as np
import pytest

from loopwise.models import Posterior
from loopwise.policies import DiversityGreedyPolicy, PolicyContext
from loopwise.utils import (
    CHUNK_THRESHOLD,
    DEFAULT_DTYPE,
    PAIRWISE_HARD_CAP,
    ResourceRefusal,
    check_resources,
    chunk_slices,
    estimate_dense_gb,
    guard_pairwise_matrix,
    make_rng,
    normalize_scores,
)

# ------------------------------------------------------------- the hard cap


def test_oversized_dpp_is_refused_without_explicit_chunking():
    n = PAIRWISE_HARD_CAP + 1
    with pytest.raises(ResourceRefusal) as excinfo:
        guard_pairwise_matrix(n, chunked=False)

    message = str(excinfo.value)
    assert "REFUSED" in message
    assert f"{n:,}" in message
    # The message must be actionable: it names both remedies.
    assert "chunked=True" in message
    assert "pre-filter" in message.lower()


def test_hard_cap_fires_regardless_of_available_memory(monkeypatch):
    """The cap is a design limit, not a memory limit.

    Pretend the machine has a terabyte free: a 20,001 x 20,001 dense matrix is
    still refused, because at that size the dense path means the caller has
    not thought about scaling.
    """
    monkeypatch.setattr("loopwise.utils._memory_snapshot", lambda: (1024.0, 1024.0, 64))
    with pytest.raises(ResourceRefusal):
        guard_pairwise_matrix(PAIRWISE_HARD_CAP + 1, chunked=False)

    # Just under the cap, with that much RAM, is fine.
    report = guard_pairwise_matrix(PAIRWISE_HARD_CAP - 1, chunked=False)
    assert report.decision in ("ok", "downcast")


def test_chunked_request_passes_the_hard_cap(monkeypatch):
    monkeypatch.setattr("loopwise.utils._memory_snapshot", lambda: (1024.0, 1024.0, 64))
    report = guard_pairwise_matrix(50_000, chunked=True)
    assert report.decision in ("ok", "downcast", "chunk")


def test_diversity_policy_refuses_an_oversized_pool_end_to_end(monkeypatch):
    """The refusal must reach the caller through the policy, not just the util.

    Uses a fake memory snapshot and a tiny feature matrix with a spoofed row
    count, so the test does not need 20k rows of real data to prove the gate
    fires.
    """
    n = PAIRWISE_HARD_CAP + 500
    rng = make_rng(0)

    class _FakeFeatures(np.ndarray):
        """An array that reports an oversized first dimension."""

    features = rng.normal(size=(50, 4)).astype(DEFAULT_DTYPE)
    posterior = Posterior(rng.random((4, 50)).astype(DEFAULT_DTYPE))

    class _Ctx(PolicyContext):
        @property
        def n_candidates(self):  # pragma: no cover - trivial override
            return n

    ctx = _Ctx(
        features=features,
        posterior=posterior,
        budget=10,
        rng=rng,
        chunked=False,
    )
    for objective in ("kdpp", "facility_location"):
        with pytest.raises(ResourceRefusal, match="REFUSED"):
            DiversityGreedyPolicy(objective=objective).select(ctx)


# ------------------------------------------------------- check_resources core


def test_decisions_track_the_fraction_of_available_ram(monkeypatch):
    monkeypatch.setattr("loopwise.utils._memory_snapshot", lambda: (16.0, 8.0, 4))

    small = check_resources((1000, 1000), task="small")  # 4 MB
    assert small.decision in ("ok", "downcast")
    assert small.recommended_chunk_rows is None

    # ~50% of 8 GiB available -> chunk
    rows = int((CHUNK_THRESHOLD * 8.0 * (1 << 30)) / (4 * 1000)) + 1000
    mid = check_resources((rows, 1000), task="mid")
    assert mid.decision == "chunk"
    assert mid.recommended_chunk_rows is not None and mid.recommended_chunk_rows > 0
    assert mid.recommended_n_jobs <= 3  # capped at cpu_count - 1

    huge = check_resources((200_000, 200_000), task="huge")
    assert huge.decision == "refuse"
    with pytest.raises(ResourceRefusal):
        huge.raise_if_refused()


def test_float64_request_is_downcast(monkeypatch):
    monkeypatch.setattr("loopwise.utils._memory_snapshot", lambda: (16.0, 8.0, 4))
    report = check_resources((2000, 2000), dtype=np.float64, task="f64")
    assert report.decision == "downcast"
    assert report.recommended_dtype is np.float32
    assert "float32" in report.message


def test_refusal_message_is_actionable(monkeypatch):
    monkeypatch.setattr("loopwise.utils._memory_snapshot", lambda: (16.0, 8.0, 4))
    report = check_resources((500_000, 500_000), task="similarity matrix")
    assert report.decision == "refuse"
    for expected in ("REFUSED", "similarity matrix", "GiB", "available"):
        assert expected in report.message
    # It must suggest something concrete to do next.
    assert any(
        hint in report.message.lower()
        for hint in ("reduce", "pre-filter", "lower the batch", "switch to")
    )


def test_no_chunking_available_means_refuse_sooner(monkeypatch):
    monkeypatch.setattr("loopwise.utils._memory_snapshot", lambda: (16.0, 8.0, 4))
    shape = (int(0.7 * 8.0 * (1 << 30) / 4), 1)
    assert check_resources(shape, task="x").decision == "chunk"
    strict = check_resources(shape, task="x", allow_chunking=False)
    assert strict.decision == "refuse"
    assert "cannot stream" in strict.message


def test_cpu_cap_leaves_a_core_free(monkeypatch):
    monkeypatch.setattr("loopwise.utils._memory_snapshot", lambda: (16.0, 2.0, 8))
    # 2e6 x 200 float32 = 1.49 GiB, i.e. 75% of the 2 GiB available.
    report = check_resources((2_000_000, 200), task="cap", n_jobs=-1)
    assert report.decision in ("chunk", "refuse")
    assert report.recommended_n_jobs == 7


def test_estimate_dense_gb():
    assert estimate_dense_gb((1 << 30 // 4,), dtype=np.float32) == pytest.approx(
        (1 << 30 // 4) * 4 / (1 << 30)
    )
    assert estimate_dense_gb((1000, 1000), dtype=np.float32, copies=2) == pytest.approx(
        2 * 1000 * 1000 * 4 / (1 << 30)
    )
    # float64 is exactly twice float32.
    assert estimate_dense_gb((100, 100), dtype=np.float64) == 2 * estimate_dense_gb(
        (100, 100), dtype=np.float32
    )


def test_psutil_absence_degrades_to_caution(monkeypatch):
    monkeypatch.setattr("loopwise.utils._HAVE_PSUTIL", False)
    with pytest.warns(RuntimeWarning, match="psutil"):
        report = check_resources((10, 10), task="no-psutil")
    assert report.ram_available_gb == pytest.approx(4.0)


# ---------------------------------------------------------------- small utils


def test_chunk_slices_covers_everything_exactly():
    assert list(chunk_slices(10, None)) == [slice(0, 10)]
    assert list(chunk_slices(10, 100)) == [slice(0, 10)]
    slices = list(chunk_slices(10, 3))
    assert slices == [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 10)]
    covered = np.concatenate([np.arange(10)[s] for s in slices])
    np.testing.assert_array_equal(covered, np.arange(10))


def test_normalize_scores():
    values = np.array([1.0, 5.0, 100.0], dtype=np.float32)
    ranked = normalize_scores(values, "rank")
    assert ranked.min() == 0.0 and ranked.max() == 1.0
    # Rank normalization is scale-free; min-max is not.
    assert normalize_scores(values, "minmax")[1] < ranked[1]

    ties = normalize_scores(np.array([2.0, 2.0, 2.0], dtype=np.float32), "rank")
    np.testing.assert_allclose(ties, [0.5, 0.5, 0.5])
    assert normalize_scores(np.array([], dtype=np.float32)).size == 0

    with pytest.raises(ValueError):
        normalize_scores(values, "nonsense")


def test_chunked_kdpp_matches_the_unchunked_selection(monkeypatch):
    """Chunking must change memory use, not the answer.

    k-DPP greedy MAP reads one kernel column per step, built in row blocks. The
    block size must not affect the result, or the "chunked path gives identical
    selections" claim in the refusal message would be false.
    """
    rng = make_rng(0)
    features = rng.normal(size=(300, 6)).astype(DEFAULT_DTYPE)
    posterior = Posterior(rng.random((8, 300)).astype(DEFAULT_DTYPE))

    def context():
        return PolicyContext(
            features=features, posterior=posterior, budget=20, rng=make_rng(5)
        )

    unchunked = DiversityGreedyPolicy(objective="kdpp").select(context())

    # The 300 x 300 kernel plus a work buffer is 0.69 MiB; claim only 1 MiB
    # available so the guard is forced down the chunked path.
    monkeypatch.setattr("loopwise.utils._memory_snapshot", lambda: (0.01, 1.0 / 1024, 4))
    chunked_ctx = context()
    chunked_ctx.chunked = True
    chunked = DiversityGreedyPolicy(objective="kdpp").select(chunked_ctx)

    assert chunked.diagnostics["resource_decision"] == "chunk"
    np.testing.assert_array_equal(unchunked.indices, chunked.indices)
