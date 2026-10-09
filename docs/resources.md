# Memory, CPU, and the resource guard

Loopwise is meant to run on the laptop next to the plate reader. The policies
that are most useful on a clustered compound library -- facility location and
k-DPP -- are also the ones that want to build a matrix of every compound
against every other compound. This document explains what that costs, what
Loopwise does about it, and what to do when it refuses.

## The problem in one line

A similarity matrix over `n` candidates holds `n^2` numbers:

| Pool size | Dense `n x n`, float32 | Realistic outcome |
| --- | --- | --- |
| 1,000 | 4 MB | fine |
| 5,000 | 95 MB | fine |
| 10,000 | 0.37 GB | fine on most machines |
| 20,000 | 1.5 GB | tight on a 8 GB laptop with a browser open |
| 50,000 | 9.3 GB | swap, then the OOM killer |
| 100,000 | 37 GB | immediate failure |

The unguarded failure mode is the bad one: the machine becomes unresponsive for
several minutes, then the process is killed by the operating system with no
message. You learn nothing about what went wrong, and if it happened overnight
you have lost the run.

## The contract

One function, called by every code path whose memory use depends on pool size:

```python
from loopwise.utils import check_resources, chunk_slices

report = check_resources(
    (n_rows, n_cols),       # what is about to be allocated
    dtype=np.float32,       # the default throughout Loopwise
    copies=2,               # how many arrays this size are held at once
    task="k-DPP kernel",    # appears in the message
    n_jobs=None,            # None/-1 means "all cores"
)
report.raise_if_refused()

for block in chunk_slices(n_rows, report.recommended_chunk_rows):
    ...                     # same loop whether or not chunking was needed
```

`check_resources` measures what the machine actually has, via `psutil`, and
returns a decision. It measures **available** RAM rather than total, because
that is the number that matters -- a browser and a chat client are typically
already holding several gigabytes.

| Fraction of available RAM | `decision` | What happens |
| --- | --- | --- |
| < 50% | `ok` or `downcast` | allocate; a float64 request comes back recommending float32 |
| >= 50% | `chunk` | stream `recommended_chunk_rows` rows at a time; `n_jobs` capped at `nproc - 1` |
| >= 90% | `refuse` | `ResourceRefusal` is raised, with a remedy named |

`copies` is not padding. A pairwise computation typically holds the matrix plus
a working buffer; an argsort holds an index array of the same length. Passing
`copies=2` is the honest estimate.

### Why cap at `nproc - 1`

Each worker in a chunked computation holds its own block, so parallelism
multiplies peak memory. And a machine with every core saturated stops
responding to the person using it. Leaving one core free costs a few percent of
throughput and keeps the laptop usable.

### Why float32 is the default

Screening data has two or three significant figures. A plate reader reporting
fluorescence to seven decimal places is reporting noise. float64 doubles the
memory for precision nobody has. `utils.DEFAULT_DTYPE` is `np.float32`, and the
places that promote to float64 do it inside a numerically sensitive computation
(entropy, log-determinants) and say so in a comment.

## The hard cap

Separately from the memory check:

> A dense `n x n` matrix is **never** materialized for `n > 20,000` unless the
> caller passes `chunked=True`.

This fires regardless of free memory. On a machine with a terabyte of RAM it
still refuses. The reason: a 400-million-entry matrix means the caller has
almost certainly not thought about scaling, and the chunked path is available,
correct, and nearly as fast.

```python
from loopwise.utils import guard_pairwise_matrix, ResourceRefusal

try:
    guard_pairwise_matrix(50_000, chunked=False)
except ResourceRefusal as exc:
    print(exc)
```

```
REFUSED. pairwise similarity matrix for n=50,000 would need a dense
50,000 x 50,000 matrix (9.3 GiB in float32). Loopwise never materializes an
n x n matrix above n=20,000 without explicit chunking. Either call this path
with chunked=True (the chunked greedy facility-location / k-DPP implementation
gives identical selections), or pre-filter the pool to the top 20,000
candidates by model score before selecting.
```

Every refusal names what was requested, what the machine has, and what to do
instead. A refusal that does not name a remedy is treated as a bug.

## What `chunked=True` actually does

The two diversity objectives behave differently, and the difference matters:

**k-DPP: exact, and genuinely cheap.** The fast greedy MAP algorithm (Chen et
al., 2018) needs one kernel *column* per selection step, not the whole matrix.
It keeps an `n x k` working array -- for 50,000 candidates and 96 wells that is
18 MB -- and streams each column in row blocks. The chunked path produces
**bit-identical** selections to the dense path, and
`tests/test_resources.py::test_chunked_kdpp_matches_the_unchunked_selection`
asserts exactly that. The claim in the refusal message is tested, not asserted.

**Facility location: approximate, and it says so.** The coverage objective sums
over every pool member, so it cannot avoid touching all of them. With chunking,
Loopwise subsamples the reference set that the sum runs over
(`max_reference_set`, default 4,000) and uses stochastic greedy (Mirzasoleiman
et al., 2015), which evaluates a random candidate subset per step for a
`(1 - 1/e - epsilon)` guarantee. The result depends on the seed. Selections
report `"exact": False` and a note in `diagnostics`, so a figure made from a
chunked run can be labelled honestly.

```python
batch = campaign.select()
print(batch.diagnostics["exact"], batch.diagnostics["note"])
```

## What to do when it refuses

In rough order of preference:

1. **Pre-filter the pool.** Rank by model score and keep the top 10,000-20,000.
   You were never going to select a compound from the bottom of the ranking
   anyway, and this is almost always the right answer.
2. **Pass `chunked=True`** (`Campaign(..., chunked=True)`). For k-DPP this is
   free and exact. For facility location, note the approximation.
3. **Use a pointwise policy.** `bald`, `uncertainty`, `greedy` and `thompson`
   are all `O(n)` in the pool and do not build a similarity matrix at all.
4. **Reduce the feature dimension.** 2,048-bit fingerprints make every kernel
   evaluation eight times more expensive than 256-bit ones, and for diversity
   purposes the extra bits rarely change which compounds are near-duplicates.

## Other guarded paths

**Gaussian processes** (`contrib.GaussianProcessAdapter`) invert an `n x n`
matrix over the *training* set, which is `O(n^3)` time. In an active-learning
loop the training set grows every round, so a GP that was fast in round 1 can
hang in round 8. The adapter refuses above `max_train=5000` with a message
naming the cubic cost and the linear-cost alternatives. This is a soft limit
you can raise, deliberately.

**Posterior sampling** allocates `(n_samples, n_items)`. With 50 samples over
100,000 candidates that is 20 MB -- usually fine, and guarded anyway.

**Fingerprint matrices** (`datasets.featurize_smiles`) allocate
`(n_molecules, n_bits)` and are guarded before featurization starts, so a
too-large request fails in a second rather than after twenty minutes of RDKit.

## Check your own machine

```
$ loopwise check
RAM: 14.8 GiB available of 15.7 GiB total
CPUs: 4 (Loopwise caps parallelism at 3 under pressure)
Dense n x n hard cap: n = 20,000 without chunked=True

   pool size   n x n float32   decision
       1,000         0.00 GiB         ok
       5,000         0.09 GiB         ok
      10,000         0.37 GiB         ok
      20,000         1.49 GiB         ok
      50,000         9.31 GiB refuse/capped
     100,000        37.25 GiB refuse/capped
```

## If you are adding code

Call `check_resources()` before the allocation, not after. Honour the returned
decision -- a guard whose verdict is ignored is worse than no guard, because it
produces a log line claiming the problem was handled. And if you add a new
`O(n^2)` path, add it to this document.

See `CLAUDE.md` and `AGENTS.md` for the rules as enforced in review.
