# AGENTS.md

Instructions for autonomous coding agents working in this repository.
`CLAUDE.md` holds the full design rationale; this file is the short, strict
version. Where they overlap, they agree -- if you find them disagreeing, that
is a bug worth reporting.

## Build, test, lint

```bash
pip install -e ".[dev]"      # core + pytest; this is what CI's gate uses
pytest -m "not network"      # MUST pass before any commit
ruff check loopwise tests    # MUST be clean
ruff format loopwise tests   # formatting is enforced
```

`pytest -m network` downloads Tox21 and BACE from the MoleculeNet mirror. It is
a separate CI job marked `continue-on-error`, because an upstream mirror outage
is not this repo's bug. Do not move network tests into the default run.

Optional dependencies (LightGBM, PyTorch, RDKit) must stay optional:
`import loopwise.contrib` has to succeed without them, and their adapters must
`pytest.importorskip` in tests.

## Hard rules

1. **Call `check_resources()` before any allocation whose size depends on the
   pool size.** See the table below. A new code path that allocates an
   `O(n^2)` array without a guard will be reverted.

2. **Never materialize a dense `n x n` matrix for `n > 20_000` without
   `chunked=True`.** `guard_pairwise_matrix()` enforces this and fires
   regardless of free RAM. The k-DPP and facility-location paths are the ones
   this applies to.

3. **Never weaken these tested invariants:**
   - every policy returns exactly `budget` unique items
   - no sample ever occupies a reserved control well; every plate carries a
     full set of controls
   - replay is bit-reproducible under a fixed seed and unaffected by the global
     NumPy random state
   - the resource guard refuses an oversized DPP with an actionable message
   - the pool DataFrame is never mutated

4. **Take randomness from the passed-in `Generator`** (`ctx.rng`,
   `campaign.rng`, `make_rng(seed)`). Never call `np.random.*`. A single
   global-state draw breaks replay determinism, which is the repo's evidence.

5. **Raise rather than silently degrade.** A model that cannot express
   uncertainty must make information-seeking policies raise
   `DegeneratePosteriorError`, not return an arbitrary batch. Refusal messages
   state what was requested, what the machine has, and what to do instead.

6. **float32 everywhere** (`utils.DEFAULT_DTYPE`). Promote to float64 only
   inside a numerically sensitive computation (entropy, log-determinant), and
   say why in a comment.

7. **Do not report single-seed policy comparisons**, in code, docs, notebooks,
   or commit messages. Seed spread is often as large as the between-policy gap.

8. **Keep the stand-ins labelled.** `SyntheticFoldRescueModel` and
   `make_glb1_like_pool` are simulators. Never present their output as GLB1
   predictions or as FoldRescue results.

## The resource-guard contract

```python
report = check_resources(shape, dtype=np.float32, copies=2, task="...", n_jobs=None)
report.raise_if_refused()
for block in chunk_slices(n_rows, report.recommended_chunk_rows):
    ...
```

Measured against **available** RAM via psutil, not total:

| Fraction of available | `report.decision` | Required behaviour |
| --- | --- | --- |
| < 50% | `ok` / `downcast` | allocate; prefer the recommended dtype |
| >= 50% | `chunk` | stream `recommended_chunk_rows`; use `recommended_n_jobs` (<= nproc-1) |
| >= 90% | `refuse` | `raise ResourceRefusal` naming a remedy |

Plus the absolute cap: dense `n x n` refused above n = 20,000 without
`chunked=True`.

## Where things live

| File | Responsibility |
| --- | --- |
| `utils.py` | resource guard, RNG discipline, feature coercion, score normalization |
| `models.py` | `Posterior` and the adapter interface every policy reads |
| `policies.py` | the eight selection policies |
| `cost.py` | per-compound and per-plate costs, replicate policy, currency budget |
| `plate.py` | 96/384 templates, control reservation, edge-effect-aware layout |
| `batch.py` | reconciling budget, cost and plate capacity into a batch size |
| `campaign.py` | `select` / `observe` / `report` state machine and metrics |
| `replay.py` | retrospective comparison on measured datasets |
| `datasets.py` | Tox21, BACE, featurization, synthetic pools |
| `contrib/` | LightGBM, PyTorch, Gaussian process, FoldRescue adapters |

Adding a model family: write one adapter in `contrib/` that produces a
`Posterior`. Do not touch `policies.py`.

Adding a policy: subclass `Policy`, emit a per-item `reason` string, register it
in `POLICY_REGISTRY` and `available_policies()`, and add it to the parametrized
contract tests.

## Commit conventions

- Imperative subject line under ~72 characters, then a body explaining *why*.
- State test results honestly: if something fails or is skipped, say so.
- Do not include model identifiers or assistant names in code, comments, or
  artifacts; attribution goes in commit trailers only.
- Do not open a pull request unless explicitly asked.
