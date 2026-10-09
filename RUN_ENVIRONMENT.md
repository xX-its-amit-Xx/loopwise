# RUN_ENVIRONMENT.md

The exact environment in which the committed cookbook notebooks and the figures
under `results/figures/` were executed. Recorded so the outputs can be
reproduced and so any numbers quoted in the README can be traced to a machine.

## Platform

| | |
| --- | --- |
| Executed | 2026-10-09 |
| Host | Claude Code cloud container (ephemeral Linux sandbox) |
| OS | Linux 6.18.44, x86_64 |
| CPUs | 4 logical cores |
| RAM | 15.7 GiB total, ~14.8 GiB available at run time |
| GPU | none -- everything here runs on CPU, including the PyTorch adapter |
| Python | 3.13.16 (CPython) |

No GPU is needed anywhere in this repository. The PyTorch adapter trains small
MLPs (two hidden layers, width 64); on this machine a 3-member deep ensemble
over a few hundred compounds fits in a couple of seconds.

## Package versions

| Package | Version |
| --- | --- |
| loopwise | 0.1.0 (editable install from this repo) |
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.1 |
| scipy | (resolved with scikit-learn) |
| matplotlib | 3.x |
| psutil | 5.9+ |
| rdkit | 2026.03.6 |
| lightgbm | 4.7.0 |
| torch | 2.14.1+cu130 (CPU execution) |
| nbformat / nbclient | current at run time |

Capture your own with:

```bash
python -c "import numpy, pandas, sklearn, rdkit, lightgbm, torch; \
print(numpy.__version__, pandas.__version__, sklearn.__version__, \
rdkit.__version__, lightgbm.__version__, torch.__version__)"
pip freeze > environment_freeze.txt
```

## Datasets

Downloaded at run time, cached in `$LOOPWISE_DATA_DIR` (default
`~/.cache/loopwise`). Neither is redistributed in this repository.

| Dataset | Source | Size |
| --- | --- | --- |
| Tox21 | `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz` | 7,831 rows; 7,258 with an NR-AR measurement after dropping 7 unparseable SMILES |
| BACE-1 | `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv` | 1,513 compounds, 590 published descriptors |

Both are distributed by MoleculeNet (Wu et al., *Chem. Sci.* 2018). Tox21 is US
government work; BACE is from Subramanian et al., *J. Chem. Inf. Model.*
56(10):1936-1949 (2016).

If the mirror is unreachable, download the files manually and place them in
`$LOOPWISE_DATA_DIR` under the names `tox21.csv.gz` and `bace.csv`.

## Reproducing the notebooks

```bash
pip install -e ".[all]"
python cookbook/build_notebooks.py            # build and execute all four
python cookbook/build_notebooks.py 02         # just the replay notebook
python cookbook/build_notebooks.py --no-exec  # build without executing
```

The notebooks are generated from `cookbook/build_notebooks.py` rather than
hand-edited, so the committed `.ipynb` files are always the executed output of
exactly that source.

Approximate wall-clock on this 4-core machine:

| Notebook | Time |
| --- | --- |
| 01 quickstart | ~1 min |
| 02 replay on public screens | ~25 min (7 policies x 5 seeds x 2 datasets, plus the noise comparison) |
| 03 plate layout | ~20 s |
| 04 driving FoldRescue | ~40 s |

## Determinism

Every number in the committed notebooks is reproducible on the same package
versions. Loopwise threads a single seeded `numpy.random.Generator` through the
campaign, the policy, the model adapters and the plate layout, and never calls
the global `np.random.*` functions -- so an unrelated library drawing a random
number cannot perturb a run.
`tests/test_replay.py::test_seed_determinism_survives_a_foreign_global_rng`
asserts exactly that.

What is *not* guaranteed bit-for-bit across machines: results can differ with a
different BLAS, a different scikit-learn version (tree-building tie-breaks
change between releases), or a different RDKit version (fingerprint generation
has changed across major versions). The qualitative findings in the README --
greedy's per-round collapse, the hits-versus-model-quality trade-off on BACE --
were stable across all five seeds and should not be version-sensitive.

## Figures

`results/figures/` holds every figure the notebooks produce. Filenames are
prefixed with the notebook that generated them. `results/tables/` holds the
per-round replay CSVs and the example plate maps, so the README's tables can be
checked against the data they came from.
