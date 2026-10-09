"""Public datasets for retrospective replay, with featurization.

Two real, fully measured public datasets ship as loaders. Both are downloaded
on first use and cached under ``~/.cache/loopwise`` (override with
``LOOPWISE_DATA_DIR``); neither is redistributed in this repository.

``load_tox21(target="NR-AR")``
    The Tox21 Data Challenge (2014): ~7,800 compounds screened against 12
    nuclear-receptor and stress-response assays. We use one assay at a time as
    a binary screen. NR-AR -- the androgen receptor -- has a 4.2% hit rate over
    7,265 measured compounds, which is realistically scarce: a greedy policy
    can look excellent while learning nothing, which is exactly the regime
    Loopwise is about. Distributed by MoleculeNet (Wu et al., 2018).

``load_bace()``
    1,513 inhibitors of human beta-secretase 1 (BACE-1) with measured IC50
    values, from Subramanian et al. (2016). This is a **dose-response** screen,
    not a yes/no one: each compound has a continuous potency, so it exercises
    the regression path and the "how many wells to find the potent ones"
    question rather than hit-counting alone. The published set ships with
    pre-computed molecular descriptors, so it works without a chemistry
    toolkit installed.

Featurization
-------------
Tox21 ships as SMILES strings -- text like ``CC(=O)Oc1ccccc1C(=O)O`` for
aspirin -- and a model cannot consume text. Two options:

* **Morgan fingerprints** (RDKit, if installed). A fingerprint asks, for every
  atom, "what does the neighbourhood within 2 bonds of this atom look like?",
  hashes each answer into one of ``n_bits`` buckets, and sets that bit. Two
  molecules sharing many substructures share many bits. This is the standard
  representation for ligand-based screening, introduced as ECFP by Rogers &
  Hahn (2010).
* **Hashed-token fallback** (no RDKit). Character n-grams of the SMILES string,
  hashed to bits. Crude -- it has no idea what a ring is -- but it is a real,
  reproducible featurization that lets the replay run anywhere, and policies
  are compared against each other on the *same* features, so the comparison
  stays fair even where the absolute numbers suffer.

Both paths are deterministic: same input, same bits, every time and on every
machine.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DEFAULT_DTYPE, check_resources

__all__ = [
    "TOX21_TARGETS",
    "data_dir",
    "featurize_smiles",
    "load_bace",
    "load_tox21",
    "make_synthetic_pool",
]

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
BACE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"

#: The 12 Tox21 assays, in the order the challenge lists them.
TOX21_TARGETS = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)


def data_dir() -> Path:
    """Cache directory for downloaded datasets."""
    path = Path(os.environ.get("LOOPWISE_DATA_DIR", Path.home() / ".cache" / "loopwise"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, filename: str) -> Path:
    target = data_dir() / filename
    if target.exists() and target.stat().st_size > 0:
        return target
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        tmp.replace(target)
    except Exception as exc:  # pragma: no cover - network dependent
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Could not download {url}: {exc}\n"
            f"Download it manually and place it at {target}, or set "
            "LOOPWISE_DATA_DIR to a directory that already contains it."
        ) from exc
    return target


# ---------------------------------------------------------------- featurizers


def _morgan_fingerprints(
    smiles: list[str], n_bits: int, radius: int
) -> tuple[np.ndarray, list[bool]]:
    """Morgan/ECFP fingerprints via RDKit. Returns (bits, parse_ok flags)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")  # the pool will contain unparseable SMILES
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=DEFAULT_DTYPE)
    ok = []
    for i, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(text))
        if mol is None:
            ok.append(False)
            continue
        fp = generator.GetFingerprintAsNumPy(mol)
        out[i] = fp.astype(DEFAULT_DTYPE)
        ok.append(True)
    return out, ok


def _hashed_ngrams(smiles: list[str], n_bits: int, ngram: int = 4) -> np.ndarray:
    """Deterministic hashed character n-grams -- the no-RDKit fallback.

    Uses BLAKE2b rather than Python's ``hash()``, which is randomized per
    process: with ``hash()`` the features would differ between runs and replay
    would not be reproducible.
    """
    out = np.zeros((len(smiles), n_bits), dtype=DEFAULT_DTYPE)
    for i, text in enumerate(smiles):
        s = str(text)
        for size in (1, 2, ngram):
            for j in range(max(len(s) - size + 1, 0)):
                token = s[j : j + size].encode("utf-8")
                digest = hashlib.blake2b(token, digest_size=8).digest()
                out[i, int.from_bytes(digest, "big") % n_bits] = 1.0
    return out


def featurize_smiles(
    smiles: list[str],
    *,
    n_bits: int = 512,
    radius: int = 2,
    use_rdkit: bool | None = None,
) -> tuple[np.ndarray, list[str], dict]:
    """Featurize SMILES to a binary matrix. Returns ``(X, column_names, info)``."""
    check_resources(
        (len(smiles), n_bits), task="fingerprint matrix", copies=2
    ).raise_if_refused()

    if use_rdkit is None:
        try:
            import rdkit  # noqa: F401

            use_rdkit = True
        except ImportError:
            use_rdkit = False

    if use_rdkit:
        X, ok = _morgan_fingerprints(smiles, n_bits, radius)
        info = {
            "featurizer": f"morgan_ecfp{2 * radius}",
            "n_bits": n_bits,
            "radius": radius,
            "n_unparseable": int(len(ok) - sum(ok)),
            "parse_ok": ok,
        }
    else:
        X = _hashed_ngrams(smiles, n_bits)
        info = {
            "featurizer": "hashed_char_ngrams",
            "n_bits": n_bits,
            "n_unparseable": 0,
            "parse_ok": [True] * len(smiles),
            "warning": (
                "RDKit is not installed, so SMILES were featurized as hashed "
                "character n-grams. Policies are still compared on identical "
                "features, but absolute model quality is well below what "
                "Morgan fingerprints give. Install rdkit for real numbers."
            ),
        }
    return X, [f"fp_{i}" for i in range(n_bits)], info


# -------------------------------------------------------------------- loaders


def load_tox21(
    target: str = "NR-AR",
    *,
    n_bits: int = 512,
    radius: int = 2,
    max_rows: int | None = None,
    use_rdkit: bool | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Load one Tox21 assay as a binary screening pool.

    Returns ``(pool, info)`` where ``pool`` has ``mol_id``, ``smiles``,
    ``label`` (0/1) and ``fp_*`` feature columns, with unmeasured compounds for
    this assay dropped -- Tox21 is sparse, and a compound with no measurement
    cannot serve as replay ground truth.
    """
    if target not in TOX21_TARGETS:
        raise ValueError(f"target must be one of {TOX21_TARGETS}, got {target!r}")
    path = _download(TOX21_URL, "tox21.csv.gz")
    raw = pd.read_csv(path)
    frame = raw.dropna(subset=[target]).reset_index(drop=True)
    if max_rows is not None and len(frame) > max_rows:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(frame), size=max_rows, replace=False)
        frame = frame.iloc[np.sort(keep)].reset_index(drop=True)

    X, cols, info = featurize_smiles(
        frame["smiles"].tolist(), n_bits=n_bits, radius=radius, use_rdkit=use_rdkit
    )
    ok = np.asarray(info.pop("parse_ok"), dtype=bool)
    pool = pd.DataFrame(
        {
            "mol_id": frame["mol_id"].astype(str),
            "smiles": frame["smiles"].astype(str),
            "label": frame[target].astype(int),
        }
    )
    pool = pd.concat([pool, pd.DataFrame(X, columns=cols)], axis=1)
    pool = pool.loc[ok].reset_index(drop=True)

    info.update(
        {
            "dataset": "Tox21",
            "target": target,
            "source_url": TOX21_URL,
            "citation": (
                "Tox21 Data Challenge 2014; distributed via MoleculeNet "
                "(Wu et al., Chem. Sci. 2018). Public domain (US government)."
            ),
            "n_compounds": len(pool),
            "n_actives": int(pool["label"].sum()),
            "hit_rate": round(float(pool["label"].mean()), 4),
            "task": "classification",
            "feature_cols": cols,
        }
    )
    return pool, info


def load_bace(
    *,
    use_descriptors: bool = True,
    n_bits: int = 512,
    radius: int = 2,
    potency_threshold: float = 7.0,
    use_rdkit: bool | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load the BACE-1 dose-response set.

    Returns ``(pool, info)`` with both label columns, so one dataset serves
    two questions:

    ``pIC50``
        The continuous potency -- negative log10 of the IC50 concentration, so
        7.0 means 100 nM and each unit is a tenfold improvement. Use with
        ``task="regression"``.
    ``label``
        ``pIC50 >= potency_threshold`` (default 7.0, i.e. 100 nM or better).
        The dose-response analogue of "is this a hit?", and the reason the
        threshold is a parameter: where you put it decides your hit rate, and
        a campaign's apparent success is partly a choice of threshold.

    ``use_descriptors=True`` uses the 590 pre-computed physicochemical
    descriptors shipped with the dataset (works with no chemistry toolkit);
    False computes Morgan fingerprints from the SMILES instead.
    """
    path = _download(BACE_URL, "bace.csv")
    raw = pd.read_csv(path)
    smiles_col = "mol" if "mol" in raw.columns else "smiles"

    pool = pd.DataFrame(
        {
            "compound_id": raw["CID"].astype(str)
            if "CID" in raw.columns
            else [f"BACE{i:04d}" for i in range(len(raw))],
            "smiles": raw[smiles_col].astype(str),
            "pIC50": raw["pIC50"].astype(float),
        }
    )
    pool["label"] = (pool["pIC50"] >= potency_threshold).astype(int)

    if use_descriptors:
        drop = {smiles_col, "CID", "Class", "Model", "pIC50"}
        numeric = raw.drop(columns=[c for c in drop if c in raw.columns]).select_dtypes(
            include=[np.number]
        )
        # Constant columns carry no information and break standardization.
        numeric = numeric.loc[:, numeric.std(numeric_only=True) > 1e-8]
        feature_cols = [f"desc_{c}" for c in numeric.columns]
        features = pd.DataFrame(numeric.to_numpy(dtype=DEFAULT_DTYPE), columns=feature_cols)
        info = {"featurizer": "published_descriptors", "n_features": len(feature_cols)}
        ok = np.ones(len(pool), dtype=bool)
    else:
        X, feature_cols, info = featurize_smiles(
            pool["smiles"].tolist(), n_bits=n_bits, radius=radius, use_rdkit=use_rdkit
        )
        ok = np.asarray(info.pop("parse_ok"), dtype=bool)
        features = pd.DataFrame(X, columns=feature_cols)

    pool = pd.concat([pool, features], axis=1).loc[ok].reset_index(drop=True)
    pool = pool.replace([np.inf, -np.inf], np.nan).fillna({c: 0.0 for c in feature_cols})

    info.update(
        {
            "dataset": "BACE-1",
            "source_url": BACE_URL,
            "citation": (
                "Subramanian, Ramsundar, Pande & Denny, J. Chem. Inf. Model. "
                "56(10):1936-1949 (2016); distributed via MoleculeNet "
                "(Wu et al., Chem. Sci. 2018)."
            ),
            "n_compounds": len(pool),
            "n_actives": int(pool["label"].sum()),
            "hit_rate": round(float(pool["label"].mean()), 4),
            "potency_threshold_pIC50": potency_threshold,
            "task": "regression or classification",
            "feature_cols": feature_cols,
        }
    )
    return pool, info


def make_synthetic_pool(
    n_items: int = 2000,
    n_features: int = 20,
    *,
    hit_rate: float = 0.05,
    n_clusters: int = 12,
    noise: float = 0.6,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """A toy pool with the two features that make screening hard.

    Used by the quickstart notebook and the tests, so neither needs network
    access. Deliberately includes:

    * **Clustering.** Compounds come in ``n_clusters`` structural families, as
      real libraries do (analogue series). This is what makes a diversity
      constraint matter: without clusters, the top-k by score are already
      spread out and diversity policies have nothing to fix.
    * **A low hit rate.** Activity depends on a few features plus noise, with
      the threshold set to give the requested hit rate. At 5%, a 96-well greedy
      batch finding 20 hits looks like a triumph and still teaches the model
      almost nothing.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=2.5, size=(n_clusters, n_features))
    assignment = rng.integers(0, n_clusters, size=n_items)
    X = centers[assignment] + rng.normal(scale=0.8, size=(n_items, n_features))

    weights = np.zeros(n_features)
    active_features = rng.choice(n_features, size=max(3, n_features // 5), replace=False)
    weights[active_features] = rng.normal(scale=1.5, size=active_features.size)
    latent = X @ weights + rng.normal(scale=noise, size=n_items)
    threshold = float(np.quantile(latent, 1.0 - hit_rate))

    pool = pd.DataFrame(
        X.astype(DEFAULT_DTYPE), columns=[f"f{i}" for i in range(n_features)]
    )
    pool.insert(0, "item_id", [f"CPD{i:05d}" for i in range(n_items)])
    pool["cluster"] = assignment
    pool["label"] = (latent > threshold).astype(int)
    pool["latent_activity"] = latent.astype(DEFAULT_DTYPE)
    info = {
        "dataset": "synthetic",
        "n_compounds": n_items,
        "n_actives": int(pool["label"].sum()),
        "hit_rate": round(float(pool["label"].mean()), 4),
        "n_clusters": n_clusters,
        "informative_features": sorted(int(i) for i in active_features),
        "feature_cols": [f"f{i}" for i in range(n_features)],
        "task": "classification",
    }
    return pool, info
