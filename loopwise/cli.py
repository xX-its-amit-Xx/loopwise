"""Command-line interface: ``loopwise <command>``.

Three commands, each the thing you would otherwise write a throwaway script
for:

``loopwise select``
    Select a batch from a pool CSV and write the batch, the plate map and the
    plate figure. The whole point of Loopwise in one command.
``loopwise replay``
    Run a retrospective comparison of policies on a labelled CSV (or one of the
    bundled public datasets) and write the comparison figure.
``loopwise check``
    Print what this machine can handle: RAM, CPUs, and the pool size at which
    the diversity policies will start chunking or refuse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--features", help="comma-separated feature columns")
    parser.add_argument("--id-col", default=None, help="compound id column")
    parser.add_argument("--seed", type=int, default=0)


def cmd_select(args: argparse.Namespace) -> int:
    from sklearn.ensemble import RandomForestClassifier

    from .campaign import Campaign
    from .cost import CostModel, ReplicatePolicy
    from .plate import PLATE_96, PLATE_384

    pool = pd.read_csv(args.pool)
    feature_cols = args.features.split(",") if args.features else None
    labels = None
    if args.labels:
        label_frame = pd.read_csv(args.labels)
        id_col = args.id_col or label_frame.columns[0]
        labels = dict(zip(label_frame[id_col], label_frame[args.label_col], strict=True))

    template = (
        PLATE_384 if args.plate == "384" else (None if args.plate == "none" else PLATE_96)
    )
    cost_model = CostModel(
        per_item_cost=args.cost_col if args.cost_col else 0.0,
        plate_fixed_cost=args.plate_cost,
        usable_wells_per_plate=template.usable_wells if template else 0,
        budget_currency=args.budget_currency,
        replicates=ReplicatePolicy(n_replicates=args.replicates),
    )
    campaign = Campaign(
        pool=pool,
        model=RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1),
        budget=args.budget,
        policy=args.policy,
        feature_cols=feature_cols,
        id_col=args.id_col,
        cost_col=args.cost_col,
        cost_model=cost_model,
        plate=template,
        initial_labels=labels,
        seed=args.seed,
        n_rounds_planned=args.n_rounds,
        n_bootstrap=args.n_bootstrap,
        chunked=args.chunked,
    )
    batch = campaign.select()
    print(batch.explain())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    batch.to_csv(str(out / "batch.csv"))
    print(f"\nwrote {out / 'batch.csv'}")
    if batch.plate_layout is not None:
        batch.plate_layout.to_csv(str(out / "plate_map.csv"))
        print(f"wrote {out / 'plate_map.csv'}")
        if not args.no_figure:
            import matplotlib

            matplotlib.use("Agg")
            batch.save_plate_map(figure_path=str(out / "plate_map.png"))
            print(f"wrote {out / 'plate_map.png'}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    from sklearn.ensemble import RandomForestClassifier

    from .replay import compare_policies, plot_comparison

    if args.dataset == "tox21":
        from .datasets import load_tox21

        pool, info = load_tox21(args.target, n_bits=args.n_bits)
        feature_cols, id_col, label_col = info["feature_cols"], "mol_id", "label"
    elif args.dataset == "bace":
        from .datasets import load_bace

        pool, info = load_bace()
        feature_cols, id_col, label_col = info["feature_cols"], "compound_id", "label"
    elif args.dataset == "synthetic":
        from .datasets import make_synthetic_pool

        pool, info = make_synthetic_pool(seed=args.seed)
        pool = pool.drop(columns=["latent_activity"])
        feature_cols, id_col, label_col = info["feature_cols"], "item_id", "label"
    else:
        pool = pd.read_csv(args.dataset)
        info = {"dataset": args.dataset}
        feature_cols = args.features.split(",") if args.features else None
        id_col, label_col = args.id_col, args.label_col

    print(json.dumps({k: v for k, v in info.items() if k != "feature_cols"}, indent=1))
    frame, results = compare_policies(
        pool,
        label_col=label_col,
        model_factory=lambda: RandomForestClassifier(
            n_estimators=100, random_state=0, n_jobs=2
        ),
        policies=args.policies.split(","),
        seeds=tuple(range(args.n_seeds)),
        feature_cols=feature_cols,
        id_col=id_col,
        budget=args.budget,
        n_rounds=args.n_rounds,
        n_initial_labels=args.n_initial,
        verbose=True,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(str(out / "replay_rounds.csv"), index=False)
    n_actives = next(iter(results.values()))[0].n_actives
    plot_comparison(
        frame,
        n_actives=n_actives,
        path=str(out / "replay_comparison.png"),
        title=f"Replay: {info.get('dataset', args.dataset)}",
    )
    print(f"\nwrote {out / 'replay_rounds.csv'} and {out / 'replay_comparison.png'}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from .utils import (
        PAIRWISE_HARD_CAP,
        _memory_snapshot,
        check_resources,
        estimate_dense_gb,
    )

    total, available, cpus = _memory_snapshot()
    print(f"RAM: {available:.1f} GiB available of {total:.1f} GiB total")
    print(f"CPUs: {cpus} (Loopwise caps parallelism at {max(1, cpus - 1)} under pressure)")
    print(f"Dense n x n hard cap: n = {PAIRWISE_HARD_CAP:,} without chunked=True\n")
    print(f"{'pool size':>12} {'n x n float32':>15} {'decision':>10}")
    for n in (1000, 5000, 10_000, 20_000, 50_000, 100_000):
        report = check_resources((n, n), copies=2, task=f"n={n}")
        gb = estimate_dense_gb((n, n))
        flag = report.decision
        if n > PAIRWISE_HARD_CAP:
            flag = f"{flag}/capped"
        print(f"{n:>12,} {gb:>12.2f} GiB {flag:>10}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="loopwise",
        description=("Active-learning experiment selection for small-n biological assays."),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_select = sub.add_parser("select", help="select a batch from a pool CSV")
    p_select.add_argument("pool", help="CSV of candidate compounds")
    p_select.add_argument("--policy", default="hybrid")
    p_select.add_argument("--budget", type=int, default=96)
    p_select.add_argument("--labels", help="CSV of measurements already in hand")
    p_select.add_argument("--label-col", default="label")
    p_select.add_argument("--cost-col", default=None)
    p_select.add_argument("--plate", default="96", choices=["96", "384", "none"])
    p_select.add_argument("--plate-cost", type=float, default=0.0)
    p_select.add_argument("--budget-currency", type=float, default=None)
    p_select.add_argument("--replicates", type=int, default=1)
    p_select.add_argument("--n-rounds", type=int, default=None)
    p_select.add_argument("--n-bootstrap", type=int, default=None)
    p_select.add_argument("--chunked", action="store_true")
    p_select.add_argument("--no-figure", action="store_true")
    p_select.add_argument("--out", default="loopwise_out")
    _add_common(p_select)
    p_select.set_defaults(func=cmd_select)

    p_replay = sub.add_parser("replay", help="compare policies retrospectively")
    p_replay.add_argument(
        "dataset", help="'tox21', 'bace', 'synthetic', or a path to a labelled CSV"
    )
    p_replay.add_argument("--target", default="NR-AR", help="Tox21 assay")
    p_replay.add_argument("--n-bits", type=int, default=512)
    p_replay.add_argument("--label-col", default="label")
    p_replay.add_argument(
        "--policies", default="random,greedy,uncertainty,bald,thompson,hybrid"
    )
    p_replay.add_argument("--budget", type=int, default=96)
    p_replay.add_argument("--n-rounds", type=int, default=8)
    p_replay.add_argument("--n-initial", type=int, default=96)
    p_replay.add_argument("--n-seeds", type=int, default=3)
    p_replay.add_argument("--out", default="loopwise_out")
    _add_common(p_replay)
    p_replay.set_defaults(func=cmd_replay)

    p_check = sub.add_parser("check", help="report this machine's resource limits")
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
