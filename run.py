#!/usr/bin/env python3

import argparse
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of a 1-D array of non-negative counts."""
    a = np.sort(counts.astype(float))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum() / (n * a.sum())) - (n + 1) / n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="guide assignment comparison metrics")
    parser.add_argument(
        "--assignments.h5ad",
        dest="assignments",
        required=True,
        help="h5ad file from one guide-assignment module",
    )
    parser.add_argument("--output_dir", "-o", required=True)
    parser.add_argument("--name", "-n", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.assignments} ...", file=sys.stderr)
    adata = ad.read_h5ad(args.assignments)

    is_unassigned = adata.obs["is_unassigned"].astype(bool).values
    is_multi = adata.obs["is_multi_infected"].astype(bool).values
    n_cells = adata.n_obs
    print(f"  {n_cells} cells, {(~is_unassigned).sum()} assigned", file=sys.stderr)

    X = adata.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float32)

    assigned_mat = None
    if "assigned" in adata.layers:
        a = adata.layers["assigned"]
        assigned_mat = a.toarray() if sp.issparse(a) else np.asarray(a)

    dataset = args.name
    rows = [
        dict(
            dataset=dataset,
            metric="coverage",
            submetric="fraction_assigned",
            value=float((~is_unassigned).mean()),
            n=n_cells,
        ),
        dict(
            dataset=dataset,
            metric="coverage",
            submetric="frac_multi_infected",
            value=float(is_multi.mean()),
            n=n_cells,
        ),
    ]

    if assigned_mat is not None and (~is_unassigned).any():
        X_a = X[~is_unassigned]
        A_a = assigned_mat[~is_unassigned].astype(np.float32)
        total_umi = X_a.sum(axis=1)
        assigned_umi = (X_a * A_a).sum(axis=1)
        valid = total_umi > 0
        frac = np.where(valid, assigned_umi / np.where(valid, total_umi, 1.0), np.nan)
        rows.append(
            dict(
                dataset=dataset,
                metric="umi",
                submetric="mean_assigned_umi_frac",
                value=float(np.nanmean(frac)),
                n=int(valid.sum()),
            )
        )

        single = ~is_unassigned & ~is_multi
        if single.any():
            top1_umi = np.argmax(X[single], axis=1)
            top1_assigned = np.argmax(assigned_mat[single], axis=1)
            rows.append(
                dict(
                    dataset=dataset,
                    metric="umi",
                    submetric="top1_match_rate",
                    value=float((top1_umi == top1_assigned).mean()),
                    n=int(single.sum()),
                )
            )

    if "target_gene" in adata.obs.columns:
        single_mask = ~is_unassigned & ~is_multi
        tg = adata.obs.loc[single_mask, "target_gene"]
        cells_per_target = tg[tg != ""].value_counts()
        n_targets = len(cells_per_target)

        if n_targets > 0:
            counts = cells_per_target.values
            rows += [
                dict(
                    dataset=dataset,
                    metric="perturbation_coverage",
                    submetric="n_targets",
                    value=float(n_targets),
                    n=int(single_mask.sum()),
                ),
                dict(
                    dataset=dataset,
                    metric="perturbation_coverage",
                    submetric="median_cells_per_target",
                    value=float(np.median(counts)),
                    n=n_targets,
                ),
                dict(
                    dataset=dataset,
                    metric="perturbation_coverage",
                    submetric="min_cells_per_target",
                    value=float(counts.min()),
                    n=n_targets,
                ),
                dict(
                    dataset=dataset,
                    metric="perturbation_coverage",
                    submetric="gini_cells_per_target",
                    value=gini(counts),
                    n=n_targets,
                ),
            ]

            nt_mask = cells_per_target.index == "non-targeting"
            nt_cells = int(cells_per_target[nt_mask].sum())
            total_single = int((~is_unassigned & ~is_multi).sum())
            rows.append(
                dict(
                    dataset=dataset,
                    metric="perturbation_coverage",
                    submetric="frac_nt_cells",
                    value=float(nt_cells / total_single) if total_single > 0 else float("nan"),
                    n=total_single,
                )
            )

    df = pd.DataFrame(rows)
    tsv_path = os.path.join(args.output_dir, f"{args.name}.tsv")
    parquet_path = os.path.join(args.output_dir, f"{args.name}.parquet")
    print(
        f"Writing {tsv_path} and {parquet_path} ({len(df)} rows) ...", file=sys.stderr
    )
    df.to_csv(tsv_path, sep="\t", index=False)
    df.to_parquet(parquet_path, index=False)


if __name__ == "__main__":
    main()
