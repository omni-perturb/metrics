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
    parser.add_argument(
        "--rawcounts.h5mu",
        dest="rawcounts",
        required=False,
        default=None,
        help="h5mu file from count-data stage (for RNA knockdown metrics)",
    )
    parser.add_argument("--output_dir", "-o", required=True)
    parser.add_argument("--name", "-n", required=True)
    return parser.parse_args()


def _dense(X) -> np.ndarray:
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def rna_knockdown_rows(adata: ad.AnnData, h5mu_path: str, dataset: str) -> list[dict]:
    import muon as mu

    rna = mu.read_h5mu(h5mu_path)["rna"]
    X = rna.X
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    lib[lib == 0] = 1.0

    gene_idx = {g: i for i, g in enumerate(rna.var_names)}
    if "gene_name" in rna.var.columns:
        for i, s in enumerate(rna.var["gene_name"]):
            if s and s not in gene_idx:
                gene_idx[s] = i

    single = (~adata.obs["is_unassigned"].astype(bool).values
              & ~adata.obs["is_multi_infected"].astype(bool).values)
    tg = adata.obs["target_gene"].fillna("").values

    # nt_mask ⊆ single (NT cells are always singly-assigned), so single covers both
    tg_s = tg[single]
    lib_s = lib[single]

    tgt_counts = (pd.Series(tg_s).value_counts()
                  .drop(["non-targeting", ""], errors="ignore"))
    targets = [(t, gene_idx[t]) for t, n in tgt_counts.items()
               if n >= 5 and t in gene_idx]
    if not targets:
        return []

    names, col_idxs = zip(*targets)
    col_idxs = np.array(col_idxs)

    # row-slice to singly-assigned cells, then extract only needed gene columns
    X_s = X[single][:, col_idxs]
    if sp.issparse(X_s):
        X_s = X_s.tocsc().astype(np.float64)
        X_s = X_s.multiply(1.0 / lib_s[:, None] * 1e4)
        X_s.data = np.log1p(X_s.data)   # log1p(0)=0: sparse zeros stay zero
    else:
        X_s = np.log1p(X_s.astype(np.float64) / lib_s[:, None] * 1e4)

    nt_s = tg_s == "non-targeting"
    nt_means = np.asarray(X_s[nt_s].mean(axis=0)).ravel()

    lfcs = [
        float(np.asarray(X_s[:, i][tg_s == name].mean()) - nt_means[i])
        for i, name in enumerate(names)
    ]

    arr = np.array(lfcs)
    print(f"  RNA: {len(arr)} targets, median LFC {np.median(arr):.3f}", file=sys.stderr)
    return [
        dict(dataset=dataset, metric="rna_knockdown", submetric="n_targets_with_rna",
             value=float(len(arr)), n=len(arr)),
        dict(dataset=dataset, metric="rna_knockdown", submetric="median_lfc_target_gene",
             value=float(np.median(arr)), n=len(arr)),
        dict(dataset=dataset, metric="rna_knockdown", submetric="frac_lfc_lt_neg1",
             value=float((arr < -1.0).mean()), n=len(arr)),
    ]


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.assignments} ...", file=sys.stderr)
    adata = ad.read_h5ad(args.assignments)

    is_unassigned = adata.obs["is_unassigned"].astype(bool).values
    is_multi = adata.obs["is_multi_infected"].astype(bool).values
    single = ~is_unassigned & ~is_multi
    n_cells = adata.n_obs
    print(f"  {n_cells} cells, {(~is_unassigned).sum()} assigned", file=sys.stderr)

    dataset = args.name
    rows = [
        dict(dataset=dataset, metric="coverage", submetric="fraction_assigned",
             value=float((~is_unassigned).mean()), n=n_cells),
        dict(dataset=dataset, metric="coverage", submetric="frac_multi_infected",
             value=float(is_multi.mean()), n=n_cells),
    ]

    if "assigned" in adata.layers and (~is_unassigned).any():
        X = _dense(adata.X).astype(np.float32)
        assigned_mat = _dense(adata.layers["assigned"])

        X_a = X[~is_unassigned]
        A_a = assigned_mat[~is_unassigned].astype(np.float32)
        total_umi = X_a.sum(axis=1)
        assigned_umi = (X_a * A_a).sum(axis=1)
        valid = total_umi > 0
        frac = np.where(valid, assigned_umi / np.where(valid, total_umi, 1.0), np.nan)
        rows.append(dict(dataset=dataset, metric="umi", submetric="mean_assigned_umi_frac",
                         value=float(np.nanmean(frac)), n=int(valid.sum())))

        if single.any():
            top1_umi = np.argmax(X[single], axis=1)
            top1_assigned = np.argmax(assigned_mat[single], axis=1)
            rows.append(dict(dataset=dataset, metric="umi", submetric="top1_match_rate",
                             value=float((top1_umi == top1_assigned).mean()),
                             n=int(single.sum())))

    if "target_gene" in adata.obs.columns:
        tg = adata.obs.loc[single, "target_gene"]
        cells_per_target = tg[tg != ""].value_counts()
        n_targets = len(cells_per_target)

        if n_targets > 0:
            counts = cells_per_target.values
            rows += [
                dict(dataset=dataset, metric="perturbation_coverage",
                     submetric="n_targets", value=float(n_targets),
                     n=int(single.sum())),
                dict(dataset=dataset, metric="perturbation_coverage",
                     submetric="median_cells_per_target", value=float(np.median(counts)),
                     n=n_targets),
                dict(dataset=dataset, metric="perturbation_coverage",
                     submetric="min_cells_per_target", value=float(counts.min()),
                     n=n_targets),
                dict(dataset=dataset, metric="perturbation_coverage",
                     submetric="gini_cells_per_target", value=gini(counts),
                     n=n_targets),
            ]

            rows.append(dict(dataset=dataset, metric="perturbation_coverage",
                             submetric="frac_nt_cells",
                             value=float(cells_per_target.get("non-targeting", 0) / single.sum()),
                             n=int(single.sum())))

    if args.rawcounts:
        print(f"Loading RNA from {args.rawcounts} ...", file=sys.stderr)
        rows.extend(rna_knockdown_rows(adata, args.rawcounts, dataset))

    df = pd.DataFrame(rows)
    tsv_path = os.path.join(args.output_dir, f"{args.name}.tsv")
    parquet_path = os.path.join(args.output_dir, f"{args.name}.parquet")
    print(f"Writing {tsv_path} and {parquet_path} ({len(df)} rows) ...", file=sys.stderr)
    df.to_csv(tsv_path, sep="\t", index=False)
    df.to_parquet(parquet_path, index=False)


if __name__ == "__main__":
    main()
