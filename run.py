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

    mdata = mu.read_h5mu(h5mu_path)
    if "rna" not in mdata.mod:
        print("  no 'rna' modality in h5mu; skipping RNA metrics", file=sys.stderr)
        return []

    rna = mdata["rna"]
    print(f"  RNA: {rna.n_obs} cells x {rna.n_vars} genes", file=sys.stderr)

    # per-cell library sizes — row sums on the sparse matrix directly
    X_rna = rna.X
    lib = np.asarray(X_rna.sum(axis=1)).flatten().astype(np.float64)
    lib = np.where(lib > 0, lib, 1.0)

    # build gene symbol → column index; prefer var_names, fall back to gene_name column
    gene_names = np.array(rna.var_names)
    gene_index: dict[str, int] = {g: i for i, g in enumerate(gene_names)}
    if "gene_name" in rna.var.columns:
        for i, sym in enumerate(rna.var["gene_name"]):
            if sym and sym not in gene_index:
                gene_index[sym] = i

    # singly-assigned cells (barcodes are aligned — no subsetting needed)
    is_unassigned = adata.obs["is_unassigned"].astype(bool).values
    is_multi = adata.obs["is_multi_infected"].astype(bool).values
    single = ~is_unassigned & ~is_multi

    tg = adata.obs["target_gene"].fillna("").values
    nt_mask = single & (tg == "non-targeting")

    targets = (
        pd.Series(tg[single])
        .value_counts()
        .drop(labels=["non-targeting", ""], errors="ignore")
    )

    # convert to CSC once so column slicing is O(nnz_per_column) not O(nnz_total)
    X_csc = X_rna.tocsc() if sp.issparse(X_rna) else X_rna

    lfcs: list[float] = []
    n_missing = 0
    for target_gene, n_assigned in targets.items():
        if n_assigned < 5:
            continue
        if target_gene not in gene_index:
            n_missing += 1
            continue
        g_idx = gene_index[target_gene]
        col = X_csc[:, g_idx].toarray().ravel() if sp.issparse(X_csc) else X_csc[:, g_idx]
        col_norm = np.log1p(col.astype(np.float64) / lib * 1e4)
        tg_mask = single & (tg == target_gene)
        lfc = float(col_norm[tg_mask].mean() - col_norm[nt_mask].mean())
        lfcs.append(lfc)

    if n_missing:
        print(f"  {n_missing} target genes not found in RNA var", file=sys.stderr)
    if not lfcs:
        return []

    arr = np.array(lfcs)
    print(
        f"  RNA knockdown: {len(arr)} targets, median LFC {np.median(arr):.3f}",
        file=sys.stderr,
    )
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
    n_cells = adata.n_obs
    print(f"  {n_cells} cells, {(~is_unassigned).sum()} assigned", file=sys.stderr)

    X = _dense(adata.X).astype(np.float32)

    assigned_mat = None
    if "assigned" in adata.layers:
        assigned_mat = _dense(adata.layers["assigned"])

    dataset = args.name
    rows = [
        dict(dataset=dataset, metric="coverage", submetric="fraction_assigned",
             value=float((~is_unassigned).mean()), n=n_cells),
        dict(dataset=dataset, metric="coverage", submetric="frac_multi_infected",
             value=float(is_multi.mean()), n=n_cells),
    ]

    if assigned_mat is not None and (~is_unassigned).any():
        X_a = X[~is_unassigned]
        A_a = assigned_mat[~is_unassigned].astype(np.float32)
        total_umi = X_a.sum(axis=1)
        assigned_umi = (X_a * A_a).sum(axis=1)
        valid = total_umi > 0
        frac = np.where(valid, assigned_umi / np.where(valid, total_umi, 1.0), np.nan)
        rows.append(dict(dataset=dataset, metric="umi", submetric="mean_assigned_umi_frac",
                         value=float(np.nanmean(frac)), n=int(valid.sum())))

        single = ~is_unassigned & ~is_multi
        if single.any():
            top1_umi = np.argmax(X[single], axis=1)
            top1_assigned = np.argmax(assigned_mat[single], axis=1)
            rows.append(dict(dataset=dataset, metric="umi", submetric="top1_match_rate",
                             value=float((top1_umi == top1_assigned).mean()),
                             n=int(single.sum())))

    if "target_gene" in adata.obs.columns:
        single_mask = ~is_unassigned & ~is_multi
        tg = adata.obs.loc[single_mask, "target_gene"]
        cells_per_target = tg[tg != ""].value_counts()
        n_targets = len(cells_per_target)

        if n_targets > 0:
            counts = cells_per_target.values
            rows += [
                dict(dataset=dataset, metric="perturbation_coverage",
                     submetric="n_targets", value=float(n_targets),
                     n=int(single_mask.sum())),
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

            nt_cells = int(cells_per_target.get("non-targeting", 0))
            total_single = int(single_mask.sum())
            rows.append(dict(dataset=dataset, metric="perturbation_coverage",
                             submetric="frac_nt_cells",
                             value=float(nt_cells / total_single) if total_single > 0 else float("nan"),
                             n=total_single))

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
