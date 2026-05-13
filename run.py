#!/usr/bin/env python3

import argparse
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


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
    method = str(adata.uns.get("guide_assignment_method", os.path.basename(args.assignments)))
    obs = adata.obs[["guide_identity", "n_guides_assigned"]]

    X = adata.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float32)

    assigned_mat = None
    if "assigned" in adata.layers:
        a = adata.layers["assigned"]
        assigned_mat = a.toarray() if sp.issparse(a) else np.asarray(a)

    print(f"  {method}: {len(obs)} cells", file=sys.stderr)

    is_assigned = (obs["guide_identity"] != "").values
    n_cells = len(obs)
    dataset = args.name

    rows = [
        dict(dataset=dataset, metric="coverage", submetric="fraction_assigned",
             method=method, value=float(is_assigned.mean()), n=n_cells),
        dict(dataset=dataset, metric="coverage", submetric="frac_multi_assigned",
             method=method, value=float((obs["n_guides_assigned"].values > 1).mean()), n=n_cells),
    ]

    if assigned_mat is not None and is_assigned.any():
        X_a = X[is_assigned]
        A_a = assigned_mat[is_assigned].astype(np.float32)
        total_umi = X_a.sum(axis=1)
        assigned_umi = (X_a * A_a).sum(axis=1)
        valid = total_umi > 0
        frac = np.where(valid, assigned_umi / np.where(valid, total_umi, 1.0), np.nan)
        rows.append(dict(
            dataset=dataset, metric="umi", submetric="mean_assigned_umi_frac",
            method=method, value=float(np.nanmean(frac)), n=int(valid.sum()),
        ))

        single = is_assigned & (obs["n_guides_assigned"].values == 1)
        if single.any():
            top1_umi = np.argmax(X[single], axis=1)
            top1_assigned = np.argmax(assigned_mat[single], axis=1)
            rows.append(dict(
                dataset=dataset, metric="umi", submetric="top1_match_rate",
                method=method, value=float((top1_umi == top1_assigned).mean()), n=int(single.sum()),
            ))

    out_path = os.path.join(args.output_dir, f"{args.name}.parquet")
    df = pd.DataFrame(rows)
    print(f"Writing {out_path} ({len(df)} rows) ...", file=sys.stderr)
    df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    main()
