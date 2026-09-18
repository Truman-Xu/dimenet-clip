"""
Similarity-search virtual-screening experiment.

For each protein target, repeat --n-replicates times:
  1. Pick a random known active as the "query".
  2. Rank every OTHER ligand of that target (all actives + all
     decoys/inactives provided for the target, not a subsample) by ECFP4
     Tanimoto similarity to the query.
  3. Score that ranking with AUC-ROC and enrichment factor at 1% and 5%.
  4. Separately, report simple hit-count/precision/recall at the fixed
     similarity thresholds 0.5 / 0.7 / 0.9 (i.e. "ligand predicted active if
     similarity to the query >= threshold").

Averages across replicates are reported and plotted per target. Supports
both DUD-E and LIT-PCBA via --dataset.

Usage:
    python vs_screening_analysis.py --dataset dude
    python vs_screening_analysis.py --dataset litpcba
"""
import argparse
import paths  # default input/output locations, see plots/paths.py
from pathlib import Path

import numpy as np
import pandas as pd

from fp_similarity_common import read_smiles_file
from vs_common import build_or_load_library, run_replicates, plot_metric_boxplot

DATASET_CONFIG = {
    "dude": {
        "base_dir": str(paths.DUDE_DIR),
        "actives_file": "actives_final.ism",
        "inactives_file": "decoys_final.ism",
        "inactive_label": "decoys",
        "output_dir": str(paths.out("dude", "vs_screening_analysis")),
    },
    "litpcba": {
        "base_dir": str(paths.LIT_PCBA_DIR),
        "actives_file": "actives.smi",
        "inactives_file": "inactives.smi",
        "inactive_label": "inactives",
        "output_dir": str(paths.out("lit-pcba", "vs_screening_analysis")),
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG))
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    parser.add_argument("--ef-fracs", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-dir", default=None, help="Override the dataset's ligand directory (e.g. for smoke tests).")
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    base_dir = Path(args.base_dir or cfg["base_dir"])
    out_dir = Path(args.output_dir or cfg["output_dir"])
    cache_dir = out_dir / "fp_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir()
                          and (p / cfg["actives_file"]).exists())
    print(f"[{args.dataset}] Found {len(target_dirs)} targets in {base_dir}")

    replicate_rows = []
    skipped = []

    for i, target_dir in enumerate(target_dirs, 1):
        target = target_dir.name
        actives = read_smiles_file(target_dir / cfg["actives_file"])
        inactives = read_smiles_file(target_dir / cfg["inactives_file"])

        cache_path = cache_dir / f"{target}.pkl"
        fps, labels = build_or_load_library(cache_path, actives, inactives, args.radius, args.n_bits)
        n_actives_valid = int((labels == 1).sum())
        n_inactives_valid = int((labels == 0).sum())

        results = run_replicates(fps, labels, args.n_replicates, args.thresholds, args.ef_fracs, args.seed + i)
        if not results:
            skipped.append(target)
            print(f"[{i}/{len(target_dirs)}] {target}: SKIPPED (need >=2 actives and >=1 inactive; "
                  f"have {n_actives_valid} actives, {n_inactives_valid} {cfg['inactive_label']})")
            continue

        for rep_i, row in enumerate(results):
            row["dataset"] = args.dataset
            row["target"] = target
            row["replicate"] = rep_i
            row["n_actives_total"] = n_actives_valid
            row["n_inactives_total"] = n_inactives_valid
            replicate_rows.append(row)

        mean_auc = np.mean([r["auc_roc"] for r in results])
        print(f"[{i}/{len(target_dirs)}] {target}: {len(results)} replicates, "
              f"n_actives={n_actives_valid}, n_{cfg['inactive_label']}={n_inactives_valid}, "
              f"mean AUC={mean_auc:.3f}")

    if skipped:
        print(f"\nSkipped {len(skipped)} targets (insufficient actives/inactives): {skipped}")

    replicate_df = pd.DataFrame(replicate_rows)
    replicate_csv = out_dir / "replicate_level_results.csv"
    replicate_df.to_csv(replicate_csv, index=False)
    print(f"\nWrote replicate-level results -> {replicate_csv}")

    # ---- per-target summary (mean/std across replicates) -------------------
    ef_cols = [c for c in replicate_df.columns if c.startswith("ef_")]
    thr_precision_cols = [c for c in replicate_df.columns if c.endswith("_precision")]
    thr_recall_cols = [c for c in replicate_df.columns if c.endswith("_recall")]
    thr_nsel_cols = [c for c in replicate_df.columns if c.endswith("_n_selected")]

    agg_map = {"auc_roc": ["mean", "std"], "n_actives_total": "first", "n_inactives_total": "first",
               "replicate": "count"}
    for c in ef_cols + thr_precision_cols + thr_recall_cols + thr_nsel_cols:
        agg_map[c] = ["mean", "std"]

    summary = replicate_df.groupby("target").agg(agg_map)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.rename(columns={"replicate_count": "n_replicates",
                                       "n_actives_total_first": "n_actives_total",
                                       "n_inactives_total_first": "n_inactives_total"})
    summary = summary.reset_index().sort_values("auc_roc_mean", ascending=False)
    summary_csv = out_dir / "per_target_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote per-target summary -> {summary_csv}")

    # ============================= PLOTS ====================================
    sort_order = summary.sort_values("auc_roc_mean")  # ascending for horizontal plots (top = best)

    per_target_auc = {t: replicate_df.loc[replicate_df["target"] == t, "auc_roc"].to_numpy()
                       for t in sort_order["target"]}
    plot_metric_boxplot(
        per_target_auc, sort_order,
        sort_col="auc_roc_mean", value_label="AUC-ROC",
        title=f"Similarity-search AUC-ROC by target — {args.dataset.upper()}\n"
              f"(random active query vs. all target ligands, {args.n_replicates} replicates)",
        out_path=out_dir / "auc_roc_by_target.png", ref_line=0.5,
    )

    for frac in args.ef_fracs:
        col = f"ef_{frac:g}"
        per_target_ef = {t: replicate_df.loc[replicate_df["target"] == t, col].to_numpy()
                          for t in sort_order["target"]}
        plot_metric_boxplot(
            per_target_ef, sort_order, sort_col=f"{col}_mean",
            value_label=f"Enrichment factor (top {frac*100:g}%)",
            title=f"Similarity-search EF{frac*100:g}% by target — {args.dataset.upper()}\n"
                  f"(random active query vs. all target ligands, {args.n_replicates} replicates)",
            out_path=out_dir / f"ef{frac*100:g}pct_by_target.png", ref_line=1.0,
        )

    print(f"\nSaved plots to {out_dir}:")
    print("  - auc_roc_by_target.png")
    for frac in args.ef_fracs:
        print(f"  - ef{frac*100:g}pct_by_target.png")


if __name__ == "__main__":
    main()
