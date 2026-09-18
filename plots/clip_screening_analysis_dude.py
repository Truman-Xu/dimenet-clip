"""
Evaluate the pre-trained DimeNet-CLIP model's virtual-screening performance
on DUD-E, from the precomputed ligand/pocket embeddings written by
eval/encode_benchmarks.py (the scripted form of eval/dude_eval.ipynb).

No GPU or model weights needed here - the embeddings are already computed;
this script just does the ligand-embedding . pocket-embedding dot product
scoring (as in eval/dude_eval.ipynb) and derives the same metrics used for the
fingerprint similarity-search experiment (AUC-ROC, EF1%, EF5%).

Unlike LIT-PCBA (multiple crystal structures/pockets per target), the DUD-E
embeddings have exactly ONE pocket per target, so there is no within-target
replicate axis for CLIP here - each target gets a single AUC/EF value (this
also means DUD-E only has CLIP embeddings for the 25 targets that had a
usable prepared pocket, out of the full 102).

Usage:
    python clip_screening_analysis_dude.py
"""
import argparse
import paths  # default input/output locations, see plots/paths.py
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fp_similarity_common import INK_MUTED
from vs_common import enrichment_factor, plot_metric_bar, plot_metric_boxplot, score_clip_pair

RED = "#d62728"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ligand-embeddings", default=str(paths.encoding("dude", "ligand")))
    parser.add_argument("--pocket-embeddings", default=str(paths.encoding("dude", "pocket")))
    parser.add_argument("--ef-fracs", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--output-dir", default=str(paths.out("dude", "clip_vs_screening_analysis")))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.ligand_embeddings, "rb") as f:
        lig_encoded = pickle.load(f)
    with open(args.pocket_embeddings, "rb") as f:
        prot_encoded = pickle.load(f)

    targets = sorted(lig_encoded.keys())
    print(f"[clip] Found {len(targets)} DUD-E targets with embeddings: {targets}")

    replicate_rows = []
    for i, target in enumerate(targets, 1):
        lig_emb = lig_encoded[target]["emb"]
        labels = lig_encoded[target]["labels"]
        n_actives_total = int(labels.sum())
        n_inactives_total = int(len(labels) - n_actives_total)

        prot_emb = prot_encoded[target]
        y_true, y_score = score_clip_pair(lig_emb, prot_emb, labels)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            print(f"[{i}/{len(targets)}] {target}: SKIPPED (single class after filtering)")
            continue

        row = {
            "dataset": "dude_clip",
            "target": target,
            "replicate": 0,
            "pdbid": "single_pocket",
            "n_scored": len(y_true),
            "auc_roc": float(roc_auc_score(y_true, y_score)),
            "n_actives_total": n_actives_total,
            "n_inactives_total": n_inactives_total,
        }
        for frac in args.ef_fracs:
            row[f"ef_{frac:g}"] = enrichment_factor(y_true, y_score, frac)
        replicate_rows.append(row)

        print(f"[{i}/{len(targets)}] {target}: n_actives={n_actives_total}, "
              f"n_decoys={n_inactives_total}, AUC={row['auc_roc']:.3f}")

    replicate_df = pd.DataFrame(replicate_rows)
    replicate_csv = out_dir / "replicate_level_results.csv"
    replicate_df.to_csv(replicate_csv, index=False)
    print(f"\nWrote replicate-level results -> {replicate_csv}")

    ef_cols = [c for c in replicate_df.columns if c.startswith("ef_")]
    agg_map = {"auc_roc": ["mean", "std"], "n_actives_total": "first", "n_inactives_total": "first",
               "replicate": "count"}
    for c in ef_cols:
        agg_map[c] = ["mean", "std"]

    summary = replicate_df.groupby("target").agg(agg_map)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.rename(columns={"replicate_count": "n_structures",
                                       "n_actives_total_first": "n_actives_total",
                                       "n_inactives_total_first": "n_inactives_total"})
    summary = summary.reset_index().sort_values("auc_roc_mean", ascending=False)
    summary_csv = out_dir / "per_target_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote per-target summary -> {summary_csv}")

    # ============================= PLOTS ====================================
    sort_order = summary.sort_values("auc_roc_mean")

    per_target_auc = {t: replicate_df.loc[replicate_df["target"] == t, "auc_roc"].to_numpy()
                       for t in sort_order["target"]}
    plot_metric_boxplot(
        per_target_auc, sort_order, sort_col="auc_roc_mean", value_label="AUC-ROC",
        title="DimeNet-CLIP virtual-screening AUC-ROC by target — DUD-E\n(single pocket structure per target - no replicate spread)",
        out_path=out_dir / "auc_roc_by_target.png", ref_line=0.5, save_pdf=True,
    )
    for frac in args.ef_fracs:
        col = f"ef_{frac:g}"
        if frac == 0.01:
            # Bar chart: DUD-E CLIP has one value per target (no replicate
            # spread), so a boxplot degenerates to a zero-width box here.
            # EF=2 is highlighted in red per request.
            plot_metric_bar(
                sort_order, sort_col=f"{col}_mean",
                value_label="Enrichment factor (top 1%)",
                title="DimeNet-CLIP EF1% by target — DUD-E\n(single pocket structure per target - no replicate spread)",
                out_path=out_dir / f"ef{frac*100:g}pct_by_target.png",
                ref_lines=[(1.0, INK_MUTED, "Random (EF=1)"), (2.0, RED, "EF=2")],
            )
            continue
        per_target_ef = {t: replicate_df.loc[replicate_df["target"] == t, col].to_numpy()
                          for t in sort_order["target"]}
        plot_metric_boxplot(
            per_target_ef, sort_order, sort_col=f"{col}_mean",
            value_label=f"Enrichment factor (top {frac*100:g}%)",
            title=f"DimeNet-CLIP EF{frac*100:g}% by target — DUD-E\n(single pocket structure per target - no replicate spread)",
            out_path=out_dir / f"ef{frac*100:g}pct_by_target.png", ref_line=1.0,
        )

    print(f"\nSaved plots to {out_dir}:")
    print("  - auc_roc_by_target.png")
    for frac in args.ef_fracs:
        print(f"  - ef{frac*100:g}pct_by_target.png")


if __name__ == "__main__":
    main()
