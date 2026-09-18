"""
Control experiment: does the DimeNet-CLIP protein-pocket embedding actually
contribute to virtual-screening performance on DUD-E, or would any pocket
embedding do?

For every DUD-E target T with precomputed embeddings (see
clip_screening_analysis_dude.py — 25 of 102 targets), this script scores
T's own ligand set (its true actives/decoys, unchanged) against:
  1. its own, CORRECT pocket embedding (one point per target — same
     definition as clip_screening_analysis_dude.py) — the null hypothesis
     being tested is whether this beats (2).
  2. every OTHER target's pocket embedding (24 MISMATCHED pockets per
     target) — a per-target null/control distribution of AUC-ROC and
     EF1%/EF5% you'd get from a pocket embedding that carries no real
     information about T's binding site.

If the correct pocket reliably outperforms the mismatched-pocket null,
the pocket embedding is contributing real, target-specific signal. If
correct and mismatched performance are indistinguishable, the ligand
embeddings alone (or some other artifact) are driving the ranking and the
pocket embedding is not being meaningfully used.

Requires the same precomputed embedding pickles as
clip_screening_analysis_dude.py (no GPU or model weights needed).

Produces, in --output-dir:
  - replicate_level_results.csv : one row per (intended_target,
    pocket_target) pair (both correct and mismatched), with auc_roc,
    ef_0.01, ef_0.05.
  - per_target_summary.csv : per intended target, the correct-pocket
    value alongside the mismatched-pocket mean+-std and the delta.
  - auc_roc_pocket_control.png, ef1pct_pocket_control.png,
    ef5pct_pocket_control.png : horizontal boxplot of the mismatched-pocket
    null distribution per target with the correct-pocket value overlaid as
    a diamond marker, sorted by correct-pocket AUC-ROC.
  - A paired Wilcoxon signed-rank test (correct vs. mean-mismatched AUC
    across targets) printed to stdout.

Usage:
    python clip_pocket_specificity_control_dude.py
"""
import argparse
import paths  # default input/output locations, see plots/paths.py
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from vs_common import enrichment_factor, plot_boxplot_with_reference_marker, score_clip_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ligand-embeddings", default=str(paths.encoding("dude", "ligand")))
    parser.add_argument("--pocket-embeddings", default=str(paths.encoding("dude", "pocket")))
    parser.add_argument("--ef-fracs", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--output-dir", default=str(paths.out("dude", "clip_pocket_control_analysis")))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.ligand_embeddings, "rb") as f:
        lig_encoded = pickle.load(f)
    with open(args.pocket_embeddings, "rb") as f:
        prot_encoded = pickle.load(f)

    targets = sorted(t for t in lig_encoded if t in prot_encoded)
    print(f"[clip pocket control] {len(targets)} DUD-E targets with embeddings: {targets}")

    def score_row(lig_emb, labels, pocket_target, prot_emb, n_actives_total, n_inactives_total):
        y_true, y_score = score_clip_pair(lig_emb, prot_emb, labels)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            return None
        row = {
            "pocket_target": pocket_target,
            "n_scored": len(y_true),
            "auc_roc": float(roc_auc_score(y_true, y_score)),
            "n_actives_total": n_actives_total,
            "n_inactives_total": n_inactives_total,
        }
        for frac in args.ef_fracs:
            row[f"ef_{frac:g}"] = enrichment_factor(y_true, y_score, frac)
        return row

    replicate_rows = []
    skipped = []
    for i, target in enumerate(targets, 1):
        lig_emb = lig_encoded[target]["emb"]
        labels = lig_encoded[target]["labels"]
        n_actives_total = int(labels.sum())
        n_inactives_total = int(len(labels) - n_actives_total)

        correct_row = score_row(lig_emb, labels, target, prot_encoded[target],
                                 n_actives_total, n_inactives_total)
        if correct_row is None:
            skipped.append(target)
            print(f"[{i}/{len(targets)}] {target}: SKIPPED (single class after filtering)")
            continue
        correct_row.update({"dataset": "dude_clip", "intended_target": target,
                             "pocket_type": "correct"})
        replicate_rows.append(correct_row)

        for other_target in targets:
            if other_target == target:
                continue
            row = score_row(lig_emb, labels, other_target, prot_encoded[other_target],
                             n_actives_total, n_inactives_total)
            if row is None:
                continue
            row.update({"dataset": "dude_clip", "intended_target": target,
                        "pocket_type": "mismatched"})
            replicate_rows.append(row)

        n_mismatched = sum(1 for r in replicate_rows
                            if r["intended_target"] == target and r["pocket_type"] == "mismatched")
        mismatched_mean_auc = np.mean([r["auc_roc"] for r in replicate_rows
                                        if r["intended_target"] == target and r["pocket_type"] == "mismatched"])
        print(f"[{i}/{len(targets)}] {target}: n_actives={n_actives_total}, n_decoys={n_inactives_total} — "
              f"correct-pocket AUC={correct_row['auc_roc']:.3f}, "
              f"mismatched-pocket AUC={mismatched_mean_auc:.3f} (n={n_mismatched} other targets)")

    if skipped:
        print(f"\nSkipped {len(skipped)} targets (single class): {skipped}")

    replicate_df = pd.DataFrame(replicate_rows)
    replicate_csv = out_dir / "replicate_level_results.csv"
    replicate_df.to_csv(replicate_csv, index=False)
    print(f"\nWrote replicate-level results -> {replicate_csv}")

    # ---- per-target summary: correct value vs mismatched mean+-std --------
    ef_cols = [c for c in replicate_df.columns if c.startswith("ef_")]
    summary_rows = []
    for target in sorted(replicate_df["intended_target"].unique()):
        sub = replicate_df[replicate_df["intended_target"] == target]
        correct = sub[sub["pocket_type"] == "correct"].iloc[0]
        mismatched = sub[sub["pocket_type"] == "mismatched"]
        row = {
            "target": target,
            "n_actives_total": int(correct["n_actives_total"]),
            "n_inactives_total": int(correct["n_inactives_total"]),
            "n_mismatched_pockets": len(mismatched),
            "correct_auc_roc": correct["auc_roc"],
            "mismatched_auc_roc_mean": mismatched["auc_roc"].mean(),
            "mismatched_auc_roc_std": mismatched["auc_roc"].std(),
            "delta_auc_roc": correct["auc_roc"] - mismatched["auc_roc"].mean(),
        }
        for col in ef_cols:
            row[f"correct_{col}"] = correct[col]
            row[f"mismatched_{col}_mean"] = mismatched[col].mean()
            row[f"mismatched_{col}_std"] = mismatched[col].std()
            row[f"delta_{col}"] = correct[col] - mismatched[col].mean()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("correct_auc_roc", ascending=False)
    summary_csv = out_dir / "per_target_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Wrote per-target summary -> {summary_csv}")

    # ---- paired test: correct vs mean-mismatched AUC across targets -------
    stat, p_value = wilcoxon(summary_df["correct_auc_roc"], summary_df["mismatched_auc_roc_mean"])
    print(f"\nAcross {len(summary_df)} targets: mean correct-pocket AUC = "
          f"{summary_df['correct_auc_roc'].mean():.3f}, "
          f"mean mismatched-pocket AUC = {summary_df['mismatched_auc_roc_mean'].mean():.3f}, "
          f"mean delta = {summary_df['delta_auc_roc'].mean():.3f} +- {summary_df['delta_auc_roc'].std():.3f}")
    print(f"Paired Wilcoxon signed-rank test (correct vs. mismatched-mean AUC per target): "
          f"W={stat:.1f}, p={p_value:.2e}")

    # ============================= PLOTS ====================================
    sort_order = summary_df.sort_values("correct_auc_roc")  # ascending -> best at top
    targets_sorted = sort_order["target"].tolist()

    mismatched_auc_by_target = {t: replicate_df.loc[(replicate_df["intended_target"] == t)
                                                     & (replicate_df["pocket_type"] == "mismatched"), "auc_roc"].to_numpy()
                                 for t in targets_sorted}
    correct_auc_by_target = dict(zip(summary_df["target"], summary_df["correct_auc_roc"]))
    plot_boxplot_with_reference_marker(
        mismatched_auc_by_target, correct_auc_by_target, targets_sorted,
        value_label="AUC-ROC",
        title="DimeNet-CLIP pocket-specificity control — DUD-E\n"
              "ligands scored against their correct pocket vs. every other target's (mismatched) pocket",
        out_path=out_dir / "auc_roc_pocket_control.png", ref_line=0.5,
        null_label="Mismatched pocket (other 24 targets)", marker_label="Correct pocket",
        save_pdf=True,
    )

    for frac in args.ef_fracs:
        col = f"ef_{frac:g}"
        mismatched_by_target = {t: replicate_df.loc[(replicate_df["intended_target"] == t)
                                                     & (replicate_df["pocket_type"] == "mismatched"), col].to_numpy()
                                 for t in targets_sorted}
        correct_by_target = dict(zip(summary_df["target"], summary_df[f"correct_{col}"]))
        plot_boxplot_with_reference_marker(
            mismatched_by_target, correct_by_target, targets_sorted,
            value_label=f"Enrichment factor (top {frac*100:g}%)",
            title=f"DimeNet-CLIP pocket-specificity control EF{frac*100:g}% — DUD-E\n"
                  "ligands scored against their correct pocket vs. every other target's (mismatched) pocket",
            out_path=out_dir / f"ef{frac*100:g}pct_pocket_control.png", ref_line=1.0,
            null_label="Mismatched pocket (other 24 targets)", marker_label="Correct pocket",
        )

    print(f"\nSaved plots to {out_dir}:")
    print("  - auc_roc_pocket_control.png")
    for frac in args.ef_fracs:
        print(f"  - ef{frac*100:g}pct_pocket_control.png")


if __name__ == "__main__":
    main()
