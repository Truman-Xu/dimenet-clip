"""
Side-by-side comparison of the two LIT-PCBA virtual-screening methods:
  1. ECFP4 fingerprint similarity search (vs_screening_analysis.py --dataset litpcba)
  2. DimeNet-CLIP ligand/pocket embedding dot-product scoring (clip_screening_analysis_litpcba.py)

Restricted to the targets both methods were evaluated on (CLIP embeddings
only exist for 10 of the 15 LIT-PCBA targets).

Requires both analyses to have already been run. Produces, in --output-dir:
  - clip_vs_fingerprint_comparison.csv : per-target mean+-std for both
    methods (AUC-ROC, EF1%, EF5%) plus the delta.
  - clip_vs_fingerprint_comparison.png : grouped boxplots (one 3-panel
    figure: AUC-ROC, EF1%, EF5%), fingerprint search vs CLIP side by side
    for every shared target.

Usage:
    python compare_clip_vs_fingerprint_litpcba.py
"""
import argparse
import paths  # default input/output locations, see plots/paths.py
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fp_similarity_common import BLUE, AQUA, style_axes, SURFACE, GRID, INK_SECONDARY


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp-replicates", default=str(paths.out("lit-pcba", "vs_screening_analysis") / "replicate_level_results.csv"))
    parser.add_argument("--clip-replicates", default=str(paths.out("lit-pcba", "clip_vs_screening_analysis") / "replicate_level_results.csv"))
    parser.add_argument("--output-dir", default=str(paths.out("lit-pcba", "clip_vs_screening_analysis")))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_df = pd.read_csv(args.fp_replicates)
    clip_df = pd.read_csv(args.clip_replicates)

    shared_targets = sorted(set(fp_df["target"]) & set(clip_df["target"]))
    print(f"Shared targets ({len(shared_targets)}): {shared_targets}")

    metrics = [("auc_roc", "AUC-ROC", 0.5), ("ef_0.01", "Enrichment factor (top 1%)", 1.0),
               ("ef_0.05", "Enrichment factor (top 5%)", 1.0)]

    # ---- summary table -------------------------------------------------
    rows = []
    for t in shared_targets:
        row = {"target": t,
               "fp_n_replicates": int((fp_df["target"] == t).sum()),
               "clip_n_structures": int((clip_df["target"] == t).sum())}
        for col, _, _ in metrics:
            fp_vals = fp_df.loc[fp_df["target"] == t, col]
            clip_vals = clip_df.loc[clip_df["target"] == t, col]
            row[f"fp_{col}_mean"] = fp_vals.mean()
            row[f"fp_{col}_std"] = fp_vals.std()
            row[f"clip_{col}_mean"] = clip_vals.mean()
            row[f"clip_{col}_std"] = clip_vals.std()
            row[f"{col}_delta_fp_minus_clip"] = fp_vals.mean() - clip_vals.mean()
        rows.append(row)
    table = pd.DataFrame(rows)
    out_csv = out_dir / "clip_vs_fingerprint_comparison.csv"
    table.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}\n")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(table[["target", "fp_auc_roc_mean", "clip_auc_roc_mean", "auc_roc_delta_fp_minus_clip",
                      "fp_ef_0.01_mean", "clip_ef_0.01_mean", "fp_ef_0.05_mean", "clip_ef_0.05_mean"]]
              .to_string(index=False))

    # ---- grouped boxplot, 3 panels --------------------------------------
    x = np.arange(len(shared_targets))
    width = 0.32

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), dpi=150)
    for ax, (col, ylabel, ref_line) in zip(axes, metrics):
        fp_data = [fp_df.loc[fp_df["target"] == t, col].to_numpy() for t in shared_targets]
        clip_data = [clip_df.loc[clip_df["target"] == t, col].to_numpy() for t in shared_targets]

        bp_fp = ax.boxplot(fp_data, positions=x - width / 2, widths=width * 0.9,
                            patch_artist=True, showfliers=False,
                            medianprops={"color": "#0b0b0b", "linewidth": 1.3})
        bp_clip = ax.boxplot(clip_data, positions=x + width / 2, widths=width * 0.9,
                              patch_artist=True, showfliers=False,
                              medianprops={"color": "#0b0b0b", "linewidth": 1.3})
        for patch in bp_fp["boxes"]:
            patch.set_facecolor(BLUE)
            patch.set_edgecolor(SURFACE)
        for patch in bp_clip["boxes"]:
            patch.set_facecolor(AQUA)
            patch.set_edgecolor(SURFACE)
        for bp in (bp_fp, bp_clip):
            for element in ("whiskers", "caps"):
                for line in bp[element]:
                    line.set_color("#898781")
                    line.set_linewidth(0.9)

        ax.axhline(ref_line, color="#898781", linestyle="--", linewidth=1, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(shared_targets, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        style_axes(ax)
        ax.legend([bp_fp["boxes"][0], bp_clip["boxes"][0]], ["Fingerprint similarity search", "DimeNet-CLIP"],
                  frameon=False, labelcolor=INK_SECONDARY, fontsize=9, loc="upper right")

    fig.suptitle("LIT-PCBA virtual screening: fingerprint similarity search vs. DimeNet-CLIP\n"
                 "(fingerprint: 10 random-query replicates; CLIP: one point per available protein structure)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = out_dir / "clip_vs_fingerprint_comparison.png"
    fig.savefig(out_png, facecolor=SURFACE)
    fig.savefig(out_png.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
    print(f"\nWrote {out_png}")


if __name__ == "__main__":
    main()
