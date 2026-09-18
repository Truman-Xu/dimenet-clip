"""
Direct DUD-E vs LIT-PCBA comparison of the similarity-search virtual-
screening experiment (see vs_screening_analysis.py), restricted to the
protein targets present in both datasets.

Requires vs_screening_analysis.py to have already been run for both
--dataset dude and --dataset litpcba.

Produces, in --output-dir:
  - shared_target_vs_comparison.csv : AUC-ROC / EF1% / EF5% mean+std for
    each dataset per shared target, plus DUD-E-minus-LIT-PCBA deltas.
  - shared_target_vs_comparison.png : grouped bar chart (AUC, EF1%, EF5%)
    comparing the two datasets on each shared target.

Usage:
    python compare_vs_shared_targets.py
"""
import argparse
import paths  # default input/output locations, see plots/paths.py
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fp_similarity_common import BLUE, AQUA, style_axes, SURFACE, GRID, INK_SECONDARY

# LIT-PCBA target folder -> matching DUD-E target key, plus a display label.
SHARED_TARGETS = [
    ("ADRB2", "adrb2", "ADRB2"),
    ("ESR1_ago", "esr1", "ESR1\n(LIT-PCBA agonist)"),
    ("ESR1_ant", "esr1", "ESR1\n(LIT-PCBA antagonist)"),
    ("MAPK1", "mk01", "MAPK1 / ERK2"),
    ("PPARG", "pparg", "PPARG"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dude-summary", default=str(paths.out("dude", "vs_screening_analysis") / "per_target_summary.csv"))
    parser.add_argument("--lit-pcba-summary", default=str(paths.out("lit-pcba", "vs_screening_analysis") / "per_target_summary.csv"))
    parser.add_argument("--output-dir", default=str(paths.out("lit-pcba", "vs_screening_analysis")))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dude_df = pd.read_csv(args.dude_summary).set_index("target")
    lit_df = pd.read_csv(args.lit_pcba_summary).set_index("target")

    rows = []
    for lit_target, dude_target, label in SHARED_TARGETS:
        d = dude_df.loc[dude_target]
        l = lit_df.loc[lit_target]
        rows.append({
            "target": label.replace("\n", " "),
            "lit_pcba_folder": lit_target,
            "dude_target": dude_target,
            "dude_n_actives": int(d["n_actives_total"]), "dude_n_inactives": int(d["n_inactives_total"]),
            "litpcba_n_actives": int(l["n_actives_total"]), "litpcba_n_inactives": int(l["n_inactives_total"]),
            "dude_auc_mean": d["auc_roc_mean"], "dude_auc_std": d["auc_roc_std"],
            "litpcba_auc_mean": l["auc_roc_mean"], "litpcba_auc_std": l["auc_roc_std"],
            "auc_delta_dude_minus_litpcba": d["auc_roc_mean"] - l["auc_roc_mean"],
            "dude_ef1pct_mean": d["ef_0.01_mean"], "dude_ef1pct_std": d["ef_0.01_std"],
            "litpcba_ef1pct_mean": l["ef_0.01_mean"], "litpcba_ef1pct_std": l["ef_0.01_std"],
            "ef1pct_delta_dude_minus_litpcba": d["ef_0.01_mean"] - l["ef_0.01_mean"],
            "dude_ef5pct_mean": d["ef_0.05_mean"], "dude_ef5pct_std": d["ef_0.05_std"],
            "litpcba_ef5pct_mean": l["ef_0.05_mean"], "litpcba_ef5pct_std": l["ef_0.05_std"],
            "ef5pct_delta_dude_minus_litpcba": d["ef_0.05_mean"] - l["ef_0.05_mean"],
        })

    table = pd.DataFrame(rows)
    out_csv = out_dir / "shared_target_vs_comparison.csv"
    table.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}\n")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table[["target", "dude_auc_mean", "litpcba_auc_mean", "auc_delta_dude_minus_litpcba",
                      "dude_ef1pct_mean", "litpcba_ef1pct_mean", "dude_ef5pct_mean", "litpcba_ef5pct_mean"]]
              .to_string(index=False))

    # ---- grouped bar chart: AUC, EF1%, EF5% -------------------------------
    labels = [r["target"] for r in rows]
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), dpi=150)
    panels = [
        ("auc_mean", "auc_std", "AUC-ROC", 0.5),
        ("ef1pct_mean", "ef1pct_std", "Enrichment factor (top 1%)", 1.0),
        ("ef5pct_mean", "ef5pct_std", "Enrichment factor (top 5%)", 1.0),
    ]
    for ax, (mean_suffix, std_suffix, ylabel, ref_line) in zip(axes, panels):
        dude_vals = table[f"dude_{mean_suffix}"].to_numpy()
        dude_err = table[f"dude_{std_suffix}"].to_numpy()
        lit_vals = table[f"litpcba_{mean_suffix}"].to_numpy()
        lit_err = table[f"litpcba_{std_suffix}"].to_numpy()

        ax.bar(x - width / 2, dude_vals, width, yerr=dude_err, capsize=3,
               color=BLUE, label="DUD-E", edgecolor=SURFACE)
        ax.bar(x + width / 2, lit_vals, width, yerr=lit_err, capsize=3,
               color=AQUA, label="LIT-PCBA", edgecolor=SURFACE)
        ax.axhline(ref_line, color="#898781", linestyle="--", linewidth=1, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        style_axes(ax)

    fig.suptitle("Similarity-search virtual screening: DUD-E vs LIT-PCBA\n(shared protein targets, mean ± std over 10 replicates)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = out_dir / "shared_target_vs_comparison.png"
    fig.savefig(out_png, facecolor=SURFACE)
    fig.savefig(out_png.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
    print(f"\nWrote {out_png}")


if __name__ == "__main__":
    main()
