"""
Assemble the manuscript figure "sims-2", a 3-panel composite:
  (a) DUD-E active-active vs. active-decoy similarity (pooled across targets)
  (b) LIT-PCBA active-active vs. active-inactive similarity (pooled across targets)
  (c) mean +- std AUC-ROC / EF1% / EF5% from similarity-search virtual
      screening, DUD-E vs. LIT-PCBA, on the 4 shared protein targets

The global-vs-within-target similarity comparison (DUD-E/LIT-PCBA) that used
to be panels (a)/(b) of this figure now lives in the SI as its own figure -
see make_figure_si_global_similarity.py.

Requires the following to have already been run (reads their saved arrays/
CSVs only - no refingerprinting or re-scoring here):
  active_decoy_similarity_distributions.py                            (a)
  active_inactive_similarity_distributions_litpcba.py                 (b)
  compare_vs_shared_targets.py                                        (c)

Usage:
    python make_figure_sims2.py
"""
import argparse
import paths  # default input/output locations, see plots/paths.py
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from fp_similarity_common import plot_two_kde_on_ax, style_axes, BLUE, AQUA, SURFACE, GRID, INK_SECONDARY


def pool_by_prefix(npz, prefix):
    """Concatenate every array in npz whose key starts with prefix__."""
    arrays = [v for k, v in npz.items() if k.startswith(f"{prefix}__") and len(v)]
    return np.concatenate(arrays)


def add_panel_label(ax, label):
    ax.text(-0.02, 1.05, label, transform=ax.transAxes, fontsize=15, fontweight="bold",
            va="bottom", ha="right")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dude-decoy-npz", default=str(paths.out("dude", "active_decoy_similarity_analysis") / "raw_similarities.npz"))
    parser.add_argument("--lit-pcba-inactive-npz", default=str(paths.out("lit-pcba", "active_inactive_similarity_analysis") / "raw_similarities.npz"))
    parser.add_argument("--shared-target-csv", default=str(paths.out("lit-pcba", "vs_screening_analysis") / "shared_target_vs_comparison.csv"))
    parser.add_argument("--output-dir", default=str(paths.FIGURES_DIR))
    parser.add_argument("--out-name", default="sims-2")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dude_decoy = np.load(args.dude_decoy_npz)
    lit_inactive = np.load(args.lit_pcba_inactive_npz)
    shared_table = pd.read_csv(args.shared_target_csv)

    fig = plt.figure(figsize=(14, 13), dpi=150)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.2, 1, 1, 1], hspace=0.55, wspace=0.28)

    # ---- (a)/(b): active-active vs. active-decoy/inactive similarity ------
    ax_a = fig.add_subplot(gs[0, 0])
    plot_two_kde_on_ax(
        ax_a, pool_by_prefix(dude_decoy, "active_active"), "Active - active",
        pool_by_prefix(dude_decoy, "active_decoy"), "Active - decoy",
        "DUD-E: active-active vs. active-decoy",
        color_a=BLUE, color_b=AQUA,
    )
    add_panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    plot_two_kde_on_ax(
        ax_b, pool_by_prefix(lit_inactive, "active_active"), "Active - active",
        pool_by_prefix(lit_inactive, "active_inactive"), "Active - inactive",
        "LIT-PCBA: active-active vs. active-inactive",
        color_a=BLUE, color_b=AQUA,
    )
    add_panel_label(ax_b, "b")

    # ---- (c): shared-target AUC-ROC / EF1% / EF5%, DUD-E vs. LIT-PCBA -----
    labels = shared_table["target"].tolist()
    x = np.arange(len(labels))
    width = 0.35
    metric_panels = [
        ("auc_mean", "auc_std", "AUC-ROC", 0.5),
        ("ef1pct_mean", "ef1pct_std", "Enrichment factor (top 1%)", 1.0),
        ("ef5pct_mean", "ef5pct_std", "Enrichment factor (top 5%)", 1.0),
    ]
    for row, (mean_suffix, std_suffix, ylabel, ref_line) in enumerate(metric_panels, start=1):
        ax = fig.add_subplot(gs[row, :])
        dude_vals = shared_table[f"dude_{mean_suffix}"].to_numpy()
        dude_err = shared_table[f"dude_{std_suffix}"].to_numpy()
        lit_vals = shared_table[f"litpcba_{mean_suffix}"].to_numpy()
        lit_err = shared_table[f"litpcba_{std_suffix}"].to_numpy()

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
        if row == 1:
            add_panel_label(ax, "c")

    out_pdf = out_dir / f"{args.out_name}.pdf"
    fig.savefig(out_pdf, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out_dir / f"{args.out_name}.png", facecolor=SURFACE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
