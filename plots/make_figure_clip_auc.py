"""
Assemble the manuscript figure "clip-auc": DimeNet-CLIP per-target AUC-ROC on
DUD-E (left/top, single pocket per target - no replicate spread) and LIT-PCBA
(right/bottom, mean +- std across available pocket structures), side by side
as one two-panel vector figure.

Requires clip_screening_analysis_dude.py and clip_screening_analysis_litpcba.py
to have already been run (reads their per_target_summary.csv and
replicate_level_results.csv - no re-scoring here).

Usage:
    python make_figure_clip_auc.py
"""
import argparse
import paths  # default input/output locations, see plots/paths.py
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from fp_similarity_common import SURFACE, INK_SECONDARY
from vs_common import plot_metric_boxplot_on_ax


def load_panel_data(summary_csv, replicate_csv):
    summary = pd.read_csv(summary_csv)
    replicate_df = pd.read_csv(replicate_csv)
    sort_order = summary.sort_values("auc_roc_mean")
    per_target_auc = {t: replicate_df.loc[replicate_df["target"] == t, "auc_roc"].to_numpy()
                       for t in sort_order["target"]}
    return per_target_auc, sort_order


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dude-dir", default=str(paths.out("dude", "clip_vs_screening_analysis")))
    parser.add_argument("--lit-pcba-dir", default=str(paths.out("lit-pcba", "clip_vs_screening_analysis")))
    parser.add_argument("--output-dir", default=str(paths.FIGURES_DIR))
    parser.add_argument("--out-name", default="clip-auc")
    args = parser.parse_args()

    dude_dir = Path(args.dude_dir)
    lit_dir = Path(args.lit_pcba_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dude_auc, dude_sort = load_panel_data(dude_dir / "per_target_summary.csv", dude_dir / "replicate_level_results.csv")
    lit_auc, lit_sort = load_panel_data(lit_dir / "per_target_summary.csv", lit_dir / "replicate_level_results.csv")

    n_targets = max(len(dude_sort), len(lit_sort))
    fig_h = max(6, 0.18 * n_targets)
    fig, axes = plt.subplots(1, 2, figsize=(15, fig_h), dpi=150)

    panels = [
        (axes[0], dude_auc, dude_sort, "DUD-E\n(single pocket structure per target - no replicate spread)", "a"),
        (axes[1], lit_auc, lit_sort, "LIT-PCBA\n(mean over all available pocket structures per target)", "b"),
    ]
    for ax, per_target_auc, sort_order, subtitle, label in panels:
        sm, norm = plot_metric_boxplot_on_ax(
            ax, per_target_auc, sort_order, sort_col="auc_roc_mean", value_label="AUC-ROC",
            title=subtitle, ref_line=0.5,
        )
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label("mean AUC-ROC", color=INK_SECONDARY, fontsize=9)
        cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        ax.text(-0.02, 1.02, label, transform=ax.transAxes, fontsize=15, fontweight="bold",
                va="bottom", ha="right")

    fig.suptitle("DimeNet-CLIP virtual-screening AUC-ROC by target", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_pdf = out_dir / f"{args.out_name}.pdf"
    fig.savefig(out_pdf, facecolor=SURFACE)
    fig.savefig(out_dir / f"{args.out_name}.png", facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
