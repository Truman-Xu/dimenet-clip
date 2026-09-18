"""
Assemble the SI figure "global vs. within-target similarity": a 2-panel
composite of
  (a) DUD-E global vs. within-target ECFP4 similarity
  (b) LIT-PCBA global vs. within-target ECFP4 similarity

These were originally panels (a)/(b) of the main-text sims-2 figure and were
moved to the SI; sims-2.pdf now starts at what used to be panel (c) - see
make_figure_sims2.py.

Requires ecfp4_similarity_analysis.py and lit_pcba_similarity_analysis.py to
have already been run (reads their saved raw_similarities.npz arrays only -
no refingerprinting here).

Usage:
    python make_figure_si_global_similarity.py
"""
import argparse
import paths  # default input/output locations, see plots/paths.py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fp_similarity_common import plot_two_kde_on_ax, SURFACE


def add_panel_label(ax, label):
    ax.text(-0.02, 1.05, label, transform=ax.transAxes, fontsize=15, fontweight="bold",
            va="bottom", ha="right")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dude-sim-npz", default=str(paths.out("dude", "similarity_analysis") / "raw_similarities.npz"))
    parser.add_argument("--lit-pcba-sim-npz", default=str(paths.out("lit-pcba", "similarity_analysis") / "raw_similarities.npz"))
    parser.add_argument("--output-dir", default=str(paths.FIGURES_DIR))
    parser.add_argument("--out-name", default="global_vs_within_target_similarity")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dude_sim = np.load(args.dude_sim_npz)
    lit_sim = np.load(args.lit_pcba_sim_npz)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    plot_two_kde_on_ax(
        axes[0], dude_sim["global_similarities"], "All ligands, global pool",
        dude_sim["within_target_similarities"], "Pooled within-target pairs",
        "DUD-E: global vs. within-target similarity",
    )
    add_panel_label(axes[0], "a")

    plot_two_kde_on_ax(
        axes[1], lit_sim["global_similarities"], "All ligands, global pool",
        lit_sim["within_target_similarities"], "Pooled within-target pairs",
        "LIT-PCBA: global vs. within-target similarity",
    )
    add_panel_label(axes[1], "b")

    fig.tight_layout()
    out_pdf = out_dir / f"{args.out_name}.pdf"
    fig.savefig(out_pdf, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out_dir / f"{args.out_name}.png", facecolor=SURFACE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
