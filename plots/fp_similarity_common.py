"""
Shared helpers for ECFP4 pairwise-Tanimoto-similarity analyses
(used by ecfp4_similarity_analysis.py, lit_pcba_similarity_analysis.py,
and compare_dude_lit_pcba.py) so both datasets are processed and plotted
with identical fingerprinting parameters and chart styling.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs

RDLogger.DisableLog("rdApp.*")

# ---- palette (validated categorical + sequential ramp, light mode) --------
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BLUE = "#2a78d6"       # series 1
AQUA = "#1baf7a"       # series 2
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK_PRIMARY)


def read_smiles_file(path):
    smiles = []
    if not path.exists():
        return smiles
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts:
                smiles.append(parts[0])
    return smiles


def sample_stratified(list_a, list_b, sample_size, rng):
    """Split sample_size roughly evenly between list_a/list_b, topping up
    from whichever has more if the other runs short."""
    n_a = min(sample_size // 2, len(list_a))
    n_b = min(sample_size - n_a, len(list_b))
    if n_b < sample_size - n_a:
        n_a = min(sample_size - n_b, len(list_a))
    a_sample = rng.sample(list_a, n_a) if n_a else []
    b_sample = rng.sample(list_b, n_b) if n_b else []
    return a_sample, b_sample


def smiles_to_fp(smi, radius, n_bits):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def all_pairs_similarity(fps):
    sims = []
    for i in range(len(fps) - 1):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    return np.array(sims, dtype=np.float32)


def cross_pairs_similarity(fps_a, fps_b):
    """Every pairwise similarity between two disjoint fingerprint lists."""
    sims = []
    for fp in fps_a:
        sims.extend(DataStructs.BulkTanimotoSimilarity(fp, fps_b))
    return np.array(sims, dtype=np.float32)


def plot_global_histogram(sims, n_ligands, title, out_path, color=BLUE):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(sims, bins=60, range=(0, 1), color=color, alpha=0.85,
            edgecolor=SURFACE, linewidth=0.5, density=True)
    kde = gaussian_kde(sims)
    xs = np.linspace(0, 1, 300)
    ax.plot(xs, kde(xs), color="#0d366b", linewidth=2)
    mean_v = sims.mean()
    ax.axvline(mean_v, color=INK_PRIMARY, linestyle="--", linewidth=1)
    ax.text(mean_v, ax.get_ylim()[1] * 0.95, f" mean={mean_v:.3f}",
            color=INK_PRIMARY, fontsize=9, va="top")
    ax.set_xlabel("Pairwise Tanimoto similarity (ECFP4)")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\n(n={n_ligands} ligands, {len(sims):,} pairs)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_per_target_boxplot(per_target_sims, summary_df, title, out_path):
    n_targets = len(summary_df)
    fig_h = max(6, 0.16 * n_targets)
    fig, ax = plt.subplots(figsize=(9, fig_h), dpi=150)
    data = [per_target_sims[t] for t in summary_df["target"]]
    positions = np.arange(1, n_targets + 1)
    bp = ax.boxplot(data, positions=positions, vert=False, widths=0.7,
                     patch_artist=True, showfliers=False,
                     medianprops={"color": INK_PRIMARY, "linewidth": 1.5})

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    medians = summary_df["median_similarity"].fillna(0).to_numpy()
    norm = Normalize(vmin=medians.min(), vmax=medians.max())
    for patch, med in zip(bp["boxes"], medians):
        patch.set_facecolor(cmap(norm(med)))
        patch.set_edgecolor(INK_MUTED)
        patch.set_linewidth(0.6)
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color(INK_MUTED)
            line.set_linewidth(0.8)

    ax.set_yticks(positions)
    ax.set_yticklabels(summary_df["target"].str.upper(), fontsize=7)
    ax.set_xlabel("Pairwise Tanimoto similarity (ECFP4)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Median similarity", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_grouped_horizontal_boxplot(data_a_by_target, label_a, data_b_by_target, label_b, targets_sorted,
                                     xlabel, title, out_path, color_a=BLUE, color_b=AQUA, ref_line=None):
    """Two boxes per target row (e.g. active-active vs active-decoy
    similarity), sorted in the caller-supplied target order."""
    n_targets = len(targets_sorted)
    fig_h = max(6, 0.2 * n_targets)
    fig, ax = plt.subplots(figsize=(9, fig_h), dpi=150)
    positions = np.arange(1, n_targets + 1)
    offset = 0.19

    data_a = [data_a_by_target[t] for t in targets_sorted]
    data_b = [data_b_by_target[t] for t in targets_sorted]

    bp_a = ax.boxplot(data_a, positions=positions - offset, vert=False, widths=0.32,
                       patch_artist=True, showfliers=False,
                       medianprops={"color": INK_PRIMARY, "linewidth": 1.2})
    bp_b = ax.boxplot(data_b, positions=positions + offset, vert=False, widths=0.32,
                       patch_artist=True, showfliers=False,
                       medianprops={"color": INK_PRIMARY, "linewidth": 1.2})
    for patch in bp_a["boxes"]:
        patch.set_facecolor(color_a)
        patch.set_edgecolor(SURFACE)
    for patch in bp_b["boxes"]:
        patch.set_facecolor(color_b)
        patch.set_edgecolor(SURFACE)
    for bp in (bp_a, bp_b):
        for element in ("whiskers", "caps"):
            for line in bp[element]:
                line.set_color(INK_MUTED)
                line.set_linewidth(0.7)

    if ref_line is not None:
        ax.axvline(ref_line, color=INK_MUTED, linestyle="--", linewidth=1, zorder=0)

    ax.set_yticks(positions)
    ax.set_yticklabels([t.upper() for t in targets_sorted], fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.legend([bp_a["boxes"][0], bp_b["boxes"][0]], [label_a, label_b],
              frameon=False, labelcolor=INK_SECONDARY, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_two_kde_on_ax(ax, sims_a, label_a, sims_b, label_b, title,
                        color_a=BLUE, color_b=AQUA, xlabel="Pairwise Tanimoto similarity (ECFP4)",
                        fontsize=9, xlim=(0, 1)):
    xs = np.linspace(xlim[0], xlim[1], 300)
    kde_a = gaussian_kde(sims_a)
    kde_b = gaussian_kde(sims_b)
    ax.plot(xs, kde_a(xs), color=color_a, linewidth=2, label=f"{label_a} (n={len(sims_a):,} pairs)")
    ax.plot(xs, kde_b(xs), color=color_b, linewidth=2, label=f"{label_b} (n={len(sims_b):,} pairs)")
    ax.fill_between(xs, kde_a(xs), color=color_a, alpha=0.12)
    ax.fill_between(xs, kde_b(xs), color=color_b, alpha=0.12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=fontsize + 2)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=fontsize)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)


def plot_two_kde_comparison(sims_a, label_a, sims_b, label_b, title, out_path,
                             color_a=BLUE, color_b=AQUA, xlabel="Pairwise Tanimoto similarity (ECFP4)",
                             xlim=(0, 1), save_pdf=False):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    plot_two_kde_on_ax(ax, sims_a, label_a, sims_b, label_b, title, color_a, color_b, xlabel, xlim=xlim)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
