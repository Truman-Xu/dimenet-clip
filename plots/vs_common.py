"""
Shared helpers for the similarity-search virtual-screening experiment:
for a random active "query" per target, rank every other ligand of that
target (actives + all decoys/inactives) by ECFP4 Tanimoto similarity to the
query, then score that ranking with AUC-ROC and enrichment factors, and
separately report simple hit-count/precision/recall stats at fixed
similarity thresholds (0.5 / 0.7 / 0.9).

Used by dude_vs_screening.py and lit_pcba_vs_screening.py.
"""
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import roc_auc_score

from rdkit import DataStructs
from matplotlib.patches import Patch

from fp_similarity_common import (
    smiles_to_fp, style_axes, SURFACE, GRID, INK_MUTED, INK_SECONDARY, SEQ_RAMP, BLUE, AQUA,
)


def build_or_load_library(cache_path: Path, actives_smiles, inactives_smiles, radius, n_bits):
    """Fingerprint every active/inactive SMILES for a target (all of them,
    not a subsample), cached to disk as a pickle of (fps, labels)."""
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            fps, labels = pickle.load(f)
        return fps, labels

    fps = []
    labels = []
    for smi in actives_smiles:
        fp = smiles_to_fp(smi, radius, n_bits)
        if fp is not None:
            fps.append(fp)
            labels.append(1)
    for smi in inactives_smiles:
        fp = smiles_to_fp(smi, radius, n_bits)
        if fp is not None:
            fps.append(fp)
            labels.append(0)
    labels = np.array(labels, dtype=np.int8)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((fps, labels), f, protocol=pickle.HIGHEST_PROTOCOL)
    return fps, labels


def score_clip_pair(lig_emb, prot_emb, labels):
    """Dot-product screening score between one pocket embedding and every
    ligand embedding for a target (matches the CLIP eval notebooks)."""
    pred = torch.matmul(lig_emb, prot_emb.T).flatten()
    finite = torch.isfinite(pred)
    y_score = pred[finite].numpy().astype(np.float64)
    y_true = labels[finite].numpy().astype(np.int8)
    return y_true, y_score


def enrichment_factor(y_true, y_score, frac):
    n = len(y_score)
    n_sel = max(1, int(np.ceil(frac * n)))
    order = np.argsort(-y_score)
    top_idx = order[:n_sel]
    hits_in_top = y_true[top_idx].sum()
    total_actives = y_true.sum()
    if total_actives == 0:
        return np.nan
    expected = n_sel * (total_actives / n)
    return float(hits_in_top / expected)


def threshold_stats(y_true, y_score, threshold):
    mask = y_score >= threshold
    n_sel = int(mask.sum())
    n_actives_sel = int(y_true[mask].sum()) if n_sel else 0
    n_inactives_sel = n_sel - n_actives_sel
    total_actives = int(y_true.sum())
    precision = n_actives_sel / n_sel if n_sel > 0 else np.nan
    recall = n_actives_sel / total_actives if total_actives > 0 else np.nan
    return {
        "n_selected": n_sel,
        "n_actives_selected": n_actives_sel,
        "n_inactives_selected": n_inactives_sel,
        "precision": precision,
        "recall": recall,
    }


def run_replicates(fps, labels, n_replicates, thresholds, ef_fracs, seed):
    """One replicate = pick a random active as query, rank every OTHER
    ligand of the target by similarity to it, score the ranking."""
    active_idx = np.where(labels == 1)[0]
    n_actives = len(active_idx)
    n_inactives = len(labels) - n_actives
    if n_actives < 2 or n_inactives < 1:
        return []  # can't hold out a query and still have both classes

    rng = random.Random(seed)
    n_rep = min(n_replicates, n_actives)
    query_indices = rng.sample(list(active_idx), n_rep)

    labels_arr = np.asarray(labels)
    results = []
    for query_idx in query_indices:
        sims_all = np.array(DataStructs.BulkTanimotoSimilarity(fps[query_idx], fps), dtype=np.float32)
        y_score = np.delete(sims_all, query_idx)
        y_true = np.delete(labels_arr, query_idx)

        row = {
            "query_idx": int(query_idx),
            "n_scored": len(y_true),
            "auc_roc": float(roc_auc_score(y_true, y_score)),
        }
        for frac in ef_fracs:
            row[f"ef_{frac:g}"] = enrichment_factor(y_true, y_score, frac)
        for thr in thresholds:
            stats = threshold_stats(y_true, y_score, thr)
            for k, v in stats.items():
                row[f"thr{thr:g}_{k}"] = v
        results.append(row)
    return results


def plot_metric_boxplot_on_ax(ax, per_target_values, summary_df, sort_col, value_label, title, ref_line=None):
    """Horizontal boxplot of a per-replicate metric (AUC, EF1%, EF5%, ...)
    across targets, sorted by sort_col, colored by that same column. Draws
    onto the given ax (no figure creation/saving) so it can be composed into
    a multi-panel figure; returns (sm, norm) so the caller can attach its
    own colorbar. See plot_metric_boxplot for the standalone-figure wrapper."""
    n_targets = len(summary_df)
    data = [per_target_values[t] for t in summary_df["target"]]
    positions = np.arange(1, n_targets + 1)
    bp = ax.boxplot(data, positions=positions, vert=False, widths=0.7,
                     patch_artist=True, showfliers=False,
                     medianprops={"color": "#0b0b0b", "linewidth": 1.5})

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    vals = summary_df[sort_col].fillna(0).to_numpy()
    norm = Normalize(vmin=vals.min(), vmax=vals.max())
    for patch, v in zip(bp["boxes"], vals):
        patch.set_facecolor(cmap(norm(v)))
        patch.set_edgecolor(INK_MUTED)
        patch.set_linewidth(0.6)
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color(INK_MUTED)
            line.set_linewidth(0.8)

    if ref_line is not None:
        ax.axvline(ref_line, color=INK_MUTED, linestyle="--", linewidth=1, zorder=0)

    ax.set_yticks(positions)
    ax.set_yticklabels(summary_df["target"].str.upper(), fontsize=7)
    ax.set_xlabel(value_label)
    ax.set_title(title)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm, norm


def plot_metric_boxplot(per_target_values, summary_df, sort_col, value_label, title, out_path, ref_line=None,
                         save_pdf=False):
    """Standalone-figure wrapper around plot_metric_boxplot_on_ax: creates
    its own fig/ax, draws the boxplot, attaches a colorbar, and saves."""
    n_targets = len(summary_df)
    fig_h = max(6, 0.18 * n_targets)
    fig, ax = plt.subplots(figsize=(9, fig_h), dpi=150)
    sm, norm = plot_metric_boxplot_on_ax(ax, per_target_values, summary_df, sort_col, value_label, title, ref_line)

    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label(f"mean {value_label}", color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)


def plot_metric_bar(summary_df, sort_col, value_label, title, out_path, ref_lines=None,
                     err_col=None, xlim=None):
    """Horizontal bar chart of a per-target metric that has no within-target
    replicate spread (e.g. DUD-E CLIP, one pocket structure per target), so
    a boxplot would just be a degenerate n=1 box. Sorted and colored by
    sort_col, same convention as plot_metric_boxplot. ref_lines is an
    optional list of (value, color, label) tuples drawn as vertical
    reference lines. err_col, if given, is a column of summary_df plotted
    as symmetric error bars (e.g. std across replicates/pocket structures -
    NaN, e.g. from a single-replicate target, is treated as 0). xlim, if
    given, overrides the default autoscaled x-axis range (e.g. to match
    another plot's scale for direct visual comparison)."""
    n_targets = len(summary_df)
    fig_h = max(6, 0.18 * n_targets)
    fig, ax = plt.subplots(figsize=(9, fig_h), dpi=150)
    vals = summary_df[sort_col].fillna(0).to_numpy()
    errs = summary_df[err_col].fillna(0).to_numpy() if err_col else None
    positions = np.arange(1, n_targets + 1)

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    norm = Normalize(vmin=vals.min(), vmax=vals.max())
    ax.barh(positions, vals, height=0.7, color=[cmap(norm(v)) for v in vals],
            edgecolor=INK_MUTED, linewidth=0.6, zorder=3,
            xerr=errs, error_kw=dict(ecolor=INK_SECONDARY, capsize=3, linewidth=1, zorder=4))

    for ref_val, ref_color, ref_label in (ref_lines or []):
        ax.axvline(ref_val, color=ref_color, linestyle="--", linewidth=1.3, zorder=5, label=ref_label)

    ax.set_yticks(positions)
    ax.set_yticklabels(summary_df["target"].str.upper(), fontsize=7)
    ax.set_xlabel(value_label)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    if ref_lines:
        ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9, loc="lower right")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label(value_label, color=INK_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_boxplot_with_reference_marker(null_values_by_target, marker_value_by_target, targets_sorted,
                                        value_label, title, out_path, ref_line=None,
                                        null_label="Mismatched pocket", marker_label="Correct pocket",
                                        box_color=AQUA, marker_color=BLUE, save_pdf=False):
    """Horizontal boxplot of a per-target null/control distribution (e.g.
    scoring a target's ligands against every OTHER target's pocket
    embedding), with a single reference value per target (e.g. the score
    using that target's own, correct pocket) overlaid as a diamond marker.
    Targets are plotted in the caller-supplied order (top = last)."""
    n_targets = len(targets_sorted)
    fig_h = max(6, 0.22 * n_targets)
    fig, ax = plt.subplots(figsize=(9, fig_h), dpi=150)
    positions = np.arange(1, n_targets + 1)

    data = [null_values_by_target[t] for t in targets_sorted]
    bp = ax.boxplot(data, positions=positions, vert=False, widths=0.6,
                     patch_artist=True, showfliers=False,
                     medianprops={"color": "#0b0b0b", "linewidth": 1.3})
    for patch in bp["boxes"]:
        patch.set_facecolor(box_color)
        patch.set_edgecolor(INK_MUTED)
        patch.set_alpha(0.55)
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color(INK_MUTED)
            line.set_linewidth(0.8)

    marker_vals = [marker_value_by_target[t] for t in targets_sorted]
    scatter = ax.scatter(marker_vals, positions, marker="D", s=42, color=marker_color,
                          edgecolor=SURFACE, linewidth=0.7, zorder=5, label=marker_label)

    if ref_line is not None:
        ax.axvline(ref_line, color=INK_MUTED, linestyle="--", linewidth=1, zorder=0)

    ax.set_yticks(positions)
    ax.set_yticklabels([t.upper() for t in targets_sorted], fontsize=7)
    ax.set_xlabel(value_label)
    ax.set_title(title)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    box_proxy = Patch(facecolor=box_color, edgecolor=INK_MUTED, alpha=0.55, label=null_label)
    ax.legend(handles=[box_proxy, scatter], frameon=False, labelcolor=INK_SECONDARY,
              fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
