"""
Featurization-agnostic helpers for the AEV / DimeNet-basis conformer-variance
experiment: RDKit 3D conformer generation, L1-distance helpers over dense feature
vectors, and the two "distance in context of ECFP4 similarity" plots shared by both
featurizations (used by aev_conformer_variance_analysis.py).
"""
import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

from fp_similarity_common import (
    style_axes, plot_two_kde_comparison,
    SURFACE, GRID, INK_SECONDARY, BLUE, AQUA,
)


class ConformerBundle:
    def __init__(self, mol, symbols, heavy_mask, coords):
        self.mol = mol
        self.symbols = symbols          # list[str], length n_atoms
        self.heavy_mask = heavy_mask    # np.ndarray[bool], length n_atoms
        self.coords = coords            # np.ndarray[float32], [n_conf, n_atoms, 3]


def embed_conformers(smiles, n_conformers, seed, prune_rms_thresh=0.5, min_heavy_atoms=3):
    """Embed up to n_conformers 3D conformers for smiles via RDKit ETKDGv3 followed by
    MMFF94 optimization. Returns a ConformerBundle, or None if the molecule couldn't be
    parsed/embedded, or has too few heavy atoms to have a meaningful local environment."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() < min_heavy_atoms:
        return None

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms_thresh
    params.numThreads = 0
    cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_conformers, params=params))
    if not cids:
        return None

    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=500, numThreads=0)
    except Exception:
        pass  # fall back to raw ETKDG coordinates if MMFF has no parameters for this molecule

    symbols = [atom.GetSymbol() for atom in mol_h.GetAtoms()]
    heavy_mask = np.array([atom.GetAtomicNum() != 1 for atom in mol_h.GetAtoms()], dtype=bool)
    coords = np.stack([mol_h.GetConformer(cid).GetPositions() for cid in cids]).astype(np.float32)

    return ConformerBundle(mol_h, symbols, heavy_mask, coords)


def l1_pairwise(vectors):
    """All pairwise L1 distances among rows of a 2D array, in the same upper-triangle
    row-major order as fp_similarity_common.all_pairs_similarity (so a distance array
    and a Tanimoto-similarity array built from the same ordered input line up
    index-for-index)."""
    vectors = np.asarray(vectors)
    if len(vectors) < 2:
        return np.array([], dtype=np.float32)
    dists = []
    for i in range(len(vectors) - 1):
        dists.append(np.abs(vectors[i + 1:] - vectors[i]).sum(axis=1))
    return np.concatenate(dists).astype(np.float32)


def l1_cross(vectors_a, vectors_b):
    """All pairwise L1 distances between two disjoint sets of feature vectors (e.g.
    every conformer of molecule A against every conformer of molecule B), for a single
    highlighted pair where a robust mean over all conformer combinations is preferable
    to picking one arbitrary representative conformer per molecule."""
    a = np.asarray(vectors_a)
    b = np.asarray(vectors_b)
    dists = np.empty((len(a), len(b)), dtype=np.float32)
    for i in range(len(a)):
        dists[i] = np.abs(b - a[i]).sum(axis=1)
    return dists.ravel()


def plot_distance_kde_comparison(intra, inter, title, out_path, xlabel, save_pdf=False):
    """L1 distances (unlike Tanimoto similarity) aren't bounded to [0, 1], so the KDE
    x-range must be sized to the actual data rather than fp_similarity_common's
    Tanimoto-oriented [0, 1] default."""
    hi = max(intra.max(initial=0.0), inter.max(initial=0.0)) * 1.05
    plot_two_kde_comparison(
        inter, "Inter-molecule (different molecules)",
        intra, "Intra-molecule (same molecule, different conformer)",
        title, out_path, xlabel=xlabel, xlim=(0, hi), save_pdf=save_pdf,
    )


def plot_tanimoto_vs_feature_distance(tanimoto, feature_dist, intra_ref, title, out_path,
                                       ylabel="Feature L1 distance", xlim=(0, 1), save_pdf=False):
    """Scatter of ECFP4 Tanimoto similarity (x) vs. feature L1 distance (y) for
    inter-molecule pairs, with the intra-molecule (conformer-only) distance's
    median/IQR overlaid as a horizontal reference band -- the "noise floor" a
    reasonable-conformation baseline needs to clear."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.scatter(tanimoto, feature_dist, s=6, alpha=0.25, color=BLUE, edgecolors="none",
               label=f"Inter-molecule pairs (n={len(tanimoto):,})")

    med = np.median(intra_ref)
    q1, q3 = np.percentile(intra_ref, [25, 75])
    ax.axhspan(q1, q3, color=AQUA, alpha=0.18, zorder=0,
               label="Intra-molecule (conformer) IQR")
    ax.axhline(med, color=AQUA, linewidth=1.5, linestyle="--",
               label="Intra-molecule (conformer) median")

    ax.set_xlim(*xlim)
    ax.set_xlabel("Pairwise Tanimoto similarity (ECFP4)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
