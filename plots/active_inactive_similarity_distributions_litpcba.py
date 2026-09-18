"""
For every LIT-PCBA target, compare two pairwise ECFP4 Tanimoto similarity
distributions:
  1. active-active   : similarity between pairs of known active ligands
  2. active-inactive  : similarity between actives and inactives (real
     screening non-hits from the same assay, not property-matched decoys)

Mirrors active_decoy_similarity_distributions.py (DUD-E, Experiment 4) so
the two datasets are directly comparable; the only substantive difference
is the provenance of the negative class (see methods.tex, "analog bias").

Reuses the full-library fingerprints already cached by
vs_screening_analysis.py --dataset litpcba
(lit-pcba/vs_screening_analysis/fp_cache/<target>.pkl) - no refingerprinting
needed. To keep the active x inactive cross product tractable, each
target's actives/inactives are randomly subsampled to --max-actives /
--max-inactives before computing exact pairwise similarity on the
subsample.

Produces, in --output-dir:
  - per_target_stats.csv : mean/median/std/n_pairs for both distributions,
    per target.
  - pooled_active_active_vs_active_inactive.png : the two distributions
    pooled across all targets, overlaid on one KDE plot.
  - per_target_active_active_vs_active_inactive_boxplot.png : grouped
    horizontal boxplot, both distributions side by side for every target.
  - raw_similarities.npz : the raw per-target arrays.

Usage:
    python active_inactive_similarity_distributions_litpcba.py
"""
import argparse
import paths  # default input/output locations, see plots/paths.py
import random
from pathlib import Path

import numpy as np
import pandas as pd

from fp_similarity_common import (
    all_pairs_similarity, cross_pairs_similarity, plot_two_kde_comparison,
    plot_grouped_horizontal_boxplot, read_smiles_file, BLUE, AQUA,
)
from vs_common import build_or_load_library


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lit-pcba-dir", default=str(paths.LIT_PCBA_DIR))
    parser.add_argument("--fp-cache-dir", default=str(paths.out("lit-pcba", "vs_screening_analysis") / "fp_cache"),
                         help="Fingerprint cache built by vs_screening_analysis.py --dataset litpcba.")
    parser.add_argument("--output-dir", default=str(paths.out("lit-pcba", "active_inactive_similarity_analysis")))
    parser.add_argument("--max-actives", type=int, default=300)
    parser.add_argument("--max-inactives", type=int, default=300)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(args.lit_pcba_dir)
    cache_dir = Path(args.fp_cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    target_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir()
                          and (p / "actives.smi").exists())
    print(f"Found {len(target_dirs)} LIT-PCBA targets in {base_dir}")

    active_active_sims = {}
    active_inactive_sims = {}
    rows = []

    for i, target_dir in enumerate(target_dirs, 1):
        target = target_dir.name
        cache_path = cache_dir / f"{target}.pkl"
        if cache_path.exists():
            fps, labels = build_or_load_library(cache_path, [], [], args.radius, args.n_bits)
        else:
            actives = read_smiles_file(target_dir / "actives.smi")
            inactives = read_smiles_file(target_dir / "inactives.smi")
            fps, labels = build_or_load_library(cache_path, actives, inactives, args.radius, args.n_bits)

        active_idx = np.where(labels == 1)[0]
        inactive_idx = np.where(labels == 0)[0]

        n_act_sample = min(args.max_actives, len(active_idx))
        n_inact_sample = min(args.max_inactives, len(inactive_idx))
        act_sample_idx = rng.sample(list(active_idx), n_act_sample)
        inact_sample_idx = rng.sample(list(inactive_idx), n_inact_sample)

        act_fps = [fps[i] for i in act_sample_idx]
        inact_fps = [fps[i] for i in inact_sample_idx]

        aa_sims = all_pairs_similarity(act_fps) if len(act_fps) > 1 else np.array([], dtype=np.float32)
        ai_sims = cross_pairs_similarity(act_fps, inact_fps) if act_fps and inact_fps else np.array([], dtype=np.float32)

        active_active_sims[target] = aa_sims
        active_inactive_sims[target] = ai_sims

        rows.append({
            "target": target,
            "n_actives_sampled": n_act_sample, "n_inactives_sampled": n_inact_sample,
            "n_active_active_pairs": len(aa_sims), "n_active_inactive_pairs": len(ai_sims),
            "active_active_mean": float(aa_sims.mean()) if len(aa_sims) else np.nan,
            "active_active_median": float(np.median(aa_sims)) if len(aa_sims) else np.nan,
            "active_active_std": float(aa_sims.std()) if len(aa_sims) else np.nan,
            "active_inactive_mean": float(ai_sims.mean()) if len(ai_sims) else np.nan,
            "active_inactive_median": float(np.median(ai_sims)) if len(ai_sims) else np.nan,
            "active_inactive_std": float(ai_sims.std()) if len(ai_sims) else np.nan,
        })
        print(f"[{i}/{len(target_dirs)}] {target}: {n_act_sample} actives, {n_inact_sample} inactives sampled — "
              f"active-active mean={rows[-1]['active_active_mean']:.3f}, "
              f"active-inactive mean={rows[-1]['active_inactive_mean']:.3f}")

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "per_target_stats.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nWrote per-target stats -> {summary_csv}")

    np.savez(out_dir / "raw_similarities.npz",
             **{f"active_active__{k}": v for k, v in active_active_sims.items()},
             **{f"active_inactive__{k}": v for k, v in active_inactive_sims.items()})
    print(f"Wrote raw similarity arrays -> {out_dir / 'raw_similarities.npz'}")

    # ============================= PLOTS ====================================

    pooled_aa = np.concatenate([v for v in active_active_sims.values() if len(v)])
    pooled_ai = np.concatenate([v for v in active_inactive_sims.values() if len(v)])

    plot_two_kde_comparison(
        pooled_aa, "Active - active",
        pooled_ai, "Active - inactive",
        "Pooled ECFP4 pairwise similarity — LIT-PCBA\nactive-active vs. active-inactive (all targets combined)",
        out_dir / "pooled_active_active_vs_active_inactive.png",
        color_a=BLUE, color_b=AQUA, save_pdf=True,
    )
    print(f"\nWrote {out_dir / 'pooled_active_active_vs_active_inactive.png'}")

    sort_order = summary_df.sort_values("active_active_median", ascending=True)["target"].tolist()
    plot_grouped_horizontal_boxplot(
        active_active_sims, "Active - active", active_inactive_sims, "Active - inactive", sort_order,
        xlabel="Pairwise Tanimoto similarity (ECFP4)",
        title="Per-target ECFP4 pairwise similarity — LIT-PCBA\nactive-active vs. active-inactive (sorted by active-active median)",
        out_path=out_dir / "per_target_active_active_vs_active_inactive_boxplot.png",
        color_a=BLUE, color_b=AQUA,
    )
    print(f"Wrote {out_dir / 'per_target_active_active_vs_active_inactive_boxplot.png'}")


if __name__ == "__main__":
    main()
