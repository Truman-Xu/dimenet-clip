"""
For every DUD-E target, compare two pairwise ECFP4 Tanimoto similarity
distributions:
  1. active-active   : similarity between pairs of known active ligands
  2. active-decoy    : similarity between actives and decoys

Reuses the full-library fingerprints already cached by vs_screening_analysis.py
(fp_cache/<target>.pkl) - no refingerprinting needed. To keep the
active x decoy cross product tractable across all 102 targets, each target's
actives/decoys are randomly subsampled to --max-actives / --max-decoys
before computing exact pairwise similarity on the subsample.

Produces, in --output-dir:
  - per_target_stats.csv : mean/median/std/n_pairs for both distributions,
    per target.
  - pooled_active_active_vs_active_decoy.png : the two distributions pooled
    across all targets, overlaid on one plot.
  - per_target_active_active_vs_active_decoy_boxplot.png : grouped
    horizontal boxplot, both distributions side by side for every target.
  - raw_similarities.npz : the raw per-target arrays.

Usage:
    python active_decoy_similarity_distributions.py
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
    parser.add_argument("--dude-dir", default=str(paths.DUDE_DIR))
    parser.add_argument("--fp-cache-dir", default=str(paths.out("dude", "vs_screening_analysis") / "fp_cache"),
                         help="Fingerprint cache built by vs_screening_analysis.py --dataset dude.")
    parser.add_argument("--output-dir", default=str(paths.out("dude", "active_decoy_similarity_analysis")))
    parser.add_argument("--max-actives", type=int, default=300)
    parser.add_argument("--max-decoys", type=int, default=300)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(args.dude_dir)
    cache_dir = Path(args.fp_cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    target_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir()
                          and (p / "actives_final.ism").exists())
    print(f"Found {len(target_dirs)} DUD-E targets in {base_dir}")

    active_active_sims = {}
    active_decoy_sims = {}
    rows = []

    for i, target_dir in enumerate(target_dirs, 1):
        target = target_dir.name
        cache_path = cache_dir / f"{target}.pkl"
        if cache_path.exists():
            fps, labels = build_or_load_library(cache_path, [], [], args.radius, args.n_bits)
        else:
            actives = read_smiles_file(target_dir / "actives_final.ism")
            decoys = read_smiles_file(target_dir / "decoys_final.ism")
            fps, labels = build_or_load_library(cache_path, actives, decoys, args.radius, args.n_bits)

        active_idx = np.where(labels == 1)[0]
        decoy_idx = np.where(labels == 0)[0]

        n_act_sample = min(args.max_actives, len(active_idx))
        n_dec_sample = min(args.max_decoys, len(decoy_idx))
        act_sample_idx = rng.sample(list(active_idx), n_act_sample)
        dec_sample_idx = rng.sample(list(decoy_idx), n_dec_sample)

        act_fps = [fps[i] for i in act_sample_idx]
        dec_fps = [fps[i] for i in dec_sample_idx]

        aa_sims = all_pairs_similarity(act_fps) if len(act_fps) > 1 else np.array([], dtype=np.float32)
        ad_sims = cross_pairs_similarity(act_fps, dec_fps) if act_fps and dec_fps else np.array([], dtype=np.float32)

        active_active_sims[target] = aa_sims
        active_decoy_sims[target] = ad_sims

        rows.append({
            "target": target,
            "n_actives_sampled": n_act_sample, "n_decoys_sampled": n_dec_sample,
            "n_active_active_pairs": len(aa_sims), "n_active_decoy_pairs": len(ad_sims),
            "active_active_mean": float(aa_sims.mean()) if len(aa_sims) else np.nan,
            "active_active_median": float(np.median(aa_sims)) if len(aa_sims) else np.nan,
            "active_active_std": float(aa_sims.std()) if len(aa_sims) else np.nan,
            "active_decoy_mean": float(ad_sims.mean()) if len(ad_sims) else np.nan,
            "active_decoy_median": float(np.median(ad_sims)) if len(ad_sims) else np.nan,
            "active_decoy_std": float(ad_sims.std()) if len(ad_sims) else np.nan,
        })
        print(f"[{i}/{len(target_dirs)}] {target}: {n_act_sample} actives, {n_dec_sample} decoys sampled — "
              f"active-active mean={rows[-1]['active_active_mean']:.3f}, "
              f"active-decoy mean={rows[-1]['active_decoy_mean']:.3f}")

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "per_target_stats.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nWrote per-target stats -> {summary_csv}")

    np.savez(out_dir / "raw_similarities.npz",
             **{f"active_active__{k}": v for k, v in active_active_sims.items()},
             **{f"active_decoy__{k}": v for k, v in active_decoy_sims.items()})
    print(f"Wrote raw similarity arrays -> {out_dir / 'raw_similarities.npz'}")

    # ============================= PLOTS ====================================

    pooled_aa = np.concatenate([v for v in active_active_sims.values() if len(v)])
    pooled_ad = np.concatenate([v for v in active_decoy_sims.values() if len(v)])

    plot_two_kde_comparison(
        pooled_aa, "Active - active",
        pooled_ad, "Active - decoy",
        "Pooled ECFP4 pairwise similarity — DUD-E\nactive-active vs. active-decoy (all targets combined)",
        out_dir / "pooled_active_active_vs_active_decoy.png",
        color_a=BLUE, color_b=AQUA, save_pdf=True,
    )
    print(f"\nWrote {out_dir / 'pooled_active_active_vs_active_decoy.png'}")

    sort_order = summary_df.sort_values("active_active_median", ascending=True)["target"].tolist()
    plot_grouped_horizontal_boxplot(
        active_active_sims, "Active - active", active_decoy_sims, "Active - decoy", sort_order,
        xlabel="Pairwise Tanimoto similarity (ECFP4)",
        title="Per-target ECFP4 pairwise similarity — DUD-E\nactive-active vs. active-decoy (sorted by active-active median)",
        out_path=out_dir / "per_target_active_active_vs_active_decoy_boxplot.png",
        color_a=BLUE, color_b=AQUA,
    )
    print(f"Wrote {out_dir / 'per_target_active_active_vs_active_decoy_boxplot.png'}")


if __name__ == "__main__":
    main()
