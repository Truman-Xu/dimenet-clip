"""
ECFP4 pairwise Tanimoto similarity analysis for the DUD-E ligand set.

For every target in <dude-dir>/all/<target>/{actives_final.ism,decoys_final.ism}
this script:
  1. Randomly (stratified actives/decoys) samples up to --target-sample-size
     ligands per target and computes ECFP4 (Morgan radius=2, 2048-bit)
     fingerprints for them.
  2. Computes all-pairs Tanimoto similarity within each target -> per-target
     distribution ("grouped by protein target").
  3. Pools a random subsample (--global-sample-size) drawn from all sampled
     ligands (regardless of target) and computes all-pairs Tanimoto similarity
     -> global distribution ("all fingerprints").
  4. Writes summary CSVs, a raw-data .npz, and three plots to --output-dir.

Usage:
    python ecfp4_similarity_analysis.py \
        --dude-dir /path/to/dude \
        --output-dir plots/output/dude/similarity_analysis
"""
import argparse
import paths  # default input/output locations, see plots/paths.py
import random
from pathlib import Path

import numpy as np
import pandas as pd

from fp_similarity_common import (
    read_smiles_file, sample_stratified, smiles_to_fp, all_pairs_similarity,
    plot_global_histogram, plot_per_target_boxplot, plot_two_kde_comparison,
)


def get_ligand_smiles(target_dir: Path):
    actives = read_smiles_file(target_dir / "actives_final.ism")
    decoys = read_smiles_file(target_dir / "decoys_final.ism")
    return actives, decoys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dude-dir", default=str(paths.DUDE_DIR),
                         help="Directory containing one subfolder per DUD-E target.")
    parser.add_argument("--output-dir", default=str(paths.out("dude", "similarity_analysis")))
    parser.add_argument("--target-sample-size", type=int, default=300,
                         help="Max ligands sampled per target (stratified actives/decoys).")
    parser.add_argument("--global-sample-size", type=int, default=3000,
                         help="Max ligands sampled for the global (all-target) pool.")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius (2 = ECFP4).")
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    dude_dir = Path(args.dude_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_dirs = sorted(p for p in dude_dir.iterdir() if p.is_dir()
                          and (p / "actives_final.ism").exists())
    print(f"Found {len(target_dirs)} DUD-E targets in {dude_dir}")

    per_target_fps = {}     # target -> list of ExplicitBitVect
    per_target_meta = []    # rows for summary csv

    for i, target_dir in enumerate(target_dirs, 1):
        target = target_dir.name
        actives, decoys = get_ligand_smiles(target_dir)
        act_sample, dec_sample = sample_stratified(actives, decoys, args.target_sample_size, rng)

        fps = []
        n_act_ok = n_dec_ok = 0
        for smi in act_sample:
            fp = smiles_to_fp(smi, args.radius, args.n_bits)
            if fp is not None:
                fps.append(fp)
                n_act_ok += 1
        for smi in dec_sample:
            fp = smiles_to_fp(smi, args.radius, args.n_bits)
            if fp is not None:
                fps.append(fp)
                n_dec_ok += 1

        per_target_fps[target] = fps
        per_target_meta.append({
            "target": target,
            "n_actives_total": len(actives),
            "n_decoys_total": len(decoys),
            "n_actives_sampled": n_act_ok,
            "n_decoys_sampled": n_dec_ok,
            "n_sampled": len(fps),
        })
        print(f"[{i}/{len(target_dirs)}] {target}: sampled {len(fps)} ligands "
              f"({n_act_ok} actives, {n_dec_ok} decoys)")

    # ---- per-target pairwise similarity ------------------------------------
    per_target_sims = {}
    for row in per_target_meta:
        target = row["target"]
        fps = per_target_fps[target]
        sims = all_pairs_similarity(fps) if len(fps) > 1 else np.array([], dtype=np.float32)
        per_target_sims[target] = sims
        row["n_pairs"] = len(sims)
        row["mean_similarity"] = float(sims.mean()) if len(sims) else np.nan
        row["median_similarity"] = float(np.median(sims)) if len(sims) else np.nan
        row["std_similarity"] = float(sims.std()) if len(sims) else np.nan

    summary_df = pd.DataFrame(per_target_meta).sort_values("median_similarity")
    summary_csv = out_dir / "per_target_similarity_stats.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nWrote per-target stats -> {summary_csv}")

    # ---- global pool: random subsample drawn from all sampled ligands -----
    all_fps = [fp for fps in per_target_fps.values() for fp in fps]
    if len(all_fps) > args.global_sample_size:
        idx = np_rng.choice(len(all_fps), size=args.global_sample_size, replace=False)
        global_fps = [all_fps[i] for i in idx]
    else:
        global_fps = all_fps
    print(f"Global pool: {len(global_fps)} ligands drawn from {len(all_fps)} total sampled")

    global_sims = all_pairs_similarity(global_fps)

    # pooled within-target similarities (union of every target's own pairs)
    within_target_sims = np.concatenate([s for s in per_target_sims.values() if len(s)])

    np.savez(out_dir / "raw_similarities.npz",
             global_similarities=global_sims,
             within_target_similarities=within_target_sims,
             **{f"target__{k}": v for k, v in per_target_sims.items()})
    print(f"Wrote raw similarity arrays -> {out_dir / 'raw_similarities.npz'}")

    # ============================= PLOTS ====================================

    plot_global_histogram(
        global_sims, len(global_fps),
        "Distribution of pairwise ECFP4 similarities — all DUD-E ligands",
        out_dir / "global_similarity_distribution.png",
    )

    plot_per_target_boxplot(
        per_target_sims, summary_df,
        "Per-target pairwise ECFP4 similarity distributions\n(sorted by median, actives+decoys sampled per target)",
        out_dir / "per_target_similarity_boxplot.png",
    )

    plot_two_kde_comparison(
        global_sims, "All ligands, global pool",
        within_target_sims, "Pooled within-target pairs",
        "Global vs. within-target ECFP4 similarity",
        out_dir / "global_vs_within_target_comparison.png",
        save_pdf=True,
    )

    print(f"\nSaved plots to {out_dir}:")
    print("  - global_similarity_distribution.png")
    print("  - per_target_similarity_boxplot.png")
    print("  - global_vs_within_target_comparison.png")


if __name__ == "__main__":
    main()
