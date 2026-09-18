"""
AEV / DimeNet-basis conformer-variance baseline for DUD-E ligands.

For a random subsample of DUD-E ligands (actives + decoys pooled across all targets --
this experiment is about molecule identity, not binder/decoy status), this script:
  1. Generates --n-conformers RDKit (ETKDGv3 + MMFF94) 3D conformers per molecule.
  2. Computes a molecular feature vector per conformer (mean over heavy atoms only)
     using TorchANI AEV and/or "raw" (untrained) DimeNet radial+angular basis features.
  3. Compares:
     - Intra-molecule distance: L1 distance between a molecule's own conformers'
       feature vectors ("conformer noise").
     - Inter-molecule distance: L1 distance between different molecules' feature
       vectors (using each molecule's first conformer as its representative, matching
       the thesis's own "first stored conformer" convention), together with the ECFP4
       Tanimoto similarity for the same molecule pairs.
  4. Writes summary CSVs, raw distance arrays, and plots to --output-dir.

Usage (needs torchani, which the main requirements do not include; see plots/README.md):
    python aev_conformer_variance_analysis.py \
        --dude-dir /path/to/dude \
        --output-dir plots/output/dude/aev_conformer_variance_analysis
"""
import argparse
import paths  # default input/output locations, see plots/paths.py
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fp_similarity_common import read_smiles_file, smiles_to_fp, all_pairs_similarity
from conformer_common import (
    embed_conformers, l1_pairwise, plot_distance_kde_comparison, plot_tanimoto_vs_feature_distance,
)
from aev_common import (
    DEFAULT_SPECIES_ORDER, make_aev_computer, make_species_converter,
    has_unsupported_elements, compute_aev_molecular_features,
)
from dimenet_basis_common import make_basis_layers, compute_dimenet_basis_molecular_features


def pool_all_smiles(dude_dir: Path):
    target_dirs = sorted(p for p in dude_dir.iterdir() if p.is_dir()
                          and (p / "actives_final.ism").exists())
    seen = set()
    pooled = []
    for target_dir in target_dirs:
        for smi in (read_smiles_file(target_dir / "actives_final.ism")
                    + read_smiles_file(target_dir / "decoys_final.ism")):
            if smi not in seen:
                seen.add(smi)
                pooled.append(smi)
    return pooled, len(target_dirs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dude-dir", default=str(paths.DUDE_DIR),
                         help="Directory containing one subfolder per DUD-E target.")
    parser.add_argument("--output-dir", default=str(paths.out("dude", "aev_conformer_variance_analysis")))
    parser.add_argument("--n-molecules", type=int, default=200,
                         help="Number of unique ligands randomly sampled across the whole pooled DUD-E library.")
    parser.add_argument("--n-conformers", type=int, default=10,
                         help="RDKit conformers embedded per molecule.")
    parser.add_argument("--species", nargs="+", default=DEFAULT_SPECIES_ORDER,
                         help="AEV species order/coverage. Molecules with any other element are skipped.")
    parser.add_argument("--featurizations", choices=["aev", "dimenet", "both"], default="both")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius (2 = ECFP4).")
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dude_dir = Path(args.dude_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_aev = args.featurizations in ("aev", "both")
    run_dimenet = args.featurizations in ("dimenet", "both")

    print(f"Pooling SMILES from all DUD-E targets in {dude_dir}...")
    pooled_smiles, n_targets = pool_all_smiles(dude_dir)
    print(f"Pooled {len(pooled_smiles)} unique ligands across {n_targets} targets")

    sample = rng.sample(pooled_smiles, min(args.n_molecules, len(pooled_smiles)))
    print(f"Sampled {len(sample)} molecules (seed={args.seed})")

    aev_computer = make_aev_computer(args.species) if run_aev else None
    species_converter = make_species_converter(args.species) if run_aev else None
    rbf_layer, sbf_layer = make_basis_layers() if run_dimenet else (None, None)

    molecules = []  # list of dicts: smiles, fp, n_heavy_atoms, n_conformers, [aev], [dimenet]
    n_embed_fail = 0
    n_unsupported = 0

    for i, smi in enumerate(sample, 1):
        bundle = embed_conformers(smi, args.n_conformers, args.seed)
        if bundle is None:
            n_embed_fail += 1
            continue
        if run_aev and has_unsupported_elements(bundle.symbols, args.species):
            n_unsupported += 1
            continue
        fp = smiles_to_fp(smi, args.radius, args.n_bits)
        if fp is None:
            n_embed_fail += 1
            continue

        entry = {"smiles": smi, "fp": fp, "n_heavy_atoms": int(bundle.heavy_mask.sum()),
                 "n_conformers": bundle.coords.shape[0]}
        if run_aev:
            entry["aev"] = compute_aev_molecular_features(
                aev_computer, species_converter, bundle.symbols, bundle.coords, bundle.heavy_mask)
        if run_dimenet:
            entry["dimenet"] = compute_dimenet_basis_molecular_features(
                rbf_layer, sbf_layer, bundle.coords, bundle.heavy_mask)
        molecules.append(entry)

        if i % 20 == 0 or i == len(sample):
            print(f"[{i}/{len(sample)}] featurized {len(molecules)} molecules so far "
                  f"({n_embed_fail} embed failures, {n_unsupported} unsupported elements)")

    print(f"\nSuccessfully featurized {len(molecules)}/{len(sample)} molecules "
          f"({n_embed_fail} embedding failures, {n_unsupported} unsupported-element skips)")

    # ---- per-molecule summary + intra-molecule (conformer) distances --------------------
    summary_rows = []
    intra = {"aev": [], "dimenet": []}
    for m in molecules:
        row = {"smiles": m["smiles"], "n_heavy_atoms": m["n_heavy_atoms"], "n_conformers": m["n_conformers"]}
        for key in ("aev", "dimenet"):
            if key not in m:
                continue
            d = l1_pairwise(m[key])
            intra[key].append(d)
            row[f"{key}_intra_mean"] = float(d.mean()) if len(d) else np.nan
            row[f"{key}_intra_median"] = float(np.median(d)) if len(d) else np.nan
            row[f"{key}_intra_std"] = float(d.std()) if len(d) else np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "per_molecule_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Wrote per-molecule summary -> {summary_csv}")

    for key in ("aev", "dimenet"):
        intra[key] = np.concatenate(intra[key]) if intra[key] else np.array([], dtype=np.float32)

    # ---- inter-molecule distances + Tanimoto (representative conformer = index 0) -------
    fps = [m["fp"] for m in molecules]
    tanimoto = all_pairs_similarity(fps)

    inter = {}
    if run_aev:
        inter["aev"] = l1_pairwise(np.stack([m["aev"][0] for m in molecules]))
    if run_dimenet:
        inter["dimenet"] = l1_pairwise(np.stack([m["dimenet"][0] for m in molecules]))

    np.savez(out_dir / "raw_distances.npz", tanimoto=tanimoto,
              **{f"intra_{k}": v for k, v in intra.items() if k in inter},
              **{f"inter_{k}": v for k, v in inter.items()})
    print(f"Wrote raw distance arrays -> {out_dir / 'raw_distances.npz'}")

    # ---- plots + summary stats -----------------------------------------------------------
    stats_lines = [
        f"n_molecules_sampled={len(sample)}",
        f"n_molecules_featurized={len(molecules)}",
        f"n_embed_failures={n_embed_fail}",
        f"n_unsupported_element_skips={n_unsupported}",
        "",
    ]
    for key, label in (("aev", "AEV"), ("dimenet", "DimeNet-basis")):
        if key not in inter:
            continue
        intra_d, inter_d = intra[key], inter[key]
        rho, pval = spearmanr(tanimoto, inter_d)
        stats_lines += [
            f"[{label}]",
            f"  intra-molecule (conformer) L1 distance: mean={intra_d.mean():.4f} "
            f"median={np.median(intra_d):.4f} std={intra_d.std():.4f} (n={len(intra_d)} pairs)",
            f"  inter-molecule L1 distance:              mean={inter_d.mean():.4f} "
            f"median={np.median(inter_d):.4f} std={inter_d.std():.4f} (n={len(inter_d)} pairs)",
            f"  inter/intra median ratio: {np.median(inter_d) / np.median(intra_d):.2f}x",
            f"  Spearman(Tanimoto similarity, {label} L1 distance): rho={rho:.4f}, p={pval:.2e}",
            "",
        ]

        # AEV plots are also saved as PDF, and the scatter's Tanimoto axis is zoomed to
        # [0, 0.4] (where this dataset's inter-molecule similarities actually fall) --
        # both by request, scoped to AEV only, not DimeNet-basis.
        plot_distance_kde_comparison(
            intra_d, inter_d,
            f"{label}: intra-molecule (conformer) vs. inter-molecule L1 distance\n"
            f"({len(molecules)} DUD-E ligands, {args.n_conformers} conformers/molecule)",
            out_dir / f"{key}_intra_vs_inter_distance_kde.png",
            xlabel=f"{label} molecular feature L1 distance",
            save_pdf=(key == "aev"),
        )
        plot_tanimoto_vs_feature_distance(
            tanimoto, inter_d, intra_d,
            f"{label} distance vs. ECFP4 Tanimoto similarity, in context of conformer noise\n"
            f"({len(molecules)} DUD-E ligands, representative conformer per molecule)",
            out_dir / f"{key}_tanimoto_vs_distance_scatter.png",
            ylabel=f"{label} molecular feature L1 distance",
            xlim=(0, 0.4) if key == "aev" else (0, 1),
            save_pdf=(key == "aev"),
        )

    stats_text = "\n".join(stats_lines)
    (out_dir / "summary_stats.txt").write_text(stats_text)
    print("\n" + stats_text)
    print(f"Saved plots and summary_stats.txt to {out_dir}")


if __name__ == "__main__":
    main()
