"""
Illustrative case studies: molecule pairs that are structurally very close (high
ECFP4 Tanimoto similarity), to visualize why the DUD-E-wide inter- vs. intra-molecule
AEV distance separation (aev_conformer_variance_analysis.py) narrows specifically for
chemically similar pairs -- among all inter-molecule comparisons in that experiment,
the "least separable" ones (smallest inter-AEV distance, closest to the conformer-noise
floor) are exactly the pairs that are chemically near-identical.

Two sources of pairs:
  1. Three synthetic pairs, each built by hand-editing a base SMILES in one of three
     ways: (1) atom-species substitution, (2) alternative branching/regiochemistry,
     (3) single<->double bond change.
  2. Two real DUD-E active-active pairs, mined for the highest ECFP4 Tanimoto
     similarity across all 102 targets -- showing the same effect in real congeneric
     series, not just synthetic edits.

For each pair: renders 2D structures side by side as true vector graphics (RDKit's
native SVG drawer -- bonds/atoms as paths and labels as text, not a rasterized image),
labeled with the intra-molecule AEV L1 distance (pooled conformer noise from both
molecules), the inter-molecule AEV L1 distance (mean across all conformer-pair
combinations between the two molecules), and the ECFP4 Tanimoto similarity. Output is
saved as both .svg and .pdf (via cairosvg) so the figure can be re-formatted in a
vector editor (e.g. Adobe Illustrator) without rasterization artifacts.

Usage (needs torchani, which the main requirements do not include; see plots/README.md):
    python aev_case_study_pairs.py
"""
import argparse
import paths  # default input/output locations, see plots/paths.py
import re
from pathlib import Path

import cairosvg
import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

from fp_similarity_common import read_smiles_file, smiles_to_fp, INK_PRIMARY, INK_SECONDARY
from conformer_common import embed_conformers, l1_pairwise, l1_cross
from aev_common import (
    DEFAULT_SPECIES_ORDER, make_aev_computer, make_species_converter,
    has_unsupported_elements, compute_aev_molecular_features,
)

# ---- three hand-crafted "minimal edit" pairs -------------------------------------
_BENZANILIDE_PARA_CL = "c1ccc(cc1)C(=O)Nc1ccc(Cl)cc1"  # shared base for cases 1 and 2

SYNTHETIC_CASES = [
    {
        "label": "Case 1: atom-species substitution (Cl -> F)",
        "smiles_a": _BENZANILIDE_PARA_CL,
        "smiles_b": "c1ccc(cc1)C(=O)Nc1ccc(F)cc1",
        "name_a": "N-(4-chlorophenyl)benzamide",
        "name_b": "N-(4-fluorophenyl)benzamide",
    },
    {
        "label": "Case 2: alternative branching (para-Cl -> ortho-Cl)",
        "smiles_a": _BENZANILIDE_PARA_CL,
        "smiles_b": "c1ccc(cc1)C(=O)Nc1ccccc1Cl",
        "name_a": "N-(4-chlorophenyl)benzamide",
        "name_b": "N-(2-chlorophenyl)benzamide",
    },
    {
        "label": "Case 3: single -> double bond (butanamide -> but-3-enamide)",
        "smiles_a": "CCCC(=O)Nc1ccccc1",
        "smiles_b": "C=CCC(=O)Nc1ccccc1",
        "name_a": "N-phenylbutanamide",
        "name_b": "N-phenylbut-3-enamide",
    },
    {
        "label": "Case 4: carbon chain extension (butanamide -> pentanamide)",
        "smiles_a": "CCCC(=O)Nc1ccccc1",
        "smiles_b": "CCCCC(=O)Nc1ccccc1",
        "name_a": "N-phenylbutanamide",
        "name_b": "N-phenylpentanamide",
    },
]


def find_high_similarity_active_pairs(dude_dir: Path, top_k=2, radius=2, n_bits=2048, max_tanimoto=0.99):
    """Scan every DUD-E target's actives for the single highest-Tanimoto active-active
    pair per target (excluding near-duplicate/identical fingerprints), returning the
    top_k pairs across distinct targets."""
    target_dirs = sorted(p for p in dude_dir.iterdir() if p.is_dir() and (p / "actives_final.ism").exists())
    best_per_target = []
    for target_dir in target_dirs:
        actives = read_smiles_file(target_dir / "actives_final.ism")
        if len(actives) < 2:
            continue
        fps, valid_smis = [], []
        for smi in actives:
            fp = smiles_to_fp(smi, radius, n_bits)
            if fp is not None:
                fps.append(fp)
                valid_smis.append(smi)
        if len(fps) < 2:
            continue

        target_best = None
        for i in range(len(fps) - 1):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
            for j, sim in enumerate(sims, start=i + 1):
                if sim < max_tanimoto and (target_best is None or sim > target_best[0]):
                    target_best = (sim, valid_smis[i], valid_smis[j])
        if target_best is not None:
            best_per_target.append((target_best[0], target_dir.name, target_best[1], target_best[2]))

    best_per_target.sort(key=lambda x: -x[0])
    return best_per_target[:top_k]


def compute_pair_metrics(smiles_a, smiles_b, n_conformers, seed, species_order,
                          aev_computer, species_converter, radius, n_bits):
    bundle_a = embed_conformers(smiles_a, n_conformers, seed)
    bundle_b = embed_conformers(smiles_b, n_conformers, seed)
    if bundle_a is None or bundle_b is None:
        return None
    if (has_unsupported_elements(bundle_a.symbols, species_order)
            or has_unsupported_elements(bundle_b.symbols, species_order)):
        return None

    feats_a = compute_aev_molecular_features(
        aev_computer, species_converter, bundle_a.symbols, bundle_a.coords, bundle_a.heavy_mask)
    feats_b = compute_aev_molecular_features(
        aev_computer, species_converter, bundle_b.symbols, bundle_b.coords, bundle_b.heavy_mask)

    intra = np.concatenate([l1_pairwise(feats_a), l1_pairwise(feats_b)])
    inter = l1_cross(feats_a, feats_b)

    fp_a = smiles_to_fp(smiles_a, radius, n_bits)
    fp_b = smiles_to_fp(smiles_b, radius, n_bits)
    tanimoto = DataStructs.TanimotoSimilarity(fp_a, fp_b)

    return {
        "intra_mean": float(intra.mean()) if len(intra) else float("nan"),
        "inter_mean": float(inter.mean()),
        "tanimoto": float(tanimoto),
    }


def _esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _mol_svg_fragment(mol, size):
    """RDKit's own vector SVG for one molecule (bonds/atoms as <path>s, labels as
    <text>s), with the outer <?xml?> prolog and <svg> wrapper stripped so it can be
    embedded as a positioned <g> inside a larger composed SVG document."""
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    drawer.drawOptions().clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return re.search(r"<svg[^>]*>(.*)</svg>", svg, re.DOTALL).group(1)


def build_case_svg(cases, mol_size=380):
    """Compose one master SVG document -- true vector molecule structures plus native
    <text> captions -- for a list of pair cases, laid out as one row per case (two
    molecules + a caption line underneath)."""
    margin, col_gap, row_gap = 30, 30, 22
    title_h, caption_h = 26, 46
    row_h = title_h + mol_size + caption_h
    width = margin + mol_size + col_gap + mol_size + margin
    height = margin + len(cases) * row_h + (len(cases) - 1) * row_gap + margin

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]
    col_x = (margin, margin + mol_size + col_gap)

    for row, case in enumerate(cases):
        mol_a = Chem.MolFromSmiles(case["smiles_a"])
        mol_b = Chem.MolFromSmiles(case["smiles_b"])
        row_top = margin + row * (row_h + row_gap)
        title_y = row_top + title_h - 6
        mol_y = row_top + title_h
        caption_y1 = mol_y + mol_size + 20
        caption_y2 = caption_y1 + 18

        for (x, mol, name) in ((col_x[0], mol_a, case["name_a"]), (col_x[1], mol_b, case["name_b"])):
            parts.append(
                f'<text x="{x + mol_size / 2}" y="{title_y}" text-anchor="middle" '
                f'font-family="Helvetica, Arial, sans-serif" font-size="13" '
                f'fill="{INK_PRIMARY}">{_esc(name)}</text>'
            )
            parts.append(f'<g transform="translate({x},{mol_y})">{_mol_svg_fragment(mol, mol_size)}</g>')

        m = case["metrics"]
        parts.append(
            f'<text x="{width / 2}" y="{caption_y1}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="11.5" font-weight="600" '
            f'fill="{INK_PRIMARY}">{_esc(case["label"])}</text>'
        )
        parts.append(
            f'<text x="{width / 2}" y="{caption_y2}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="10.5" '
            f'fill="{INK_SECONDARY}">'
            f'Intra-molecule AEV distance: {m["intra_mean"]:.2f}   |   '
            f'Inter-molecule AEV distance: {m["inter_mean"]:.2f}   |   '
            f'ECFP4 Tanimoto similarity: {m["tanimoto"]:.2f}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def save_svg_and_pdf(svg_text, out_path_no_ext):
    svg_path = out_path_no_ext.with_suffix(".svg")
    pdf_path = out_path_no_ext.with_suffix(".pdf")
    svg_path.write_text(svg_text)
    cairosvg.svg2pdf(bytestring=svg_text.encode(), write_to=str(pdf_path))
    return svg_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dude-dir", default=str(paths.DUDE_DIR))
    parser.add_argument("--output-dir", default=str(paths.out("dude", "aev_case_study_pairs")))
    parser.add_argument("--n-conformers", type=int, default=10)
    parser.add_argument("--n-real-pairs", type=int, default=2,
                         help="Number of real DUD-E active-active high-similarity pairs to mine and include.")
    parser.add_argument("--species", nargs="+", default=DEFAULT_SPECIES_ORDER)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dude_dir = Path(args.dude_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aev_computer = make_aev_computer(args.species)
    species_converter = make_species_converter(args.species)

    print(f"Mining DUD-E actives across all targets for {args.n_real_pairs} "
          f"highest-Tanimoto active-active pairs...")
    real_pairs = find_high_similarity_active_pairs(dude_dir, top_k=args.n_real_pairs,
                                                     radius=args.radius, n_bits=args.n_bits)
    real_cases = []
    for sim, target, smi_a, smi_b in real_pairs:
        print(f"  {target}: Tanimoto={sim:.3f}")
        real_cases.append({
            "label": f"Real DUD-E active-active pair (target: {target.upper()})",
            "smiles_a": smi_a,
            "smiles_b": smi_b,
            "name_a": f"{target.upper()} active A",
            "name_b": f"{target.upper()} active B",
        })

    all_cases = SYNTHETIC_CASES + real_cases
    for case in all_cases:
        print(f"Computing AEV/Tanimoto metrics for: {case['label']}")
        metrics = compute_pair_metrics(
            case["smiles_a"], case["smiles_b"], args.n_conformers, args.seed, args.species,
            aev_computer, species_converter, args.radius, args.n_bits,
        )
        if metrics is None:
            raise RuntimeError(f"Failed to compute metrics for case: {case['label']}")
        case["metrics"] = metrics
        print(f"  intra={metrics['intra_mean']:.3f}  inter={metrics['inter_mean']:.3f}  "
              f"tanimoto={metrics['tanimoto']:.3f}")

    svg_path, pdf_path = save_svg_and_pdf(
        build_case_svg(all_cases), out_dir / "aev_similar_pair_case_studies")
    print(f"\nSaved combined case-study figure -> {svg_path} / {pdf_path}")

    for case in SYNTHETIC_CASES:
        safe_label = case["label"].split(":")[0].strip().lower().replace(" ", "_")
        svg_path, pdf_path = save_svg_and_pdf(build_case_svg([case]), out_dir / safe_label)
        print(f"Saved individual figure -> {svg_path} / {pdf_path}")


if __name__ == "__main__":
    main()
