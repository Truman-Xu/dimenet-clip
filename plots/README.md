# Manuscript figures

Scripts that reproduce every data-driven figure in the manuscript, plus
Table 1. They are the original analysis scripts. The only changes are
that input and output locations now come from [`paths.py`](paths.py), not
hardcoded paths, and that the DimeNet-CLIP figures read the embeddings
written by [`../eval/encode_benchmarks.py`](../eval/encode_benchmarks.py).

## Figure → script

| Manuscript figure | Produced by | Needs |
|---|---|---|
| `aev_tanimoto_vs_distance_scatter.pdf` | `aev_conformer_variance_analysis.py` | DUD-E, torchani |
| `aev_intra_vs_inter_distance_kde.pdf` (SI) | `aev_conformer_variance_analysis.py` | DUD-E, torchani |
| `aev_conf.pdf` (SI) | `aev_case_study_pairs.py` ¹ | DUD-E, torchani, cairosvg |
| `sims-2.pdf` | `make_figure_sims2.py` | DUD-E, LIT-PCBA |
| `global_vs_within_target_similarity.pdf` (SI) | `make_figure_si_global_similarity.py` | DUD-E, LIT-PCBA |
| `clip-auc.pdf` | `make_figure_clip_auc.py` | embeddings |
| `pocket_specificity.pdf` | `clip_pocket_specificity_control_dude.py` | DUD-E embeddings |
| `fp_vs_clip.pdf` (SI) | `compare_clip_vs_fingerprint_litpcba.py` | LIT-PCBA, embeddings |
| Table 1 | `summary_metrics_table.py` | all of the above |

¹ Produces the same six molecule pairs with the same AEV distances and
Tanimoto similarities as the manuscript. The manuscript version was
re-laid-out from this output in a vector editor, so the arrangement differs.

The figure assemblers (`make_figure_*.py`, `summary_metrics_table.py`) read
results written by the analysis scripts earlier in the pipeline.
[`make_all.sh`](make_all.sh) runs everything in dependency order. The two
schematic figures (`dimenet_clip_workflow`, `model-pipelines`) are not
generated from data and are not included here.

## Inputs

1. **DUD-E**: the per-target folders from the
   [DUD-E download](http://dude.docking.org/), each containing
   `actives_final.ism` and `decoys_final.ism`.
2. **LIT-PCBA**: the per-target folders from the
   [LIT-PCBA download](https://drugdesign.unistra.fr/LIT-PCBA/), each
   containing `actives.smi` and `inactives.smi`.
3. **Benchmark embeddings** (DimeNet-CLIP figures only), written by
   `eval/encode_benchmarks.py` with the released model:

   ```bash
   python eval/encode_benchmarks.py --dataset dude --out_prefix dude \
       --model_dir weights/clip_pdbbind_finetuned --epoch 3 \
       --pocket_pkl /path/to/dude-pocket-4.pkl \
       --ligand_pkl /path/to/dude-ligand-z-pos.pkl --output_dir encodings
   python eval/encode_benchmarks.py --dataset litpcba --out_prefix lit \
       --model_dir weights/clip_pdbbind_finetuned --epoch 3 \
       --pocket_pkl /path/to/h_pocket_4.pkl \
       --ligand_pkl /path/to/h_ligands.pkl --output_dir encodings
   ```

   The LIT-PCBA pickles are built by `eval/lit_pcba_labels_prep.ipynb`. Use
   the hydrogen-bearing `h_pocket_4.pkl` / `h_ligands.pkl`. The manuscript's
   numbers come from those files, not from `pcba_sep_pocket_vecs.pkl`.

## Environment

The main `requirements.txt` covers everything except the two AEV scripts.
Those also need `torchani` and `cairosvg`:

```bash
pip install torchani cairosvg
```

If those clash with your main environment, install them in a separate one
and point `AEV_PYTHON` at it (see below).

## Running

Set the locations (defaults in [`paths.py`](paths.py)), then run the driver
from anywhere:

```bash
export DUDE_DIR=/path/to/dude                # per-target DUD-E folders
export LIT_PCBA_DIR=/path/to/lit-pcba        # per-target LIT-PCBA folders
export ENCODINGS_DIR=/path/to/encodings      # eval/encode_benchmarks.py output
export MODEL_EPOCH=3                         # epoch suffix of those pickles
export AEV_PYTHON=/path/to/torchani-env/bin/python   # optional
./plots/make_all.sh
```

Results are written under `plots/output/` (or `$PLOTS_OUTPUT_DIR`), one
directory per analysis. The manuscript-named figures are collected in
`plots/output/figures/`, and Table 1 is in `plots/output/table1/` as CSV and
LaTeX. Set `SKIP_AEV=1` to skip the torchani step.

Each script also runs on its own, and its command-line flags override the
`paths.py` defaults, e.g. `python plots/vs_screening_analysis.py --help`.

The first run fingerprints the full DUD-E (~1.4M) and LIT-PCBA (~2.9M)
libraries, so budget 10–20 minutes per dataset. Fingerprints are cached, so
reruns take seconds.

## Reproducibility

All sampling is seeded (seed 42), including RDKit conformer embedding. Run
from scratch on the original data, this pipeline reproduced every result
behind the manuscript:

- The fingerprint and CLIP per-target results, all four similarity
  distributions, the AEV distances, and Table 1 matched the originals.
- All eight figures render pixel-identical to the manuscript's, except
  `aev_conf.pdf`. That one is pixel-identical to the original script output,
  which was re-laid-out by hand for the manuscript.

Two caveats apply:

- **Enrichment factors from fingerprint search depend slightly on the CPU.**
  `vs_common.enrichment_factor` ranks ligands with numpy's default
  (unstable) sort. Tanimoto similarities have many ties, and numpy picks a
  CPU-specific sort routine, so tied ligands can fall on either side of the
  top-1% / 5% cutoff on different hardware. The manuscript's values
  reproduce exactly on the CPU they were computed on (Intel Xeon E5-2650 v3,
  which has no AVX-512). On an AVX-512 node, per-target EF moved by up to
  0.32, and Table 1 shifted in the last digit (DUD-E EF1 std 12.76 → 12.75,
  LIT-PCBA EF1 mean 4.23 → 4.22). AUC-ROC is unaffected, because it averages
  over ties. DimeNet-CLIP EF is also unaffected, because continuous
  embedding scores essentially never tie.
- **DimeNet-CLIP AUC-ROC can differ by ~1e-7** between machines, from
  float32 dot-product rounding. The rounded values don't change.

The AEV results were reproduced with torchani 2.2.4, cairosvg 2.9.0 and
RDKit 2024.03.6. Conformer embedding can change between RDKit releases, so
other versions may shift the AEV distances slightly.

## Files

| File | Role |
|---|---|
| `paths.py` | Default input/output locations, overridable by environment variables |
| `fp_similarity_common.py` | ECFP-4 fingerprints, sampling, Tanimoto helpers, shared plot styling |
| `vs_common.py` | Fingerprint-library cache, replicate AUC/EF loop, CLIP scoring |
| `conformer_common.py` | RDKit conformer generation (ETKDGv3 + MMFF94), L1 distances |
| `aev_common.py` | TorchANI AEV featurization (ANI-1x hyperparameters, 10 species) |
| `dimenet_basis_common.py` | Untrained DimeNet radial/angular basis features |
| `vs_screening_analysis.py` | Fingerprint similarity search, AUC-ROC / EF1% / EF5% per target |
| `compare_vs_shared_targets.py` | Targets shared by DUD-E and LIT-PCBA (sims-2, panel c) |
| `active_decoy_similarity_distributions.py` | Active–active vs active–decoy similarity, DUD-E |
| `active_inactive_similarity_distributions_litpcba.py` | Same, LIT-PCBA |
| `ecfp4_similarity_analysis.py` | Global vs within-target similarity, DUD-E |
| `lit_pcba_similarity_analysis.py` | Same, LIT-PCBA |
| `clip_screening_analysis_dude.py` | DimeNet-CLIP AUC-ROC / EF per target, DUD-E |
| `clip_screening_analysis_litpcba.py` | Same, LIT-PCBA (mean over pocket structures) |
| `clip_pocket_specificity_control_dude.py` | Correct- vs mismatched-pocket control |
| `compare_clip_vs_fingerprint_litpcba.py` | DimeNet-CLIP vs fingerprint search, LIT-PCBA |
| `aev_conformer_variance_analysis.py` | Conformer vs molecule AEV distances |
| `aev_case_study_pairs.py` | AEV case studies on similar molecule pairs |
| `make_figure_*.py`, `summary_metrics_table.py` | Assemble the multi-panel figures and Table 1 |
