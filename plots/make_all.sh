#!/usr/bin/env bash
# Reproduce every data-driven figure in the manuscript, plus Table 1.
#
# Point the pipeline at your data first (see plots/paths.py for defaults):
#   DUDE_DIR, LIT_PCBA_DIR   raw benchmark folders
#   ENCODINGS_DIR, MODEL_EPOCH   output of eval/encode_benchmarks.py
#   PLOTS_OUTPUT_DIR         where results go (default: plots/output)
# and choose interpreters:
#   PYTHON       main environment (default: python)
#   AEV_PYTHON   environment with torchani + cairosvg for the two AEV scripts
#                (default: $PYTHON)
#   SKIP_AEV=1   skip the AEV experiments (aev_* figures)
#
# The manuscript-named figures are collected in $PLOTS_OUTPUT_DIR/figures and
# Table 1 in $PLOTS_OUTPUT_DIR/table1. The first run fingerprints the full DUD-E
# (~1.4M) and LIT-PCBA (~2.9M) libraries, so budget 10-20 min per dataset;
# fingerprints are cached, and later runs take seconds.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="${PYTHON:-python}"
AEV_PY="${AEV_PYTHON:-$PY}"
OUT="$("$PY" -c 'import paths; print(paths.OUTPUT_DIR)')"
FIG="$OUT/figures"
run() { echo; echo "==> $*"; "$@"; }

# 1. Fingerprint similarity search (Table 1 fingerprint rows; caches fingerprints used below)
run "$PY" vs_screening_analysis.py --dataset dude
run "$PY" vs_screening_analysis.py --dataset litpcba
run "$PY" compare_vs_shared_targets.py

# 2. Active-active vs active-decoy/inactive similarity (sims-2, panels a-b)
run "$PY" active_decoy_similarity_distributions.py
run "$PY" active_inactive_similarity_distributions_litpcba.py

# 3. Global vs within-target similarity (SI)
run "$PY" ecfp4_similarity_analysis.py
run "$PY" lit_pcba_similarity_analysis.py

# 4. DimeNet-CLIP screening from the benchmark embeddings
run "$PY" clip_screening_analysis_dude.py
run "$PY" clip_screening_analysis_litpcba.py
run "$PY" clip_pocket_specificity_control_dude.py
run "$PY" compare_clip_vs_fingerprint_litpcba.py

# 5. Assembled figures and Table 1
run "$PY" make_figure_sims2.py
run "$PY" make_figure_si_global_similarity.py
run "$PY" make_figure_clip_auc.py
run "$PY" summary_metrics_table.py --output-dir "$OUT/table1"

# 6. AEV conformer-variance experiment and case studies (torchani)
if [ "${SKIP_AEV:-0}" != 1 ]; then
    run "$AEV_PY" aev_conformer_variance_analysis.py
    run "$AEV_PY" aev_case_study_pairs.py
fi

# 7. Collect the figures that are written under analysis-specific names
mkdir -p "$FIG"
cp "$OUT/dude/clip_pocket_control_analysis/auc_roc_pocket_control.pdf"            "$FIG/pocket_specificity.pdf"
cp "$OUT/lit-pcba/clip_vs_screening_analysis/clip_vs_fingerprint_comparison.pdf" "$FIG/fp_vs_clip.pdf"
if [ "${SKIP_AEV:-0}" != 1 ]; then
    cp "$OUT/dude/aev_conformer_variance_analysis/aev_tanimoto_vs_distance_scatter.pdf" "$FIG/"
    cp "$OUT/dude/aev_conformer_variance_analysis/aev_intra_vs_inter_distance_kde.pdf"  "$FIG/"
    # Same six cases and values as the manuscript's aev_conf.pdf, which was
    # re-laid-out from this output in a vector editor.
    cp "$OUT/dude/aev_case_study_pairs/aev_similar_pair_case_studies.pdf"             "$FIG/aev_conf.pdf"
fi
echo; echo "Manuscript figures in $FIG:"; ls "$FIG"
echo "Table 1: $OUT/table1/summary_metrics_table.tex"
