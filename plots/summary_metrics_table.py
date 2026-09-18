"""
Top-level summary table: mean +/- std of AUC-ROC, EF1%, and EF5%, aggregated
across ALL targets, for every (method, dataset) combination evaluated in
this analysis:
  - ECFP4 fingerprint similarity search (Experiment 2) on DUD-E and LIT-PCBA
  - DimeNet-CLIP embedding dot-product scoring (Experiment 3) on DUD-E and
    LIT-PCBA

Reads each method/dataset's already-computed per_target_summary.csv (one
row per target, with that target's own mean+-std over its replicates /
pocket structures - see vs_screening_analysis.py and
clip_screening_analysis_{dude,litpcba}.py). The "mean" for a target is
that target's auc_roc_mean / ef_*_mean column; this script then aggregates
those per-target means across all targets in the dataset, reporting:
  - mean-of-target-means (the headline number)
  - std-of-target-means (target-to-target variability - NOT replicate
    variability, which is already averaged out per target)

This mirrors how such summary tables are conventionally reported in
virtual-screening benchmark papers (mean +- std *across targets*).

Requires all four per_target_summary.csv files to already exist (i.e. both
vs_screening_analysis.py --dataset {dude,litpcba} and both
clip_screening_analysis_{dude,litpcba}.py must have been run first).

Produces, in --output-dir:
  - summary_metrics_table.csv : one row per (method, dataset), columns
    n_targets, auc_mean, auc_std, ef1pct_mean, ef1pct_std, ef5pct_mean,
    ef5pct_std.
  - summary_metrics_table.tex : the same table rendered as a standalone
    LaTeX booktabs table (mean $\\pm$ std cells), ready to \\input{} or
    copy into a paper.

Usage:
    python summary_metrics_table.py
"""
import argparse
import paths  # default input/output locations, see plots/paths.py
from pathlib import Path

import pandas as pd

METHODS = [
    ("Fingerprint search", "DUD-E",
     str(paths.out("dude", "vs_screening_analysis") / "per_target_summary.csv")),
    ("Fingerprint search", "LIT-PCBA",
     str(paths.out("lit-pcba", "vs_screening_analysis") / "per_target_summary.csv")),
    ("DimeNet-CLIP", "DUD-E",
     str(paths.out("dude", "clip_vs_screening_analysis") / "per_target_summary.csv")),
    ("DimeNet-CLIP", "LIT-PCBA",
     str(paths.out("lit-pcba", "clip_vs_screening_analysis") / "per_target_summary.csv")),
]


def fmt(mean, std, decimals):
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def build_latex_table(table: pd.DataFrame) -> str:
    header = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{llrccc}",
        r"\toprule",
        r"Method & Dataset & $N$ targets & AUC-ROC & EF (top 1\%) & EF (top 5\%) \\",
        r"\midrule",
    ]

    data_lines = []
    prev_method = None
    for _, row in table.iterrows():
        if row["method"] != prev_method and prev_method is not None:
            data_lines.append(r"\midrule")
        method_cell = row["method"] if row["method"] != prev_method else ""
        prev_method = row["method"]
        auc_cell = fmt(row["auc_mean"], row["auc_std"], 3)
        ef1_cell = fmt(row["ef1pct_mean"], row["ef1pct_std"], 2)
        ef5_cell = fmt(row["ef5pct_mean"], row["ef5pct_std"], 2)
        data_lines.append(f"{method_cell} & {row['dataset']} & {int(row['n_targets'])} & "
                           f"{auc_cell} & {ef1_cell} & {ef5_cell} \\\\")

    lines = header + data_lines + [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Virtual-screening performance (mean $\pm$ std across targets) for ECFP4 "
        r"fingerprint similarity search and DimeNet-CLIP embedding scoring, on DUD-E and "
        r"LIT-PCBA. Fingerprint per-target means are taken over 10 random-query replicates; "
        r"DimeNet-CLIP per-target means are taken over available pocket structures "
        r"(1 for DUD-E, 1--14 for LIT-PCBA).}",
        r"\label{tab:summary-metrics}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp-dude", default=METHODS[0][2])
    parser.add_argument("--fp-litpcba", default=METHODS[1][2])
    parser.add_argument("--clip-dude", default=METHODS[2][2])
    parser.add_argument("--clip-litpcba", default=METHODS[3][2])
    parser.add_argument("--output-dir", default=str(paths.out("lit-pcba", "clip_vs_screening_analysis")))
    args = parser.parse_args()

    paths = {
        ("Fingerprint search", "DUD-E"): args.fp_dude,
        ("Fingerprint search", "LIT-PCBA"): args.fp_litpcba,
        ("DimeNet-CLIP", "DUD-E"): args.clip_dude,
        ("DimeNet-CLIP", "LIT-PCBA"): args.clip_litpcba,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (method, dataset), path in paths.items():
        df = pd.read_csv(path)
        rows.append({
            "method": method,
            "dataset": dataset,
            "n_targets": len(df),
            "auc_mean": df["auc_roc_mean"].mean(),
            "auc_std": df["auc_roc_mean"].std(),
            "ef1pct_mean": df["ef_0.01_mean"].mean(),
            "ef1pct_std": df["ef_0.01_mean"].std(),
            "ef5pct_mean": df["ef_0.05_mean"].mean(),
            "ef5pct_std": df["ef_0.05_mean"].std(),
        })
        print(f"{method:20s} {dataset:10s} n_targets={len(df):3d}  "
              f"AUC={rows[-1]['auc_mean']:.3f}+-{rows[-1]['auc_std']:.3f}  "
              f"EF1%={rows[-1]['ef1pct_mean']:.2f}+-{rows[-1]['ef1pct_std']:.2f}  "
              f"EF5%={rows[-1]['ef5pct_mean']:.2f}+-{rows[-1]['ef5pct_std']:.2f}")

    table = pd.DataFrame(rows)
    out_csv = out_dir / "summary_metrics_table.csv"
    table.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    latex = build_latex_table(table)
    out_tex = out_dir / "summary_metrics_table.tex"
    out_tex.write_text(latex + "\n")
    print(f"Wrote {out_tex}\n")
    print(latex)


if __name__ == "__main__":
    main()
