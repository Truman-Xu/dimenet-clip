"""Default input and output locations for the figure pipeline in ``plots/``.

Every script in this directory takes its default locations from here, so
pointing the whole pipeline at your copy of the data only requires setting a
few environment variables (each script's own ``--*-dir`` / ``--output-dir``
flags still override these per run):

  DUDE_DIR          DUD-E per-target folders, each holding actives_final.ism
                    and decoys_final.ism          (default: <repo>/data/dude)
  LIT_PCBA_DIR      LIT-PCBA per-target folders, each holding actives.smi and
                    inactives.smi                  (default: <repo>/data/lit-pcba)
  ENCODINGS_DIR     embedding pickles written by eval/encode_benchmarks.py
                                                   (default: <repo>/encodings)
  MODEL_EPOCH       epoch suffix of those pickles  (default: 3, the released model)
  PLOTS_OUTPUT_DIR  where results and figures go   (default: <repo>/plots/output)
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _dir(var, default):
    return Path(os.environ.get(var, default)).expanduser()


DUDE_DIR = _dir("DUDE_DIR", REPO_ROOT / "data" / "dude")
LIT_PCBA_DIR = _dir("LIT_PCBA_DIR", REPO_ROOT / "data" / "lit-pcba")
ENCODINGS_DIR = _dir("ENCODINGS_DIR", REPO_ROOT / "encodings")
MODEL_EPOCH = int(os.environ.get("MODEL_EPOCH", 3))
OUTPUT_DIR = _dir("PLOTS_OUTPUT_DIR", REPO_ROOT / "plots" / "output")
FIGURES_DIR = OUTPUT_DIR / "figures"


def out(dataset, analysis):
    """Results directory of one analysis, e.g. ``out("dude", "vs_screening_analysis")``."""
    return OUTPUT_DIR / dataset / analysis


def encoding(prefix, kind):
    """Embedding pickle from ``eval/encode_benchmarks.py --out_prefix <prefix>``."""
    return ENCODINGS_DIR / f"{prefix}_{kind}_encoded_ep{MODEL_EPOCH}.pkl"
