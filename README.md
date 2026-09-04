# DimeNetCLIP

Code, evaluation notebooks, and model weights for a DimeNet-based joint
ligand–pocket embedding model, trained with a CLIP-style contrastive
objective for protein–ligand binding affinity / virtual screening. This
repository accompanies the accompanying manuscript's Methods section and
implements its five-stage pipeline:

1. **Ligand structure data preparation** — sample and serialize ligand
   conformers from a Uni-Mol ligand LMDB database.
2. **Pocket structure data preparation** — serialize protein pocket
   structures from Uni-Mol pocket LMDB databases.
3. **Self-supervised denoising pre-training** — pre-train the DimeNet
   backbone (ligand and pocket domains, separately) with a coordinate
   denoising / implicit force-matching objective, initialized from a
   QM9-pretrained backbone.
4. **Contrastive pre-training on SAIR** — train a joint ligand–pocket
   DimeNetCLIP model with an affinity-weighted, unique-protein-per-batch
   CLIP objective on a large affinity-labeled dataset.
5. **Contrastive fine-tuning on PDBBind** — fine-tune the SAIR-pretrained
   model on PDBBind.

See the manuscript's Methods section for the full mathematical description
of each stage (noise schedule, loss functions, batch sampling, optimizer
settings, etc.); this README covers how to run the code.

## Repository layout

```
dimenet_clip.py                    # Model code: DimeNetBackbone, DimeNetReadout,
                                    # DimeNetCLIP, SigmoidWeightedCLIPLoss,
                                    # DimeNetPretrainer + denoising loss, all-gather utils

data_prep/
  ligand_prep.py                   # Stage 1
  pocket_prep.py                   # Stage 2
  atom_name_table.txt              # PDB atom-nomenclature reference table used by pocket_prep.py
  qm9_pretrain.py                  # Extracts the QM9-pretrained backbone/readout init
                                    # used by train_denoising.py (see weights/qm9_pretrained)

train_denoising.py                 # Stage 3
train_clip_dimenet_sair.py         # Stage 4
train_clip_dimenet_pdbbind.py      # Stage 5

eval/
  dude_eval.ipynb                  # DUDE virtual-screening benchmark (AUC / enrichment factor)
  lit_pcba_eval.ipynb              # LIT-PCBA benchmark (notebook version)
  lit_pcba_eval.py                 # LIT-PCBA benchmark (script version, argparse)
  lit_pcba_labels_prep.ipynb       # Builds the LIT-PCBA evaluation pickles from docked poses
  eval_denoising.ipynb             # Sanity-checks the denoising-pretrained backbones

slurm/                             # Example SLURM job scripts for each stage above
                                    # (edit the #SBATCH directives and paths for your cluster)

weights/                           # Published checkpoints (see "Model weights" below)
```

## Installation

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric's optional wheels (`torch-scatter`,
`torch-sparse`, `torch-cluster`) must be installed separately, matched to
your CUDA version — see the [PyG install
guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

`data_prep/ligand_prep.py`, `data_prep/pocket_prep.py`, and the evaluation
notebooks also depend on
[`crimm`](https://crimm.readthedocs.io/) (periodic-table lookup, PDB
parsing, topology building), which is included in `requirements.txt`.

## Data and datasets

**No training data is bundled in this repository** — the denoising
pre-training corpus (millions of ligand/pocket conformers) and the SAIR /
PDBBind contrastive-training datasets are far too large to publish here.
Every script takes the relevant data path(s) as required command-line
arguments; point them at your own copies:

| Stage | Script | Data argument(s) |
|---|---|---|
| 1. Ligand prep | `data_prep/ligand_prep.py` | `--lmdb_dir`, `--output_dir` |
| 2. Pocket prep | `data_prep/pocket_prep.py` | `--lmdb_dir`, `--output_dir` |
| 3. Denoising pre-training | `train_denoising.py` | `--dataset_path` (output of stage 1/2) |
| 4. SAIR CLIP pre-training | `train_clip_dimenet_sair.py` | `--data_path` |
| 5. PDBBind CLIP fine-tuning | `train_clip_dimenet_pdbbind.py` | `--data_path` |
| DUDE / LIT-PCBA evaluation | `eval/*.ipynb`, `eval/lit_pcba_eval.py` | paths set in a config cell / `--data_path` |

Raw source data (Uni-Mol ligand/pocket LMDB databases, SAIR, PDBBind, DUDE,
LIT-PCBA) must be obtained separately from their respective providers.

Note on stage 3 input naming: `train_denoising.py`'s single-process `train()`
path reads `h_{name}_pos_train.pkl` / `h_z_{name}_train.pkl` (the direct
output of `ligand_prep.py` / `pocket_prep.py` with hydrogens retained), while
its multi-GPU `ddp_train()` path (used for the reported runs) reads
`{name}_pos.pkl` / `z_{name}.pkl` from the same `--dataset_path` directory —
name your training split accordingly depending on which entry point you use.

## Model weights

Full checkpoint histories (up to 100 epochs per stage, ~1.8 GB per stage)
are not published here. `weights/` instead contains one representative
checkpoint per stage:

| Directory | Contents | Notes |
|---|---|---|
| `weights/qm9_pretrained/` | `backbone_U.pt`, `readout_U.pt` | QM9-pretrained DimeNet, split into backbone/readout (initialization for Stage 3). Reproduce with `data_prep/qm9_pretrain.py`. |
| `weights/denoising_ligand/` | `backbone_epoch_29.pt`, `ener_readout_epoch_29.pt` | Ligand-domain denoising pre-training, final saved epoch. |
| `weights/denoising_pocket/` | `backbone_epoch_26.pt`, `ener_readout_epoch_26.pt` | Pocket-domain denoising pre-training, final saved epoch (unfrozen readout, lr=1e-6, as reported). |
| `weights/clip_sair_pretrained/` | `dimenet_clip_epoch_98.pth`, `val_losses.npy` | SAIR contrastive pre-training, best validation-loss epoch (98/99, val loss 0.885). |
| `weights/clip_pdbbind_finetuned/` | `dimenet_clip_epoch_2.pth`, `val_losses.npy` | PDBBind contrastive fine-tuning checkpoint used for the DUDE / LIT-PCBA evaluation notebooks in this repo (val loss 392.6; from a short 6-epoch run — see caveat below). |

All `.pth` DimeNetCLIP checkpoints are plain `state_dict`s (already unwrapped
from `DistributedDataParallel`) and load directly with
`model.load_state_dict(torch.load(path))`.

**Caveat on `clip_pdbbind_finetuned`:** several PDBBind fine-tuning runs with
different ablation settings were produced during development. The checkpoint
published here is the one the DUDE/LIT-PCBA evaluation notebooks in `eval/`
were actually run against. A separate, fully-converged 100-epoch run (best
validation loss 94.5 at epoch 51) also exists; if you are trying to
reproduce a specific number from the manuscript and it doesn't match, this
is the first place to check.

## Running the pipeline

```bash
# 1. Ligand data prep
python data_prep/ligand_prep.py --lmdb_dir /path/to/unimol/ligands --output_dir /path/to/ligand_data

# 2. Pocket data prep
python data_prep/pocket_prep.py --lmdb_dir /path/to/unimol/pockets --output_dir /path/to/pocket_data

# 3. Denoising pre-training (single GPU)
python train_denoising.py --name pocket --dataset_path /path/to/pocket_data \
    --qm9_weights_dir weights/qm9_pretrained --unfreeze_readout --lr 1e-6 --batch_size 8

# 4. CLIP pre-training on SAIR (multi-GPU DDP; see slurm/submit_clip_train_sair.sh)
python train_clip_dimenet_sair.py --data_path /path/to/sair_preprocessed.pkl \
    --pocket_backbone_path weights/denoising_pocket/backbone_epoch_26.pt \
    --ligand_backbone_path weights/denoising_ligand/backbone_epoch_29.pt \
    --freeze_ligand --world_size 8

# 5. CLIP fine-tuning on PDBBind (multi-GPU DDP; see slurm/submit_clip_train_pdbbind.sh)
python train_clip_dimenet_pdbbind.py --data_path /path/to/pdbbind_preprocessed.pkl \
    --load_weight_path weights/clip_sair_pretrained/dimenet_clip_epoch_98.pth --world_size 8
```

See `slurm/` for complete example job scripts matching the hyperparameters
reported in the manuscript.

## Evaluation

`eval/dude_eval.ipynb` and `eval/lit_pcba_eval.{ipynb,py}` load a
DimeNetCLIP checkpoint, encode a benchmark's ligands/pockets, and report
ROC-AUC and enrichment factor per target. `eval/lit_pcba_labels_prep.ipynb`
shows how the LIT-PCBA input pickles are built from docked poses.
`eval/eval_denoising.ipynb` sanity-checks the denoising-pretrained backbones
before they're used to initialize Stage 4. Edit the configuration cell at
the top of each notebook (or the `--model_dir`/`--data_path` flags of
`lit_pcba_eval.py`) to point at your local copies of the benchmark data.

## License

MIT — see [LICENSE](LICENSE).
