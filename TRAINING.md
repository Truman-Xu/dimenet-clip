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

Note on stage 3 input naming: both `train_denoising.py` entry points read the
same four files that `ligand_prep.py` / `pocket_prep.py` write —
`{prefix}{name}_pos_train.pkl`, `{prefix}z_{name}_train.pkl`,
`{prefix}{name}_pos_valid.pkl`, `{prefix}z_{name}_valid.pkl` — where the prefix
is set with `--file_prefix`. Use `--file_prefix h_` for data prepared with
hydrogens retained (the prep scripts' default) and `--file_prefix ""` (the
default) for data prepared with `--remove_hs`.

Hydrogens: the released `weights/denoising_*` backbones were pre-trained on
**hydrogen-stripped** structures, i.e. the prep scripts run with `--remove_hs`,
even though the later CLIP stages and all benchmark evaluation use structures
that retain hydrogens. Pass `--remove_hs` to both prep scripts to reproduce the
released backbones.

Reproducibility: `ligand_prep.py` samples `--n_total` molecules from the LMDB
without replacement; pass `--seed` (default 42) to make that sample
deterministic.

Resuming: all three training scripts accept `--resume`, which restores model,
optimizer, scheduler and epoch from `last_checkpoint.pt` in the output
directory. It is safe to pass on a first run. Stage 3 takes days on 8 GPUs, so
submit it with `--resume` from the start.

## Model weights

Full checkpoint histories (up to 100 epochs per stage, ~1.8 GB per stage)
are not published here. `weights/` instead contains one representative
checkpoint per stage:

| Directory | Contents | Notes |
|---|---|---|
| `weights/qm9_pretrained/` | `backbone_U.pt`, `readout_U.pt` | QM9-pretrained DimeNet, split into backbone/readout (initialization for Stage 3). Reproduce with `data_prep/qm9_pretrain.py`. |
| `weights/denoising_ligand/` | `backbone_epoch_29.pt`, `ener_readout_epoch_29.pt` | Ligand-domain denoising pre-training, final saved epoch (readout trained jointly with the backbone). |
| `weights/denoising_pocket/` | `backbone_epoch_26.pt`, `ener_readout_epoch_26.pt` | Pocket-domain denoising pre-training, final saved epoch (readout frozen at its QM9 values, lr=1e-6). |
| `weights/clip_sair_pretrained/` | `dimenet_clip_epoch_5.pth`, `val_losses.npy` | SAIR contrastive pre-training, epoch 5 (val loss 0.416) — the checkpoint the reported PDBBind model was fine-tuned from. `val_losses.npy` covers the whole 100-epoch run (best: epoch 99, 0.136). |
| `weights/clip_pdbbind_finetuned/` | `dimenet_clip_epoch_3.pth`, `val_losses.npy` | PDBBind contrastive fine-tuning — the model behind every DimeNet-CLIP number in the manuscript. Epoch 3 is the validation minimum (22.98) of a 9-epoch run. |

All `.pth` DimeNetCLIP checkpoints are plain `state_dict`s (already unwrapped
from `DistributedDataParallel`) and load directly with
`model.load_state_dict(torch.load(path))`.

**Provenance:** each checkpoint above was traced to the exact run and epoch
behind the manuscript's results, by re-encoding the benchmarks and matching
embeddings to ~1e-7 and by nearest-neighbour parameter search for training
parents. See [`weights/PROVENANCE.md`](weights/PROVENANCE.md). Two things
about the released lineage are worth knowing when retraining: the ligand
backbone is frozen in *both* contrastive stages (it is bit-identical to
`denoising_ligand/backbone_epoch_29.pt`), and PDBBind fine-tuning started
from an intermediate SAIR checkpoint (epoch 5), not the converged one.

## Running the pipeline

```bash
# 1. Ligand data prep (--remove_hs reproduces the released backbones)
python data_prep/ligand_prep.py --lmdb_dir /path/to/unimol/ligands \
    --output_dir /path/to/denoising_data --remove_hs --seed 42

# 2. Pocket data prep
python data_prep/pocket_prep.py --lmdb_dir /path/to/unimol/pockets \
    --output_dir /path/to/denoising_data --remove_hs

# 3. Denoising pre-training (8-GPU DDP; see slurm/submit_denoise_train.sh).
#    The released backbones trained the ligand readout and froze the pocket one.
python train_denoising.py --name ligand --dataset_path /path/to/denoising_data \
    --qm9_weights_dir weights/qm9_pretrained --save_dir /path/to/ligand_models \
    --world_size 8 --unfreeze_readout --lr 1e-6 --batch_size 16 --resume
python train_denoising.py --name pocket --dataset_path /path/to/denoising_data \
    --qm9_weights_dir weights/qm9_pretrained --save_dir /path/to/pocket_models \
    --world_size 8 --lr 1e-6 --batch_size 8 --resume

# 4. CLIP pre-training on SAIR (multi-GPU DDP; see slurm/submit_clip_train_sair.sh)
python train_clip_dimenet_sair.py --data_path /path/to/sair_preprocessed.pkl \
    --pocket_backbone_path weights/denoising_pocket/backbone_epoch_26.pt \
    --ligand_backbone_path weights/denoising_ligand/backbone_epoch_29.pt \
    --freeze_ligand --world_size 8 --batch_size 4 --affinity_cutoff -1.0 \
    --max_pocket_atoms 300 --max_lig_atoms 100 --resume

# 5. CLIP fine-tuning on PDBBind (multi-GPU DDP; see slurm/submit_clip_train_pdbbind.sh)
python train_clip_dimenet_pdbbind.py --data_path /path/to/pdbbind_preprocessed.pkl \
    --load_weight_path weights/clip_sair_pretrained/dimenet_clip_epoch_5.pth \
    --world_size 8 --batch_size 16 --n_epochs 30 --affinity_cutoff 0 \
    --freeze_ligand --resume

# 6. Encode a benchmark with the fine-tuned checkpoint (bridges stage 5 to the figures)
python eval/encode_benchmarks.py --model_dir clip-pdbbind-finetuned --epoch <best> \
    --pocket_pkl /path/to/dude-pocket-4.pkl \
    --ligand_pkl /path/to/dude-ligand-z-pos.pkl \
    --output_dir /path/to/encodings
```

Pick `<best>` as the `argmin` of `val_losses.npy` in the stage-4 / stage-5
output directory. Stage 4 is initialized from the denoising backbones; passing
`--load_weight_path` instead overrides that and initializes the entire CLIP
model from an existing checkpoint — the two initialization paths are mutually
exclusive.

See `slurm/` for complete example job scripts matching the hyperparameters
reported in the manuscript.

## Evaluation

`eval/dude_eval.ipynb` and `eval/lit_pcba_eval.{ipynb,py}` load a
DimeNetCLIP checkpoint, encode a benchmark's ligands/pockets, and report
ROC-AUC and enrichment factor per target. `eval/lit_pcba_labels_prep.ipynb`
shows how the LIT-PCBA input pickles are built from docked poses.
`eval/encode_benchmarks.py` is the headless, scripted form of the encoding
cells in `dude_eval.ipynb`: it writes the `*_ligand_encoded_ep{N}.pkl` and
`*_pocket_encoded_ep{N}.pkl` pickles that the downstream per-target
AUC-ROC / enrichment analysis consumes.
`eval/eval_denoising.ipynb` sanity-checks a denoising backbone before it is
used to initialize Stage 4 (note: as committed it loads
`weights/qm9_pretrained/` and defines its own all-atom variant of
`compute_denoising_loss`, so point it at `weights/denoising_*` and use the
library loss if you want numbers comparable to training-time loss). Edit the configuration cell at
the top of each notebook (or the `--model_dir`/`--data_path` flags of
`lit_pcba_eval.py`) to point at your local copies of the benchmark data.

## License

MIT — see [LICENSE](LICENSE).
