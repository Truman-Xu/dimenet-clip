# DimeNetCLIP

DimeNetCLIP is a joint ligand–pocket embedding model for protein–ligand
binding: a [DimeNet](https://arxiv.org/abs/2003.03123) 3D graph neural
network encodes a small-molecule ligand and a protein binding pocket into a
shared latent space, trained with a CLIP-style contrastive objective so that
binders end up close together and non-binders end up far apart. The
similarity between a ligand embedding and a pocket embedding is a
binding/affinity score, useful for virtual screening and pose/compound
ranking.


## Installation

```bash
git clone <this-repo-url>
cd dimenet-clip
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric's optional wheels (`torch-scatter`,
`torch-sparse`, `torch-cluster`) must be installed separately, matched to
your CUDA version — see the [PyG install
guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

Tested with Python ≥3.10, PyTorch Geometric 2.7, and a CUDA-capable GPU (CPU
also works for inference on small inputs, just pass `device='cpu'` below).

## Model weights

Pretrained checkpoints are included in [`weights/`](weights/). For
inference / scoring, use the fully fine-tuned model:

| Checkpoint | Use it for |
| --- | --- |
| `weights/clip_pdbbind_finetuned/dimenet_clip_epoch_3.pth` | Ligand–pocket scoring / virtual screening — the model behind every DimeNet-CLIP number in the manuscript |
| `weights/clip_sair_pretrained/dimenet_clip_epoch_5.pth` | Earlier-stage checkpoint, contrastively pre-trained on SAIR only (the checkpoint the model above was fine-tuned from) |

The lineage of every checkpoint, and how it was verified, is documented in
[`weights/PROVENANCE.md`](weights/PROVENANCE.md).


## Quickstart: score a ligand against a pocket

```python
import torch
from torch_geometric.data import Data, Batch
from dimenet_clip import DimeNetCLIP

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DimeNetCLIP(hidden_channels=128, out_channels=128, num_blocks=6,
                     use_conformers=False)
state_dict = torch.load(
    'weights/clip_pdbbind_finetuned/dimenet_clip_epoch_3.pth',
    map_location=device,
)
# Checkpoints were saved from DistributedDataParallel; strip the prefix.
state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.to(device).eval()

# z: atomic numbers (LongTensor), pos: 3D coordinates in Angstroms (FloatTensor)
ligand = Data(z=ligand_z, pos=ligand_pos).to(device)
pocket = Data(z=pocket_z, pos=pocket_pos).to(device)

with torch.no_grad():
    ligand_batch = Batch.from_data_list([ligand])
    pocket_batch = Batch.from_data_list([pocket])
    v_ligand = model.encode_ligand(ligand_batch.z, ligand_batch.pos, ligand_batch.batch)
    v_pocket = model.encode_pocket(pocket_batch.z, pocket_batch.pos, pocket_batch.batch)

score = (v_ligand @ v_pocket.T).item()  # higher = more likely to bind
```

`encode_ligand`/`encode_pocket` accept PyG-style batches, so you can encode
many ligands or pockets in one call by batching several `Data` objects
together and ranking rows of the resulting similarity matrix — this is what
the benchmark evaluation below does.

## Reproducing the benchmark figure

[`eval/dude_eval.ipynb`](eval/dude_eval.ipynb) loads the released
`clip_pdbbind_finetuned` checkpoint, encodes every ligand and pocket in the
[DUD-E](http://dude.docking.org/) virtual-screening benchmark, ranks ligands
by predicted score against each target's pocket, and reports ROC-AUC / 1%
enrichment factor per target, plus a per-target recall curve (cumulative
actives recovered vs. number of top-ranked predictions, against the
chance-level diagonal) — the main results figure for this model.

To reproduce it:

1. Obtain DUD-E separately (not bundled here — see the [DUD-E
   website](http://dude.docking.org/)) and preprocess it into two pickles:
   - a pocket pickle: `{target: (z_pocket, pocket_pos)}`
   - a ligand pickle: `{target: [(z_ligand, ligand_pos, label), ...]}`
     (`label` is 1 for an active, 0 for a decoy)
2. Open `eval/dude_eval.ipynb` and edit the configuration cell at the top:
   ```python
   DUDE_POCKET_PKL = '/path/to/dude-pocket-4.pkl'
   DUDE_LIGAND_PKL = '/path/to/dude-ligand-z-pos.pkl'
   MODEL_DIR = '../weights/clip_pdbbind_finetuned'
   MODEL_EPOCH = 3
   ```
3. Run all cells. The "Results" section prints mean/max/min AUC and EF1
   across targets, and plots the per-target recall curve.

To reproduce the manuscript's [LIT-PCBA](https://drugdesign.unistra.fr/LIT-PCBA/)
numbers, encode the benchmark with the hydrogen-bearing pocket and ligand
pickles (`h_pocket_4.pkl`, `h_ligands.pkl`) that
`eval/lit_pcba_labels_prep.ipynb` builds from docked poses:

```bash
python eval/encode_benchmarks.py --dataset litpcba --out_prefix lit \
    --model_dir weights/clip_pdbbind_finetuned --epoch 3 \
    --pocket_pkl /path/to/h_pocket_4.pkl --ligand_pkl /path/to/h_ligands.pkl \
    --output_dir /path/to/encodings
```

`eval/encode_benchmarks.py --dataset dude` does the same for DUD-E headlessly.

Additionally, [`eval/lit_pcba_eval.py`](eval/lit_pcba_eval.py) scores a *different*,
hydrogen-stripped per-complex pocket set (`pcba_sep_pocket_vecs.pkl`):

```bash
python eval/lit_pcba_eval.py \
    --model_dir weights/clip_pdbbind_finetuned --epoch 3 \
    --data_path /path/to/pcba_sep_pocket_vecs.pkl
```

## Reproducing the manuscript figures

[`plots/`](plots/) contains the scripts behind every data-driven figure in the
manuscript and Table 1, with a driver that runs them in order
(`plots/make_all.sh`). See [plots/README.md](plots/README.md) for the
figure-to-script map and the inputs each figure needs.

## Retraining or reproducing the manuscript's numbers exactly

The full five-stage training pipeline (ligand/pocket data prep, denoising
pre-training, SAIR contrastive pre-training, PDBBind fine-tuning), SLURM job
scripts, and per-checkpoint provenance notes are documented in
[TRAINING.md](TRAINING.md).

## License

MIT — see [LICENSE](LICENSE).
