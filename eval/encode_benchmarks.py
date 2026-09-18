#!/usr/bin/env python
"""
Encode a DUD-E benchmark set with a trained DimeNet-CLIP checkpoint and dump the
ligand/pocket embedding pickles that the figure pipeline consumes.

This is the scripted equivalent of the encoding cells in dude_eval.ipynb, so the
chain from checkpoint to manuscript figure can be run headlessly and reproducibly.

Input data formats (--dataset selects the layout):
  dude     pocket pkl: {target: (z_pocket, pocket_pos)}
           ligand pkl: {target: [(z_ligand, ligand_pos, label), ...]}
  litpcba  pocket pkl: {target: {pdb_id: (z_pocket, pocket_pos)}}   (several structures/target)
           ligand pkl: {target: {ligand_id: (z_ligand, ligand_pos, label)}}
  label 1/True = active, 0/False = decoy or inactive

Output (consumed by database/analysis/clip_screening_analysis_dude.py and
clip_pocket_specificity_control_dude.py):
  {out_prefix}_pocket_encoded_ep{epoch}.pkl : {target: Tensor[1, D]}          (dude)
                                              {target: {pdb_id: Tensor[1, D]}} (litpcba)
  {out_prefix}_ligand_encoded_ep{epoch}.pkl : {target: {'emb': Tensor[N, D], 'labels': Tensor[N]}}

Usage:
  python encode_benchmarks.py --model_dir ../weights/clip_pdbbind_finetuned --epoch 3 \
      --pocket_pkl /path/to/dude-pocket-4.pkl \
      --ligand_pkl /path/to/dude-ligand-z-pos.pkl \
      --output_dir /path/to/encodings
"""

import os
import sys
import argparse
import pickle

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dimenet_clip import DimeNetCLIP


def load_ddp_state_dict(model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    return {
        (k[len('module.'):] if k.startswith('module.') else k): v
        for k, v in state_dict.items()
    }


def encode_one_pocket(model, z, pos, device):
    d = Data(z=torch.tensor(z), pos=torch.tensor(pos, dtype=torch.float)).to(device)
    batch = Batch.from_data_list([d])
    return model.encode_pocket(d.z, d.pos, batch.batch).cpu()


def batch_encode_ligands(model, ligand_data, batch_size=32, device='cuda'):
    # accept either a list of (z, pos, label) or a dict keyed by ligand id
    if isinstance(ligand_data, dict):
        ligand_data = list(ligand_data.values())
    n_chunks = int(np.ceil(len(ligand_data) / batch_size))
    chunk_ids = np.array_split(range(len(ligand_data)), n_chunks)
    encoded = []
    labels = []
    for ids in chunk_ids:
        data_list = []
        for idx in ids:
            z, pos, label = ligand_data[idx]
            data_list.append(Data(z=torch.tensor(z), pos=torch.tensor(pos, dtype=torch.float)))
            labels.append(bool(label))
        batch = Batch.from_data_list(data_list).to(device)
        encoded.append(model.encode_ligand(batch.z, batch.pos, batch.batch).cpu())
    return torch.cat(encoded), torch.tensor(labels)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory holding dimenet_clip_epoch_{epoch}.pth')
    parser.add_argument('--epoch', type=int, required=True,
                        help='Checkpoint epoch to encode with (pick the argmin of val_losses.npy)')
    parser.add_argument('--pocket_pkl', type=str, required=True)
    parser.add_argument('--ligand_pkl', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='dude', choices=('dude', 'litpcba'),
                        help='Input layout: dude = one pocket per target; '
                             'litpcba = several pocket structures per target')
    parser.add_argument('--out_prefix', type=str, default='dude',
                        help="Basename stem for the two output pickles")
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--out_channels', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    with open(args.pocket_pkl, 'rb') as f:
        protein_vecs = pickle.load(f)
    with open(args.ligand_pkl, 'rb') as f:
        ligand_vecs = pickle.load(f)

    model = DimeNetCLIP(
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_blocks=args.num_blocks,
        use_conformers=False,
    )
    ckpt = os.path.join(args.model_dir, f'dimenet_clip_epoch_{args.epoch}.pth')
    model.load_state_dict(load_ddp_state_dict(ckpt))
    model.to(args.device).eval()
    print(f'Loaded {ckpt}')

    prot_encoded = {}
    lig_encoded = {}
    skipped = []
    with torch.no_grad():
        for t in tqdm(list(ligand_vecs.keys()), desc='targets'):
            if t not in protein_vecs or protein_vecs[t] is None:
                skipped.append(t)
                continue
            if args.dataset == 'litpcba':
                # several crystal structures per target, each encoded separately
                prot_encoded[t] = {
                    pdb_id: encode_one_pocket(model, z, pos, args.device)
                    for pdb_id, (z, pos) in protein_vecs[t].items()
                }
            else:
                z_prot, prot_coord = protein_vecs[t]
                prot_encoded[t] = encode_one_pocket(model, z_prot, prot_coord, args.device)

            emb, labels = batch_encode_ligands(
                model, ligand_vecs[t], batch_size=args.batch_size, device=args.device)
            lig_encoded[t] = {'emb': emb, 'labels': labels}

    os.makedirs(args.output_dir, exist_ok=True)
    pocket_out = os.path.join(
        args.output_dir, f'{args.out_prefix}_pocket_encoded_ep{args.epoch}.pkl')
    ligand_out = os.path.join(
        args.output_dir, f'{args.out_prefix}_ligand_encoded_ep{args.epoch}.pkl')
    with open(pocket_out, 'wb') as f:
        pickle.dump(prot_encoded, f)
    with open(ligand_out, 'wb') as f:
        pickle.dump(lig_encoded, f)

    print(f'Encoded {len(lig_encoded)} targets ({len(skipped)} skipped for missing pocket)')
    if skipped:
        print(f'  skipped: {sorted(skipped)}')
    print(f'Wrote {pocket_out}')
    print(f'Wrote {ligand_out}')


if __name__ == '__main__':
    main()
