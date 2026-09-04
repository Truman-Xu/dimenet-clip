#!/usr/bin/env python
"""
Evaluate CLIP model on LIT-PCBA dataset (no-hydrogen, sep-pocket version).

Input data format (pcba_sep_pocket_vecs.pkl):
  {target: {(pdb_id, ligand_id): {z_lig, lig_pos, z_pocket, pocket_pos, label}}}

Usage:
  python lit_pcba_eval.py --model_dir ../weights/clip_pdbbind_finetuned --epoch 51 \
      --data_path /path/to/pcba_sep_pocket_vecs.pkl
"""

import os
import sys
import argparse
import numpy as np
import pickle
import torch
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dimenet_clip import DimeNetCLIP


def load_ddp_state_dict(model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    return {
        (k[len('module.'):] if k.startswith('module.') else k): v
        for k, v in state_dict.items()
    }


def calculate_ef(labels, scores, fraction=0.01):
    combined = sorted(zip(scores, labels), reverse=True, key=lambda x: x[0])
    sorted_labels = [label for _, label in combined]
    n_total = len(sorted_labels)
    n_x = max(1, int(n_total * fraction))
    hits_x = sum(sorted_labels[:n_x])
    hits_total = sum(sorted_labels)
    if hits_total == 0:
        return 0.0
    return (hits_x / n_x) / (hits_total / n_total)


def extract_pockets_and_ligands(target_data):
    """
    From {(pdb_id, ligand_id): data_dict} extract:
      - unique_pockets: {pdb_id: (z_pocket, pocket_pos)}
      - ligands: list of (z_lig, lig_pos, label), deduplicated by ligand_id
    """
    unique_pockets = {}
    ligands_by_id = {}
    for (pdb_id, ligand_id), d in target_data.items():
        if pdb_id not in unique_pockets:
            unique_pockets[pdb_id] = (d['z_pocket'], d['pocket_pos'])
        if ligand_id not in ligands_by_id:
            ligands_by_id[ligand_id] = (d['z_lig'], d['lig_pos'], d['label'])
    return unique_pockets, list(ligands_by_id.values())


def batch_encode_ligands(model, ligand_list, batch_size=32, device='cuda'):
    n_chunks = max(1, int(np.ceil(len(ligand_list) / batch_size)))
    chunk_ids = np.array_split(range(len(ligand_list)), n_chunks)
    encoded, labels = [], []
    for ids in chunk_ids:
        data_list = []
        for idx in ids:
            z, pos, label = ligand_list[idx]
            data_list.append(Data(z=torch.tensor(z), pos=torch.tensor(pos, dtype=torch.float)))
            labels.append(label)
        batch = Batch.from_data_list(data_list).to(device)
        emb = model.encode_ligand(batch.z, batch.pos, batch.batch)
        encoded.append(emb.cpu())
    return torch.cat(encoded), torch.tensor(labels)


def main():
    parser = argparse.ArgumentParser(description='Evaluate CLIP model on LIT-PCBA (no-H sep-pocket)')
    parser.add_argument(
        '--model_dir', required=True,
        help='Directory containing dimenet_clip_epoch_{epoch}.pth, e.g. weights/clip_pdbbind_finetuned')
    parser.add_argument('--epoch', type=int, default=51, help='Checkpoint epoch to load (see --model_dir)')
    parser.add_argument(
        '--data_path', required=True,
        help='Path to a preprocessed LIT-PCBA pickle: {target: {(pdb_id, ligand_id): '
             '{z_lig, lig_pos, z_pocket, pocket_pos, label}}}')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_encodings', action='store_true',
                        help='Save pocket/ligand encodings to pickle files')
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}")
    with open(args.data_path, 'rb') as f:
        sep_data = pickle.load(f)
    targets = list(sep_data.keys())
    print(f"Targets: {targets}\n")

    model = DimeNetCLIP(hidden_channels=128, out_channels=128, num_blocks=6, use_conformers=False)
    model_path = os.path.join(args.model_dir, f"dimenet_clip_epoch_{args.epoch}.pth")
    print(f"Loading model from {model_path}")
    model.load_state_dict(load_ddp_state_dict(model_path))
    model.to(args.device)
    model.eval()
    print("Model loaded.\n")

    prot_encoded = {}
    lig_encoded = {}

    with torch.no_grad():
        for t in tqdm(targets, desc='Encoding targets'):
            unique_pockets, ligand_list = extract_pockets_and_ligands(sep_data[t])

            prot_encoded[t] = {}
            for pdb_id, (z_prot, prot_coord) in unique_pockets.items():
                prot_data = Data(
                    z=torch.tensor(z_prot),
                    pos=torch.tensor(prot_coord, dtype=torch.float)
                ).to(args.device)
                batch = Batch.from_data_list([prot_data])
                prot_emb = model.encode_pocket(prot_data.z, prot_data.pos, batch.batch)
                prot_encoded[t][pdb_id] = prot_emb.cpu()

            cur_encoded, labels = batch_encode_ligands(
                model, ligand_list, batch_size=args.batch_size, device=args.device
            )
            lig_encoded[t] = {'emb': cur_encoded, 'labels': labels}

    if args.save_encodings:
        out_dir = os.path.dirname(args.data_path)
        with open(os.path.join(out_dir, f'sep_pocket_encoded_ep{args.epoch}.pkl'), 'wb') as f:
            pickle.dump(prot_encoded, f)
        with open(os.path.join(out_dir, f'sep_ligand_encoded_ep{args.epoch}.pkl'), 'wb') as f:
            pickle.dump(lig_encoded, f)
        print("Encodings saved.\n")

    # Compute similarity scores and metrics
    preds = {}
    for t in targets:
        preds[t] = {}
        lig_emb = lig_encoded[t]['emb']
        labels = lig_encoded[t]['labels']
        for pdb_id, prot_emb in prot_encoded[t].items():
            scores = torch.matmul(lig_emb.to(args.device), prot_emb.to(args.device).T).flatten()
            valid_ids = torch.where(scores.isfinite())[0].cpu()
            filtered_labels = labels[valid_ids]
            filtered_scores = scores[valid_ids].cpu()
            auc = roc_auc_score(filtered_labels, filtered_scores)
            ef1 = calculate_ef(filtered_labels.tolist(), filtered_scores.tolist())
            preds[t][pdb_id] = {
                'auc': auc, 'ef1': ef1,
                'preds': filtered_scores, 'labels': filtered_labels
            }

    # Report
    print(f"\n{'='*60}")
    print(f"Model: epoch {args.epoch}")
    print(f"Model dir: {args.model_dir}")
    print(f"Data: {args.data_path}")
    print('='*60)

    all_aucs, all_ef1s = [], []
    for t, by_pdbid in preds.items():
        aucs = np.array([r['auc'] for r in by_pdbid.values()])
        ef1s = np.array([r['ef1'] for r in by_pdbid.values()])
        n_ligs = list(by_pdbid.values())[0]['labels'].shape[0]
        all_aucs.append(aucs.mean())
        all_ef1s.append(ef1s.mean())
        print(f"\n{t}")
        print(f"  Num Structures: {len(aucs)}, Num Ligands: {n_ligs}")
        print(f"  AUC mean: {aucs.mean():.3f}, std: {aucs.std():.3f}, "
              f"max: {aucs.max():.3f}, min: {aucs.min():.3f}")
        print(f"  EF1 mean: {ef1s.mean():.3f}, std: {ef1s.std():.3f}, "
              f"max: {ef1s.max():.3f}, min: {ef1s.min():.3f}")

    all_aucs = np.array(all_aucs)
    all_ef1s = np.array(all_ef1s)
    print(f"\n{'='*60}")
    print(f"Overall Mean AUC: {all_aucs.mean():.4f} ± {all_aucs.std():.4f}")
    print(f"Overall Mean EF1: {all_ef1s.mean():.4f} ± {all_ef1s.std():.4f}")
    print('='*60)


if __name__ == '__main__':
    main()
