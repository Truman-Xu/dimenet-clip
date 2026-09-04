"""
DDP training of DimeNetCLIP (contrastive learning) on the SAIR dataset
using UniqueProteinSampler: each batch contains one sample per unique
protein target, forming a valid CLIP contrastive matrix.

Dataset format (from extract_pocket.py):
    pickle: (z_ligs, lig_pos, z_pockets, pocket_pos, labels, protein_ids)
    labels: standardized pIC50 (zero-mean, unit-variance)

Note on affinity_cutoff:
    SigmoidWeightedCLIPLoss uses sigmoid(steepness * (label - cutoff)) to
    weight positive pairs. With standardized labels, use cutoff=0.0 (default).
    If you retrain the dataset with raw pIC50 labels, use cutoff~6.0 instead.
"""

import os
import random
import pickle
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from dimenet_clip import DimeNetCLIP, SigmoidWeightedCLIPLoss


# ==============================================================================
# Differentiable AllGather (for pooling embeddings across GPUs in CLIP loss)
# ==============================================================================

class DifferentiableAllGather(torch.autograd.Function):
    """
    Gathers tensors from all GPUs while keeping the computational graph intact.
    Forward:  all_gather  — each GPU gets the full concatenated tensor.
    Backward: reduce_scatter_tensor — each GPU receives the summed gradient
              for its own slice, matching the multi-GPU CLIP loss semantics.
    """
    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        return tuple(gathered)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_out = torch.zeros_like(grad_outputs[0])
        stacked = torch.stack([g.contiguous() for g in grad_outputs], dim=0)
        dist.reduce_scatter_tensor(grad_out, stacked)
        return grad_out


def all_gather_with_gradients(tensor):
    """Returns [world_size * B, D] with gradients flowing to the local slice."""
    return torch.cat(DifferentiableAllGather.apply(tensor), dim=0)


def all_gather_no_grad(tensor):
    """Plain all_gather for use inside torch.no_grad() (e.g. validation)."""
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


# ==============================================================================
# Dataset
# ==============================================================================

class SAIRDataset(Dataset):
    def __init__(self, vecs_path):
        with open(vecs_path, 'rb') as f:
            z_ligs, lig_pos, z_pockets, pocket_pos, labels, protein_ids = pickle.load(f)
        assert len(z_ligs) == len(lig_pos) == len(z_pockets) == len(pocket_pos) == len(labels)

        self.protein_ids = protein_ids  # list of strings, one per sample

        self.lig_slices = []
        self.pocket_slices = []
        self.lig_sizes = []      # number of atoms per ligand
        self.pocket_sizes = []   # number of atoms per pocket
        start = 0
        for arr in z_ligs:
            n = len(arr)
            self.lig_slices.append(slice(start, start + n))
            self.lig_sizes.append(n)
            start += n
        start = 0
        for arr in z_pockets:
            n = len(arr)
            self.pocket_slices.append(slice(start, start + n))
            self.pocket_sizes.append(n)
            start += n

        self.z_ligs = torch.tensor(np.concatenate(z_ligs), dtype=torch.int)
        self.lig_pos = torch.tensor(np.concatenate(lig_pos), dtype=torch.float)
        self.z_pockets = torch.tensor(np.concatenate(z_pockets), dtype=torch.int)
        self.pocket_pos = torch.tensor(np.concatenate(pocket_pos), dtype=torch.float)
        self.labels = torch.tensor(np.array(labels), dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'z_lig': self.z_ligs[self.lig_slices[idx]],
            'lig_pos': self.lig_pos[self.lig_slices[idx]],
            'z_pocket': self.z_pockets[self.pocket_slices[idx]],
            'pocket_pos': self.pocket_pos[self.pocket_slices[idx]],
            'label': self.labels[idx],
            'protein_id': self.protein_ids[idx],
        }


# ==============================================================================
# Collate Function
# ==============================================================================

def collate_fn(batch):
    z_ligs, lig_pos_list, z_pockets, pocket_pos_list = [], [], [], []
    labels, lig_batch_idx, pocket_batch_idx = [], [], []

    for i, sample in enumerate(batch):
        z_ligs.append(sample['z_lig'])
        lig_pos_list.append(sample['lig_pos'])
        z_pockets.append(sample['z_pocket'])
        pocket_pos_list.append(sample['pocket_pos'])
        labels.append(sample['label'])
        lig_batch_idx.append(torch.full(sample['z_lig'].shape, i, dtype=torch.long))
        pocket_batch_idx.append(torch.full(sample['z_pocket'].shape, i, dtype=torch.long))

    return {
        'z_lig': torch.cat(z_ligs),
        'lig_pos': torch.cat(lig_pos_list),
        'z_pocket': torch.cat(z_pockets),
        'pocket_pos': torch.cat(pocket_pos_list),
        'label': torch.stack(labels),
        'lig_batch': torch.cat(lig_batch_idx),
        'pocket_batch': torch.cat(pocket_batch_idx),
    }


# ==============================================================================
# Distributed Unique Protein Batch Sampler
# ==============================================================================

class DistributedUniqueProteinSampler(Sampler):
    """
    Batch sampler for DDP training that guarantees every batch contains
    exactly one sample from each of `batch_size` distinct protein targets.

    Proteins are partitioned across ranks (rank 0 gets proteins [0, W, 2W, ...],
    rank 1 gets [1, W+1, 2W+1, ...], etc.) so no two ranks share a protein
    within the same training step. This ensures the CLIP diagonal is valid
    on each GPU independently.

    Args:
        dataset:          SAIRDataset or Subset thereof.
        batch_size:       Number of unique protein-ligand pairs per batch.
        num_replicas:     DDP world size.
        rank:             Current DDP rank.
        seed:             Base random seed; combined with epoch for shuffling.
        max_pocket_atoms: Skip samples whose pocket exceeds this atom count.
                          Prevents CUDA OOM from very large pockets (default 150).
        max_lig_atoms:    Skip samples whose ligand exceeds this atom count (default 100).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        max_pocket_atoms: int = 150,
        max_lig_atoms: int = 100,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        # Resolve underlying SAIRDataset and index mapping for Subset
        if isinstance(dataset, torch.utils.data.Subset):
            underlying = dataset.dataset
            index_map = list(dataset.indices)  # local_idx -> global_idx
        else:
            underlying = dataset
            index_map = list(range(len(dataset)))

        # Build protein -> [local_indices] map, filtering by atom count
        n_total = len(index_map)
        n_filtered = 0
        protein_to_indices = defaultdict(list)
        for local_idx, global_idx in enumerate(index_map):
            if underlying.pocket_sizes[global_idx] > max_pocket_atoms:
                n_filtered += 1
                continue
            if underlying.lig_sizes[global_idx] > max_lig_atoms:
                n_filtered += 1
                continue
            pid = underlying.protein_ids[global_idx]
            protein_to_indices[pid].append(local_idx)

        if rank == 0:
            print(
                f"[Sampler] Filtered {n_filtered}/{n_total} samples by atom count "
                f"(max_pocket={max_pocket_atoms}, max_lig={max_lig_atoms}). "
                f"Remaining: {n_total - n_filtered}"
            )

        self.protein_to_indices = dict(protein_to_indices)

        # Partition proteins across ranks deterministically
        all_proteins = sorted(self.protein_to_indices.keys())
        self.local_proteins = all_proteins[rank::num_replicas]

        if len(self.local_proteins) < batch_size:
            raise ValueError(
                f"Rank {rank}: {len(self.local_proteins)} proteins assigned, "
                f"but batch_size={batch_size}. Reduce --batch_size or --world_size."
            )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        local_proteins = list(self.local_proteins)

        # Budget: one pass over local samples
        total_local_samples = sum(
            len(self.protein_to_indices[p]) for p in local_proteins
        )
        total_batches = total_local_samples // self.batch_size

        # Sync to the minimum across all ranks so every rank iterates the same
        # number of batches — prevents DDP collective mismatch / deadlock when
        # proteins are unevenly distributed after atom-count filtering.
        total_batches_t = torch.tensor(total_batches, dtype=torch.long,
                                       device=self.rank)
        all_reduce(total_batches_t, op=ReduceOp.MIN)
        total_batches = int(total_batches_t.item())

        for _ in range(total_batches):
            selected = rng.sample(local_proteins, self.batch_size)
            yield [rng.choice(self.protein_to_indices[p]) for p in selected]

    def __len__(self):
        total_local = sum(
            len(self.protein_to_indices[p]) for p in self.local_proteins
        )
        return total_local // self.batch_size

def load_pretrained_model_weights(model: DimeNetCLIP, pocket_backbone_path: str, ligand_backbone_path: str):
    p_state_dict = torch.load(pocket_backbone_path)
    l_state_dict = torch.load(ligand_backbone_path)
    model.backbone_pocket.load_state_dict(p_state_dict)
    model.backbone_ligand.load_state_dict(l_state_dict)

def load_profsa_model_weights(model: DimeNetCLIP, model_weights_path: str):
    model.load_state_dict(torch.load(model_weights_path))
# ==============================================================================
# DDP Setup / Teardown
# ==============================================================================

def ddp_setup(rank: int, world_size: int, port: str = "29500"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["MASTER_PORT"] = port
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


# ==============================================================================
# Training Worker
# ==============================================================================

def ddp_train(
    rank,
    world_size,
    n_epochs,
    data_path,
    model_save_path,
    batch_size,
    hidden_channels,
    out_channels,
    num_blocks,
    affinity_cutoff,
    seed,
    max_pocket_atoms,
    max_lig_atoms,
    freeze_ligand,
    freeze_pocket,
    load_weight_path,
    pocket_backbone_path,
    ligand_backbone_path,
):
    print(f"[rank {rank}] Starting DDP training.")
    ddp_setup(rank, world_size)

    # ---- Model ----
    # use_conformers=False: dataset has a single RDKit conformer per ligand
    model = DimeNetCLIP(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_blocks=num_blocks,
        use_conformers=False,
    )

    if load_weight_path is not None:
        # Warm-start from a full CLIP checkpoint (e.g. one pre-trained on ProfSA).
        load_profsa_model_weights(model, load_weight_path)
    elif pocket_backbone_path is not None or ligand_backbone_path is not None:
        # Fall back to initializing each branch from its denoising-pretrained
        # backbone (see train_denoising.py / weights/denoising_{ligand,pocket}).
        load_pretrained_model_weights(model, pocket_backbone_path, ligand_backbone_path)

    # Freeze ligand or pocket branch (backbone) before DDP wrapping
    total_frozen = 0
    total = sum(p.numel() for p in model.parameters())
    if freeze_ligand:
        ligand_modules = [
            model.backbone_ligand,
            # model.readout_ligand,
            # model.proj_ligand,
        ]
        if model.conformer_agg is not None:
            ligand_modules.append(model.conformer_agg)
        for m in ligand_modules:
            for p in m.parameters():
                p.requires_grad_(False)
        if rank == 0:
            total_frozen += sum(p.numel() for m in ligand_modules for p in m.parameters())

    if freeze_pocket:
        for p in model.backbone_pocket.parameters():
            p.requires_grad_(False)
        if rank == 0:
            total_frozen += sum(p.numel() for p in model.backbone_pocket.parameters())

    if total_frozen > 0 and rank == 0:
        print(f"[rank 0] Num frozen: {total_frozen:,} / {total:,} params")

    model.to(rank)
    # DDP only syncs gradients for parameters that require grad
    model = DDP(model, device_ids=[rank])

    loss_fn = SigmoidWeightedCLIPLoss(cutoff=affinity_cutoff).to(rank)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # ---- Dataset & Splits ----
    full_dataset = SAIRDataset(data_path)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [0.95, 0.05],
        generator=torch.Generator().manual_seed(42)
    )
    if rank == 0:
        print(
            f"Dataset sizes — total: {len(full_dataset)}, "
            f"train: {len(train_dataset)}, val: {len(val_dataset)}"
        )

    # ---- Samplers & Loaders ----
    train_sampler = DistributedUniqueProteinSampler(
        train_dataset,
        batch_size=batch_size,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        max_pocket_atoms=max_pocket_atoms,
        max_lig_atoms=max_lig_atoms,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,   # handles both batching and sampling
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    # ---- Epoch Loop ----
    os.makedirs(model_save_path, exist_ok=True)
    val_losses = []

    for epoch in tqdm(range(n_epochs), disable=(rank != 0), desc="Epochs"):
        train_sampler.set_epoch(epoch)

        # -- Training --
        model.train()
        nan_count = 0
        progress_bar = tqdm(train_loader, disable=(rank != 0), leave=False, desc="Train")
        for batch in progress_bar:
            optimizer.zero_grad()

            data_lig = SimpleNamespace(
                z=batch['z_lig'].to(rank),
                pos=batch['lig_pos'].to(rank),
                batch=batch['lig_batch'].to(rank),
            )
            data_pkt = SimpleNamespace(
                z=batch['z_pocket'].to(rank),
                pos=batch['pocket_pos'].to(rank),
                batch=batch['pocket_batch'].to(rank),
            )
            labels = batch['label'].to(rank)

            v_l, v_p, logit_scale = model(data_lig, data_pkt)

            nan_found = torch.tensor(
                [not torch.isfinite(v_l).all() or not torch.isfinite(v_p).all()],
                device=rank,
            )
            all_reduce(nan_found, op=ReduceOp.MAX)
            if nan_found:
                nan_count += 1
                optimizer.zero_grad()
                continue

            # Pool embeddings and labels from all GPUs for a larger CLIP matrix
            all_v_l = all_gather_with_gradients(v_l)
            all_v_p = all_gather_with_gradients(v_p)
            all_labels = all_gather_no_grad(labels)

            loss = loss_fn(all_v_l, all_v_p, logit_scale, all_labels)

            skip = torch.tensor(
                [not torch.isfinite(loss) or loss > 1e4],
                device=rank,
            )
            all_reduce(skip, op=ReduceOp.MAX)
            if skip:
                nan_count += 1
                optimizer.zero_grad()
                continue

            progress_bar.set_postfix({"loss": loss.item(), "NaN count": nan_count})
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch} | train NaN batches: {nan_count}")

        # -- Validation --
        model.eval()
        total_loss = torch.tensor(0.0, device=rank)
        nan_count = 0
        progress_bar = tqdm(val_loader, disable=(rank != 0), leave=False, desc='Test')
        with torch.no_grad():
            for batch in progress_bar:
                data_lig = SimpleNamespace(
                    z=batch['z_lig'].to(rank),
                    pos=batch['lig_pos'].to(rank),
                    batch=batch['lig_batch'].to(rank),
                )
                data_pkt = SimpleNamespace(
                    z=batch['z_pocket'].to(rank),
                    pos=batch['pocket_pos'].to(rank),
                    batch=batch['pocket_batch'].to(rank),
                )
                labels = batch['label'].to(rank)

                v_l, v_p, logit_scale = model(data_lig, data_pkt)
                nan_found = torch.tensor([not torch.isfinite(v_l).all() or not torch.isfinite(v_p).all()], device=rank)
                all_reduce(nan_found, op=ReduceOp.MAX)
                if nan_found:
                    nan_count += 1
                    continue

                all_v_l = all_gather_no_grad(v_l)
                all_v_p = all_gather_no_grad(v_p)
                all_labels = all_gather_no_grad(labels)

                loss = loss_fn(all_v_l, all_v_p, logit_scale, all_labels)
                nan_found = torch.tensor([not torch.isfinite(loss).all()], device=rank)
                large_loss = torch.tensor([(loss > 1e3).any()], device=rank)
                skip = torch.logical_or(nan_found, large_loss).to(rank)
                all_reduce(skip, op=ReduceOp.MAX)
                if skip:
                    nan_count += 1
                    continue
                total_loss += loss
                progress_bar.set_postfix({"loss": f'{loss.item():.3e}', "NaN count": nan_count})

        all_reduce(total_loss, op=ReduceOp.SUM)
        avg_val_loss = (total_loss / (len(val_dataset)/batch_size)).item()
        scheduler.step(avg_val_loss)

        if rank == 0:
            print(
                f"Epoch {epoch} | val NaN batches: {nan_count} | "
                f"val loss: {avg_val_loss:.4e} | logit_scale: {logit_scale.item():.3f}"
            )
            val_losses.append(avg_val_loss)
            torch.save(
                model.module.state_dict(),
                os.path.join(model_save_path, f"dimenet_clip_epoch_{epoch}.pth")
            )
            np.save(
                os.path.join(model_save_path, "val_losses.npy"),
                np.array(val_losses)
            )
            torch.cuda.empty_cache()

    destroy_process_group()


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train DimeNetCLIP on SAIR with DDP + UniqueProteinSampler'
    )
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to the preprocessed SAIR pickle file (ligand/pocket coords + affinity labels)'
    )
    parser.add_argument(
        '--model_save_path', type=str, default='clip-models/',
        help='Directory to save model checkpoints'
    )
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Number of unique protein-ligand pairs per batch per GPU'
    )
    parser.add_argument('--world_size', type=int, default=8)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--out_channels', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=6)
    parser.add_argument(
        '--affinity_cutoff', type=float, default=0.0,
        help=(
            'Sigmoid cutoff for SigmoidWeightedCLIPLoss. '
            'Default 0.0 for standardized pIC50 labels. '
            'Use ~6.0 if labels are raw pIC50.'
        )
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--max_pocket_atoms', type=int, default=200,
        help='Skip pockets with more than this many atoms (prevents CUDA OOM on 11 GB GPUs)'
    )
    parser.add_argument(
        '--max_lig_atoms', type=int, default=80,
        help='Skip ligands with more than this many atoms'
    )
    parser.add_argument(
        '--freeze_ligand', action='store_true', default=False,
        help='Freeze ligand-branch backbone weights during training'
    )
    parser.add_argument(
        '--freeze_pocket', action='store_true', default=False,
        help='Freeze pocket-branch backbone weights during training'
    )
    parser.add_argument(
        '--load_weight_path', type=str, default=None,
        help=(
            'Optional path to a full DimeNetCLIP checkpoint to warm-start from '
            '(e.g. a ProfSA-pretrained checkpoint). If omitted, the ligand/pocket '
            'backbones are instead initialized from --ligand_backbone_path / '
            '--pocket_backbone_path.'
        )
    )
    parser.add_argument(
        '--pocket_backbone_path', type=str, default=None,
        help='Path to a denoising-pretrained pocket backbone .pt (see weights/denoising_pocket); '
             'used only if --load_weight_path is not set'
    )
    parser.add_argument(
        '--ligand_backbone_path', type=str, default=None,
        help='Path to a denoising-pretrained ligand backbone .pt (see weights/denoising_ligand); '
             'used only if --load_weight_path is not set'
    )

    args = parser.parse_args()

    mp.spawn(
        ddp_train,
        args=(
            args.world_size,
            args.n_epochs,
            args.data_path,
            args.model_save_path,
            args.batch_size,
            args.hidden_channels,
            args.out_channels,
            args.num_blocks,
            args.affinity_cutoff,
            args.seed,
            args.max_pocket_atoms,
            args.max_lig_atoms,
            args.freeze_ligand,
            args.freeze_pocket,
            args.load_weight_path,
            args.pocket_backbone_path,
            args.ligand_backbone_path,
        ),
        nprocs=args.world_size,
        join=True,
    )
