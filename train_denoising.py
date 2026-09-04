import torch
# torch.autograd.set_detect_anomaly(True)
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import pickle
from dimenet_clip import (
    DimeNetPretrainer, compute_denoising_loss
) 

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp

def collate_fn(batch):
    z_pockets = []
    pocket_pos_list = []
    pocket_batch = []
    
    for i, sample in enumerate(batch):
        z = torch.tensor(sample[0], dtype=torch.int)
        z_pockets.append(z)
        pocket_pos = torch.tensor(sample[1], dtype=torch.float)
        pocket_pos_list.append(pocket_pos)
        pocket_batch.append(torch.full(z.shape, i, dtype=torch.long))
        
    return {
        'z': torch.cat(z_pockets, dim=0),
        'pos': torch.cat(pocket_pos_list, dim=0),
        'batch': torch.cat(pocket_batch, dim=0)
    }

class PocketDataset(Dataset):
    def __init__(self, pocket_pos_list, z_pocket_list):
        self.pocket_pos_list = pocket_pos_list
        self.z_pocket_list = z_pocket_list

    def __len__(self):
        return len(self.pocket_pos_list)

    def __getitem__(self, idx):
        return self.z_pocket_list[idx], self.pocket_pos_list[idx]

def load_qm9_model_weights(model: DimeNetPretrainer, qm9_weights_dir: str):
    bb_state_dict = torch.load(f'{qm9_weights_dir}/backbone_U.pt')
    ener_state_dict = torch.load(f'{qm9_weights_dir}/readout_U.pt')
    model.backbone.load_state_dict(bb_state_dict)
    model.energy_readout.load_state_dict(ener_state_dict)

def freeze_readout_weights(model: DimeNetPretrainer):
    for param in model.energy_readout.parameters():
        param.requires_grad = False

# ==============================================================================
# 1. Main Training Loop
# ==============================================================================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- A. Load Data ---
    # Replace this with your actual Unlabeled Dataset class
    # It must return PyG Data objects with 'z' (atomic numbers) and 'pos' (coords)
    print(f"Loading Dataset: {args.dataset_path}...")
    # dataset = MyMoleculeDataset(root=args.dataset_path) 
    # For demo purposes, I'll create a fake list
    batch_size = args.batch_size
    data_dir = args.dataset_path
    mol_type = args.name
    with open(os.path.join(data_dir, f'h_{mol_type}_pos_train.pkl'), 'rb') as f:
        pocket_pos_list = pickle.load(f)
    with open(os.path.join(data_dir, f'h_z_{mol_type}_train.pkl'), 'rb') as f:
        z_pocket_list = pickle.load(f)

    with open(os.path.join(data_dir, f'{mol_type}_pos_valid.pkl'), 'rb') as f:
        valid_pocket_pos_list = pickle.load(f)
    with open(os.path.join(data_dir, f'z_{mol_type}_valid.pkl'), 'rb') as f:
        valid_z_pocket_list = pickle.load(f)
    # n_train = int(4e6)
    # valid_pocket_pos_list = pocket_pos_list[n_train:]
    # valid_z_pocket_list = z_pocket_list[n_train:]
    # pocket_pos_list = pocket_pos_list[:n_train]
    # z_pocket_list = z_pocket_list[:n_train]

    train_dataset = PocketDataset(
        pocket_pos_list, z_pocket_list
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    valid_dataset = PocketDataset(
        valid_pocket_pos_list, valid_z_pocket_list
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # --- B. Initialize Model ---
    model = DimeNetPretrainer(
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks
    )
    load_qm9_model_weights(model, args.qm9_weights_dir)
    if not args.unfreeze_readout:
        freeze_readout_weights(model)
    model.to(device)
    training_weights = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(
        training_weights, 
        lr=args.lr, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # --- C. Training Loop ---
    
    print("Starting Pre-training...")
    nan_check = 0
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        model.train()
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # --- Denoising Step ---
            loss = compute_denoising_loss(model, batch, noise_std=args.noise_std, device=device)
            progress_bar.set_postfix({"loss": loss.item(), "NaN count": nan_check})
            if loss.isnan().any() or (loss>1e3).any():
                # loss = sum([p.sum() for p in model.parameters()]) * 0.0
                nan_check += loss.isnan().sum().item()
                nan_check += (loss>1e3).sum().item()
                continue

            loss.backward()
            
            # Gradient Clipping (Important for geometric gradients)
            torch.nn.utils.clip_grad_norm_(training_weights, max_norm=10.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
        
        model.eval()
        val_loss = 0.0
        for batch in valid_loader:
            loss = compute_denoising_loss(model, batch, noise_std=args.noise_std, device=device)
            val_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        scheduler.step(avg_val_loss)
        
        # --- D. Save Checkpoint ---
        # We save ONLY the backbone weights, which we will load into the CLIP model later.
        if epoch % args.save_interval == 0:
            save_path = f"{args.save_dir}/backbone_epoch_{epoch}.pt"
            torch.save(model.backbone.state_dict(), save_path)
            save_path = f"{args.save_dir}/ener_readout_epoch_{epoch}.pt"
            torch.save(model.energy_readout.state_dict(), save_path)
            print(f"Saved pre train model to {save_path}")

def ddp_setup(rank: int, world_size: int):
    """
    Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["NCCL_SOCKET_IFNAME"]= "lo"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    # initialize the process group
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def get_dataloader(data_dir, mol_type, rank, batch_size, world_size, test_run=False):
    with open(os.path.join(data_dir, f'{mol_type}_pos.pkl'), 'rb') as f:
        pocket_pos_list = pickle.load(f)
    with open(os.path.join(data_dir, f'z_{mol_type}.pkl'), 'rb') as f:
        z_pocket_list = pickle.load(f)

    with open(os.path.join(data_dir, f'{mol_type}_pos_valid.pkl'), 'rb') as f:
        valid_pocket_pos_list = pickle.load(f)
    with open(os.path.join(data_dir, f'z_{mol_type}_valid.pkl'), 'rb') as f:
        valid_z_pocket_list = pickle.load(f)
    
    # n_train = int(4e6)
    # valid_pocket_pos_list = pocket_pos_list[n_train:]
    # valid_z_pocket_list = z_pocket_list[n_train:]
    # pocket_pos_list = pocket_pos_list[:n_train]
    # z_pocket_list = z_pocket_list[:n_train]
    
    train_dataset = PocketDataset(
        pocket_pos_list, z_pocket_list
    )
    valid_dataset = PocketDataset(
        valid_pocket_pos_list, valid_z_pocket_list, 
    )
    if rank == 0:
        print(f'Num train samples: {len(train_dataset)}')
        print(f'Num eval samples: {len(valid_dataset)}')

    if test_run:
        train_dataset = torch.utils.data.Subset(train_dataset, range(256))
        valid_dataset = torch.utils.data.Subset(valid_dataset, range(256))

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn, sampler=train_sampler
    )
    
    
    val_sampler = DistributedSampler(
        valid_dataset, num_replicas=world_size, rank=rank
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn, sampler=val_sampler
    )
    return train_loader, valid_loader

def ddp_train(rank, args):
    # This function will be called by each process in DDP
    print(f"Running DDP on rank {rank}.")
    world_size = args.world_size
    ddp_setup(rank, world_size)
    epochs = args.epochs
    if args.test_run:
        epochs = 2
    train_loader, val_loader = get_dataloader(
        args.dataset_path, args.name, rank, args.batch_size, world_size, test_run=args.test_run
    )

    model = DimeNetPretrainer(
        hidden_channels=args.hidden_channels, num_blocks=args.num_blocks
    )
    load_qm9_model_weights(model, args.qm9_weights_dir)
    if not args.unfreeze_readout:
        freeze_readout_weights(model)
    model.to(rank)
    model = DDP(
        model, device_ids=[rank], 
        # find_unused_parameters=True
    )
    training_weights = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(
        training_weights, 
        lr=args.lr, weight_decay=1e-5
    )
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=1000)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    
    for epoch in tqdm(range(epochs), desc="Overall Training Progress", disable=rank!=0):
        batch_progress_bar = tqdm(train_loader, disable=rank!=0, desc=f"Epoch {epoch+1}/{args.epochs}")
        # train_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        nan_check = 0
        for batch_data in batch_progress_bar:
            optimizer.zero_grad()
            
            # Compute denoising loss
            loss = compute_denoising_loss(model, batch_data, noise_std=args.noise_std, device=rank)
            nan_found = torch.tensor([not torch.isfinite(loss).all()], device=rank)
            large_loss = torch.tensor([(loss > 1e3).any()], device=rank)
            skip = torch.logical_or(nan_found, large_loss).to(rank)
            all_reduce(skip, op=ReduceOp.MAX)
            if skip:
                nan_check += 1
                optimizer.zero_grad() 
                continue
            
            avg_loss = loss.item()/args.batch_size
            batch_progress_bar.set_postfix_str(f"Val Loss: {avg_loss:.4e}, NaN Count: {nan_check}")
            loss.backward()
            
            # Gradient Clipping (Important for geometric gradients)
            torch.nn.utils.clip_grad_norm_(training_weights, max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()

            if warmup_scheduler.last_epoch < 1000:
                warmup_scheduler.step()

        # Average the training loss across all processes
        avg_train_loss = torch.tensor(train_loss / len(train_loader)).to(rank)
        train_nan_check = torch.tensor(nan_check).to(rank)
        all_reduce(avg_train_loss, op=ReduceOp.SUM)
        all_reduce(train_nan_check, op=ReduceOp.SUM)
        avg_train_loss /= world_size

        model.eval()
        val_loss = 0.0
        nan_check = 0
        batch_progress_bar = tqdm(val_loader, disable=rank!=0, desc=f"Validation Epoch {epoch+1}/{args.epochs}")
        for batch_data in batch_progress_bar:
            loss = compute_denoising_loss(model, batch_data, noise_std=args.noise_std, device=rank)
            nan_found = torch.tensor([not torch.isfinite(loss).all()], device=rank)
            if nan_found:
                nan_check += loss.isnan().sum().item()
                nan_check += loss.isinf().sum().item()
                continue
            avg_loss = loss.item()/args.batch_size
            batch_progress_bar.set_postfix_str(f"Val Loss: {avg_loss:.4e}, NaN Count: {nan_check}")
            val_loss += loss.item()

        avg_val_loss = torch.tensor(val_loss / len(val_loader)).to(rank)
        val_nan_check = torch.tensor(nan_check).to(rank)
        all_reduce(avg_val_loss, op=ReduceOp.SUM)
        all_reduce(val_nan_check, op=ReduceOp.SUM)
        avg_val_loss /= world_size

        if rank == 0:
            print(
                f"\nEpoch {epoch+1}"
                f"\nAvg Train Loss: {avg_train_loss.item():.6e} | Avg Val Loss: {avg_val_loss.item():.6e} | LR: {optimizer.param_groups[0]['lr']:.2e}"
                f"\nTrain NaN Count: {train_nan_check} | Val NaN Count: {val_nan_check.item()}"
            )


        scheduler.step(avg_val_loss)
        if epoch % args.save_interval == 0 and rank == 0:
            save_path = f"{args.save_dir}/backbone_epoch_{epoch}.pt"
            torch.save(model.module.backbone.state_dict(), save_path)
            save_path = f"{args.save_dir}/ener_readout_epoch_{epoch}.pt"
            torch.save(model.module.energy_readout.state_dict(), save_path)
            print(f"Saved pre train model to {save_path}")

    destroy_process_group()

# ==============================================================================
# 4. Entry Point
# ==============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # Data params
    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help=(
            'Directory containing the ligand/pocket denoising data produced by '
            'data_prep/ligand_prep.py or data_prep/pocket_prep.py '
            '(h_{name}_pos_train.pkl, h_z_{name}_train.pkl, {name}_pos_valid.pkl, '
            'z_{name}_valid.pkl)'))
    parser.add_argument('--name', type=str, default='ligand', help='ligand or pocket')
    parser.add_argument(
        '--qm9_weights_dir', type=str, required=True,
        help='Directory with QM9-pretrained backbone_U.pt / readout_U.pt (see weights/qm9_pretrained)')
    
    # Model params
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=6)
    
    # Training params
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--noise_std', type=float, default=0.1, help='Standard deviation of noise to inject')
    # parser.add_argument('--save_dir', type=str, default='./denoise_models', help='Directory to save backbone checkpoints')
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs for DDP')
    parser.add_argument('--test_run', action='store_true', help='If set, runs a quick test with a subset of data')
    parser.add_argument(
        '--unfreeze_readout', action='store_true', 
        help='If set, unfreeze the qm9 pretrained model weights of the energy output layer',
    )
    
    args = parser.parse_args()
    save_dir = os.path.abspath(f"./{args.name}_models_4")
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir
    if args.world_size == 1:
        train(args)
    else:
        torch.multiprocessing.spawn(
            ddp_train, 
            args=(args, ), 
            nprocs=args.world_size, 
            join=True
        )