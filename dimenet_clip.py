import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Sequential, Parameter
from torch_geometric.nn import GlobalAttention
from torch_geometric.utils import scatter, softmax
from torch_geometric.nn.resolver import activation_resolver
from typing import List, Optional, Tuple, Union, Callable

from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer, 
    SphericalBasisLayer, 
    EmbeddingBlock, 
    InteractionBlock, 
    OutputBlock, 
    radius_graph, 
    triplets
)

# ==============================================================================
# 1. DimeNetBackbone (Modular Feature Extractor)
# ==============================================================================
class DimeNetBackbone(torch.nn.Module):
    r"""
    The geometric "Brain" of the model. 
    Extracts rich atom-level features but performs NO aggregation.
    Useful for Unsupervised Pre-training (Denoising).
    """
    def __init__(
        self,
        hidden_channels: int,
        num_blocks: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        act: Union[str, Callable] = 'swish',
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        
        act = activation_resolver(act)

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff,
                                       envelope_exponent)

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionBlock(
                hidden_channels,
                num_bilinear,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                act,
            ) for _ in range(num_blocks)
        ])

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[List[Tensor], Tensor, Tensor, int, Optional[Tensor]]:
        
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances and angles
        dist_sq = (pos[i] - pos[j]).pow(2).sum(dim=-1)
        # Clamp distances to avoid numerical instability 
        # (especially in denoising pre-training when atoms can be very close)
        # dist = torch.sqrt(dist_sq + 1e-4)
        dist = dist_sq.sqrt()

        pos_ji, pos_ki = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_i]
        a = (pos_ji * pos_ki).sum(dim=-1)

        # torch.cross gives a vector. We calculate its magnitude safely.
        cross_prod = torch.cross(pos_ji, pos_ki, dim=1)
        cross_prod_sq = cross_prod.pow(2).sum(dim=-1)
        # clamp to avoid sqrt of zero, which can cause NaNs 
        # in denoising pre-training when atoms are very close
        # b = torch.sqrt(cross_prod_sq + 1e-4)
        b = cross_prod_sq.sqrt()
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding Block
        x = self.emb(z, rbf, i, j)
        
        # Save intermediate states for Readout
        xs = [x]

        # Interaction Blocks
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            xs.append(x)

        return xs, rbf, i, z.size(0), batch


# ==============================================================================
# 2. DimeNetReadout (Atom-Level Attention)
# ==============================================================================
class DimeNetReadout(torch.nn.Module):
    r"""
    The "Mouth" of the model.
    Applies OutputBlocks and uses Attention Pooling (instead of Sum) 
    to focus on critical atoms (Warheads/Pharmacophores).
    """
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        num_radial: int,
        num_output_layers: int = 3,
        act: Union[str, Callable] = 'swish',
    ):
        super().__init__()
        act = activation_resolver(act)
        
        self.output_blocks = torch.nn.ModuleList([
            OutputBlock(
                num_radial,
                hidden_channels,
                out_channels,
                num_output_layers,
                act,
                'glorot_orthogonal',
            ) for _ in range(num_blocks + 1)
        ])
        
        # Attention Mechanism: Gate -> Score
        # We process the final summed features to decide importance
        # self.attn_gate = Linear(out_channels, 1)

    def forward(
        self,
        xs: List[Tensor],
        rbf: Tensor,
        i: Tensor,
        num_nodes: int,
        batch: Optional[Tensor] = None
    ) -> Tensor:
        
        # 1. Compute Atom Features from all blocks
        x_init = xs[0]
        P = self.output_blocks[0](x_init, rbf, i, num_nodes=num_nodes)
        for x, output_block in zip(xs[1:], self.output_blocks[1:]):
            P = P + output_block(x, rbf, i, num_nodes=num_nodes)
            
        # P is now [num_atoms, out_channels]
        
        # 2. Attention Pooling
        # Calculate attention scores for each atom
        # alpha = self.attn_gate(P).tanh() # [num_atoms, 1]
        
        
        # Softmax over the batch (atoms in the same molecule)
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=P.device)
            
        # Global Attention automatically handles the softmax per graph in the batch
        # But GlobalAttention usually expects a separate NN for gating. 
        # Here we implement it manually for clarity or use PyG's utility.
        
        # Standard PyG Global Attention Pattern:
        # out = global_attention(P, batch, gate_nn=self.attn_gate)
        # However, to keep it consistent with DimeNet internals, let's look at P:
        
        # Simple Global Attention:
        # attn_weights = softmax(alpha, batch)
        # attn_weights = self.attn_gate(x_init).sigmoid() # [num_atoms, 1]
        # print(P.shape, attn_weights.shape)
        # out = scatter(P * attn_weights, batch, dim=0, reduce='sum')
        out = scatter(P, batch, dim=0, reduce='sum')
        
        return out # [batch_size, out_channels]


# ==============================================================================
# 3. Multi-Instance Aggregation (Conformer-Level Attention)
# ==============================================================================
class ConformerAggregator(torch.nn.Module):
    r"""
    Aggregates multiple conformers of the same molecule into a single representation.
    Learns to pick the 'Bioactive' conformer.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate_nn = Sequential(
            Linear(channels, channels // 2),
            torch.nn.Tanh(),
            Linear(channels // 2, 1)
        )

    def forward(self, x: Tensor, batch_indices: Tensor) -> Tensor:
        r"""
        x: [Total_Conformers, Channels]
        batch_indices: [Total_Conformers] -> Indicates which molecule this conformer belongs to
                       (e.g., [0, 0, 0, 1, 1, 1, 2, ...])
        """
        # Calculate attention score for each conformer
        alpha = self.gate_nn(x)
        
        # Softmax over the molecule instances
        # (e.g., normalize scores among the 10 conformers of Molecule A)
        alpha = softmax(alpha, batch_indices)
        
        # Weighted Sum
        out = scatter(x * alpha, batch_indices, dim=0, reduce='sum')
        return out # [Batch_Size (Molecules), Channels]


# ==============================================================================
# 4. DimeNetCLIP (Updated with Optional Conformer Aggregation)
# ==============================================================================
class DimeNetCLIP(torch.nn.Module):
    def __init__(
        self, 
        hidden_channels: int = 128, 
        out_channels: int = 128, 
        num_blocks: int = 6, 
        act: str = 'swish',
        use_conformers: bool = True  # <--- NEW TOGGLE
    ):
        super().__init__()
        self.use_conformers = use_conformers
        
        # --- LIGAND BRANCH ---
        self.backbone_ligand = DimeNetBackbone(hidden_channels, num_blocks=num_blocks, 
                                               num_bilinear=8, num_spherical=7, num_radial=6, act=act)
        
        # Atom Attention Readout
        self.readout_ligand = DimeNetReadout(hidden_channels, hidden_channels, 
                                               num_blocks=num_blocks, num_radial=6, act=act)
        
        # Conformer Attention Aggregator (Only init if used)
        if self.use_conformers:
            self.conformer_agg = ConformerAggregator(hidden_channels)
        else:
            self.conformer_agg = None
            
        # Projection to Latent Space
        self.proj_ligand = Linear(hidden_channels, out_channels)


        # --- POCKET BRANCH ---
        self.backbone_pocket = DimeNetBackbone(hidden_channels, num_blocks=num_blocks, 
                                               num_bilinear=8, num_spherical=7, num_radial=6, act=act)
        self.readout_pocket = DimeNetReadout(hidden_channels, hidden_channels, 
                                               num_blocks=num_blocks, num_radial=6, act=act)
        
        self.proj_pocket = Linear(hidden_channels, out_channels)
        
        # Learnable Temperature (Start at 0.07)
        self.logit_scale = Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_ligand(self, z, pos, batch_atoms, batch_conformers=None):
        """
        z: Atom types
        pos: Coordinates
        batch_atoms: Maps atoms -> graph_id (standard PyG batch)
        batch_conformers: Maps graph_id -> molecule_id (only needed if use_conformers=True)
        """
        # 1. Backbone & Atom Readout (Get graph-level vectors)
        xs, rbf, i, n, b = self.backbone_ligand(z, pos, batch_atoms)
        vec_graphs = self.readout_ligand(xs, rbf, i, n, b) # [Batch_Size_Graphs, Hidden]
        
        # 2. Conformer Aggregation (Optional)
        if self.use_conformers:
            if batch_conformers is None:
                raise ValueError("Model initialized with use_conformers=True, but batch_conformers was None during forward pass.")
            
            # Aggregate multiple graphs (conformers) into single molecule vectors
            vec_molecules = self.conformer_agg(vec_graphs, batch_conformers)
        else:
            # 1-to-1 mapping: Each graph is treated as a distinct molecule
            vec_molecules = vec_graphs 
            
        # 3. Project and Normalize
        return F.normalize(self.proj_ligand(vec_molecules), dim=-1)

    def encode_pocket(self, z, pos, batch):
        xs, rbf, i, n, b = self.backbone_pocket(z, pos, batch)
        vec = self.readout_pocket(xs, rbf, i, n, b)
        return F.normalize(self.proj_pocket(vec), dim=-1)

    def forward(self, data_ligand, data_pocket):
        """
        Expects PyG data objects. 
        """
        # Check if we have conformer batch indices
        batch_conformers = getattr(data_ligand, 'batch_conformers', None)
        
        v_l = self.encode_ligand(
            data_ligand.z, 
            data_ligand.pos, 
            data_ligand.batch, 
            batch_conformers
        )
        
        v_p = self.encode_pocket(
            data_pocket.z, 
            data_pocket.pos, 
            data_pocket.batch
        )
        
        return v_l, v_p, self.logit_scale.exp()


# ==============================================================================
# 5. Sigmoid-Weighted CLIP Loss (The Training Logic)
# ==============================================================================
class SigmoidWeightedCLIPLoss(torch.nn.Module):
    def __init__(self, cutoff=6.0, steepness=2.0):
        super().__init__()
        self.cutoff = cutoff
        self.steepness = steepness

    def forward(self, v_ligand, v_pocket, logit_scale, affinities):
        """
        v_ligand, v_pocket: Normalized vectors [Batch, Dim]
        affinities: Real-valued labels (pIC50) [Batch]
        """
        # 1. Compute Soft Weights based on Affinity
        # Weights -> 1.0 for high affinity, -> 0.0 for low affinity
        weights = torch.sigmoid(self.steepness * (affinities - self.cutoff))

        # 2. Compute Similarity Matrix
        logits = torch.matmul(v_ligand, v_pocket.T) * logit_scale

        # 3. Compute Cross Entropy
        # We manually compute it to inject the weights for the positive pairs
        log_probs_row = F.log_softmax(logits, dim=1) # Ligand -> Pocket
        log_probs_col = F.log_softmax(logits.T, dim=1) # Pocket -> Ligand

        # Extract diagonals (Correct pairs)
        log_probs_row_pos = torch.diag(log_probs_row)
        log_probs_col_pos = torch.diag(log_probs_col)

        # 4. Weighted Loss
        # We treat weak binders as "Low Confidence Positives"
        loss_row = -(log_probs_row_pos * weights).sum() / (weights.sum() + 1e-6)
        loss_col = -(log_probs_col_pos * weights).sum() / (weights.sum() + 1e-6)

        return (loss_row + loss_col) / 2

# ==============================================================================
# 6. Wrapper Class for Pre-training (Denoising Task)
# ==============================================================================
class DimeNetPretrainer(torch.nn.Module):
    """
    Wraps a DimeNetBackbone with an Energy Readout head for the 
    Coordinate Denoising pre-training task.
    """
    def __init__(self, hidden_channels=128, num_blocks=6, act='swish'):
        super().__init__()
        
        # The Backbone (This is what we want to save/transfer)
        self.backbone = DimeNetBackbone(
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            act=act
        )
        
        # The Pre-training Head (Predicts Scalar Energy)
        # We discard this after pre-training.
        self.energy_readout = DimeNetReadout(
            hidden_channels=hidden_channels,
            out_channels=1, # Scalar output
            num_blocks=num_blocks,
            num_radial=6,
            num_output_layers=3,
            act=act
        )

        # # Initialize the final linear layer of each OutputBlock to a small gain
        # for block in self.energy_readout.output_blocks:
        #     # PyG's OutputBlock stores its final linear layer in 'block.lin'
        #     if hasattr(block, 'lin'):
        #         torch.nn.init.uniform_(block.lin.weight, -0.01, 0.01)
        #         if block.lin.bias is not None:
        #             torch.nn.init.zeros_(block.lin.bias)

    def forward(self, z, pos, batch):
        # 1. Backbone features
        xs, rbf, i, n, b = self.backbone(z, pos, batch)
        
        # 2. Energy prediction (Sum over all atoms to get system energy)
        # Note: We use the Readout to map features -> atomic energies, then sum.
        atomic_energies = self.energy_readout(xs, rbf, i, n, b)
        system_energy = atomic_energies.sum()
        
        return system_energy

# ==============================================================================
# 7. The Denoising Loss Function
# ==============================================================================
def compute_denoising_loss(model, data, noise_std=0.1, perturb_frac = 0.1, device='cuda'):
    """
    1. Add noise to coordinates.
    2. Predict Energy of noisy state.
    3. Calculate Force (-Gradient of Energy).
    4. Minimize MSE(Force, Restoring_Vector).
    """
    z, pos, batch = data['z'].to(device), data['pos'].to(device), data['batch'].to(device)

    # Ensure gradients are enabled for force calculation (even in eval mode)
    with torch.enable_grad():
        # 1. Generate Noise
        # noise = torch.randn_like(pos) * noise_std
        n_atoms = pos.shape[0]
        n_perturb = int(np.ceil(n_atoms*perturb_frac))
        randind = torch.randperm(n_atoms)[:n_perturb]

        noise = torch.randn((n_perturb, 3), device=device) * noise_std
        noise_mask = torch.zeros_like(pos, device=device)
        noise_mask[randind] += noise
        pos_noisy = pos + noise_mask
        pos_noisy.requires_grad_(True) # Crucial for calculating force

        # 2. Forward Pass (Get Energy)
        energy = model(z, pos_noisy, batch)

        # 3. Calculate Gradient (Force field)
        # We only need create_graph=True if we are training (to backprop through force)
        is_training = model.training
        force = -torch.autograd.grad(
            outputs=energy,
            inputs=pos_noisy,
            create_graph=is_training, 
            retain_graph=is_training
        )[0]

    # 4. Target: The vector that points back to the clean position
    # If pos_noisy = pos + noise, then (pos - pos_noisy) = -noise
    target_force = -noise_mask  

    # Optional: You can scale the target by 1/sigma^2 (Score Matching theory), 
    # but for simple pre-training, matching the vector directly works well.

    # MSE Loss
    loss = torch.nn.functional.mse_loss(force, target_force)
    return loss