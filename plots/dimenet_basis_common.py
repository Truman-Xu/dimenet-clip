"""
"Raw" (untrained) DimeNet radial+angular basis featurization for the conformer-variance
experiment. Uses the exact BesselBasisLayer/SphericalBasisLayer classes and the same
per-atom angle convention as the production DimeNetCLIP model
(dimenet_clip.py at the repository root, DimeNetBackbone.forward: angle is
computed at the receiving atom idx_i, via vectors to its edge neighbors idx_j and idx_k)
but stops short of the learned Embedding/Interaction blocks, so the resulting features
are a fixed geometric descriptor, directly comparable to AEV.

Caveat: unlike AEV, these RBF/SBF basis functions carry no atomic species information --
they are a function of pairwise distances and triplet angles only. Species-awareness in
the real DimeNetCLIP model comes entirely from the learned Embedding Block (a per-species
embedding table), which is deliberately excluded here to keep this a fixed-descriptor
comparison against AEV (which is also fully non-learned).
"""
import torch
from torch_geometric.nn.models.dimenet import BesselBasisLayer, SphericalBasisLayer, radius_graph
from torch_geometric.utils import scatter

# Matches DimeNetCLIP's actual production hyperparameters (dimenet_clip.py).
NUM_RADIAL = 6
NUM_SPHERICAL = 7
CUTOFF = 5.0
ENVELOPE_EXPONENT = 5
MAX_NUM_NEIGHBORS = 32

FEATURE_LENGTH = NUM_RADIAL + NUM_SPHERICAL * NUM_RADIAL  # 6 + 42 = 48


def _triplets(edge_index, num_nodes):
    """Pure-PyTorch equivalent of torch_geometric.nn.models.dimenet.triplets(), which
    otherwise requires the optional torch-sparse extension (not installed here). For
    every edge (j -> i), finds every edge (k -> j) with k != i, giving the k -> j -> i
    triplets DimeNet's angular term needs. Small-molecule graphs (tens of atoms, a few
    hundred edges) make the Python-level grouping below fast enough; semantics match
    the original function exactly (verified against its SparseTensor-based logic)."""
    row, col = edge_index  # edge e: row[e] -> col[e], i.e. j -> i
    device = edge_index.device

    incoming = [[] for _ in range(num_nodes)]  # incoming[n] = [(edge_idx, source), ...]
    for e, (j, i) in enumerate(zip(row.tolist(), col.tolist())):
        incoming[i].append((e, j))

    idx_i, idx_j, idx_k, idx_kj, idx_ji = [], [], [], [], []
    for e, (j, i) in enumerate(zip(row.tolist(), col.tolist())):
        for e2, k in incoming[j]:
            if k == i:
                continue
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)
            idx_kj.append(e2)
            idx_ji.append(e)

    to_long = lambda vals: torch.tensor(vals, dtype=torch.long, device=device)
    return col, row, to_long(idx_i), to_long(idx_j), to_long(idx_k), to_long(idx_kj), to_long(idx_ji)


def make_basis_layers():
    """Fresh, default-initialized basis layers -- no trained checkpoint is loaded, so
    these remain a fixed geometric featurization (BesselBasisLayer.freq is technically a
    torch.nn.Parameter, but at default init it's exactly n*pi, reproducing the canonical
    fixed Bessel radial basis)."""
    rbf = BesselBasisLayer(NUM_RADIAL, CUTOFF, ENVELOPE_EXPONENT)
    sbf = SphericalBasisLayer(NUM_SPHERICAL, NUM_RADIAL, CUTOFF, ENVELOPE_EXPONENT)
    rbf.eval()
    sbf.eval()
    return rbf, sbf


def _single_conformer_atom_features(rbf_layer, sbf_layer, pos, n_atoms):
    edge_index = radius_graph(pos, r=CUTOFF, max_num_neighbors=MAX_NUM_NEIGHBORS)
    if edge_index.shape[1] == 0:
        return torch.zeros(n_atoms, FEATURE_LENGTH)

    i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = _triplets(edge_index, num_nodes=n_atoms)

    dist = (pos[i] - pos[j]).norm(dim=-1)
    rbf = rbf_layer(dist)                                             # [num_edges, NUM_RADIAL]
    radial_atom = scatter(rbf, i, dim=0, dim_size=n_atoms, reduce="mean")

    if idx_i.numel() == 0:
        angular_atom = torch.zeros(n_atoms, NUM_SPHERICAL * NUM_RADIAL)
    else:
        pos_ji = pos[idx_j] - pos[idx_i]
        pos_ki = pos[idx_k] - pos[idx_i]
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)
        sbf = sbf_layer(dist, angle, idx_kj)                          # [num_triplets, NUM_SPHERICAL*NUM_RADIAL]
        angular_atom = scatter(sbf, idx_i, dim=0, dim_size=n_atoms, reduce="mean")

    return torch.cat([radial_atom, angular_atom], dim=-1)             # [n_atoms, FEATURE_LENGTH]


def compute_dimenet_basis_molecular_features(rbf_layer, sbf_layer, coords, heavy_mask):
    """coords: np.ndarray [n_conf, n_atoms, 3]. Returns np.ndarray [n_conf, FEATURE_LENGTH]
    -- the mean basis feature over heavy atoms only, one row per conformer."""
    n_conf, n_atoms, _ = coords.shape
    heavy = torch.as_tensor(heavy_mask, dtype=torch.bool)
    rows = []
    with torch.no_grad():
        for c in range(n_conf):
            pos = torch.as_tensor(coords[c], dtype=torch.float32)
            atom_feats = _single_conformer_atom_features(rbf_layer, sbf_layer, pos, n_atoms)
            rows.append(atom_feats[heavy].mean(dim=0))
    return torch.stack(rows).numpy()
