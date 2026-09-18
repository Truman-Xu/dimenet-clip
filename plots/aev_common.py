"""
TorchANI AEVComputer setup and molecular AEV featurization for the conformer-variance
experiment. Uses AEVComputer.cover_linearly with the documented ANI-1x basis-function
hyperparameters, but a custom, broader species list (H, C, N, O, F, P, S, Cl, Br, I)
sized to actually cover drug-like DUD-E ligands -- the literal 4-species ANI-1x model
(H, C, N, O only) would drop the large majority of DUD-E molecules, which routinely
contain halogens and sulfur (e.g. sulfonamides, trifluoromethyl groups).
"""
import torch
from torchani import AEVComputer
from torchani.utils import ChemicalSymbolsToInts

DEFAULT_SPECIES_ORDER = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

# ANI-1x basis-function hyperparameters, per AEVComputer.cover_linearly's own
# docstring: "(5.2, 3.5, 16.0, 8.0, 16, 4, 32.0, 8, 4)" reproduces the ANI-1x AEV.
# Only the trailing num_species argument is swapped out below, for DEFAULT_SPECIES_ORDER.
_RADIAL_CUTOFF = 5.2
_ANGULAR_CUTOFF = 3.5
_RADIAL_ETA = 16.0
_ANGULAR_ETA = 8.0
_RADIAL_DIST_DIVISIONS = 16
_ANGULAR_DIST_DIVISIONS = 4
_ZETA = 32.0
_ANGLE_SECTIONS = 8


def make_species_converter(species_order=DEFAULT_SPECIES_ORDER):
    return ChemicalSymbolsToInts(species_order)


def make_aev_computer(species_order=DEFAULT_SPECIES_ORDER):
    return AEVComputer.cover_linearly(
        _RADIAL_CUTOFF, _ANGULAR_CUTOFF, _RADIAL_ETA, _ANGULAR_ETA,
        _RADIAL_DIST_DIVISIONS, _ANGULAR_DIST_DIVISIONS, _ZETA, _ANGLE_SECTIONS,
        len(species_order),
    )


def has_unsupported_elements(symbols, species_order=DEFAULT_SPECIES_ORDER):
    allowed = set(species_order)
    return any(sym not in allowed for sym in symbols)


def compute_aev_molecular_features(aev_computer, species_converter, symbols, coords, heavy_mask):
    """coords: np.ndarray [n_conf, n_atoms, 3]. Returns np.ndarray [n_conf, aev_length]
    -- the mean AEV over heavy atoms only, one row per conformer."""
    n_conf, n_atoms, _ = coords.shape
    species_row = species_converter(symbols)                        # LongTensor [n_atoms]
    species = species_row.unsqueeze(0).expand(n_conf, n_atoms).contiguous()
    coords_t = torch.as_tensor(coords, dtype=torch.float32)

    with torch.no_grad():
        _, aevs = aev_computer((species, coords_t))                 # [n_conf, n_atoms, aev_length]

    heavy = torch.as_tensor(heavy_mask, dtype=torch.bool)
    mol_aev = aevs[:, heavy, :].mean(dim=1)                          # [n_conf, aev_length]
    return mol_aev.numpy()
