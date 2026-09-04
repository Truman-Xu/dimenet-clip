"""Extract a QM9-pretrained DimeNet backbone/readout checkpoint pair for use
as the denoising-pretraining initialization in train_denoising.py
(load_qm9_model_weights). Downloads PyG's official QM9-pretrained DimeNet
(internal energy at 0K target, 'U'), then splits its weights into the
DimeNetBackbone / DimeNetReadout modules defined in dimenet_clip.py.
Produces backbone_U.pt and readout_U.pt under --output_dir; see
weights/qm9_pretrained for the published checkpoint.
"""
import argparse
import os
import sys

import torch
from torch_geometric.datasets import QM9
from torch_geometric.nn.models.dimenet import DimeNet

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dimenet_clip import DimeNetBackbone, DimeNetReadout  # noqa: E402

# Index of the internal energy at 0K (U) target among the 12 QM9 regression
# targets kept after PyG's from_qm9_pretrained-compatible column selection
# below (matches qm9_target_dict[U_TARGET_INDEX] == 'U').
U_TARGET_INDEX = 8

DIMENET_KWARGS = dict(
    hidden_channels=128,
    num_blocks=6,
    num_bilinear=8,
    num_spherical=7,
    num_radial=6,
    cutoff=5.0,
)


def main(qm9_root, output_dir):
    dataset = QM9(qm9_root)
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
    dataset.data.y = dataset.data.y[:, idx]

    stock_model, _ = DimeNet.from_qm9_pretrained(qm9_root, dataset, U_TARGET_INDEX)

    backbone = DimeNetBackbone(**DIMENET_KWARGS)
    readout = DimeNetReadout(
        hidden_channels=DIMENET_KWARGS['hidden_channels'],
        out_channels=1,
        num_blocks=DIMENET_KWARGS['num_blocks'],
        num_radial=DIMENET_KWARGS['num_radial'],
    )

    for module_name in ('rbf', 'sbf', 'emb', 'interaction_blocks'):
        getattr(backbone, module_name).load_state_dict(getattr(stock_model, module_name).state_dict())
    readout.output_blocks.load_state_dict(stock_model.output_blocks.state_dict())

    os.makedirs(output_dir, exist_ok=True)
    torch.save(backbone.state_dict(), os.path.join(output_dir, 'backbone_U.pt'))
    torch.save(readout.state_dict(), os.path.join(output_dir, 'readout_U.pt'))
    print(f"Saved backbone_U.pt and readout_U.pt to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract a QM9-pretrained DimeNet backbone/readout checkpoint')
    parser.add_argument(
        '--qm9_root', type=str, required=True,
        help='Directory to download/cache the QM9 dataset (via torch_geometric.datasets.QM9)')
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to write backbone_U.pt / readout_U.pt to')
    args = parser.parse_args()
    main(args.qm9_root, args.output_dir)
