import argparse
import os
import pickle

import lmdb
import numpy as np
from crimm.Data.ptable import PERIODIC_TABLE
from numpy.random import choice
from tqdm.auto import tqdm


def process_data(data, remove_hs=False):
    atoms = data['atoms']
    coords = data['coordinates'][0]
    selected_coords = []
    atomic_numbers = []
    failed_atoms = []
    for i, atom in enumerate(atoms):
        atom = atom.capitalize()
        if remove_hs and atom.startswith('H'):
            continue
        atom_coords = coords[i]
        if atom not in PERIODIC_TABLE:
            failed_atoms.append(atom)
            continue
        atomic_number = PERIODIC_TABLE[atom]['number']
        selected_coords.append(atom_coords)
        atomic_numbers.append(atomic_number)
    if len(failed_atoms) > 0:
        return None, None, failed_atoms
    selected_coords = np.array(selected_coords)
    atomic_numbers = np.array(atomic_numbers)
    return selected_coords, atomic_numbers, None


def main(dataset_type, lmdb_dir, output_dir, n_total, n_train, remove_hs=False):
    database_path = os.path.join(lmdb_dir, f'{dataset_type}.lmdb')
    env = lmdb.open(
        database_path, readonly=True, lock=False, subdir=False,
        readahead=False, meminit=False, max_readers=256
    )
    failed = []
    ligand_pos = []
    valid_pos = []
    z_ligand = []
    z_valid = []

    with env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
        chosen_keys = choice(keys, size=n_total, replace=False)
        train_keys = chosen_keys[:n_train]
        valid_keys = chosen_keys[n_train:]

        for idx in tqdm(train_keys):
            datapoint_pickled = txn.get(idx)
            data = pickle.loads(datapoint_pickled)
            coords, atomic_numbers, failed_atoms = process_data(data)
            if failed_atoms is not None:
                failed.append((data['smi'], failed_atoms))
            else:
                ligand_pos.append(coords)
                z_ligand.append(atomic_numbers)

        for idx in tqdm(valid_keys):
            datapoint_pickled = txn.get(idx)
            data = pickle.loads(datapoint_pickled)
            coords, atomic_numbers, failed_atoms = process_data(data, remove_hs)
            if failed_atoms is not None:
                failed.append((data['smi'], failed_atoms))
            else:
                valid_pos.append(coords)
                z_valid.append(atomic_numbers)

    print(f"Failed to process {len(failed)} ligands")
    os.makedirs(output_dir, exist_ok=True)
    prefix = 'h_' if not remove_hs else ''

    with open(os.path.join(output_dir, f'{prefix}ligand_pos_{dataset_type}.pkl'), 'wb') as f:
        pickle.dump(ligand_pos, f)
    with open(os.path.join(output_dir, f'{prefix}z_ligand_{dataset_type}.pkl'), 'wb') as f:
        pickle.dump(z_ligand, f)

    print(f'{dataset_type}: {len(failed)} failed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare ligand conformer data from a Uni-Mol ligand LMDB database '
                     '(Section: Ligand Structure Data Preparation).'
    )
    parser.add_argument(
        '--lmdb_dir', type=str, required=True,
        help="Directory containing the Uni-Mol ligand LMDB files, named '<dataset_type>.lmdb'")
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to write the output pickle files to')
    parser.add_argument(
        '--dataset_type', type=str, default='train',
        help="Name of the LMDB file (without extension) to read, e.g. 'train'")
    parser.add_argument(
        '--n_total', type=int, default=int(5e6),
        help='Total number of ligands to sample without replacement')
    parser.add_argument(
        '--n_train', type=int, default=int(4.5e6),
        help='Number of sampled ligands assigned to the training split '
             '(the remainder becomes the validation split)')
    parser.add_argument('--remove_hs', action='store_true', help='Strip hydrogens from validation split')
    args = parser.parse_args()

    main(
        args.dataset_type, args.lmdb_dir, args.output_dir,
        args.n_total, args.n_train, remove_hs=args.remove_hs,
    )
