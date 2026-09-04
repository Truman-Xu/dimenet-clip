import argparse
import os
import pickle

import lmdb
import numpy as np
from crimm.Data.ptable import PERIODIC_TABLE
from tqdm.auto import tqdm


def load_atom_names(atom_name_table_path):
    atom_names = set()
    with open(atom_name_table_path, 'r') as f:
        for l in f.read().splitlines():
            if l.startswith('#'):
                continue
            l = l.split('\t')
            if l == ['']:
                continue
            atom_name = l[2]
            atom_names.add(atom_name)
    atom_names.add('HN')
    atom_names.add('OXT')
    return atom_names


def process_data(data, atom_names, remove_hs=False):
    atoms = data['atoms']
    coords = data['coordinates'][0]
    selected_coords = []
    atomic_numbers = []
    failed_atoms = []
    for i, atom in enumerate(atoms):
        if remove_hs and atom.startswith('H'):
            continue
        atom_coords = coords[i]
        if atom not in atom_names:
            failed_atoms.append(atom)
            continue
        # The first character represents the element
        element = atom[0]
        atomic_number = PERIODIC_TABLE[element]['number']
        selected_coords.append(atom_coords)
        atomic_numbers.append(atomic_number)
    if len(failed_atoms) > 0:
        return None, None, failed_atoms
    selected_coords = np.array(selected_coords)
    atomic_numbers = np.array(atomic_numbers)
    return selected_coords, atomic_numbers, None


def main(dataset_type, lmdb_dir, output_dir, atom_names, remove_hs=False):
    if dataset_type not in ('train', 'valid'):
        raise ValueError(f'Invalid dataset name {dataset_type}')
    database_path = os.path.join(lmdb_dir, f'{dataset_type}.lmdb')
    env = lmdb.open(
        database_path, readonly=True, lock=False, subdir=False,
        readahead=False, meminit=False, max_readers=256
    )
    failed = []
    pocket_pos = []
    z_pocket = []
    with env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
        for idx in tqdm(keys):
            datapoint_pickled = txn.get(idx)
            data = pickle.loads(datapoint_pickled)
            coords, atomic_numbers, failed_atoms = process_data(data, atom_names, remove_hs)
            if failed_atoms is not None:
                failed.append((data['pdbid'], failed_atoms))
            else:
                pocket_pos.append(coords)
                z_pocket.append(atomic_numbers)
    print(f"Failed to process {len(failed)} pockets")
    os.makedirs(output_dir, exist_ok=True)
    prefix = 'h_' if not remove_hs else ''

    with open(os.path.join(output_dir, f'{prefix}pocket_pos_{dataset_type}.pkl'), 'wb') as f:
        pickle.dump(pocket_pos, f)
    with open(os.path.join(output_dir, f'{prefix}z_pocket_{dataset_type}.pkl'), 'wb') as f:
        pickle.dump(z_pocket, f)

    for pdbid, failed_atoms in failed:
        print(f"{pdbid}: {failed_atoms}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare protein pocket structure data from Uni-Mol pocket LMDB databases '
                     '(Section: Pocket Structure Data Preparation).'
    )
    parser.add_argument(
        '--lmdb_dir', type=str, required=True,
        help="Directory containing the Uni-Mol pocket LMDB files, named 'train.lmdb' and 'valid.lmdb'")
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to write the output pickle files to')
    parser.add_argument(
        '--atom_name_table', type=str,
        default=os.path.join(os.path.dirname(__file__), 'atom_name_table.txt'),
        help='Path to the PDB atom-nomenclature reference table')
    parser.add_argument('--remove_hs', action='store_true', help='Strip hydrogens from both splits')
    args = parser.parse_args()

    atom_names = load_atom_names(args.atom_name_table)
    for data_type in ('train', 'valid'):
        main(data_type, args.lmdb_dir, args.output_dir, atom_names, remove_hs=args.remove_hs)
