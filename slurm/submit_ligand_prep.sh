#!/bin/bash
# Example SLURM job for Stage 1: ligand structure data preparation.
# Edit the #SBATCH directives, CONDA_ENV, and the --lmdb_dir/--output_dir
# arguments below for your cluster before submitting.
#SBATCH --job-name=ligand-prep
#SBATCH --partition cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=50G
#SBATCH --output=./slurm_out/%x-%j.out
#SBATCH --error=./slurm_out/%x-%j.err

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "working directory="$SLURM_SUBMIT_DIR

CONDA_ENV=/path/to/conda/envs/dimenet-clip
module load anaconda
conda activate "$CONDA_ENV"
cd $SLURM_SUBMIT_DIR
which python

"$CONDA_ENV/bin/python" ../data_prep/ligand_prep.py \
    --lmdb_dir /path/to/unimol/ligands \
    --output_dir /path/to/output/ligand_data \
    --dataset_type train \
    > ligand_prep_$SLURM_JOBID.log 2>&1
