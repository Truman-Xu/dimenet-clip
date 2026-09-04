#!/bin/bash
# Example SLURM job for Stage 3: self-supervised denoising pre-training
# (pocket domain, matching the manuscript's reported configuration:
# lr=1e-6, batch_size=8/GPU, unfrozen readout, world_size=8).
# Edit the #SBATCH directives, CONDA_ENV, and the --dataset_path/
# --qm9_weights_dir arguments below for your cluster before submitting.
#SBATCH --job-name=pocket_denoising
#SBATCH --partition ada5000
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --gres=gpu:8
#SBATCH --time=96:00:00
#SBATCH --mem=300G
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

"$CONDA_ENV/bin/python" ../train_denoising.py \
    --name "pocket" \
    --dataset_path /path/to/output/pocket_data \
    --qm9_weights_dir ../weights/qm9_pretrained \
    --world_size 8 --lr "1e-6" --batch_size 8 --unfreeze_readout \
    > denoising_train_$SLURM_JOBID.log 2>&1
