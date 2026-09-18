#!/bin/bash
# Example SLURM job for evaluating a DimeNetCLIP checkpoint on LIT-PCBA.
# Edit the #SBATCH directives, CONDA_ENV, and --data_path below for your
# cluster before submitting.
#SBATCH --job-name=lit_pcba_eval
#SBATCH --partition gpu2080
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=80G
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

"$CONDA_ENV/bin/python" ../eval/lit_pcba_eval.py \
    --model_dir ../weights/clip_pdbbind_finetuned \
    --epoch 3 \
    --data_path /path/to/pcba_sep_pocket_vecs.pkl
