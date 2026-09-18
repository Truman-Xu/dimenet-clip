#!/bin/bash
# Example SLURM job for Stage 5: CLIP-style contrastive fine-tuning on
# PDBBind, starting from the SAIR-pretrained checkpoint (Stage 4).
# batch_size=16/GPU as in the original run scripts, world_size=8,
# affinity_cutoff=0, ligand backbone frozen (as in the released model), up to
# 30 epochs with the validation-loss minimum selected.
# Edit the #SBATCH directives, CONDA_ENV, and --data_path below for your
# cluster before submitting.
#SBATCH --job-name=pdbbind-clip-dimenet
#SBATCH --partition ada6000
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --time=96:00:00
#SBATCH --mem=100G
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
export PYTORCH_ALLOC_CONF=expandable_segments:True

"$CONDA_ENV/bin/python" ../train_clip_dimenet_pdbbind.py \
    --data_path /path/to/pdbbind_preprocessed.pkl \
    --model_save_path "clip-pdbbind-finetuned/" \
    --batch_size 16 \
    --n_epochs 30 \
    --world_size 8 \
    --hidden_channels 128 \
    --out_channels 128 \
    --num_blocks 6 \
    --affinity_cutoff 0 \
    --load_weight_path ../weights/clip_sair_pretrained/dimenet_clip_epoch_5.pth \
    --freeze_ligand \
    > train_clip_$SLURM_JOBID.log 2>&1
