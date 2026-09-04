#!/bin/bash
# Example SLURM job for Stage 4: CLIP-style contrastive pre-training on SAIR.
# Matches the manuscript's reported configuration: batch_size=4 unique
# targets/GPU, world_size=8, affinity_cutoff=-1.0, ligand branch frozen.
# Edit the #SBATCH directives, CONDA_ENV, and --data_path below for your
# cluster before submitting.
#SBATCH --job-name=clip-dimenet-sair
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

"$CONDA_ENV/bin/python" ../train_clip_dimenet_sair.py \
    --data_path /path/to/sair_preprocessed.pkl \
    --model_save_path "clip-sair-pretrained/" \
    --batch_size 4 \
    --n_epochs 100 \
    --world_size 8 \
    --hidden_channels 128 \
    --out_channels 128 \
    --num_blocks 6 \
    --affinity_cutoff '-1.0' \
    --max_pocket_atoms 300 \
    --max_lig_atoms 100 \
    --freeze_ligand \
    --pocket_backbone_path ../weights/denoising_pocket/backbone_epoch_26.pt \
    --ligand_backbone_path ../weights/denoising_ligand/backbone_epoch_29.pt \
    > train_clip_$SLURM_JOBID.log 2>&1
    # To warm-start from an existing full CLIP checkpoint (e.g. one pretrained
    # on ProfSA, as in the manuscript) instead, replace the two
    # --*_backbone_path flags above with:
    #   --load_weight_path /path/to/profsa_pretrained_checkpoint.pth
