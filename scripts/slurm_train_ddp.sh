#!/bin/bash
#SBATCH --job-name=dp3_ddp
#SBATCH --output=logs/%j_ddp.out
#SBATCH --error=logs/%j_ddp.err
#SBATCH --partition=clear-l40s
#SBATCH --account=clear
#SBATCH --qos=clear-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# =============================================================================
# DP3 DDP training on SLURM (4 GPUs, 1 node)
# Submit from repo root: sbatch scripts/slurm_train_ddp.sh
# =============================================================================

# 1. Environment
source /data/scratch/mlewkowicz/miniconda3/bin/activate dynaguide_v2
export WANDB_MODE=online
export OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1

# Set if wandb needs API key
# export WANDB_API_KEY=your_key_here

# 2. Training config (edit these)
ALG_NAME=dp3
TASK_NAME=calvin
ADDITION_INFO=0322
SEED=0
EXP_NAME="${TASK_NAME}-${ALG_NAME}-${ADDITION_INFO}"
RUN_DIR="data/outputs/${EXP_NAME}_seed${SEED}"
DEBUG=false
SAVE_CKPT=true

# Override dataset path (optional; else set in config e.g. task/calvin.yaml)
# Use Hydra override: task.dataset.zarr_path=/path/to/file.zarr
# ZARR_PATH="/data/scratch/mlewkowicz/calvin_dp3.zarr"

# 3. DDP env for PyTorch (srun launches one process per task; each gets one GPU)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}

# 4. Run training: one Python process per GPU, RANK/LOCAL_RANK/WORLD_SIZE set by SLURM
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
cd 3D-Diffusion-Policy

# Build Hydra overrides (dataset path override if ZARR_PATH is set)
OVERRIDES="task=${TASK_NAME} hydra.run.dir=${RUN_DIR} training.debug=${DEBUG} training.seed=${SEED} exp_name=${EXP_NAME} logging.mode=online checkpoint.save_ckpt=${SAVE_CKPT}"
if [ -n "${ZARR_PATH:-}" ]; then
  OVERRIDES="${OVERRIDES} task.dataset.zarr_path=${ZARR_PATH}"
fi

srun bash -c 'export RANK=$SLURM_PROCID; export LOCAL_RANK=$SLURM_LOCALID; export WORLD_SIZE=$SLURM_NTASKS; exec python train.py --config-name='"${ALG_NAME}"'.yaml '"${OVERRIDES}"
