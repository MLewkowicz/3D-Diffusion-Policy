#!/bin/bash --login
#SBATCH --job-name=calvin_process
#SBATCH --partition=clear-l40s  # Matching your dyna_ddp partition
#SBATCH --account=clear         # Matching your dyna_ddp account
#SBATCH --gres=gpu:1            # 1 GPU for the FPS speedup
#SBATCH --cpus-per-task=8       # File IO
#SBATCH --mem=64G               # High RAM
#SBATCH --time=04:00:00
#SBATCH --output=logs/process_%j.out

# 1. Load Cuda (Matches your dyna script environment)
export OMP_NUM_THREADS=4

# 2. Setup Python
# Since 'calvin' is a local tarball, we point directly to it
ENV_PATH="./calvin"
PYTHON_EXEC="$ENV_PATH/bin/python"

# Sanity Check
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "ERROR: Could not find python at $PYTHON_EXEC"
    exit 1
fi

# 3. Config
export ROOT_DIR="/data/scratch/mlewkowicz/calvin/task_D_D/"
export SAVE_PATH="data/calvin_D_D.zarr"
export CAMERA_PARAMS="camera_params.json"

echo "Starting GPU-accelerated conversion..."

# 4. Run
# We do NOT use --no_cuda here because we WANT the GPU speedup
$PYTHON_EXEC scripts/convert_calvin_to_dp3.py \
    --root_dir "$ROOT_DIR" \
    --save_path "$SAVE_PATH" \
    --overwrite \
    --process_both_splits \
    --no_env \
    --camera_params "$CAMERA_PARAMS"