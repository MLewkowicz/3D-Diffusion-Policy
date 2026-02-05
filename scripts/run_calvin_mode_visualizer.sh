#!/bin/bash
#SBATCH --job-name=calvin_viz
#SBATCH --output=logs/calvin_mode_visualizer_%j.out
#SBATCH --error=logs/calvin_mode_visualizer_%j.err
#SBATCH --partition=clear-l40s
#SBATCH --account=clear
#SBATCH --qos=clear-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source /data/scratch/aryannav/.miniforge3/bin/activate base

cd /home/aryannav/mit/research/diffusion_guidance/3D-Diffusion-Policy
mkdir -p logs

python -m utils.calvin_mode_visualizer /data/scratch/mlewkowicz/calvin/dataset/task_D_D --split training -o out
