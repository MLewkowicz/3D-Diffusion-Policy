# CALVIN Dataset Setup for DP3

This document describes how to configure DP3 to run on the CALVIN dataset.

## Overview

The CALVIN dataset has been integrated into DP3 with the following components:

1. **Dataset Converter**: `scripts/convert_calvin_to_dp3.py` - Converts CALVIN dataset to DP3 Zarr format
2. **Dataset Loader**: `diffusion_policy_3d/dataset/calvin_dataset.py` - Loads CALVIN data for training
3. **Environment Wrapper**: `diffusion_policy_3d/env/calvin/calvin_wrapper.py` - Gym-compatible CALVIN environment (optional, for evaluation)
4. **Environment Runner**: `diffusion_policy_3d/env_runner/calvin_runner.py` - Runs evaluation on CALVIN (optional)
5. **Config File**: `diffusion_policy_3d/config/task/calvin.yaml` - Task configuration

## Step 1: Convert CALVIN Dataset to DP3 Format

First, convert your CALVIN dataset to the DP3 Zarr format:

```bash
python scripts/convert_calvin_to_dp3.py \
    --root_dir /home/mlewkowicz/calvin/dataset/calvin_debug_dataset/ \
    --save_path data/calvin_dp3.zarr \
    --overwrite
```

**Parameters:**
- `--root_dir`: Path to CALVIN dataset root (e.g., `./calvin/dataset/task_ABC_D`)
- `--save_path`: Output path for the Zarr file (e.g., `data/calvin_dp3.zarr`)
- `--split`: Dataset split to process (`training` or `validation`)
- `--tasks`: (Optional) List of specific tasks to filter
- `--no_cuda`: (Optional) Disable CUDA for FPS sampling
- `--overwrite`: Overwrite existing output file

**What the converter does:**
- Crops images: 200×200 → 160×160 (static), 84×84 → 68×68 (gripper)
- Adjusts camera intrinsics for crop offset
- Deprojects depth maps to 3D point clouds
- Samples 1024 points from each view using FPS
- Concatenates to 2048 points total (2048, 6) where 6 = [X, Y, Z, R, G, B]
- Saves to Zarr format with: `img`, `point_cloud`, `depth`, `action`, `state`, `episode_ends`

## Step 2: Update Config File

Edit `diffusion_policy_3d/config/task/calvin.yaml` to point to your converted dataset:

```yaml
dataset:
  _target_: diffusion_policy_3d.dataset.calvin_dataset.CalvinDataset
  zarr_path: data/calvin_dp3.zarr  # Update this path
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  seed: 42
  val_ratio: 0.02
  max_train_episodes: null  # Set to limit episodes, or null for all
```

## Step 3: Train DP3 on CALVIN

Train the policy using the standard DP3 training script:

```bash
bash scripts/train_policy.sh dp3 calvin 0322 0 0
```

**Parameters:**
- `dp3`: Algorithm name (config file: `dp3.yaml`)
- `calvin`: Task name (uses `calvin.yaml` config)
- `0322`: Additional info for experiment name
- `0`: Random seed
- `0`: GPU ID

Or directly:

```bash
cd 3D-Diffusion-Policy
export CUDA_VISIBLE_DEVICES=0
python train.py --config-name=dp3.yaml \
                task=calvin \
                hydra.run.dir=data/outputs/calvin-dp3-0322_seed0 \
                training.seed=0 \
                training.device="cuda:0"
```

## Data Format

The converted CALVIN dataset has the following structure:

- **Point Cloud**: (N, 2048, 6) - XYZRGB point clouds
  - First 3 channels: XYZ coordinates
  - Last 3 channels: RGB values (normalized to [0, 1])
  
- **Images**: (N, 160, 160, 3) - Cropped static RGB images (uint8)
- **Depth**: (N, 160, 160) - Cropped static depth maps (float64)
- **Actions**: (N, 7) - Relative actions from CALVIN (float32)
  - Format: [x, y, z, euler_x, euler_y, euler_z, gripper]
- **State**: (N, 15) - Robot observations from CALVIN (float32)
  - Format: [tcp_pos(3), tcp_orient(3), gripper_width(1), arm_joints(7), gripper_action(1)]
- **Episode Ends**: Array of indices marking episode boundaries

## Evaluation (Optional)

If you have access to the CALVIN simulation environment, you can enable evaluation by:

1. Uncommenting the `env_runner` section in `calvin.yaml`
2. Setting `dataset_path` to your CALVIN dataset path
3. The runner will use the CALVIN simulation for evaluation

**Note**: Since CALVIN is primarily a real robot dataset, evaluation typically requires:
- Access to a real CALVIN robot, OR
- The CALVIN simulation environment properly configured

## Troubleshooting

### Import Errors
If you get import errors for `calvin_env`:
- Install the CALVIN environment package
- Or set `env_runner: null` in the config to skip evaluation

### CUDA Out of Memory
- Use `--no_cuda` flag in the converter to disable CUDA for FPS
- Reduce `max_train_episodes` in the config

### Dataset Path Issues
- Ensure the Zarr file path in `calvin.yaml` is correct
- Check that the converted Zarr file exists and is readable

## Files Created

- `scripts/convert_calvin_to_dp3.py` - Dataset converter
- `diffusion_policy_3d/dataset/calvin_dataset.py` - Dataset loader
- `diffusion_policy_3d/env/calvin/calvin_wrapper.py` - Environment wrapper
- `diffusion_policy_3d/env/calvin/__init__.py` - Package init
- `diffusion_policy_3d/env_runner/calvin_runner.py` - Environment runner
- `diffusion_policy_3d/config/task/calvin.yaml` - Task configuration

## References

- CALVIN Dataset: https://github.com/mees/calvin
- DP3 Paper: 3D-Diffusion-Policy
- Diffuser Actor Paper: Describes the cropping and sampling strategy used
