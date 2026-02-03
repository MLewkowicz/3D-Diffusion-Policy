# CALVIN Training Guide for DP3

This guide provides step-by-step instructions for training DP3 on the CALVIN dataset.

## Prerequisites

1. **Converted Zarr Dataset**: You should have already converted your CALVIN dataset to Zarr format using `scripts/convert_calvin_to_dp3.py`

2. **Verify Zarr File**: Make sure your Zarr file exists and contains:
   - `data/img`: (N, 160, 160, 3) - RGB images
   - `data/point_cloud`: (N, 2048, 6) - XYZRGB point clouds
   - `data/depth`: (N, 160, 160) - Depth maps
   - `data/action`: (N, 7) - Relative actions
   - `data/state`: (N, 15) - Robot observations
   - `meta/episode_ends`: Episode boundaries

## Step-by-Step Training Sequence

### Step 1: Update Config File (if needed)

Edit `3D-Diffusion-Policy/diffusion_policy_3d/config/task/calvin.yaml` to point to your Zarr file:

```yaml
dataset:
  _target_: diffusion_policy_3d.dataset.calvin_dataset.CalvinDataset
  zarr_path: data/calvin_dp3.zarr  # Update this to your Zarr file path
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  seed: 42
  val_ratio: 0.02
  max_train_episodes: null  # Set to limit episodes, or null for all
```

**Note**: The config is already set up correctly with:
- Point cloud shape: `[2048, 3]` (XYZ only, RGB is stored but not used)
- Agent position shape: `[15]` (robot_obs from CALVIN)
- Action shape: `[7]` (rel_actions from CALVIN)
- Environment runner: `null` (data-only training, no simulation evaluation)

### Step 2: Train DP3

Use the training script:

```bash
bash scripts/train_policy.sh dp3 calvin 0322 0 0
```

**Parameters:**
- `dp3`: Algorithm name (uses `dp3.yaml` config)
- `calvin`: Task name (uses `calvin.yaml` task config)
- `0322`: Additional info for experiment name (can be any string, e.g., date)
- `0`: Random seed
- `0`: GPU ID

**Alternative: Direct Python Command**

If you prefer to run directly:

```bash
cd 3D-Diffusion-Policy
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

python train.py --config-name=dp3.yaml \
                task=calvin \
                hydra.run.dir=data/outputs/calvin-dp3-0322_seed0 \
                training.debug=False \
                training.seed=0 \
                training.device="cuda:0" \
                exp_name=calvin-dp3-0322 \
                logging.mode=online \
                checkpoint.save_ckpt=True
```

### Step 3: Monitor Training

Training outputs will be saved to:
- `data/outputs/calvin-dp3-0322_seed0/`

Checkpoints will be saved every 200 epochs (configurable in `dp3.yaml`).

## Configuration Verification

✅ **Dataset Loader** (`calvin_dataset.py`):
- Correctly loads Zarr data
- Extracts XYZ from point cloud (2048, 6) → (2048, 3)
- Returns observations: `point_cloud` (T, 2048, 3) and `agent_pos` (T, 15)
- Returns actions: (T, 7)

✅ **Task Config** (`calvin.yaml`):
- Point cloud shape: `[2048, 3]` ✓
- Agent position shape: `[15]` ✓
- Action shape: `[7]` ✓
- Environment runner: `null` (for data-only training) ✓

✅ **DP3 Config** (`dp3.yaml`):
- Uses PointNet encoder with 3 channels (XYZ only)
- `use_pc_color: false` (matches our setup)
- Horizon: 16, n_obs_steps: 2, n_action_steps: 8

## Training Parameters (from dp3.yaml)

- **Batch size**: 128
- **Learning rate**: 1.0e-4
- **Optimizer**: AdamW
- **Epochs**: 3000
- **Checkpoint frequency**: Every 200 epochs
- **Validation**: Every 1 epoch
- **Horizon**: 16 timesteps
- **Observation steps**: 2
- **Action steps**: 8

## Troubleshooting

### Import Errors
If you get import errors:
```bash
# Make sure you're in the project root
cd /home/mlewkowicz/3D-Diffusion-Policy
```

### CUDA Out of Memory
- Reduce batch size in `dp3.yaml`: `dataloader.batch_size: 64`
- Reduce `max_train_episodes` in `calvin.yaml`

### Dataset Path Not Found
- Verify the Zarr file path in `calvin.yaml` is correct
- Use absolute path if relative path doesn't work:
  ```yaml
  zarr_path: /home/mlewkowicz/3D-Diffusion-Policy/data/calvin_dp3.zarr
  ```

### Shape Mismatch Errors
- Verify your Zarr file has the correct shapes:
  - `point_cloud`: (N, 2048, 6)
  - `state`: (N, 15)
  - `action`: (N, 7)
- The dataset loader automatically extracts XYZ from point cloud

## Expected Output

During training, you should see:
- Dataset loading messages
- Training loss decreasing
- Validation metrics
- Checkpoints saved to `data/outputs/calvin-dp3-0322_seed0/checkpoints/`

## Next Steps

After training:
1. Evaluate the policy (if you have CALVIN simulation environment)
2. Use the trained checkpoint for inference
3. Adjust hyperparameters if needed

## Quick Reference

**Full training command:**
```bash
bash scripts/train_policy.sh dp3 calvin 0322 0 0
```

**Check training logs:**
```bash
tail -f data/outputs/calvin-dp3-0322_seed0/train.log
```

**List checkpoints:**
```bash
ls data/outputs/calvin-dp3-0322_seed0/checkpoints/
```
