#!/bin/bash

# Convert CALVIN debug dataset to DP3 Zarr format

ROOT_DIR="/home/mlewkowicz/calvin/dataset/calvin_debug_dataset"
SAVE_PATH="data/calvin_test.zarr"
SPLIT="training"

echo "Converting CALVIN debug dataset..."
echo "Root dir: $ROOT_DIR"
echo "Save path: $SAVE_PATH"
echo "Split: $SPLIT"

python scripts/convert_calvin_to_dp3.py \
    --root_dir $ROOT_DIR \
    --save_path $SAVE_PATH \
    --visualize_samples \
    --visualize_every_n 1 \
    --visualize_save_dir visualizations/calvin

echo "Conversion complete!"
