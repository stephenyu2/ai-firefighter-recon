#!/bin/bash
# Entry point for Baseten training jobs. Checkpoints go to the platform's checkpoint folder ($BT_CHECKPOINT_DIR), and the
# frames ship inside the job in data/rgb (deploy/assemble.py puts them there). TRAIN_ARGS limits the variants.
set -eux
export OUTPUT_DIR="$BT_CHECKPOINT_DIR"
export FLAME3_RGB_DIR="$PWD/data/rgb"
exec ./run.sh ${TRAIN_ARGS:-}
