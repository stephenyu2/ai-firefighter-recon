#!/bin/bash
set -eux
pip install -q "transformers==5.18.0" "peft==0.21.1" "bitsandbytes>=0.45" "accelerate>=1.0" scikit-learn pandas pillow
nvidia-smi --query-gpu=name,memory.total --format=csv
python train_cv.py "$@"
