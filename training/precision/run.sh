#!/bin/bash
set -eux
pip install -q --upgrade torch torchvision torchao "transformers==5.18.0" "peft==0.21.1" scikit-learn pandas pillow
pip uninstall -y torchaudio || true
python -c "import torch, torchao; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchao', torchao.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
python precision_eval.py "$@"
