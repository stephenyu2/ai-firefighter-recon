## Main table (default threshold, all 738 frames, out-of-fold)

| Model | Input | Tier | Recall | Precision | Specificity | ROC-AUC | Recall, mixed flights | Recall, small fires | False alarms | Missed fires |
|---|---|---|---|---|---|---|---|---|---|---|
| Max temp above 150 C | thermal TIFF | no-learning baseline | 0.998 | 1.000 | 1.000 | 1.000 | 0.988 | 0.982 | 0 | 1 |
| Thermal JPG darkness (artifact) | thermal JPG | no-learning baseline | 0.593 | 1.000 | 1.000 | 1.000 | 0.195 | 0.164 | 0 | 253 |
| Always say fire | - | no-learning baseline | 1.000 | 0.843 | 0.000 | 0.500 | 1.000 | 1.000 | 116 | 0 |
| Qwen3-VL 8B, yes/no | RGB | off the shelf, no FLAME training | 0.973 | 1.000 | 1.000 | 1.000 | 0.963 | 0.945 | 0 | 17 |
| Molmo 2 8B, yes/no | RGB | off the shelf, no FLAME training | 0.608 | 1.000 | 1.000 | 1.000 | 0.890 | 0.491 | 0 | 244 |
| Qwen3.5 4B, yes/no | RGB | off the shelf, no FLAME training | 0.952 | 1.000 | 1.000 | 1.000 | 0.939 | 0.927 | 0 | 30 |
| CLIP ViT-L/14 zero-shot | RGB | off the shelf, no FLAME training | 0.987 | 1.000 | 1.000 | 1.000 | 0.927 | 0.909 | 0 | 8 |
| SigLIP 2 zero-shot | RGB | off the shelf, no FLAME training | 0.918 | 1.000 | 1.000 | 0.999 | 0.585 | 0.564 | 0 | 51 |
| Qwen3.5 9B, yes/no | RGB | off the shelf, no FLAME training | 0.961 | 1.000 | 1.000 | 0.997 | 0.988 | 0.982 | 0 | 24 |
| Forest-fire SigLIP 2 classifier (HF) | RGB | off the shelf, no FLAME training | 0.859 | 1.000 | 1.000 | 0.994 | 0.293 | 0.400 | 0 | 88 |
| Qwen3.5 2B, yes/no | RGB | off the shelf, no FLAME training | 0.955 | 1.000 | 1.000 | 0.994 | 0.988 | 0.982 | 0 | 28 |
| Molmo 2 4B, yes/no | RGB | off the shelf, no FLAME training | 0.400 | 1.000 | 1.000 | 0.993 | 0.549 | 0.382 | 0 | 373 |
| Gemma 4 E4B, yes/no | RGB | off the shelf, no FLAME training | 0.839 | 1.000 | 1.000 | 0.978 | 0.988 | 0.964 | 0 | 100 |
| Qwen3.5 0.8B, yes/no | RGB | off the shelf, no FLAME training | 0.910 | 0.993 | 0.966 | 0.977 | 0.988 | 0.964 | 4 | 56 |
| SigLIP 2 zero-shot | thermal JPG | off the shelf, no FLAME training | 0.084 | 1.000 | 1.000 | 0.965 | 0.000 | 0.000 | 0 | 570 |
| CLIP ViT-L/14 zero-shot | thermal JPG | off the shelf, no FLAME training | 0.619 | 1.000 | 1.000 | 0.957 | 0.220 | 0.436 | 0 | 237 |
| Fire ViT classifier (HF) | RGB | off the shelf, no FLAME training | 0.680 | 1.000 | 1.000 | 0.931 | 0.171 | 0.382 | 0 | 199 |
| DINOv2-B linear probe | RGB + thermal | trained on FLAME 3 | 0.995 | 1.000 | 1.000 | 1.000 | 0.963 | 0.964 | 0 | 3 |
| DINOv2-B + LoRA | thermal TIFF | trained on FLAME 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| DINOv2-B linear probe | thermal TIFF | trained on FLAME 3 | 0.992 | 1.000 | 1.000 | 1.000 | 0.963 | 0.945 | 0 | 5 |
| SigLIP 2 linear probe | RGB | trained on FLAME 3 | 0.987 | 1.000 | 1.000 | 1.000 | 0.988 | 0.982 | 0 | 8 |
| ResNet-50 full fine-tune | RGB | trained on FLAME 3 | 0.968 | 1.000 | 1.000 | 0.999 | 0.768 | 0.727 | 0 | 20 |
| CLIP ViT-L/14 linear probe | RGB | trained on FLAME 3 | 0.987 | 0.997 | 0.983 | 0.999 | 0.963 | 0.945 | 2 | 8 |
| DINOv2-B linear probe | RGB | trained on FLAME 3 | 0.981 | 0.997 | 0.983 | 0.999 | 0.927 | 0.909 | 2 | 12 |
| DINOv2-B + LoRA | RGB | trained on FLAME 3 | 0.990 | 1.000 | 1.000 | 0.997 | 0.927 | 0.909 | 0 | 6 |
| Fire ViT + LoRA | RGB | trained on FLAME 3 | 0.918 | 0.983 | 0.914 | 0.964 | 0.549 | 0.509 | 10 | 51 |


## Recall-tuned threshold (picked on training or validation frames, target recall 0.98)

| Model | Input | Recall | Specificity | Precision | Recall, mixed flights |
|---|---|---|---|---|---|
| Qwen3-VL 8B, yes/no | RGB | 0.973 | 1.000 | 1.000 | 0.976 |
| Molmo 2 8B, yes/no | RGB | 0.982 | 1.000 | 1.000 | 0.988 |
| Qwen3.5 4B, yes/no | RGB | 0.963 | 1.000 | 1.000 | 0.963 |
| CLIP ViT-L/14 zero-shot | RGB | 0.987 | 1.000 | 1.000 | 0.927 |
| SigLIP 2 zero-shot | RGB | 0.976 | 0.991 | 0.998 | 0.902 |
| Qwen3.5 9B, yes/no | RGB | 0.968 | 1.000 | 1.000 | 0.988 |
| Forest-fire SigLIP 2 classifier (HF) | RGB | 0.974 | 0.914 | 0.984 | 0.829 |
| Qwen3.5 2B, yes/no | RGB | 0.971 | 0.991 | 0.998 | 0.988 |
| Molmo 2 4B, yes/no | RGB | 0.966 | 0.888 | 0.979 | 1.000 |
| Gemma 4 E4B, yes/no | RGB | 0.966 | 0.672 | 0.941 | 0.988 |
| Qwen3.5 0.8B, yes/no | RGB | 0.968 | 0.629 | 0.933 | 0.988 |
| SigLIP 2 zero-shot | thermal JPG | 0.973 | 0.784 | 0.960 | 0.817 |
| CLIP ViT-L/14 zero-shot | thermal JPG | 0.977 | 0.543 | 0.920 | 0.878 |
| Fire ViT classifier (HF) | RGB | 0.929 | 0.250 | 0.869 | 0.463 |
| DINOv2-B linear probe | RGB + thermal | 0.992 | 1.000 | 1.000 | 0.939 |
| DINOv2-B + LoRA | thermal TIFF | 1.000 | 1.000 | 1.000 | 1.000 |
| DINOv2-B linear probe | thermal TIFF | 0.987 | 1.000 | 1.000 | 0.927 |
| SigLIP 2 linear probe | RGB | 0.981 | 1.000 | 1.000 | 0.988 |
| ResNet-50 full fine-tune | RGB | 0.982 | 0.991 | 0.998 | 0.866 |
| CLIP ViT-L/14 linear probe | RGB | 0.979 | 0.966 | 0.993 | 0.963 |
| DINOv2-B linear probe | RGB | 0.979 | 0.957 | 0.992 | 0.927 |
| DINOv2-B + LoRA | RGB | 0.984 | 1.000 | 1.000 | 0.927 |
| Fire ViT + LoRA | RGB | 0.939 | 0.681 | 0.940 | 0.537 |


## 16 labeled frames per class (mean over seeds)

| Run | Seeds | Recall | Specificity | ROC-AUC | Recall, mixed flights | Recall, small fires |
|---|---|---|---|---|---|---|
| L1_dinov2b_lora_rgb_16shot_PURGED | 3 | 0.981 | 1.000 | 0.996 | 0.902 | 0.873 |
| L2_firevit_lora_rgb_16shot_PURGED | 3 | 0.857 | 0.986 | 0.950 | 0.305 | 0.418 |
| P1_dinov2b_probe_rgb_16shot_PURGED | 3 | 0.968 | 1.000 | 0.999 | 0.866 | 0.824 |
| P2_dinov2b_probe_thermal_16shot_PURGED | 3 | 0.971 | 0.997 | 0.998 | 0.801 | 0.776 |
| P4_clipL14_probe_rgb_16shot_PURGED | 3 | 0.966 | 1.000 | 0.999 | 0.963 | 0.945 |
| P5_siglip2_probe_rgb_16shot_PURGED | 3 | 0.979 | 1.000 | 1.000 | 0.963 | 0.945 |


## Random split (leaky) for comparison

| Run | Recall | Precision | Specificity | ROC-AUC | Recall, mixed flights | Recall, small fires |
|---|---|---|---|---|---|---|
| F1_resnet50_fullft_rgb_RANDOMSPLIT | 0.990 | 1.000 | 1.000 | 1.000 | 0.939 | 0.945 |
