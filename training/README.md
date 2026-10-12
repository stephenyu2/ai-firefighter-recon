# Detector training

Fine-tuning for the RGB fire detector the gateway calls. Both scripts read the 738-frame FLAME 3 CV subset (Sycan
Marsh) listed in `flame3-lora/index.csv`. `index.csv` holds the label, capture time, 60-second window, flight, mixed-flight
flag and the share of thermal pixels at or above 80 C for each frame.

| Script | What it does | Outputs (under `$OUTPUT_DIR`) |
|---|---|---|
| `flame3-lora/train_cv.py` | DINOv2-B LoRA (basic and strong augmentation), DINOv2-L and DINOv2-g QLoRA (4-bit). Each is scored under purged 5-fold CV (30 s purge around each test window) and a day holdout; DINOv2-B (both) and DINOv2-g are then retrained on all frames for serving | `results/summary.csv`, `results/scores_<variant>_<protocol>.csv`, `final_<variant>/` (adapter, `head.pt`, `serving_config.json`) |
| `precision/precision_eval.py` | Paired precision study: train once per purged fold in 16-bit, then score the held-out fold in FP16, BF16, FP8, NVFP4 and NVFP4 weight-only. Reports recall, false alarms, decision flips against FP16, and GPU time per frame | `results/precision_summary.csv`, `results/scores_<variant>_<mode>.csv` |

## Run

```bash
export RECON_DATA_ROOT=/path/to/data
cd training/flame3-lora
OUTPUT_DIR=./ckpts bash run.sh
```

- `RECON_DATA_ROOT` should contain `flame3_cv/FLAME 3 CV Dataset (Sycan Marsh)/`. It defaults to the repo's
  git-ignored `data/`. On a remote GPU box, `FLAME3_RGB_DIR` can point at flat `<uid>.jpg` copies of the frames
  instead.
- `SMOKE=1` runs a 120-frame, one-epoch check.
- `FINAL_ONLY=1` skips the CV and trains only the final checkpoints. Each threshold comes from that variant's CV
  scores in `$OUTPUT_DIR/results/`, or from the committed `prior_scores_<variant>_purged_cv.csv` (DINOv2-g only), so
  run it after a CV run or for `dinov2g_qlora`.
- Pass variant names to `run.sh` or `train_cv.py` to train a subset, e.g. `bash run.sh dinov2g_qlora`.

The precision study needs a Blackwell GPU (B200 or RTX PRO 6000) for NVFP4: from `training/flame3-lora`, run
`cd ../precision && bash run.sh`. Configs for running both as remote jobs are in `deploy/training/`.
Its result from our B200 run is
`experiments/full-burn-benchmark/results/precision/cv_precision_b200.txt`.

`final_dinov2b_lora_strongaug` is the detector the gateway serves as `dinov2_lora`, and `final_dinov2g_qlora` is
`dinov2g_qlora`. See `serving/README.md` for how to load them.
