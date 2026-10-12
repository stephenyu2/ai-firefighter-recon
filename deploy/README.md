# Deployment on Baseten

Every hosted run in this project ran on Baseten:

- the detector training jobs and the precision study;
- the fire classifier on T4, L4 and RTX PRO 6000;
- each Qwen3.8 build on its own RTX PRO 6000;
- Clef, Clef-Flash and Kai.

This folder holds the configs for those runs and a script that turns them into folders you can push. The model code is
not duplicated here: `assemble.py` copies it in from `serving/`, `training/` and `experiments/`, so the code that
ran on Baseten and the code in the rest of the repo are the same files. The rest of the repo does not depend on
Baseten; any host that can run these servers works.

## What ran where

| Build | Hardware | Used for |
|---|---|---|
| `models/fire-classifier-t4`, `-l4`, `-rtx6000` | T4, L4, RTX PRO 6000 | Full-burn benchmark timing and accuracy. The RTX PRO 6000 deployment was also the classifier behind the gateway in the model trials |
| `models/fire-classifier-rtx6000-nvfp4` | RTX PRO 6000 | Compiled FP16 and NVFP4 detector variants (`QUANT_VARIANTS`) in the full-burn report |
| `models/precision-probe-rtx6000` | RTX PRO 6000 | `experiments/full-burn-benchmark/results/precision/probe_rtx6000_*.json`. The probe runs once at startup and `predict` returns its results. The committed config limits it to FP16 and FP8 (`PROBE_MODES`, the second run); remove that line to run every mode |
| `models/llm-qwen38-27b-bf16`, `-fp8`, `-int4`, `-nvfp4` | RTX PRO 6000 each | Gateway model trials. vLLM 0.31.0 docker server; weights pulled from Hugging Face at a pinned revision |
| `models/llm-qwen3vl-4b` | L4 | The earlier Qwen3-VL-4B baseline in the trials |
| `models/clef-27b-rtx6000`, `models/clef-flash-rtx6000` | RTX PRO 6000 | Clef and Clef-Flash in the trials, running the models' own release code |
| `training/flame3-lora` | H100 | Detector fine-tuning: `config.py` runs every variant, `config_final_giant.py` only the final DINOv2-g QLoRA |
| `training/precision` | B200 (`config_b200.py`), RTX PRO 6000 (`config.py`) | Paired precision study; the B200 run is `experiments/full-burn-benchmark/results/precision/cv_precision_b200.txt` |

Kai (Decision-2.0-Kai-0.6B) was deployed from the Baseten model library on an L4 and an RTX PRO 6000, so there is no
config for it here. To host it yourself, use `serving/kai/model.py`.

## Push a model

```bash
pip install --upgrade truss
truss login
python deploy/assemble.py fire-classifier-rtx6000 --detector ckpts/final_dinov2b_lora_strongaug --detector-giant ckpts/final_dinov2g_qlora
truss push deploy/build/fire-classifier-rtx6000
python deploy/assemble.py llm-qwen38-27b-fp8
truss push deploy/build/llm-qwen38-27b-fp8
```

- `assemble.py` writes `deploy/build/NAME/`, which is git-ignored.
- **Fire classifier:** pass the `final_*` folders from `training/flame3-lora/train_cv.py`; weights are not in git.
  `--detector-giant` is required for the NVFP4 build, the precision probe, and any classifier build meant to
  reproduce the DINOv2-g full-burn rows. Leave it out only when serving DINOv2-B alone.
- **Precision probe:** the 64 frames listed in `frames.txt` are rebuilt from `RECON_DATA_ROOT`, resized to the
  detector input.
- **Scale to zero and cold starts:** deployments scale to zero when idle, and the first call after that can take
  several minutes while the weights load. The clients retry 408, 429 and 5xx, and the health-check thresholds in the
  vLLM and Clef configs allow for long startups.

## Run a training job

```bash
python deploy/assemble.py flame3-lora
cd deploy/build/flame3-lora
truss train push config.py --tail
```

The job folder contains:

- the training script and the frame index;
- the 738 FLAME 3 CV frames, copied from `RECON_DATA_ROOT` into `data/rgb/`;
- `run_baseten.sh`, which points `OUTPUT_DIR` at the job's checkpoint folder (`$BT_CHECKPOINT_DIR`) and `FLAME3_RGB_DIR`
  at `data/rgb`.

Use `truss train push config_final_giant.py` for the final-giant job, or `python deploy/assemble.py precision` and
`config_b200.py` for the precision study. Checkpoints and `results/` come back through the job's checkpoint folder.

## Point the gateway at Baseten deployments

In `gateway/litellm/config.yaml`, set each route's URL to the deployment's endpoint and use the Baseten key. Chat
routes use the OpenAI-compatible `sync/v1` path. Pass-through routes use `predict` and need the `Api-Key` scheme:

```yaml
  - model_name: qwen38-27b-fp8
    litellm_params:
      model: openai/Qwen/Qwen3.8-27B
      api_base: https://model-MODEL_ID.api.baseten.co/environments/production/sync/v1
      api_key: os.environ/BASETEN_API_KEY
# under general_settings.pass_through_endpoints:
    - path: /recon/classifier
      target: https://model-MODEL_ID.api.baseten.co/environments/production/predict
      headers:
        Authorization: Api-Key os.environ/BASETEN_API_KEY
        content-type: application/json
```

Replace `MODEL_ID` with the model's id from the Baseten dashboard. To call deployments directly without the gateway,
set `RECON_API_KEY` to your Baseten key and `RECON_AUTH_SCHEME=Api-Key`.

`experiments/full-burn-benchmark/kai_eval.py` read server-side time from Baseten's
`x-baseten-container-duration-minimum-ms` response header. Set `SERVER_TIME_HEADER` to that name to get the same column.
