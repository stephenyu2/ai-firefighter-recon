# FLAME 3 fire / no-fire baselines

Experiments comparing no-learning rules, off-the-shelf models, vision-language models, and models fine-tuned
on FLAME 3 for one question per drone frame: is there fire? Run on a MacBook (Apple GPU via PyTorch MPS).

**Results:** open `report/flame3_results.html` in a browser. It has the findings, every table, run times,
and a methodology checklist. The underlying numbers are in `results/` (`report_table.csv` is the main table).

## Data
- FLAME 3 CV subset (Sycan Marsh). The Kaggle mirror downloads without a login:
  `curl -L -o flame3_cv.zip https://www.kaggle.com/api/v1/datasets/download/brycehopkins/flame-3-computer-vision-subset-sycan-marsh`
- Unzip it and set `FLAME3_ROOT` to the folder that contains `FLAME 3 CV Dataset (Sycan Marsh)`.
- 738 image sets: corrected-FOV RGB 640x512, raw RGB 4000x3000, thermal JPG (normalized per image, do not use
  for modeling), Celsius TIFF 640x512 float32. Labels are folder names only.
- Cite Hopkins et al. 2024 (arXiv 2412.02831) and the IEEE DataPort DOI 10.21227/w0mz-aq48.

**Splits:** file numbers are shuffled relative to capture time, so never build time blocks from file order.
`index_dataset.py` reads each frame's EXIF capture time and `common.load_index()` groups frames into
60-second windows. Trained models also drop training frames within 30 seconds of any test frame.

## Environments
- Main: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Molmo 2 needs its own environment (its remote code targets transformers 4.57.1):
  `pip install -r requirements-molmo.txt` in a second venv.
- The shell queues read `VENV` and `MOLMO_VENV` for those two environments.
- Run GPU jobs one at a time. Running a large VLM next to training pushed a 48 GB Mac into swap.
  On a Mac, wrap long runs in `caffeinate -i -s` so idle sleep does not freeze them.

## Pipeline (from this folder)
| Step | Script | What it does |
|---|---|---|
| 1 | `index_dataset.py` | EXIF time, drone altitude and gimbal, thermal stats per frame |
| 2 | `common.py` | time-window groups, folds, 30 s purge, day holdout, thermal encoding, metrics |
| 3 | `make_figures.py` | separability histogram, timeline, mask overlays |
| 4 | `run_baselines.py` | always-fire, max-temperature rule, thermal-JPG darkness |
| 5 | `run_zeroshot.py` | CLIP / SigLIP 2 zero-shot, two Hugging Face fire classifiers |
| 6 | `run_vlm.py`, `run_molmo.py`, `vlm_to_scores.py` | VLM yes/no scoring (Qwen3.5, Qwen3-VL, Gemma 4, Molmo 2) |
| 7 | `run_probes.py [grouped|purged|day]`, `run_probes_fewshot.py 16 purged` | frozen features + logistic regression |
| 8 | `run_finetune.py`, `run_finetune_queue2.sh` | LoRA (DINOv2-B, fire ViT) and ResNet-50 full fine-tune |
| 9 | `run_after.sh`, `run_qwen_sizes.sh` | remaining VLMs and the latency benchmark, in sequence |
| 10 | `leak_check.py` | shuffled-label audit (test AUC should drop to about 0.5) |
| 11 | `bench_latency.py`, `bench_latency_molmo.py` | clean per-frame latency, batch size 1 |
| 12 | `failure_cases.py`, `make_report.py`, `report/build_html.py` | hard-frame figure, tables, the HTML report |

## Protocol
- 5-fold StratifiedGroupKFold over 60-second capture windows, plus a 30-second purge for trained models.
- Day holdout: train on October 25-26, test on all October 27 frames (recall only; no no-fire frames that day).
- Default threshold 0.5 or the model's own answer. Recall-tuned threshold picked on training or inner-validation
  frames for recall >= 0.98, never on the test fold.
- Slices: fire frames in the two flights that also contain no-fire frames (n=82), and small fires with under
  0.25% of pixels above 80 C (n=55).
