# Full-burn detector benchmark

Every frame of two prescribed burns through the fire classifier, one request at a time:

- **Sycan Marsh, Oregon:** 738 frames from the FLAME 3 CV subset, which the detectors trained on.
- **Hanna Hammock, Florida:** 1,353 frames from FLAME 3 NADIR, which the detectors never saw.

We measured accuracy against the dataset's labels (Sycan) and thermal labels (Florida), and the time each burn takes end to end and on the server. The benchmark
covers T4, L4 and RTX PRO 6000 GPUs, 4-bit precision (NVFP4), smaller uploads, and the Kai decision model as a
replacement for the detector threshold.

**Report:** [`report/full_burn_report.html`](report/full_burn_report.html) (open in a browser).

## In short

- On the RTX PRO 6000 a whole burn takes 4 to 12 seconds of server time. The upload, not the GPU, sets the end-to-end
  time. Shrinking frames to the detector's input size and sending full-color JPEG cut end-to-end time by about 65%.
- On Florida, the detectors miss small fires that the thermal camera sees. Recall is 0.81 on the thermal-view crop
  and 0.99 on the full RGB frame (DINOv2-B). No configuration flagged any of the 170 pre-burn frames.
- NVFP4 makes the DINOv2-g GPU step 1.8x faster with no loss in recall. DINOv2-B is already fast enough in 16-bit.
- Kai followed the RGB detectors and did not recover the small fires.
- **Recommendation:** decide fire status from thermal when a frame has it, and use the RGB detector otherwise.

## Files

| Step | Script | Output |
|---|---|---|
| Prepare Florida | `prep_florida.py` | thermal-view crops, full frames, labels from the raw thermal TIFFs (fire = any pixel at or above 80 C); `results/florida_index.csv` is the committed copy of its index |
| Thermal field of view | `check_fov.py` | where the thermal camera's view sits in the RGB frame (the result is hard-coded as `FOV_W` and `FOV_H` in `prep_florida.py`) |
| Benchmark | `bench_full.py` | `results/full_runs.jsonl`, `results/preds_<dataset>_<gpu>_<model>.csv` |
| Kai evidence | `collect_evidence.py` | `results/kai/evidence_<dataset>.csv` (detector, CLIP and thermal per frame) |
| Kai decisions | `kai_eval.py` | `results/kai/kai_runs.jsonl`, `results/kai/kai_scores_<label>.csv` (state text dropped to save space; `kai_eval.state()` rebuilds it) |
| Precision probe | `precision_probe.py` | `results/precision/probe_rtx6000_*.json` |
| Tables | `summarize_full.py` | `results/full_runs_summary.csv` |
| Report | `build_report.py` | `report/full_burn_report.html` (reads only `results/`, `report_template.html` and `report_findings.html`) |

The B200 precision study in `results/precision/cv_precision_b200.txt` comes from `training/precision/precision_eval.py`.

## Reproduce

The report rebuilds offline from the committed results:

```bash
python summarize_full.py
python build_report.py
```

New runs need the FLAME 3 data under `RECON_DATA_ROOT` (default: the repo's git-ignored `data/`, with
`flame3_cv/` and `flame3_nadir/Hanna Hammock/`). They also need a fire classifier server (`serving/`). The benchmark
calls the server directly, not through the gateway, so the server-side timing in its response is not mixed with
proxy time. Set `RECON_API_KEY` if the server needs one.

```bash
export RECON_DATA_ROOT=/path/to/data
python prep_florida.py
python bench_full.py RTX_PRO_6000 http://127.0.0.1:8010/predict dinov2_lora 32 "$RECON_DATA_ROOT/flame3_nadir_prepped/florida_crop.csv"
CLIENT_RESIZE=308x252 CLIENT_QUALITY=90 CLIENT_SUBSAMPLING=0 python bench_full.py RTX_PRO_6000 http://127.0.0.1:8010/predict dinov2_lora 32 "$RECON_DATA_ROOT/flame3_nadir_prepped/florida_full.csv"
python collect_evidence.py http://127.0.0.1:8010/predict "$RECON_DATA_ROOT/flame3_nadir_prepped/florida_crop.csv"
python kai_eval.py RTX_PRO_6000 http://127.0.0.1:8011/predict 8
```

Things to know:

- **No manifest:** `bench_full.py` runs the 738 Sycan frames.
- **Server-side timing:** `kai_eval.py` reads it from the response header named by `SERVER_TIME_HEADER`, or else
  from the `timing_ms` field that `serving/kai/model.py` returns.
