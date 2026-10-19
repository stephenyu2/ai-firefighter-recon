# Decision cascade trials

Author: Spencer Theissen-Vang (experiment design, model deployment, testing, analysis and writeup).

I tested two new open decision models from the vLLM Semantic Router project, d3 (26B) and d3-mini (4.5B), plus
d3-flash (8.4B), as the fire-status decider for the recon gateway. They read images directly and answer typed
questions with a probability for every option. I compared them with Kai, Clef, Clef-Flash and the Qwen3.8-27B builds
on the same frames and sequences as the gateway model trials. Then I built a cascade: d3 makes the call when it is
confident, and Qwen3.8-27B takes a deeper look when it is not. I ran the cascade on the 20 test sequences and on every
frame of the Florida burn.

**Report:** [`report/decision_cascade_trials.html`](report/decision_cascade_trials.html) (open in a browser).

## In short

- **Cascade, test sequences:** with d3 first, the cascade got 17 of 20 sequences right and 4 of 7 small RGB-only fires,
  with no false alarms. That ties the best earlier method, the measurement rule on guided Qwen3.8 BF16 image checks.
- **Cascade, full Florida burn:** all 101 fire windows and all 22 clean windows were called correctly, with no errors
  or retries. 38 of 170 windows (22%) went to the LLM.
- **Time:** the whole Florida day (1,348 images, about 102 minutes of flying) took 49.5 minutes, 2.2 s per image.
  When d3 decides, a window takes a median 5.6 s; a deeper pass takes a median 53 s.
- **d3 alone:** the best single-frame recall and ranking of any model tested (0.42 recall, ROC AUC 0.976, about half of
  the small fires). d3-mini and d3-flash miss nearly all small fires.
- **Measurement text hurts:** with the RGB detector's counts in the request, d3 missed every small fire; with the images
  and the guidance alone it caught 2 of 7.
- **The gap:** d3 called 3 of the 7 small RGB-only fires "no fire" with over 90% confidence, so no confidence threshold
  catches them. The likely cause is resolution: 8000 by 6000 frames are sent as 640 by 512 crops.

## The cascade

`src/recon_gateway/cascade.py`:

1. **Measure:** the gateway runs its usual measurements.
2. **Ask d3:** d3 reads all 8 frames, the observer guidance and the thermal facts only, and answers two questions:
   fire status, and whether active fire is visible.
3. **Decide:** if d3's top status probability is at least 0.80 and the answer is not "unclear", that is the call.
4. **Escalate:** otherwise `agent.run(..., probe_all=True, extra_evidence=...)` runs Qwen3.8-27B with guided image
   checks on every frame and d3's probabilities as evidence. The measurement rule, with thermal as the authority, can
   raise the call to fire. Failed passes are retried up to 2 times.

The threshold and the retry limit were fixed before any run.

## Files

| Script | What it does | Results |
|---|---|---|
| `../gateway-model-trials/frame_eval.py` (with `RESULTS_DIR` set here) | Single-frame "flames visible?" check for the d3 family on the 240- and 221-frame samples, with and without guidance, plus a repeat | `results/frames_v2b.jsonl`, `results/frames_v2b_*.csv` |
| `d3_sequences.py` | The 20 sequences in three request variants: measurements plus first and last frame, all 8 frames plus guidance plus measurements, all 8 frames plus guidance only | `results/sequence_variants.jsonl` |
| `../gateway-model-trials/run_briefs.py` (`DECIDERS=d3,d3_mini`, `RESULTS_DIR` set here) | Full Qwen3.8 BF16 pipeline rerun with d3 and d3-mini as in-brief deciders | `results/briefs_v3.jsonl` |
| `cascade_eval.py` | The cascade on the 20 sequences, with d3 or d3-mini first | `results/cascade.jsonl` |
| `fullburn_cascade.py` | The cascade over every frame of the Florida burn in 170 eight-frame windows | `results/fullburn.jsonl` |
| `../gateway-model-trials/runs_store.py pack v3` (with `RESULTS_DIR` set here) | Packs the saved full runs (rerun briefs and every escalated window) | `results/runs_v3.jsonl.gz` |
| `summarize.py` | Every number in the report, from saved results only | `results/summary.json` |
| `run_all.sh` | Every step above in order | |

Run names:

| Name | Model (Hugging Face, revision tested) | Setup |
|---|---|---|
| `d3` | [vllm-sr/d3](https://huggingface.co/vllm-sr/d3/tree/093cffbaa7be6f5dfe14252ae7e80c396c06dbd9) at `093cffba` | one H100 |
| `d3_rtx` | [vllm-sr/d3](https://huggingface.co/vllm-sr/d3/tree/093cffbaa7be6f5dfe14252ae7e80c396c06dbd9) at `093cffba` | one RTX PRO 6000, `serving/d3/model.py`; a cross-check that gave the same answers |
| `d3_mini` | [vllm-sr/d3-mini](https://huggingface.co/vllm-sr/d3-mini/tree/0e78bbdbe3cf480ed1461c185c9e0c4fa9258011) at `0e78bbdb` | one L4 |
| `d3_flash` | [vllm-sr/d3-flash](https://huggingface.co/vllm-sr/d3-flash/tree/f6075bff3c0f7d5704e353d0fb2ebf94888d1691) at `f6075bff` | one L4, `serving/d3/model.py` |

Each model repository holds the weights, the model's own runtime and its serving instructions (see the model cards).
To serve a model yourself, use those instructions or `serving/d3/model.py`.

The earlier methods are compared from their saved results in `../gateway-model-trials/results`; they were not rerun.
In the saved escalated runs, Qwen3.8's writeups quote the decision model by its earlier run label, `d3g`, which is `d3`
here.

## Reproduce

The report's numbers rebuild offline from the committed results:

```bash
python summarize.py
```

A full rerun needs:

- the gateway proxy with the Qwen3.8 BF16 route, the classifier and the four d3 routes (see `gateway/` and `serving/`;
  the models come from the Hugging Face repositories above);
- the FLAME 3 data under `RECON_DATA_ROOT`, with the Florida frames prepared by `../full-burn-benchmark/prep_florida.py`.

Then run `./run_all.sh`. The full burn alone takes about 50 minutes, one window at a time.
