# Gateway model trials

This experiment asks which model should run the recon gateway's planning, image checks and writeup. The candidates
are Qwen3.8-27B at four precisions (BF16, FP8, INT4, NVFP4), each on one RTX PRO 6000, plus the earlier Qwen3-VL-4B
baseline.

It also asks two related questions:

- Should a dedicated decision model (Kai 0.6B, Clef-Flash 9B, Clef 27B) make the fire call instead of the LLM?
- Does a generic observer guidance prompt help models notice fire?

Every brief and frame check went through the LiteLLM gateway (`gateway/`) with fixed sampling, validated tool calls
and run tags.

**Report:** [`report/gateway_model_trials.html`](report/gateway_model_trials.html) (open in a browser).

## In short

- 600 of 600 briefs completed with no failed or unfinished tool-calling runs. A repeated per-frame check changed 0 of
  1,680 answers.
- **Thermal:** with thermal data, every Qwen3.8 build gets fire status right on all 24 thermal briefs.
- **Small RGB fires** are the hard case. Each model's own call caught at most 1 of 7 small-fire sequences. The
  measurement rule on guided BF16 image checks caught 4 of 7 with no false alarms, which is 17 of 20 sequences
  overall against 14 for the best LLM call.
- **Guidance:** the observer guidance (`src/recon_gateway/prompts.py`) is written once from general aerial-observation
  practice, with no burn-specific values. It raised single-frame recall on two separate frame samples for every
  Qwen3.8 build (FP8 only slightly) and for both Clef models, with no new false alarms (BF16: 0.38 to 0.48 on the second
  sample). With guidance on everywhere, the writeup's own call did not improve for any Qwen3.8 build.
- **Decision models:** Kai, Clef and Clef-Flash got 12, 11 and 13 of 20. None should replace the LLM plus thermal.
- **Speed:** single-request decode is 26 tok/s on BF16, 45 on FP8, 52 on NVFP4 and 74 on INT4. A full brief takes a
  median 43, 30, 25 and 26 s.
- **Recommendation:** BF16 with guided image checks when image judgment matters most, FP8 when speed matters (same
  writeup accuracy, 1.7x faster decode). Avoid INT4 for image judgments, and keep thermal authoritative.

## Test set

`sequences.json` lists 20 eight-frame sequences:

- 16 from Hanna Hammock, Florida (FLAME 3 NADIR), built by `make_sequences.py`;
- 4 from Sycan Marsh (FLAME 3 CV).

8 have thermal frames and 12 are RGB only. The RGB-only set has 7 small-fire sequences with the thermal data held
back, which is the hardest case. Folder paths are relative to `RECON_DATA_ROOT`.

Models never see dataset ids. Frames and sequences are renamed to neutral ids (`S`, `f01`...) before any prompt,
because names like `plot1_preburn_0040` give the answer away.

## Files

| Script | What it does | Results |
|---|---|---|
| `run_briefs.py` | Full briefs per route, with Kai, Clef and Clef-Flash called on the same evidence. `REPEATS`, `RECON_GUIDANCE`, `VERSION` | `results/briefs_v2.jsonl`, `results/runs/v2/` (git-ignored; `python runs_store.py pack v2` packs it into the committed `results/runs_v2.jsonl.gz`) |
| `deciders_with_probes.py` | Decision models given the same fixed image probes the writeup gets | `results/deciders_probes_v2.jsonl` |
| `frame_eval.py` | Yes/no fire check per model on 240 frames (221 on the `SEED=1` sample): LLM routes from answer-letter log-probs, Clef from its yes probability. AUC on log-odds; `GUIDED`, `SEED`, `REP` | `results/frames_v2b.jsonl`, `results/frames_v2b_*.csv` |
| `throughput.py` | Streaming decode speed, 1 and 8 requests at once | `results/throughput.jsonl` |
| `summarize_v2.py` | All tables. Re-applies the current policy to every saved raw brief | `results/summary_v2.json` |
| `build_report.py` | The HTML report, from `results/`, `sequences.json`, `report_text.json`, the guidance text in `src/recon_gateway/prompts.py` and the full-burn report's styles | `report/gateway_model_trials.html` |
| `run_all.sh` | Every step above in order | |

## Reproduce

The tables and report rebuild offline from the committed results, byte for byte:

```bash
python summarize_v2.py
python build_report.py
```

A full rerun needs:

- the gateway proxy with every route in `run_all.sh` (see `gateway/` and `serving/`);
- the FLAME 3 data under `RECON_DATA_ROOT`;
- the Florida frames prepared by `../full-burn-benchmark/prep_florida.py` and `make_sequences.py`.

Then run `./run_all.sh`. It sources `gateway_env.sh`, which reads the proxy key from
`gateway/litellm/.env.local`.

A rerun takes several hours, mostly the 600 briefs (25 to 46 s each, one at a time per route). Each name in a
`frame_eval.py` command also sends one request at a time, because vLLM log-probs can shift with batch composition.
