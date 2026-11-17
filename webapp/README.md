# Recon Replay console

Static front end for the AI Firefighter demo. The offline pipeline (DINOv2
detector, CLIP, thermal tools, the d3 decision cascade, Qwen3.8-27B) runs on drone frames and
saves its output; this page only reads that saved output and replays it. There
is no live GPU server, so the demo cannot break mid-run. It is a replay of real
pipeline output, which matches the "in-flight feed to brief" pitch without
pretending to run inference on stage.

## Run it

No build step and no server needed. Open `index.html` in a browser, or serve the
folder:

    python3 -m http.server --directory webapp 8000
    # then open http://localhost:8000

Deploys as-is to Netlify (publish directory `webapp/`).

## Where the data comes from

The page reads `window.DEMO`, loaded from `data.js`. Two generators write that
file in the same shape:

- `export_demo.py` (real): reads the committed result files from the real runs
  and writes the per-sequence JSON. This is what ships.

      python3 webapp/export_demo.py

  Sources (both committed in the repo):
  - `experiments/decision-cascade/results/cascade.jsonl`: the d3-first cascade
    on the 20 test sequences: what d3 answered, its probabilities, whether the
    cascade accepted that call or escalated to Qwen3.8-27B, the final status and
    timing.
  - `experiments/gateway-model-trials/results/runs_v2.jsonl.gz`: the full
    Qwen3.8-27B BF16 pipeline run on the same sequences: per-frame DINOv2 /
    CLIP / thermal measurements, hot-area trend, and the structured brief.

  The two are joined by sequence id. `SELECTED` at the top of the script picks
  which sequences appear and in what order.

- `make_mock_data.py` (synthetic, fallback): the old stand-in that writes
  invented sequences in the same shape. Kept for offline UI work when the result
  files are not present.

The viewer draws the real drone stills, bundled under `frames/` and indexed by
`frames/manifest.js`, with the thermal hot-area PNG registered on top (see
`frames/README.md`). Every number shown (measurements, cascade probabilities,
brief) is real. If an image has not loaded yet the frame falls back to a
drawn-from-the-numbers stand-in, so the page never shows blank.

Qwen's planner tool calls on the escalated sequences are in `tool_calls.js`,
pulled from `experiments/decision-cascade/results/runs_v3.jsonl.gz` (label
`cascade-d3`): the `decide` / `finish` calls and the probe answers. Only
sequences that fall under d3's 0.80 confidence bar reach Qwen, so only those
carry tool calls. They feed the written brief.

## What the page shows per sequence

1. **Decision pipeline (step through it).** Press "Begin analysis", then "Next
   step", to walk the real routing one node at a time. The same five nodes show
   every time; only the gate branch differs:
   - **Measure** takes three readings on every frame: the DINOv2 detector, CLIP,
     and the deterministic 80 C thermal hot-area. Only the thermal reading feeds
     the decision; DINOv2 and CLIP are evidence for the writeup.
   - **d3** reads all frames at once and returns a fire status
     (fire / smoke only / none / unclear) with a confidence.
   - **Confident?** if d3's top probability clears 0.80 and is not "unclear",
     the cascade takes that call and skips the LLM. Otherwise it escalates to a
     deeper **Qwen 3.8-27B** pass that writes the brief.
2. **Per-frame measurements.** DINOv2 p(fire), CLIP p(fire) and, when thermal is
   present, the hot-area percent and hottest pixel, replayed on a scrubber, a
   chart, and toggleable chips over the frame.
3. **Result.** Faithful to the cascade:
   - **d3 decided:** a short d3 result (its call, confidence and timing). The
     fast path runs no LLM, so no brief is shown.
   - **escalated:** the structured Qwen 3.8-27B brief: fire/smoke status with
     confidence and evidence, extent (with the measured hot-area percent),
     location in frame, trend, overall confidence, observations, inferences,
     limitations, and a summary.

Spread direction is intentionally not reported. On this data the frame interval
is too short and the hot-area center shift stays inside the camera-stability
margin, so the pipeline returns "indeterminate"; it is left out of the demo
rather than shown as a permanent "indeterminate".

## The contract (what `data.js` holds)

Per sequence:

    sequence_id        stable id, also the file name
    display_name       shown in the picker
    case               scene category (derived)
    truth              ground-truth label: fire | no_fire
    source             { dataset, site, flight, has_thermal, n_frames, span_s,
                         frame_interval_sec, llm_model }
    frames             ordered per-frame measurements:
                         index, id, t_sec, clock,
                         detector_p_fire (0..1), clip_p_fire (0..1),
                         thermal_hot_percent (or null), thermal_max_c (or null),
                         fire_touches_edge (bool or null),
                         fire (DINOv2 call, bool),
                         radius_frac (0..1, from thermal hot area, or null),
                         centroid ([x,y] normalized, or null), overlay_png
    thermal_reference  { available, fire_present, trend }
    cascade            { decider, decider_label, accept_p,
                         decider_status, decider_top_p, status_probabilities,
                         p_fire_visible, decider_seconds,
                         path (decision | escalated), escalated (bool),
                         llm_label, final_status, total_seconds,
                         correct (final matches truth) }
    brief              { presence, presence_confidence, presence_evidence,
                         extent, hot_area_percent (or null), location_in_frame,
                         trend, trend_basis, overall_confidence,
                         limitations[], observations[], inferences[], summary }

## Display approximations (not measurements)

A few on-screen elements are drawn for the demo and are not pipeline outputs:

- The frame background (a thermal-style field with HUD framing, since no real
  drone stills are bundled). Real frames are the drone RGB plus `overlay_png`
  when present.
- The hot-blob position. The pipeline does not store a per-frame centroid, so
  `export_demo.py` places the blob from the brief's `location_in_frame` word.
  The blob size is real (from the measured hot-area share).

## Faithful to the cascade

The fast path does not run the LLM, so a d3-decided sequence shows only d3's
call, confidence and timing, no brief. Only escalated sequences, where Qwen 3.8
actually ran, show a written brief. Nothing is borrowed from another run and
nothing is invented: the brief appearing is itself the signal that the case was
hard enough to escalate.
