# Recon Replay console

Static front end for the AI Firefighter demo. Drone imagery goes through the
offline pipeline (detector, Molmo 2, thermal reference); the pipeline writes one
JSON file per sequence plus overlay PNGs; this page only reads those files and
replays them. There is no live GPU server, so the demo cannot break mid-run. It
is a replay of pipeline output, which matches the "in-flight feed to brief"
pitch without pretending to run inference on stage.

## Run it

No build step and no server needed. Open `index.html` in a browser, or serve the
folder:

    python3 -m http.server --directory webapp 8000
    # then open http://localhost:8000

Deploys as-is to Netlify (publish directory `webapp/`).

## How it is wired right now

The page reads `window.DEMO`, which is loaded from `data.js`. While the backend
is still being built, `make_mock_data.py` stands in for the real export and
writes example data in the exact contract below:

    python3 webapp/make_mock_data.py

That writes `data/index.json`, `data/sequences/<id>.json`, and the bundled
`data.js`. The four mock sequences each exercise a different case: a clear
growing fire, a small fire at the frame edge, a smoke-obscured fire where the
correct answer is "indeterminate", and a clean no-fire scene.

The frame images here are placeholders drawn from the numbers, not real drone
stills, so the page runs with zero assets. When real `overlay_png` paths are
present the viewer can show those instead.

## The contract (what the backend produces per sequence)

This is the only coupling between front end and pipeline. Keep it stable. Match
it and the page needs no changes. The split that matters: `frames` holds the
shared per-frame MEASUREMENTS (detector + thermal), and `models` holds each
model's CONCLUSION about the sequence. Switching models in the UI swaps the
brief and the agreement badge, not the measurements.

    sequence_id        stable id, also the file name
    display_name       shown in the picker
    case               clear_growing_fire | small_edge_fire | smoke_obscured | no_fire
    source             { dataset, flight, has_thermal, frame_interval_sec,
                         pose_stable, gsd_m_per_px }
                       pose_stable: from the EXIF pose check. If false, direction
                         is frame-relative only, never a compass bearing.
                       gsd_m_per_px: ground sample distance. null when altitude
                         and FOV are not logged. When null, radius stays in
                         percent of frame and meters are not reported. Do not
                         invent meters.
    frames             ordered per-frame measurements, shared by all models:
                         index, t_sec, clock, detection_score (0..1),
                         fire (bool), radius_frac (0..1 of frame),
                         radius_m (number or null),
                         centroid ([x, y] normalized, origin top-left, or null),
                         overlay_png (path, or null)
    thermal_reference  the Tier 2 thermal-derived reference conclusion, the
                       yardstick every model is scored against:
                         { available, fire_present, trend, direction, angle_deg }
    models             map of model_id -> that model's conclusion:
                         label, family,
                         brief { presence, extent,
                                 direction { value, frame_relative, compass,
                                             angle_deg },
                                 trend (growing | stable | receding |
                                        indeterminate | not applicable),
                                 confidence (high | medium | low),
                                 limitations [string] },
                         size_estimate_frac  final predicted radius as a
                                 fraction of frame, or null if the model cannot
                                 estimate it,
                         agreement { presence, trend, direction }  (vs thermal),
                         reference  true for the deterministic thermal baseline,
                                 which is derived from the same thermal masks as
                                 thermal_reference. It is the truth, not an
                                 independent model, so the page draws it stepped
                                 (no interpolated estimate) and does not score it
                                 against itself.
    default_model      model_id to show first

    angle_deg is a FRAME-RELATIVE bearing: degrees clockwise from the top of the
    frame (top 0, right 90, bottom 180, left 270). It is not a compass bearing;
    that needs the drone heading, so compass stays null until heading is logged.
    value is the coarse word for the same angle.

Extent on the card is the measured `radius_frac` at the current frame, so it
fills in as the replay plays. Direction, trend, confidence, and limitations come
from the selected model. The badge compares the selected model's `agreement` to
`thermal_reference`.

Per-frame vs summary: `frames[]` is measured every frame (truth). A model's
size and direction are a single summary for the whole sequence, because a VLM is
asked once, not per frame. The page shows the model's claim as a white blob that
travels out along the heading arrow and grows to `size_estimate_frac`. That
motion between the start and the final size is a demo interpolation of the
summary, and the arrow length is a chosen look, since the model states a
direction and a final size but not a per-frame path or a distance. Only add real
per-frame model geometry (a `track` array) if a model actually produces it, for
example a segmentation detector.

Adding a model (Qwen-VL, Gemma, etc.) is one more entry under `models`. The UI
builds the selector from whatever ids are present.

## Where each field comes from (hand this to the pipeline team)

Checked against the current scripts on the dataset-prep branch (`baselines/`,
`eval/`) and the Milestone 2 protocol. Nothing here asks for a new capability
except one optional field.

Already produced by the current pipeline:
- `frames[].detection_score`, `frames[].fire` — Tier 1 detector, per frame.
- `frames[].radius_frac`, `frames[].centroid` — thermal mask per frame (80 C
  threshold). The mask gives area and centroid; radius is sqrt(area / pi)
  normalized to the frame. Send area or radius, the conversion is trivial.
- `thermal_reference.trend`, `.direction` — mask_tracking_baseline (first-to-last
  area change for trend; 8 direction bins plus "no clear direction").
- `models[].brief.trend`, `.direction` — molmo2_baseline returns exactly
  {trend, direction}. Direction is the same 8 bins, already defined
  frame-relative (north = top of frame).
- `models[].agreement` — compare the model's trend/direction to thermal (or let
  the page compute it).

Derived, no new model work, just formatting or logic:
- `brief.presence` — from the Tier 1 fire/no-fire result. Send a `fire_present`
  bool; the page writes the sentence.
- `brief.extent` — from `radius_frac`. The prose is page formatting.
- `brief.confidence` — per the deployment design: agreement between Molmo and
  thermal (disagree or unparseable = low).
- `brief.limitations` — authored or templated (smoke, edge, parse failure), not
  a model output.
- `direction.angle_deg` — TRUTH: computed from the centroid displacement, free.
  MODEL: Molmo emits only an 8-way bin, so the model angle is the bin center
  (NW = 315, etc.). Do not advertise sub-bin precision for a model.

One real ask, and it is optional:
- `models[].size_estimate_frac` — Molmo does NOT currently output a size. To
  fill it, add one size line to the Molmo prompt (a coarse estimate). If the
  team would rather not, drop it and the page shows the thermal size only.

Demo-only, do NOT ask the backend for these:
- the model blob's position and size between its start and its final size (that
  motion is interpolation),
- the arrow length,
- the placeholder terrain (real frames are the drone RGB plus `overlay_png`).
  The truth path itself is real (per-frame centroids); only the mock's wobble is
  synthetic.

Label mapping the page handles itself:
- direction bins N/NE/.../NW and "no clear direction" map to the UI phrases and
  "indeterminate".
- trend "unparseable" maps to "indeterminate"; no-fire maps to "not applicable".

## Switching from mock data to the pipeline

When the pipeline writes real per-sequence JSON, point the loader at it instead
of the bundled `data.js`: fetch `data/index.json` and each
`data/sequences/<id>.json` at startup and assign the same `window.DEMO` shape.
Nothing else in the page changes.
