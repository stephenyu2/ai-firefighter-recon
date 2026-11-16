# Real frames for the replay console

The drone frames behind the five demo sequences, written by `../export_frames.py`. These are the same 640 x 512
RGB images the pipeline was given, in the same order, with the same neutral ids (`f01` to `f08`) that `data.js`
uses in `frames[].id`.

    frames/<sequence_id>/f01.jpg          RGB frame
    frames/<sequence_id>/f01_hot.png      thermal pixels at or above 80 C, transparent elsewhere, same size
    frames/<sequence_id>/f01_truth.png    same, for sequences whose thermal was withheld from the pipeline
    frames/manifest.json                  paths and measured hot area per frame
    frames/manifest.js                    the same content as window.FRAMES, for pages opened from disk

An overlay exists only when the frame has at least one pixel at or above 80 C, so the takeoff sequence has none.

## Using them in the page

Load `frames/manifest.js` before `app.js`, then look a frame up by sequence id and index:

    var fr = window.FRAMES.sequences[seq.sequence_id].frames[i];
    fr.rgb            // "frames/FL-A_plot1_burn_thermal/f01.jpg"
    fr.overlay        // "frames/FL-A_plot1_burn_thermal/f01_hot.png" or null
    fr.truth_overlay  // set only on the RGB-only sequences, see below
    fr.thermal        // { hot_percent, max_c, centroid: [x, y], bbox: [x0, y0, x1, y1], seen_by_pipeline } or null

Draw the JPEG to fill the canvas and draw the PNG on top at the same size. Both are 640 x 512, so they line up
without any offset. `thermal.centroid` is the measured center of the hot area (normalized, origin top-left), so
the hot area no longer has to be placed from the brief's location word.

## The two RGB-only sequences

`FL-G_plot2_smallfire_rgb` and `FL-H_plot2_smallfire_rgb` have thermal images in the dataset, but the pipeline was
not given them. That was the test: small fires with RGB only. Their overlays are named `_truth.png` and their
`thermal.seen_by_pipeline` is `false`. They show where the fire was. Label them as ground truth if they are shown,
because no model saw them.

## Checks

For every frame the pipeline saw with thermal, the hot-area percent and hottest pixel in `manifest.json` match the
values already in `data.js` (`thermal_hot_percent`, `thermal_max_c`).

## Source and credit

- `SY-2_5`: FLAME 3 Computer Vision Subset (Sycan Marsh, Oregon).
- All `FL-` sequences: FLAME 3 NADIR thermal plot subset (Hanna Hammock, Florida).

Both are from the FLAME 3 dataset. The images belong to its authors and are included here only as a small sample
for this course demo. If you use them, cite:

Hopkins, B., ONeill, L., Marinaccio, M., Rowell, E., Parsons, R., Flanary, S., Nazim, I., Seielstad, C., Afghah, F.
(2024). FLAME 3 Dataset: Unleashing the Power of Radiometric Thermal UAV Imagery for Wildfire Management.
arXiv:2412.02831.
