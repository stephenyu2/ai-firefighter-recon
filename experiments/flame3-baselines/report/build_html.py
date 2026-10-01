"""Assemble the shareable results page: report/flame3_results.html (standalone) + report/artifact.html (fragment)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "report"))
import numpy as np
import pandas as pd

import charts
from build_lib import E, FIGS, chip, f3, img_data_uri, mmss, pct, rd, table, train_logs, vlm_wallclock
from common import load_index

df = load_index()
T, CMP, D = rd("report_table.csv"), rd("report_purge_compare.csv"), rd("report_dayholdout.csv")
FS, RS, LK, LAT = rd("report_fewshot.csv"), rd("report_randomsplit.csv"), rd("leak_check.csv"), rd("latency.csv")
TI = T.set_index("key")
row = lambda k: TI.loc[k] if k in TI.index else None
VERS = (ROOT / "results/versions.txt").read_text().strip() if (ROOT / "results/versions.txt").exists() else ""
n_models = len(T)
sec = []

# ---------- header ----------
TMIN, TMAX = float(df.t_min.min()), float(df.t_max.max())
pos = lambda t: 100 * (t - TMIN) / (TMAX - TMIN)
lo, hi = df[df.y == 0].t_max.max(), df[df.y == 1].t_max.min()
ticks = [(TMIN, f"{TMIN:.1f}", "first"), (lo, f"{lo:.0f}", ""), (hi, f"{hi:.0f}", ""), (187.4, "187", ""), (TMAX, f"{TMAX:.1f} C", "last")]
sec.append(f"""<header>
<p class="eyebrow">CIS 5980 Wildfire Recon &middot; Modeling workstream &middot; Spencer Theissen-Vang &middot; Sept 30, 2026</p>
<h1>FLAME 3 Fire Detection Baselines</h1>
<p class="lede">I tested {n_models} ways to answer one question about each drone frame: is there fire? They range from a one-line temperature rule to off-the-shelf vision models, vision-language models, and models fine-tuned on FLAME 3 with LoRA. Everything ran on a MacBook (Apple M5 Pro GPU). This page covers the data, the results, how fast each model runs, and whether the method holds up.</p>
<div class="tscale" aria-label="Temperature scale of this dataset">
<div class="bar"></div><div class="ticks">{''.join(f'<span class="{c}" style="left:{pos(t):.2f}%">{E(l)}</span>' for t, l, c in ticks)}</div></div>
<p class="cap">Temperature range of the thermal camera in this set, from {TMIN:.1f} C to {TMAX:.1f} C. Every no-fire frame peaks at or below {lo:.0f} C, every fire frame reaches at least {hi:.0f} C, and the October 25 flights cap at 187 C.</p>
<nav class="toc" aria-label="Sections"><a href="#findings">Findings</a><a href="#data">Data</a><a href="#results">Results</a><a href="#speed">Run times</a><a href="#method">Methodology</a><a href="#predictions">Week 4 predictions</a><a href="#next">Next steps</a><a href="#reproduce">Reproduce</a></nav>
</header>""")

# ---------- findings ----------
b1, z1, z3, l2, v1 = row("B1_maxtemp_rule"), row("Z1_clipL14_zeroshot_rgb"), row("Z3_firevit_edbianchi_rgb"), row("L2_firevit_lora_rgb"), row("V1_molmo2_4b")
F = []
F.append(f"<strong>Whole-frame fire vs no fire is close to solved on this subset.</strong> The hottest pixel alone gets {b1.recall:.3f} recall with {int(b1.fp)} false alarms. Zero-shot CLIP on RGB, with no training on our data, gets {z1.recall:.3f} recall with {int(z1.fp)} false alarms.")
F.append(f"<strong>The labels come from temperature.</strong> No-fire frames peak at {lo:.0f} C and fire frames start at {hi:.0f} C, so thermal models are close to perfect by construction. The informative tests are RGB-only models and the two hard slices: fire frames from the flights that also have no-fire frames, and very small fires.")
l1r, l1t = row("L1_dinov2b_lora_rgb"), row("L1_dinov2b_lora_thermal")
if l2 is not None and l1r is not None:
    F.append(f"<strong>The backbone matters more than the adapter.</strong> LoRA lifts a public fire classifier, trained on ground-level photos, from {z3.recall:.2f} to {l2.recall:.2f} recall, but it adds {int(l2.fp)} false alarms and still catches only {l2.mixed:.0%} of the mixed-flight fires. DINOv2 with the same LoRA setup reaches {l1r.recall:.3f} recall with {int(l1r.fp)} false alarms" + (f", and on thermal it makes no mistakes at all." if l1t is not None and int(l1t.fp) + int(l1t.fn) == 0 else "."))
if len(D):
    DI = D.set_index("key")
    probes = D[(D.trained_on == "Oct 25-26 only") & D.key.str.startswith("P") & (D.input == "RGB")]
    keep = [k for k in ["L1_dinov2b_lora_rgb", "F1_resnet50_fullft_rgb"] if k in DI.index]
    if len(probes) and keep and "Z1_clipL14_zeroshot_rgb" in DI.index:
        F.append(f"<strong>New flights separate the trained models.</strong> Trained only on October 25 and 26 and tested on the six October 27 flights, the RGB linear probes fall to {probes.recall.min():.2f} to {probes.recall.max():.2f} recall. DINOv2 with LoRA ({DI.loc['L1_dinov2b_lora_rgb'].recall:.3f}) and the fully fine-tuned ResNet-50 ({DI.loc['F1_resnet50_fullft_rgb'].recall:.3f}) hold up, and so does zero-shot CLIP ({DI.loc['Z1_clipL14_zeroshot_rgb'].recall:.3f}).")
rs_, f1p = (RS.set_index("key").iloc[0] if len(RS) else None), row("F1_resnet50_fullft_rgb")
if rs_ is not None and f1p is not None:
    F.append(f"<strong>A random split would have flattered us.</strong> Splitting frames at random, ResNet-50 appears to catch {rs_.mixed:.0%} of the mixed-flight fires. With time-based folds and a 30-second buffer it catches {f1p.mixed:.0%}. Neighboring frames are near-copies, so a random split mostly measures how well a model remembers frames it has already seen.")
if v1 is not None:
    F.append(f"<strong>Vision-language models rank frames well but answer badly.</strong> Molmo 2 4B answers yes on only {v1.recall:.0%} of fire frames, yet its yes-probability ranks frames with ROC-AUC {v1.roc_auc:.3f}. A threshold set on other frames lifts recall to {v1.t_recall:.2f} at {v1.t_spec:.2f} specificity. Use the probability, not the text answer." + (
        f" Molmo 2 8B ranks every frame correctly (ROC-AUC {row('V2_molmo2_8b').roc_auc:.3f}); calibrated, it reaches {row('V2_molmo2_8b').t_recall:.2f} recall with {row('V2_molmo2_8b').t_spec:.2f} specificity, while its literal answer still misses {1 - row('V2_molmo2_8b').recall:.0%} of fire frames."
        if row("V2_molmo2_8b") is not None else ""))
F.append("<strong>Dense smoke blinds RGB.</strong> In frames engulfed by smoke, the RGB image is a flat gray wall and the RGB models say no fire. The thermal TIFF shows tens of thousands of burning pixels in the same frames. The detector should be thermal-first.")
if len(LAT):
    lat = LAT.assign(spm=60 * LAT.ms_median / 1000)
    ok = lat[lat.spm <= 30]
    F.append(f"<strong>Speed.</strong> On this laptop, {len(ok)} of {len(lat)} benchmarked models stay under our 30 seconds per minute of footage budget at 1 frame per second. The fastest is the temperature rule at {lat.ms_median.min():.1f} ms per frame.")
drop = (CMP.mixed_grouped - CMP.mixed_purged) if len(CMP) else pd.Series(dtype=float)
big = CMP.assign(d=drop).sort_values("d", ascending=False).head(2) if len(CMP) else CMP
fix_tail = (" Most trained models moved by a point or two, but " + " and ".join(f"{r.model} lost {100 * r.d:.0f} points" for r in big.itertuples())
            + " of mixed-flight recall, so they had been leaning on near-duplicate frames.") if len(big) else ""
F.append("<strong>The method needed two fixes, and I reran everything they touched.</strong> Adjacent frames 3 seconds apart could land on both sides of a split, so trained models now drop training frames within 30 seconds of any test frame. I also added a full-day holdout." + fix_tail)
sec.append('<section id="findings"><h2>Key findings</h2><ul class="findings">' + "".join(f"<li>{x}</li>" for x in F) + "</ul></section>")

# ---------- data ----------
nf_flights = sorted(df[df.y == 0].flight.unique())
sec.append(f"""<section id="data"><h2>The data</h2><div class="prose">
<p>The open FLAME 3 computer-vision subset is one prescribed burn at Sycan Marsh, Oregon, captured October 25 to 27, 2022 with a DJI Matrice 30T (dates from the photo metadata). It holds {len(df)} image sets: {int(df.y.sum())} labeled fire and {int((df.y == 0).sum())} labeled no fire. Each set has a 4000x3000 RGB photo, a 640x512 RGB crop aligned to the thermal camera, a colorized thermal JPG, and a thermal TIFF with a temperature in Celsius for every pixel. Labels are folder names only. There are no masks or boxes.</p>
</div>
<figure><p class="fig-title">The hottest pixel separates the labels with room to spare</p>{charts.maxtemp_hist(df)}
<figcaption>Each bar counts frames by their hottest pixel. Nothing peaks between {lo:.0f} and {hi:.0f} C. The spike at 187 C is the October 25 flights, where the camera saturated at a lower ceiling.</figcaption></figure>
<figure><p class="fig-title">All {int((df.y == 0).sum())} no-fire frames come from {len(nf_flights)} of {df.flight.nunique()} flights</p>{charts.flights(df)}
<figcaption>Flights are separated by gaps of more than 60 seconds between frames. Only flights {" and ".join(str(f) for f in nf_flights)} contain both labels, so a model could learn which flight it is looking at instead of whether there is fire. The two hard slices below test that.</figcaption></figure>
<figure><p class="fig-title">Masks from the thermal TIFF: 80 C works, 200 C does not</p><img src="{img_data_uri(FIGS / 'fig3_mask_overlays.png')}" alt="Five example frames with RGB, thermal, and temperature masks at 80 and 200 C">
<figcaption>Columns: RGB crop, thermal JPG, pixels at or above 80 C, pixels at or above 200 C. The 200 C mask is empty for October 25 frames because of the 187 C ceiling. The thermal JPG is rescaled per image, so a cold no-fire scene renders bright yellow.</figcaption></figure>
<h3>Things that change how we should use this data</h3>
<ul class="plain">
<li>File numbers are shuffled relative to capture time. Splitting by file number is a random split, so splits must use the photo timestamps.</li>
<li>The thermal JPG is normalized per image and loses absolute temperature. Its overall darkness alone ranks fire vs no fire with ROC-AUC {f3(row("B2_thermaljpg_darkness").roc_auc) if row("B2_thermaljpg_darkness") is not None else ""}, which is an artifact, not a signal. Model on the Celsius TIFF instead.</li>
<li>Masks should use a cut near 80 C. The paper's 200 C flaming heuristic gives empty masks for every October 25 frame.</li>
<li>Our proposal says FLAME 3 is from Arizona. The open subset is Oregon. Arizona is FLAME 2 and three of the burns that are not public.</li>
<li>The Kaggle mirror lists an MIT license. IEEE DataPort's terms say CC BY. Either way, cite the paper and the DataPort DOI.</li>
</ul></section>""")

# ---------- results ----------
TIER_ORDER = ["no-learning baseline", "off the shelf, no FLAME training", "trained on FLAME 3"]
TIER_LABEL = {"no-learning baseline": "No-learning baselines", "off the shelf, no FLAME training": "Off the shelf (no training on FLAME 3)",
              "trained on FLAME 3": "Trained on FLAME 3 (purged cross-validation)"}
Ts = T.assign(o=T.tier.map({t: i for i, t in enumerate(TIER_ORDER)})).sort_values(["o", "mixed"], ascending=[True, False])
skip_chart = {"B2_thermaljpg_darkness", "Z1_clipL14_zeroshot_thermal", "Z2_siglip2_zeroshot_thermal", "P3_dinov2b_probe_fused"}
crow = [dict(group=TIER_LABEL[r.tier], label=r.model, sub="" if r.input == "-" else r.input, mixed=r.mixed, small=r.small) for r in Ts.itertuples() if r.key not in skip_chart]
n_mixed = int(((df.y == 1) & df.mixed_flight).sum()); n_small = int(((df.y == 1) & (df.frac_gt80 < 0.0025)).sum())
chart_slices = charts.slice_bars(crow, [("mixed", "Fire frames in the mixed flights", f"{n_mixed} frames from the 2 flights that also have no-fire frames"),
                                        ("small", "Small fires", f"{n_small} frames with under 0.25% of pixels above 80 C")])
main_rows = [[E(r.model), E(r.input), r.recall, r.precision, r.specificity, r.roc_auc, r.mixed, r.small, str(int(r.fp)), str(int(r.fn))] for r in Ts.itertuples()]
main_tbl = table(["Model", "Input", "Recall", "Precision", "Specificity", "ROC-AUC", "Mixed flights", "Small fires", "False alarms", "Missed fires"],
                 main_rows, num=range(2, 10), groups=[TIER_LABEL[r.tier] for r in Ts.itertuples()])
tuned_rows = [[E(r.model), E(r.input), r.t_recall, r.t_spec, r.t_prec, r.t_mixed] for r in Ts.itertuples() if r.tier != "no-learning baseline"]
tuned_tbl = table(["Model", "Input", "Recall", "Specificity", "Precision", "Mixed flights"], tuned_rows, num=range(2, 6),
                  groups=[TIER_LABEL[r.tier] for r in Ts.itertuples() if r.tier != "no-learning baseline"])
day_html = ""
if len(D):
    Dd = D.assign(o=D.tier.map({t: i for i, t in enumerate(TIER_ORDER)})).sort_values(["o", "recall"], ascending=[True, False])
    day_html = table(["Model", "Input", "Trained on", "Recall", "Recall, tuned threshold", "Small fires"],
                     [[E(r.model), E(r.input), E(r.trained_on), r.recall, r.recall_tuned, r.small] for r in Dd.itertuples()],
                     num=range(3, 6), groups=[{**TIER_LABEL, "trained on FLAME 3": "Trained on FLAME 3 (October 25 and 26 only)"}[r.tier] for r in Dd.itertuples()])
WC = vlm_wallclock()
SIZE = {"V1_molmo2_4b": "4B", "V2_molmo2_8b": "8B", "V3_qwen35_0p8b": "0.8B", "V4_qwen35_2b": "2B", "V5_qwen35_4b": "4B", "V6_qwen35_9b": "9B",
        "V7_qwen3vl_8b": "8B", "V8_gemma4_e4b": "E4B (4B effective)"}
V = Ts[Ts.key.str.startswith("V")]
_have = set(V.key)
VLM_NOTE = "Gemma 4 12B was left out because Gemma 4 E4B alone used 35 GB of the laptop's 48 GB of shared memory. " + (
    "Qwen3.5 4B first ran next to training jobs, pushed the laptop into swap, and slowed to about 38 seconds per frame. Run alone it takes under a second, so it was rerun cleanly." if "V5_qwen35_4b" in _have else
    "Qwen3.5 4B was stopped after it pushed the laptop into swap while running next to training jobs.")
vlm_tbl = table(["Model", "Size", "ROC-AUC", "Recall, literal", "Specificity, literal", "Recall, calibrated", "Specificity, calibrated", "Full run"],
                [[E(r.model), SIZE.get(r.key, ""), r.roc_auc, r.recall, r.specificity, r.t_recall, r.t_spec, mmss(WC[r.key][0]) if r.key in WC else ""] for r in V.itertuples()],
                num=range(2, 8))
few_html = ""
if len(FS):
    F2 = FS.copy(); F2["base"] = F2.key.str.replace(r"_seed\d", "", regex=True).str.replace("_PURGED", "", regex=False).str.replace("_16shot", "", regex=False)
    agg = F2.groupby("base").agg(seeds=("key", "size"), recall=("recall", "mean"), spec=("specificity", "mean"), mixed=("mixed", "mean"), small=("small", "mean")).reset_index()
    def full(b):
        k = b.replace("_PURGED", "")
        return TI.loc[k] if k in TI.index else None
    fr = []
    for r in agg.itertuples():
        fu = full(r.base)
        name = (fu.model + " (" + fu.input + ")") if fu is not None else r.base
        fr.append([E(name), str(r.seeds), r.recall, fu.recall if fu is not None else float("nan"), r.mixed, fu.mixed if fu is not None else float("nan"), r.spec])
    few_html = table(["Model", "Seeds", "Recall, 16 per class", "Recall, all data", "Mixed flights, 16 per class", "Mixed flights, all data", "Specificity, 16 per class"], fr, num=range(1, 7))
sec.append(f"""<section id="results"><h2>Results</h2><div class="prose">
<p>Every model scores the same {len(df)} frames. Trained models are scored with 5-fold cross-validation, so each frame is predicted by a model that never trained on it. Recall is the share of fire frames a model flags, specificity is the share of no-fire frames it correctly leaves alone, and ROC-AUC measures how well its scores rank fire above no fire at any threshold.</p></div>
<figure><p class="fig-title">Where the models differ: the hardest fire frames</p>{chart_slices}
<figcaption>Hover or tab to a bar for its value. The full table follows. The thermal rule and the trained models stay near the right edge, while the off-the-shelf fire classifiers and the literal Molmo 2 answers miss most of these frames.</figcaption></figure>
<h3>All models at their default threshold</h3>{main_tbl}
<p class="cap">Default threshold is 0.5 on the model's fire probability, or the model's own answer. Trained rows use purged cross-validation; the unpurged numbers are in the methodology section.</p>
<h3>At a recall-tuned threshold</h3><div class="prose"><p>Here each model's threshold is chosen on training or validation frames to keep recall at or above 0.98, then applied to the held-out frames. This is how we would set the operating point in the real system.</p></div>{tuned_tbl}
<h3>New flights: train on October 25 and 26, test on October 27</h3><div class="prose"><p>All {int((df.ts.dt.day == 27).sum())} frames from October 27 come from six flights the trained models never saw. That day has no no-fire frames, so this measures recall only.</p></div>{day_html}
<h3>Vision-language models</h3><div class="prose"><p>Each model gets the same prompt, "Is there any fire (flames, or smoke from something burning) visible in this aerial drone image? Answer with only yes or no," and I read its probability of answering yes. The literal answer is yes when that probability is above 0.5. The calibrated columns use a threshold set on other frames. Full run is the wall-clock time for all 738 frames. {VLM_NOTE}</p></div>{vlm_tbl}
<h3>With only 16 labeled frames per class</h3><div class="prose"><p>Averaged over three random draws of 16 fire and 16 no-fire training frames, with the same purged folds.</p></div>{few_html}
<figure class="narrow"><p class="fig-title">What RGB models miss</p><img src="{img_data_uri(FIGS / 'fig4_rgb_hard_fire_frames.png', max_w=900)}" alt="Fire frames missed by most RGB models, with thermal overlays showing the fire">
<figcaption>Fire frames that two or more of the strong RGB models missed, with pixels at or above 80 C in magenta. Two patterns: a thin grass-fire line at the edge of the frame on October 25, and frames fully engulfed in smoke where the RGB image shows nothing.</figcaption></figure>
</section>""")

# ---------- run times ----------
speed_html = '<div class="prose"><p>The clean benchmark runs after all other jobs finish.</p></div>'
if len(LAT):
    L2 = LAT.assign(spm=60 * LAT.ms_median / 1000).sort_values("ms_median")
    nm = lambda k: (TI.loc[k].model, TI.loc[k].input) if k in TI.index else (k, "")
    srows = [dict(label=nm(r.model)[0], sub=nm(r.model)[1], ms=r.ms_median) for r in L2.itertuples()]
    lat_tbl = table(["Model", "Input", "Load time", "Per frame (median)", "Per frame (p90)", "Per minute of footage", "GPU memory", "Within 30 s budget"],
                    [[E(nm(r.model)[0]), E(nm(r.model)[1]), ("under 1 s" if r.load_s < 1 else mmss(r.load_s)) if r.load_s else "none",
                      f"{r.ms_median:.1f} ms", f"{r.ms_p90:.1f} ms", "under 0.1 s" if r.spm < 0.1 else f"{r.spm:.1f} s",
                      "CPU only" if r.gpu_mem_gb == 0 else f"{r.gpu_mem_gb:.1f} GB", chip("Yes", "ok") if r.spm <= 30 else chip("No", "bad")] for r in L2.itertuples()],
                    num=range(2, 7))
    speed_html = (f'<figure><p class="fig-title">Compute per minute of footage on a laptop</p>{charts.speed_dots(srows)}'
                  '<figcaption>Each dot is one model: median latency for one frame, end to end, times 60 frames for a minute of video sampled at one frame per second. The line is the 30 seconds per minute target from our proposal.</figcaption></figure>' + lat_tbl)
TL = train_logs()
tl_html = ""
if len(TL):
    TL = TL[~TL.run.str.contains("_seed")]
    tl_html = table(["Run", "Training frames per fold", "Trainable parameters", "Training time, all folds", "Per fold"],
                    [[E(r.run), str(r.n_train), f"{r.params / 1e6:.2f} M", mmss(r.train_s), mmss(r.per_fold)] for r in TL.itertuples()], num=range(1, 5))
sec.append(f"""<section id="speed"><h2>Run times</h2><div class="prose">
<p>Speed matters for the brief. Our proposal promises a brief in under 30 seconds per minute of footage on one GPU. These numbers come from my laptop (Apple M5 Pro, 48 GB shared memory), measured one frame at a time with nothing else running: read the image, preprocess it, run the model, and wait for the GPU. A cloud GPU would be faster.</p></div>
{speed_html}
<h3>Measured training time</h3><div class="prose"><p>Wall-clock time for the actual runs behind the tables above. Other jobs shared the GPU during most of these, so treat them as upper bounds.</p></div>{tl_html}
</section>""")

# ---------- methodology ----------
near = "83 of 738 test frames (11%)"
CHECK = [
    ("Held-out test data for every trained model", ("Followed", "ok"), "5-fold cross-validation. Each frame is predicted once, by a model that never trained on it."),
    ("Split by capture time, not by frame or file number", ("Followed", "ok"), "File numbers are shuffled, so frames are grouped into 60-second windows from the photo timestamps. No window spans train and test."),
    ("Time buffer between train and test", ("Added after audit", "info"), f"Before the fix, {near} had a training frame 3 to 6 seconds away. Trained models now drop training frames within 30 seconds of any test frame. Every trained model was rerun."),
    ("Whole flights held out", ("Added after audit", "info"), "Day holdout: train on October 25 and 26, test on all frames from the six October 27 flights. Measures recall only, since that day has no no-fire frames."),
    ("Thresholds chosen away from the test fold", ("Followed", "ok"), "Recall-tuned thresholds come from inner validation frames for trained models and from the training folds for zero-shot models."),
    ("No tuning on the test set", ("Mostly", "warn"), "Settings were fixed before running: probe regularization 0.1, LoRA rank 16, learning rate 3e-4, 8 epochs, one prompt per model. I changed the threshold-picking rule once after seeing results because it sat too close to the training data. That affects only the recall-tuned rows."),
    ("Preprocessing fit on training data only", ("Followed", "ok"), "Feature scaling is fit inside each fold. Image normalization uses fixed ImageNet statistics."),
    ("Class imbalance handled", ("Followed", "ok"), "Balanced class weights in every trained model. Results report recall, precision, specificity, and false alarms instead of accuracy, since always saying fire scores 84% accuracy."),
    ("Stratified folds", ("Followed", "ok"), "Every test fold has 20 to 30 no-fire frames."),
    ("No label leakage", ("Verified", "ok"), "Models receive only pixels and a fixed prompt. File paths contain Fire and No Fire but never reach a model. Retraining probes on shuffled labels drops test ROC-AUC to chance (table below)."),
    ("Same frames for every model", ("Followed", "ok"), "Every row scores all 738 frames, so numbers compare directly."),
    ("Uncertainty reported", ("Partly", "warn"), "16-shot runs use three seeds. Full-data runs use one seed per fold. With 116 no-fire frames, one false alarm moves specificity by about 0.009, so small gaps between top models are noise."),
    ("Reproducible", ("Followed", "ok"), "Fixed seeds, pinned package versions, one script per step. Apple GPU kernels are not bit-exact, so reruns can differ slightly."),
    ("Hard slices defined in advance", ("Partly", "warn"), "The mixed-flight slice came from data inspection. The small-fire cut was added after the first zero-shot results. Treat both as diagnostics."),
    ("Generalization beyond one burn", ("Not yet", "bad"), "One burn, and every no-fire frame comes from two flights. Boreal's empty frames would test false alarms on new terrain, but Fairdata needs a browser download."),
    ("Pretraining overlap", ("Unknown", "warn"), "The off-the-shelf models do not publish their training data, so some may have seen FLAME images."),
]
check_tbl = table(["Practice", "Status", "What I did"], [[E(a), chip(*s), E(c)] for a, s, c in CHECK])
leak_tbl = table(["Features", "Test ROC-AUC, real labels", "Test ROC-AUC, shuffled labels (mean of 5)", "Shuffled range"],
                 [[E(r.features), r.real_auc, r.shuffled_auc_mean, f"{r.shuffled_auc_min:.2f} to {r.shuffled_auc_max:.2f}"] for r in LK.itertuples()], num=range(1, 4)) if len(LK) else ""
cmp_tbl = table(["Model", "Input", "Recall, no buffer", "Recall, 30 s buffer", "Specificity, no buffer", "Specificity, 30 s buffer", "Mixed flights, no buffer", "Mixed flights, 30 s buffer"],
                [[E(r.model), E(r.input), r.recall_grouped, r.recall_purged, r.spec_grouped, r.spec_purged, r.mixed_grouped, r.mixed_purged] for r in CMP.itertuples()], num=range(2, 8)) if len(CMP) else ""
rnd_tbl = ""
if len(RS) and "F1_resnet50_fullft_rgb" in TI.index:
    rr, pp = RS.set_index("key").iloc[0], TI.loc["F1_resnet50_fullft_rgb"]
    rnd_tbl = table(["ResNet-50 split", "Recall", "Precision", "Specificity", "Mixed flights", "Small fires"],
                    [["Random frames (leaky)", rr.recall, rr.precision, rr.specificity, rr.mixed, rr.small],
                     ["Time windows + 30 s buffer", pp.recall, pp.precision, pp.specificity, pp.mixed, pp.small]], num=range(1, 6))
sec.append(f"""<section id="method"><h2>Methodology and best practices</h2><div class="prose">
<p>The checklist below covers each practice, whether it was followed, and the evidence. Two gaps turned up in an audit partway through: near-duplicate frames across the split, and flights that appeared on both sides. Both were fixed and every affected model was rerun. The tables on this page use the fixed protocol.</p></div>
{check_tbl}
<h3>Label leakage audit</h3><div class="prose"><p>If any test label leaked into training, a probe trained on shuffled labels would still score well on the test folds. It drops to chance.</p></div>{leak_tbl}
<h3>Effect of the 30-second buffer</h3>{cmp_tbl}
<h3>Random split vs time-based split</h3><div class="prose"><p>What we would have reported with a naive random split of frames.</p></div>{rnd_tbl}
<h3>What this does not show</h3><ul class="plain">
<li>Fire presence per frame only. Masks and spread direction are not evaluated yet.</li>
<li>One prescribed burn in good weather. No night flights, no real wildfire, no heavy smoke from other sources.</li>
<li>One prompt per vision-language model. A different prompt could change their numbers.</li>
<li>Hosted commercial vision models were not run. They need a paid API key.</li>
</ul></section>""")

# ---------- predictions ----------
b0, f1r = row("B0_constant_fire"), row("F1_resnet50_fullft_rgb")
PRED = [
    ("Saying fire for every frame gets recall 1.00, precision 0.84, F1 0.92.", f"{b0.recall:.3f}, {b0.precision:.3f}, {2 * b0.precision * b0.recall / (b0.precision + b0.recall):.3f}", ("Correct", "ok"), "Pure arithmetic from the class balance."),
    ("A rule flagging max temperature above 150 C hits recall 0.98 and precision 0.95.", f"{b1.recall:.3f}, {b1.precision:.3f}", ("Correct", "ok"), "Better than predicted. The one miss is a smoldering frame that peaks at 108 C."),
]
if f1r is not None:
    PRED.append(("ResNet-50 on RGB only, with a time-based split, lands near recall 0.95 and precision 0.88, barely above constant.",
                 f"{f1r.recall:.3f}, {f1r.precision:.3f}", ("Wrong", "bad"), "Far above constant. Within one burn, visible smoke makes RGB easy. The weakness shows up on new flights and in dense smoke instead."))
pred_tbl = table(["Prediction", "Actual (recall, precision)", "Verdict", "Why"], [[E(a), E(b), chip(*c), E(d)] for a, b, c, d in PRED])
sec.append(f'<section id="predictions"><h2>Week 4 predictions, checked</h2>{pred_tbl}</section>')

# ---------- next steps ----------
ms4 = LAT.set_index("model").ms_median.get("V1_molmo2_4b") if len(LAT) else None
molmo_speed = f" On this laptop Molmo 2 4B takes about {ms4:.0f} ms per frame." if ms4 else ""
sec.append(f"""<section id="next"><h2>What this means for each of us</h2><div class="cols">
<div><h3>Data (Caroline)</h3><ul class="plain"><li>Build splits from photo timestamps with a 30-second buffer, not file numbers.</li><li>Make masks from the Celsius TIFF at about 80 C. Do not feed the thermal JPG to models.</li><li>Download Boreal's 256 empty frames through the Etsin web page so we can test false alarms on new terrain.</li></ul></div>
<div><h3>AI ops (Stephen)</h3><ul class="plain"><li>Use Molmo 2's yes-probability with a calibrated threshold, not its text answer.{E(molmo_speed)}</li><li>Expect the same calibration problem in spread-direction prompts, and plan to calibrate those too.</li></ul></div>
<div><h3>Evaluation (Joe)</h3><ul class="plain"><li>Frame-level recall is saturated. Report the mixed-flight slice, the small-fire slice, a held-out day, and false-alarm counts.</li><li>Add harder negatives before we claim anything about precision.</li></ul></div>
<div><h3>Modeling (Spencer)</h3><ul class="plain"><li>Make the detector thermal-first, with RGB as a backup channel.</li><li>Move to segmentation on temperature masks, with DINOv2 plus LoRA as the main fine-tune.</li><li>Always check held-out flights, not just held-out frames.</li></ul></div>
</div></section>""")

# ---------- reproduce + sources ----------
sec.append(f"""<section id="reproduce"><h2>Reproduce</h2><div class="prose"><p>Code and results live in <code>experiments/flame3-baselines</code> in the team repo, and the README there lists every step. Download the FLAME 3 CV subset from Kaggle, point <code>FLAME3_ROOT</code> at the unzipped folder, then run:</p></div>
<pre>cd experiments/flame3-baselines
pip install -r requirements.txt
export FLAME3_ROOT=/path/to/flame3_cv
python index_dataset.py
python run_baselines.py
python run_zeroshot.py
python run_probes.py purged
python run_probes.py day
./run_finetune_queue2.sh
./run_after.sh
python make_report.py
python report/build_html.py</pre>
<p class="cap">{E(VERS)}. Molmo 2 runs in a separate environment pinned to transformers 4.57.1 (requirements-molmo.txt).</p></section>""")
SRC = [("FLAME 3 paper (Hopkins et al., arXiv 2412.02831)", "https://arxiv.org/abs/2412.02831"),
       ("FLAME 3 on IEEE DataPort, DOI 10.21227/w0mz-aq48", "https://ieee-dataport.org/open-access/flame-3-radiometric-thermal-uav-imagery-wildfire-management"),
       ("FLAME 3 CV subset, Kaggle mirror", "https://www.kaggle.com/datasets/brycehopkins/flame-3-computer-vision-subset-sycan-marsh"),
       ("RGB and infrared segmentation on FLAME 3 (Kovaleski et al., arXiv 2609.01390)", "https://arxiv.org/abs/2609.01390"),
       ("Molmo 2 4B model card", "https://huggingface.co/allenai/Molmo2-4B"),
       ("DINOv2 base model card", "https://huggingface.co/facebook/dinov2-base"),
       ("Boreal Forest Fire dataset (Pesonen et al., Scientific Data 2025)", "https://www.nature.com/articles/s41597-025-05634-0")]
sec.append('<footer><p>Sources</p><ul class="plain">' + "".join(f'<li><a href="{u}" target="_blank" rel="noopener">{E(t)}</a></li>' for t, u in SRC) + "</ul></footer>")

# ---------- write ----------
css = (ROOT / "report/style.css").read_text(); js = (ROOT / "report/script.js").read_text()
FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600&family=IBM+Plex+Mono:wght@400;500;600&family=Public+Sans:wght@400;600;700&display=swap">'
head = f"<title>FLAME 3 Fire Detection Baselines</title>\n{FONTS}\n<style>{css}</style>\n"
body = '<div class="wrap">' + "\n".join(sec) + '</div>\n<div id="tip" role="tooltip" hidden></div>\n<script>' + js + "</script>\n"
(ROOT / "report/artifact.html").write_text(head + body)
standalone = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
              '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + head +
              "<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}img{max-width:100%}[hidden]{display:none!important}</style>\n"
              "</head><body>\n" + body + "</body></html>\n")
(ROOT / "report/flame3_results.html").write_text(standalone)
print("wrote report/flame3_results.html", f"{len(standalone) / 1e6:.2f} MB")
