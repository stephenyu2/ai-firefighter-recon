"""Build RESULTS.md tables + the hard-slice summary chart from every scores_*.csv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, RESULTS, binary_metrics, load_index, ranking_metrics

df = load_index().set_index("uid")
MIXED = df.mixed_flight.values
SMALL = (df.y.values == 1) & (df.frac_gt80.values < 0.0025)
# Tier colors: the dataviz reference palette's first three slots, unchanged. The palette doc records this trio
# as passing the all-pairs checks (CVD dE 9.2, normal-vision dE 24.0, light mode); no JS runtime was available
# to re-run the validator here. Aqua is under 3:1 contrast, so every bar is also labeled and a table exists.
TIER_COLOR = {"trained on FLAME 3": "#2a78d6", "off the shelf, no FLAME training": "#eb6834",
              "no-learning baseline": "#1baf7a"}
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"

NAMES = {
    "B0_constant_fire": ("Always say fire", "no-learning baseline", "-"),
    "B1_maxtemp_rule": ("Max temp above 150 C", "no-learning baseline", "thermal TIFF"),
    "B2_thermaljpg_darkness": ("Thermal JPG darkness (artifact)", "no-learning baseline", "thermal JPG"),
    "Z1_clipL14_zeroshot_rgb": ("CLIP ViT-L/14 zero-shot", "off the shelf, no FLAME training", "RGB"),
    "Z1_clipL14_zeroshot_thermal": ("CLIP ViT-L/14 zero-shot", "off the shelf, no FLAME training", "thermal JPG"),
    "Z2_siglip2_zeroshot_rgb": ("SigLIP 2 zero-shot", "off the shelf, no FLAME training", "RGB"),
    "Z2_siglip2_zeroshot_thermal": ("SigLIP 2 zero-shot", "off the shelf, no FLAME training", "thermal JPG"),
    "Z3_firevit_edbianchi_rgb": ("Fire ViT classifier (HF)", "off the shelf, no FLAME training", "RGB"),
    "Z4_forestfire_siglip2_rgb": ("Forest-fire SigLIP 2 classifier (HF)", "off the shelf, no FLAME training", "RGB"),
    "V1_molmo2_4b": ("Molmo 2 4B, yes/no", "off the shelf, no FLAME training", "RGB"),
    "V2_molmo2_8b": ("Molmo 2 8B, yes/no", "off the shelf, no FLAME training", "RGB"),
    "V3_qwen35_0p8b": ("Qwen3.5 0.8B, yes/no", "off the shelf, no FLAME training", "RGB"),
    "V4_qwen35_2b": ("Qwen3.5 2B, yes/no", "off the shelf, no FLAME training", "RGB"),
    "V5_qwen35_4b": ("Qwen3.5 4B, yes/no", "off the shelf, no FLAME training", "RGB"),
    "V6_qwen35_9b": ("Qwen3.5 9B, yes/no", "off the shelf, no FLAME training", "RGB"),
    "V7_qwen3vl_8b": ("Qwen3-VL 8B, yes/no", "off the shelf, no FLAME training", "RGB"),
    "V8_gemma4_e4b": ("Gemma 4 E4B, yes/no", "off the shelf, no FLAME training", "RGB"),
    "V9_gemma4_12b": ("Gemma 4 12B, yes/no", "off the shelf, no FLAME training", "RGB"),
    "P1_dinov2b_probe_rgb": ("DINOv2-B linear probe", "trained on FLAME 3", "RGB"),
    "P2_dinov2b_probe_thermal": ("DINOv2-B linear probe", "trained on FLAME 3", "thermal TIFF"),
    "P3_dinov2b_probe_fused": ("DINOv2-B linear probe", "trained on FLAME 3", "RGB + thermal"),
    "P4_clipL14_probe_rgb": ("CLIP ViT-L/14 linear probe", "trained on FLAME 3", "RGB"),
    "P5_siglip2_probe_rgb": ("SigLIP 2 linear probe", "trained on FLAME 3", "RGB"),
    "L1_dinov2b_lora_rgb": ("DINOv2-B + LoRA", "trained on FLAME 3", "RGB"),
    "L1_dinov2b_lora_thermal": ("DINOv2-B + LoRA", "trained on FLAME 3", "thermal TIFF"),
    "L2_firevit_lora_rgb": ("Fire ViT + LoRA", "trained on FLAME 3", "RGB"),
    "F1_resnet50_fullft_rgb": ("ResNet-50 full fine-tune", "trained on FLAME 3", "RGB"),
}


def metrics(key):
    s = pd.read_csv(RESULTS / f"scores_{key}.csv").set_index("uid").loc[df.index]
    y, sc, td = s.y.values, s.score.values, s.thr_default.values
    d, t, r = binary_metrics(y, sc, td), binary_metrics(y, sc, s.thr_tuned.values), ranking_metrics(y, sc)
    tm = binary_metrics(y[MIXED], sc[MIXED], s.thr_tuned.values[MIXED])
    return dict(recall=d["recall"], precision=d["precision"], specificity=d["specificity"], fp=d["fp"], fn=d["fn"],
                roc_auc=r["roc_auc"], pr_auc=r["pr_auc"],
                mixed=binary_metrics(y[MIXED], sc[MIXED], td[MIXED])["recall"],
                small=binary_metrics(y[SMALL], sc[SMALL], td[SMALL])["recall"],
                t_recall=t["recall"], t_spec=t["specificity"], t_prec=t["precision"], t_mixed=tm["recall"])


avail = {p.stem.replace("scores_", "") for p in RESULTS.glob("scores_*.csv")}
rows, cmp_rows = [], []
for key, (name, tier, inp) in NAMES.items():
    use = key + "_PURGED" if tier == "trained on FLAME 3" and key + "_PURGED" in avail else key
    if use in avail:
        rows.append(dict(key=key, run=use, model=name, tier=tier, input=inp, **metrics(use)))
    if tier == "trained on FLAME 3" and key in avail and key + "_PURGED" in avail:
        a, b = metrics(key), metrics(key + "_PURGED")
        cmp_rows.append(dict(model=name, input=inp, recall_grouped=a["recall"], recall_purged=b["recall"],
                             spec_grouped=a["specificity"], spec_purged=b["specificity"],
                             mixed_grouped=a["mixed"], mixed_purged=b["mixed"]))
T = pd.DataFrame(rows)
CMP = pd.DataFrame(cmp_rows)
CMP.to_csv(RESULTS / "report_purge_compare.csv", index=False)


def day_metrics(key):
    """Recall on every Oct 27 frame (6 flights never used for training in the day holdout)."""
    from common import pick_threshold_for_recall
    d27 = df.ts.dt.day.values == 27
    small27 = d27 & SMALL
    hold = key + "_DAYHOLDOUT"
    src = hold if hold in avail else key
    s = pd.read_csv(RESULTS / f"scores_{src}.csv").set_index("uid").loc[df.index]
    y, sc = s.y.values, s.score.values
    if src == hold:
        thr_t = s.thr_tuned.values
    else:
        early = ~d27
        thr_t = np.full(len(y), pick_threshold_for_recall(y[early], sc[early], 0.98))
    td = s.thr_default.values
    return dict(recall=binary_metrics(y[d27], sc[d27], td[d27])["recall"],
                recall_tuned=binary_metrics(y[d27], sc[d27], thr_t[d27])["recall"],
                small=binary_metrics(y[small27], sc[small27], td[small27])["recall"], trained_on="Oct 25-26 only" if src == hold else "nothing (zero-shot)" if T.set_index("key").loc[key, "tier"] != "trained on FLAME 3" else "")


D = pd.DataFrame([dict(key=r.key, model=r.model, input=r.input, tier=r.tier, **day_metrics(r.key)) for r in T.itertuples()
                  if r.tier != "trained on FLAME 3" or r.key + "_DAYHOLDOUT" in avail])
D.to_csv(RESULTS / "report_dayholdout.csv", index=False)
T.to_csv(RESULTS / "report_table.csv", index=False)


def few(prefix):
    ks = [k for k in avail if k.startswith(prefix) and "16shot" in k and "PURGED" in k]
    return pd.DataFrame([dict(key=k, **metrics(k)) for k in sorted(ks)])


F = pd.concat([few("P1_"), few("P2_"), few("P4_"), few("P5_"), few("L1_"), few("L2_")], ignore_index=True) if avail else None
if F is not None and len(F):
    F.to_csv(RESULTS / "report_fewshot.csv", index=False)
R = pd.DataFrame([dict(key=k, **metrics(k)) for k in sorted(avail) if "RANDOMSPLIT" in k])
if len(R):
    R.to_csv(RESULTS / "report_randomsplit.csv", index=False)

# ---- chart: recall on the two hard slices, default threshold, one bar per model ----
chart_keys = ["B0_constant_fire", "B1_maxtemp_rule", "Z3_firevit_edbianchi_rgb", "Z4_forestfire_siglip2_rgb",
              "Z2_siglip2_zeroshot_rgb", "Z1_clipL14_zeroshot_rgb", "V1_molmo2_4b", "V2_molmo2_8b", "V6_qwen35_9b",
              "V7_qwen3vl_8b", "V9_gemma4_12b", "P1_dinov2b_probe_rgb", "P5_siglip2_probe_rgb", "L2_firevit_lora_rgb",
              "L1_dinov2b_lora_rgb", "F1_resnet50_fullft_rgb", "P2_dinov2b_probe_thermal", "L1_dinov2b_lora_thermal"]
C = T.set_index("key").loc[[k for k in chart_keys if k in set(T.key)]].reset_index()
C["label"] = C.model + "  (" + C.input + ")"
order = {"no-learning baseline": 0, "off the shelf, no FLAME training": 1, "trained on FLAME 3": 2}
C = C.assign(o=C.tier.map(order)).sort_values(["o", "mixed"], ascending=[False, True]).reset_index(drop=True)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.36 * len(C) + 1.9), sharey=True, facecolor=SURF)
for ax, col, title in [(axes[0], "mixed", f"Fire frames in the 2 flights that also have no-fire frames (n={int((df.y.values==1)[MIXED].sum())})"),
                       (axes[1], "small", f"Small fires: under 0.25% of pixels above 80 C (n={int(SMALL.sum())})")]:
    ax.set_facecolor(SURF)
    ypos = np.arange(len(C))
    ax.barh(ypos, C[col], height=0.62, color=[TIER_COLOR[t] for t in C.tier], edgecolor=SURF, linewidth=2)
    for yv, v in zip(ypos, C[col]):
        ax.text(min(v + 0.012, 0.93), yv, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 1.05); ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.xaxis.grid(True, color="#e4e3df", lw=0.8); ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#b9b8b2"); ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_xlabel("recall at the model's default threshold", fontsize=8, color=INK2)
axes[0].set_yticks(np.arange(len(C))); axes[0].set_yticklabels(C.label, fontsize=8, color=INK)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TIER_COLOR.values()]
fig.legend(handles, TIER_COLOR.keys(), loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.6, 1.0))
fig.suptitle("Where the models differ: recall on the hardest FLAME 3 fire frames", x=0.01, ha="left", fontsize=11, color=INK, y=1.0 + 0.35 / len(C))
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(FIGS / "fig5_hard_slice_recall.png", dpi=150, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

pd.set_option("display.width", 250)
print(T.drop(columns=["key", "pr_auc"]).round(3).to_string(index=False))
if F is not None and len(F):
    print("\nfew-shot\n", F[["key", "recall", "specificity", "roc_auc", "mixed", "small"]].round(3).to_string(index=False))
if len(R):
    print("\nrandom split\n", R[["key", "recall", "precision", "specificity", "roc_auc", "mixed", "small"]].round(3).to_string(index=False))


# ---- markdown tables ----
def fmt(v, nd=3):
    return "" if pd.isna(v) else f"{v:.{nd}f}"


def md_table(frame, cols, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in frame.iterrows():
        out.append("| " + " | ".join(str(r[c]) if not isinstance(r[c], float) else fmt(r[c]) for c in cols) + " |")
    return "\n".join(out)


tier_order = {"no-learning baseline": 0, "off the shelf, no FLAME training": 1, "trained on FLAME 3": 2}
M = T.assign(o=T.tier.map(tier_order)).sort_values(["o", "roc_auc"], ascending=[True, False])
M["false_alarms"] = M.fp.astype(int).astype(str); M["missed"] = M.fn.astype(int).astype(str)
main = md_table(M, ["model", "input", "tier", "recall", "precision", "specificity", "roc_auc", "mixed", "small", "false_alarms", "missed"],
                ["Model", "Input", "Tier", "Recall", "Precision", "Specificity", "ROC-AUC", "Recall, mixed flights", "Recall, small fires", "False alarms", "Missed fires"])
tuned = md_table(M[M.tier != "no-learning baseline"], ["model", "input", "t_recall", "t_spec", "t_prec", "t_mixed"],
                 ["Model", "Input", "Recall", "Specificity", "Precision", "Recall, mixed flights"])
parts = ["## Main table (default threshold, all 738 frames, out-of-fold)\n", main,
         "\n\n## Recall-tuned threshold (picked on training or validation frames, target recall 0.98)\n", tuned]
if F is not None and len(F):
    F2 = F.copy(); F2["base"] = F2.key.str.replace(r"_seed\d", "", regex=True)
    agg = F2.groupby("base").agg(n=("key", "size"), recall=("recall", "mean"), specificity=("specificity", "mean"),
                                 roc_auc=("roc_auc", "mean"), mixed=("mixed", "mean"), small=("small", "mean")).reset_index()
    agg["seeds"] = agg.n.astype(str)
    parts += ["\n\n## 16 labeled frames per class (mean over seeds)\n",
              md_table(agg, ["base", "seeds", "recall", "specificity", "roc_auc", "mixed", "small"],
                       ["Run", "Seeds", "Recall", "Specificity", "ROC-AUC", "Recall, mixed flights", "Recall, small fires"])]
if len(R):
    parts += ["\n\n## Random split (leaky) for comparison\n",
              md_table(R, ["key", "recall", "precision", "specificity", "roc_auc", "mixed", "small"],
                       ["Run", "Recall", "Precision", "Specificity", "ROC-AUC", "Recall, mixed flights", "Recall, small fires"])]
(RESULTS / "tables.md").write_text("\n".join(parts) + "\n")
print("wrote results/tables.md")
