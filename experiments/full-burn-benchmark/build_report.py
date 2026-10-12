"""Builds report/full_burn_report.html from the full-burn runs, the RTX precision probes and the B200 precision job.
Usage: python build_report.py"""
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent / "results"
OUT = HERE.parent / "report" / "full_burn_report.html"
IDX = HERE / "florida_index.csv"

GPU_NAME = {"T4": "T4", "L4": "L4", "RTX_PRO_6000": "RTX PRO 6000"}
MODEL_NAME = {"dinov2_lora": "DINOv2-B LoRA", "dinov2g_qlora": "DINOv2-g QLoRA",
              "dinov2_lora_fp16c": "DINOv2-B, FP16 compiled", "dinov2_lora_nvfp4": "DINOv2-B, NVFP4 compiled",
              "dinov2g_qlora_fp16c": "DINOv2-g, FP16 compiled", "dinov2g_qlora_nvfp4": "DINOv2-g, NVFP4 compiled"}
DATASET_NAME = {"sycan_marsh": "Sycan Marsh, OR (738 frames)", "florida_crop": "Hanna Hammock, FL, thermal-view crop (1,353 frames)",
                "florida_full": "Hanna Hammock, FL, full RGB frame (1,353 frames)"}


def esc(x):
    return html.escape(str(x))


def fmt(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:,.{d}f}"


def load_runs():
    runs = pd.DataFrame([json.loads(l) for l in (HERE / "full_runs.jsonl").read_text().splitlines() if l.strip()])
    runs["dataset"] = runs["dataset"].fillna("sycan_marsh") if "dataset" in runs else "sycan_marsh"
    runs = runs[runs.frames.isin([738, 1353])]
    runs = runs.drop_duplicates(["dataset", "gpu", "model"], keep="last")
    rows = []
    for r in runs.to_dict("records"):
        pf = HERE / f"preds_{r['dataset']}_{r['gpu']}_{r['model']}.csv"
        if pf.exists():
            p = pd.read_csv(pf)
            fire, neg = p[p.y == 1], p[p.y == 0]
            r.update(recall=fire.fire.mean(), missed=int((1 - fire.fire).sum()), n_fire=len(fire),
                     fa=int(neg.fire.sum()), n_neg=len(neg), preds=p)
        rows.append(r)
    return rows


def table(head, body, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def stacked_bars(rows, title):
    """Server-side ms per frame split into decode, preprocess and GPU forward, one bar per run."""
    W, left, right, bh, gap, top = 760, 210, 70, 18, 10, 30
    mx = max(r["decode_ms_per_frame"] + r["preprocess_ms_per_frame"] + r["gpu_forward_ms_per_frame"] for r in rows)
    step = 10 if mx > 30 else 5 if mx > 12 else 2
    mx = step * int(np.ceil(mx / step))
    sx = lambda v: left + (W - left - right) * v / mx
    H = top + len(rows) * (bh + gap) + 30
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}" class="chart">']
    for t in range(0, mx + 1, step):
        x = sx(t)
        parts.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{H - 26}" class="grid"/>'
                     f'<text x="{x:.1f}" y="{H - 10}" class="tick" text-anchor="middle">{t}</text>')
    parts.append(f'<text x="{W - right:.1f}" y="{top - 14}" class="tick" text-anchor="end">ms per frame</text>')
    for i, r in enumerate(rows):
        y = top + i * (bh + gap)
        parts.append(f'<text x="{left - 10}" y="{y + bh - 5}" class="lab" text-anchor="end">{esc(r["label"])}</text>')
        x0 = 0.0
        for k, cls in (("decode_ms_per_frame", "s-dec"), ("preprocess_ms_per_frame", "s-pre"), ("gpu_forward_ms_per_frame", "s-gpu")):
            v = r[k]
            parts.append(f'<rect x="{sx(x0):.1f}" y="{y}" width="{max(sx(x0 + v) - sx(x0), 0.5):.1f}" height="{bh}" class="{cls}"/>')
            x0 += v
        parts.append(f'<text x="{sx(x0) + 6:.1f}" y="{y + bh - 5}" class="val">{x0:.1f}</text>')
    parts.append("</svg>")
    legend = ('<div class="legend"><span><i class="s-dec"></i>JPEG decode</span><span><i class="s-pre"></i>resize and normalize</span>'
              '<span><i class="s-gpu"></i>GPU forward pass</span></div>')
    return "".join(parts) + legend


def main():
    runs = load_runs()
    by = {(r["dataset"], r["gpu"], r["model"]): r for r in runs}
    idx = pd.read_csv(IDX)

    # ---- timing tables
    timing_sections = []
    for ds in ("sycan_marsh", "florida_crop", "florida_full"):
        rs = [r for r in runs if r["dataset"] == ds and r["model"] in ("dinov2_lora", "dinov2g_qlora") and r["gpu"] in GPU_NAME]
        rs.sort(key=lambda r: (r["model"], ["T4", "L4", "RTX_PRO_6000"].index(r["gpu"]) if r["gpu"] in GPU_NAME else 9))
        if not rs:
            continue
        body = [[esc(MODEL_NAME[r["model"]]), esc(GPU_NAME.get(r["gpu"], r["gpu"])), fmt(r["e2e_total_s"]), fmt(r["server_total_s"]),
                 f'{100 * r["network_and_overhead_s"] / r["e2e_total_s"]:.0f}%', fmt(r["server_ms_per_frame"], 2),
                 fmt(r["decode_ms_per_frame"], 2), fmt(r["preprocess_ms_per_frame"], 2), fmt(r["gpu_forward_ms_per_frame"], 2)] for r in rs]
        def _srv(g, m):
            r = by.get((ds, g, m)); return r["server_total_s"] if r else float("nan")
        share = [100 * r["network_and_overhead_s"] / r["e2e_total_s"] for r in rs]
        note = (f'<p class="result"><strong>Result:</strong> Server-side, the RTX PRO 6000 took {_srv("RTX_PRO_6000", "dinov2_lora"):.1f} s with DINOv2-B and '
                f'{_srv("RTX_PRO_6000", "dinov2g_qlora"):.1f} s with DINOv2-g' +
                (f', against {_srv("T4", "dinov2g_qlora"):.1f} s for DINOv2-g on the T4' if ("T4" in {r["gpu"] for r in rs}) else "") +
                f'. Upload was {min(share):.0f} to {max(share):.0f}% of the end-to-end time.</p>')
        timing_sections.append(f"<h3>{esc(DATASET_NAME[ds])}</h3>" + table(
            ["Model", "GPU", "End-to-end, whole set (s)", "Server-side, whole set (s)", "Share spent on network",
             "Server ms/frame", "Decode ms", "Preprocess ms", "GPU ms"], body, "num") + note)
    bars = [dict(label=f'{GPU_NAME[r["gpu"]]}, {"DINOv2-B" if r["model"] == "dinov2_lora" else "DINOv2-g"}', **r)
            for m in ("dinov2_lora", "dinov2g_qlora") for g in ("T4", "L4", "RTX_PRO_6000") if (r := by.get(("florida_crop", g, m)))]
    chart = stacked_bars(bars, "Server-side time per frame on the Florida set, split into decode, preprocess and GPU forward")

    # ---- Florida accuracy
    acc_body = []
    for ds in ("florida_crop", "florida_full"):
        for m in ("dinov2_lora", "dinov2g_qlora"):
            r = by.get((ds, "RTX_PRO_6000", m))
            if not r or "preds" not in r:
                continue
            p = r["preds"]
            rec = lambda sl: p[(p.y == 1) & (p["slice"] == sl)].fire.mean()
            flag = lambda sl: p[(p.y == -1) & (p["slice"] == sl)].fire.mean()
            pre = p[(p.y == 0) & p["slice"].str.endswith("preburn")]
            dneg = p[(p.y == 0) & (p["slice"] == "plot2_duringburn")]
            acc_body.append([esc("Thermal-view crop" if ds == "florida_crop" else "Full RGB frame"), esc(MODEL_NAME[m]),
                             f'{r["recall"]:.3f} <span class="sub">({r["missed"]} missed of {r["n_fire"]})</span>',
                             f"{rec('plot1_duringburn'):.3f}", f"{rec('plot2_duringburn'):.3f}",
                             f"{int(pre.fire.sum())} of {len(pre)}", f"{int(dneg.fire.sum())} of {len(dneg)}",
                             f"{100 * flag('plot1_postburn'):.0f}% / {100 * flag('plot2_postburn'):.0f}%", f"{100 * flag('plot3_unknown'):.0f}%"])
    acc_table = table(["Input", "Model", "Recall, all fire frames", "Plot 1 burn", "Plot 2 burn", "Flagged, pre-burn",
                       "Flagged, burn frames with no hot pixel", "Flagged, post-burn (plot 1 / 2)", "Flagged, plot 3"], acc_body, "num")
    bucket_rows = []
    edges = [0, 0.001, 0.005, 0.02, 0.05, 1.0]
    names = ["under 0.1%", "0.1 to 0.5%", "0.5 to 2%", "2 to 5%", "over 5%"]
    for i, nm in enumerate(names):
        cells = [nm]
        n = None
        for ds in ("florida_crop", "florida_full"):
            for m in ("dinov2_lora", "dinov2g_qlora"):
                r = by.get((ds, "RTX_PRO_6000", m))
                if not r or "preds" not in r:
                    cells.append("n/a"); continue
                p = r["preds"].merge(idx[["uid", "frac_gt80"]], on="uid")
                g = p[(p.y == 1) & (p.frac_gt80 > edges[i] if i else p.frac_gt80 >= 0) & (p.frac_gt80 <= edges[i + 1])]
                n = len(g)
                cells.append(f"{g.fire.mean():.2f}")
        bucket_rows.append([cells[0], str(n)] + cells[1:])
    bucket_table = table(["Hot area in the thermal view (pixels at or above 80 C)", "Fire frames", "Crop, DINOv2-B", "Crop, DINOv2-g",
                          "Full frame, DINOv2-B", "Full frame, DINOv2-g"], bucket_rows, "num")

    # ---- precision: probe on the RTX
    probe_rows = []
    for f in ("probe_rtx6000_run1.json", "probe_rtx6000_fp8.json"):
        fp = HERE / "precision" / f
        if fp.exists():
            s = json.loads(fp.read_text())
            probe_rows += [dict(r, source=f) for r in s.get("rows", [])]
    pr = {}
    for r in probe_rows:
        if r["precision"] == "fp16" and (r["model"], "fp16") in pr:
            continue
        if "error" in r and (r["model"], r["precision"]) in pr:
            continue
        pr[(r["model"], r["precision"])] = r
    pbody = []
    for m in ("dinov2_lora", "dinov2g_qlora"):
        for prec in ("fp16", "bf16", "fp8", "nvfp4", "nvfp4_weight_only"):
            r = pr.get((m, prec))
            if not r:
                continue
            label = {"fp16": "FP16", "bf16": "BF16", "fp8": "FP8", "nvfp4": "NVFP4 (4-bit weights and activations)",
                     "nvfp4_weight_only": "NVFP4, weights only"}[prec]
            if "error" in r:
                pbody.append([esc(MODEL_NAME[m]), label, f'<span class="bad">failed: {esc(r["error"][:70])}</span>', "", "", "", "", ""])
                continue
            note = " <span class=\"sub\">(per-tensor scales)</span>" if "per-tensor" in r.get("note", "") else ""
            pbody.append([esc(MODEL_NAME[m]), label + note, fmt(r.get("eager_ms_per_frame_b32"), 2), fmt(r.get("compiled_ms_per_frame_b32"), 2),
                          fmt(r.get("eager_ms_per_frame_b1"), 1), fmt(r.get("compiled_ms_per_frame_b1"), 1),
                          f'{r["max_abs_score_change"]:.3f}', f'{r["flips_vs_fp16"]} of {r["n_frames"]}'])
    probe_table = table(["Model", "Precision", "Batch 32, eager", "Batch 32, compiled", "Batch 1, eager", "Batch 1, compiled",
                         "Largest score change vs FP16", "Decisions changed vs FP16"], pbody, "num")

    # ---- precision: full-burn recall, compiled FP16 vs NVFP4 on the RTX
    fb = []
    for base in ("dinov2_lora", "dinov2g_qlora"):
        for prec in ("fp16c", "nvfp4"):
            m = f"{base}_{prec}"
            cells = [esc(MODEL_NAME[m])]
            for ds in ("sycan_marsh", "florida_crop", "florida_full"):
                r = by.get((ds, "RTX_PRO_6000", m))
                cells.append("n/a" if not r or "recall" not in r else
                             f'{r["recall"]:.3f} <span class="sub">({r["missed"]} missed, {r["fa"]} flagged clean)</span>')
            r = by.get(("florida_crop", "RTX_PRO_6000", m))
            cells += [fmt(r["gpu_forward_ms_per_frame"], 2) if r else "n/a", fmt(r["server_ms_per_frame"], 2) if r else "n/a"]
            fb.append(cells)
    flips = []
    for base in ("dinov2_lora", "dinov2g_qlora"):
        for ds in ("sycan_marsh", "florida_crop", "florida_full"):
            a, b = by.get((ds, "RTX_PRO_6000", f"{base}_fp16c")), by.get((ds, "RTX_PRO_6000", f"{base}_nvfp4"))
            if a and b and "preds" in a and "preds" in b:
                j = a["preds"].merge(b["preds"], on="uid", suffixes=("_16", "_4"))
                flips.append((base, ds, int((j.fire_16 != j.fire_4).sum()), len(j)))
    fb_table = table(["Model and precision", "Sycan recall", "Florida crop recall", "Florida full-frame recall",
                      "GPU ms/frame (Florida)", "Server ms/frame (Florida)"], fb, "num")
    flip_text = "; ".join(f'{"DINOv2-B" if b == "dinov2_lora" else "DINOv2-g"} on {DATASET_NAME[d].split(" (")[0]}: {n} of {t}' for b, d, n, t in flips)

    # ---- B200 cross-validated precision (training-protocol scores)
    b200 = (HERE / "precision" / "cv_precision_b200.txt").read_text().splitlines()
    hdr = b200[0].split()
    brow = []
    for line in b200[1:]:
        parts = line.split()
        if len(parts) >= len(hdr) and parts[2] == "ok":
            d = dict(zip(hdr, parts))
            brow.append([esc("DINOv2-B LoRA" if d["variant"].startswith("dinov2b") else "DINOv2-g LoRA"), esc(d["precision"].upper().replace("_WEIGHT_ONLY", ", weights only")),
                         d["recall"], d["false_alarms"], d["missed"], d["roc_auc"], d["flips_vs_fp16"]])
    b200_table = table(["Model", "Precision", "Recall (5-fold, purged)", "False alarms", "Missed", "ROC AUC", "Decisions changed vs FP16"], brow, "num")

    # ---- 60 FPS projection
    rtx_g = by.get(("florida_crop", "RTX_PRO_6000", "dinov2g_qlora"))
    rtx_g16 = by.get(("florida_crop", "RTX_PRO_6000", "dinov2g_qlora_fp16c"))
    rtx_g4 = by.get(("florida_crop", "RTX_PRO_6000", "dinov2g_qlora_nvfp4"))
    burn_min = 25.2
    frames60 = burn_min * 3600
    proj = []
    for label, r in [("T4, DINOv2-g", by.get(("florida_crop", "T4", "dinov2g_qlora"))), ("L4, DINOv2-g", by.get(("florida_crop", "L4", "dinov2g_qlora"))),
                     ("RTX PRO 6000, DINOv2-g FP16 (served)", rtx_g), ("RTX PRO 6000, DINOv2-g FP16 compiled", rtx_g16),
                     ("RTX PRO 6000, DINOv2-g NVFP4 compiled", rtx_g4), ("RTX PRO 6000, DINOv2-B FP16", by.get(("florida_crop", "RTX_PRO_6000", "dinov2_lora")))]:
        if not r:
            continue
        g, s = r["gpu_forward_ms_per_frame"], r["server_ms_per_frame"]
        live = '<span class="ok">yes</span>' if s <= 1000 / 60 else '<span class="bad">no</span>'
        proj.append([esc(label), fmt(1000 / g, 0), fmt(1000 / s, 0), live, fmt(frames60 * g / 60000, 1), fmt(frames60 * s / 60000, 1)])
    proj_table = table(["Setup", "Frames/s, GPU only", "Frames/s, server-side", "Keeps up with one 60 FPS stream",
                        "Plot 1 burn at 60 FPS: GPU minutes", "Plot 1 burn at 60 FPS: server minutes"], proj, "num")

    fl_crop_b = by.get(("florida_crop", "RTX_PRO_6000", "dinov2_lora"))
    fl_full_g = by.get(("florida_full", "RTX_PRO_6000", "dinov2g_qlora"))
    # client-side resize: what is sent over the network, and what it costs in recall
    up_full = next((r["upload_mb"] for r in runs if r["dataset"] == "florida_crop" and r["gpu"] in GPU_NAME
                    and isinstance(r.get("upload_mb"), float) and not np.isnan(r["upload_mb"])), None)
    variants = [("RTX_PRO_6000", "640 by 512 JPEG (as stored), resized on the server"),
                ("RTX_PRO_6000_clientresize", "308 by 252, JPEG quality 90, 4:2:0 color (default)"),
                ("RTX_PRO_6000_clientresize_q90_444", "308 by 252, JPEG quality 90, full color (4:4:4)"),
                ("RTX_PRO_6000_clientresize_q95_444", "308 by 252, JPEG quality 95, full color (4:4:4)")]
    rz = []
    for m in ("dinov2_lora", "dinov2g_qlora"):
        for g, label in variants:
            r = by.get(("florida_crop", g, m))
            if not r or "recall" not in r:
                continue
            up = r.get("upload_mb") if isinstance(r.get("upload_mb"), float) and not np.isnan(r.get("upload_mb")) else up_full
            rz.append([esc("DINOv2-B" if m == "dinov2_lora" else "DINOv2-g"), esc(label), fmt(up, 0), fmt(r["e2e_total_s"], 0),
                       fmt(r["server_ms_per_frame"], 2), f'{r["recall"]:.3f} <span class="sub">({r["missed"]} missed)</span>', str(r["fa"])])
    resize_line = ("<h3>Sending smaller frames</h3><div class=\"prose\"><p>The detectors shrink every frame to 308 by 252 anyway, so the "
                   "sender can do it first and upload far less. JPEG settings at that size decide how much small-fire detail survives. "
                   "Florida thermal-view crop, RTX PRO 6000.</p></div>" +
                   table(["Model", "What is sent", "Upload (MB)", "End-to-end (s)", "Server ms/frame", "Recall", "False alarms"], rz, "num") +
                   '<p class="result"><strong>Result:</strong> Full-color JPEG at quality 95 keeps recall within 0.003 of sending the stored frames '
                   'while cutting the upload by 57% and end-to-end time by about 65%. Default JPEG settings lose 12 to 14 small-fire frames.</p>') if rz else ""

    # ---- Kai 0.6B decision layer
    kai_table = kai_lat = ""
    kp = HERE / "kai" / "kai_runs.jsonl"
    if kp.exists():
        kr = pd.DataFrame([json.loads(l) for l in kp.read_text().splitlines() if l.strip()]).drop_duplicates(["deploy", "dataset", "mode"], keep="last")
        kd = {(r["deploy"], r["dataset"], r["mode"]): r for r in kr.to_dict("records")}
        mode_name = {"rgb_b": "Kai reading DINOv2-B + CLIP", "rgb_g": "Kai reading DINOv2-g + CLIP",
                     "rgb_b_thermal_v2": "Kai reading DINOv2-B + CLIP + thermal"}
        body = []
        for ds in ("sycan_marsh", "florida_crop", "florida_full"):
            name = DATASET_NAME[ds].split(" (")[0]
            for det_key, det_label in (("dinov2_lora", "DINOv2-B alone"), ("dinov2g_qlora", "DINOv2-g alone")):
                r = by.get((ds, "RTX_PRO_6000", det_key))
                if r and "recall" in r:
                    body.append([esc(name), esc(det_label), f'{r["recall"]:.3f} <span class="sub">({r["missed"]} missed)</span>', str(r["fa"]), "n/a", "n/a"])
            for mode in ("rgb_b", "rgb_g", "rgb_b_thermal_v2"):
                r = kd.get(("L4", ds, mode))
                nf = (by.get((ds, "RTX_PRO_6000", "dinov2_lora")) or {}).get("n_fire")
                if r:
                    rec = 1 - r["missed"] / nf if nf else r["recall"]
                    body.append([esc(name), esc(mode_name[mode]), f'{rec:.3f} <span class="sub">({r["missed"]} missed)</span>',
                                 str(r["false_alarms"]), f'{r["auc_status"]:.3f}', f'{100 * r["agree_with_detector"]:.1f}%'])
        kai_table = table(["Set", "Decision", "Recall", "False alarms", "ROC AUC of P(fire)", "Agrees with its detector"], body, "num")
        lat = []
        for dep, gpu in (("L4", "L4"), ("RTX_PRO_6000", "RTX PRO 6000")):
            rs = kr[kr.deploy == dep]
            if len(rs):
                lat.append([esc(gpu), fmt(rs.server_median_ms.median(), 0), fmt(rs.sequential_median_ms.iloc[0], 0), fmt(1000 / rs.server_median_ms.median(), 0)])
        kai_lat = table(["Kai on", "Server-side ms per decision (median)", "Round trip ms, one at a time (median)", "Decisions/s the server can do"], lat, "num")

    fill = dict(
        timing="".join(timing_sections), chart=chart, acc_table=acc_table, bucket_table=bucket_table, probe_table=probe_table,
        fb_table=fb_table, flip_text=esc(flip_text), b200_table=b200_table, proj_table=proj_table, resize_line=resize_line,
        crop_recall=f'{fl_crop_b["recall"]:.2f}' if fl_crop_b else "n/a", full_recall=f'{fl_full_g["recall"]:.3f}' if fl_full_g else "n/a",
        frames60=f"{frames60:,.0f}", kai_table=kai_table, kai_lat=kai_lat)
    ff = HERE.parent / "report_findings.html"
    fill["findings"] = ff.read_text() if ff.exists() else "<p>Findings pending.</p>"
    page = TEMPLATE
    for k, v in fill.items():
        page = page.replace("%%" + k + "%%", v)
    assert "%%" not in page, re.findall(r"%%\w+%%", page)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page)
    print("wrote", OUT, len(page))


TEMPLATE = open(Path(__file__).with_name("report_template.html")).read()

if __name__ == "__main__":
    main()
