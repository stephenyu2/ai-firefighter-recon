"""Inline SVG charts for the results page. Text and marks take colors from CSS tokens (both themes)."""
import html
import math

import numpy as np

E = lambda s: html.escape(str(s), quote=True)


def bar_v(x, y_top, w, y_base, r=4):
    h = y_base - y_top
    r = max(0.0, min(r, w / 2, h))
    return (f"M{x:.1f},{y_base:.1f}L{x:.1f},{y_top + r:.1f}Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f}"
            f"L{x + w - r:.1f},{y_top:.1f}Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f}L{x + w:.1f},{y_base:.1f}Z")


def bar_h(x0, y, length, t, round_end=True, r=4):
    if length <= 0.5:
        return ""
    r = max(0.0, min(r, t / 2, length)) if round_end else 0.0
    x1 = x0 + length
    return (f"M{x0:.1f},{y:.1f}L{x1 - r:.1f},{y:.1f}Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f}"
            f"L{x1:.1f},{y + t - r:.1f}Q{x1:.1f},{y + t:.1f} {x1 - r:.1f},{y + t:.1f}L{x0:.1f},{y + t:.1f}Z")


def hit(x, y, w, h, tip):
    return f'<rect class="hit" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" tabindex="0" data-tip="{E(tip)}"></rect>'


def svg(w, h, label, body):
    return (f'<div class="scroll"><svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{E(label)}">'
            + "".join(body) + "</svg></div>")


def maxtemp_hist(df):
    W, H, L, R, TOP, B = 960, 320, 56, 24, 44, 52
    pw, ph = W - L - R, H - TOP - B
    xmax = 620
    sx = lambda v: L + v / xmax * pw
    edges = np.arange(0, 630, 10)
    c_no, _ = np.histogram(df.t_max[df.y == 0], bins=edges)
    c_fire, _ = np.histogram(df.t_max[df.y == 1], bins=edges)
    ytop = int(math.ceil(max(c_no.max(), c_fire.max()) / 40) * 40)
    sy = lambda v: TOP + ph - v / ytop * ph
    b = []
    for v in range(0, ytop + 1, 40):
        b.append(f'<line class="gridline" x1="{L}" x2="{W - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>')
        b.append(f'<text class="ax" x="{L - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{v}</text>')
    lo, hi = df[df.y == 0].t_max.max(), df[df.y == 1].t_max.min()
    b.append(f'<rect class="band" x="{sx(lo):.1f}" y="{TOP}" width="{sx(hi) - sx(lo):.1f}" height="{ph}"/>')
    b.append(f'<text class="note" x="{sx(lo):.1f}" y="{TOP - 10}">No frame peaks between {lo:.0f} and {hi:.0f} C</text>')
    bw = pw / (len(edges) - 1)
    for counts, cls, name in [(c_no, "mk-blue", "No fire"), (c_fire, "mk-orange", "Fire")]:
        for i, n in enumerate(counts):
            if n:
                x = sx(edges[i]) + 1
                b.append(f'<path class="{cls}" d="{bar_v(x, sy(n), bw - 2, sy(0))}"/>')
                b.append(hit(x - 1, TOP, bw, ph, f"{n} frames|{name}, hottest pixel {edges[i]} to {edges[i + 1]} C"))
    b.append(f'<line class="refline" x1="{sx(80):.1f}" x2="{sx(80):.1f}" y1="{TOP}" y2="{TOP + ph}"/>')
    b.append(f'<text class="note" x="{sx(80) + 5:.1f}" y="{TOP + 14}">80 C mask cut</text>')
    n187 = int(c_fire[18])
    b.append(f'<text class="note" x="{sx(196):.1f}" y="{sy(n187) + 4:.1f}">Oct 25 flights cap at 187 C ({n187} frames)</text>')
    b.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{TOP + ph}" y2="{TOP + ph}"/>')
    for v in range(0, 601, 100):
        b.append(f'<text class="ax" x="{sx(v):.1f}" y="{TOP + ph + 18}" text-anchor="middle">{v} C</text>')
    b.append(f'<text class="lab2" x="{L + pw / 2:.1f}" y="{H - 8}" text-anchor="middle">Hottest pixel in the frame, from the radiometric TIFF</text>')
    b.append(f'<text class="lab2" x="{L - 8}" y="{TOP - 26}" text-anchor="end">frames</text>')
    lx = W - R - 250
    b.append(f'<rect class="mk-blue" x="{lx}" y="{TOP - 30}" width="12" height="12" rx="2"/>'
             f'<text class="lab" x="{lx + 18}" y="{TOP - 20}">No fire ({int((df.y == 0).sum())})</text>'
             f'<rect class="mk-orange" x="{lx + 130}" y="{TOP - 30}" width="12" height="12" rx="2"/>'
             f'<text class="lab" x="{lx + 148}" y="{TOP - 20}">Fire ({int((df.y == 1).sum())})</text>')
    return svg(W, H, "Histogram of the hottest pixel per frame, split by label", b)


def flights(df):
    fl = df.groupby("flight").agg(n=("y", "size"), fire=("y", "sum"), t0=("ts", "min"), t1=("ts", "max")).reset_index()
    fl["nofire"] = fl.n - fl.fire
    W, rowh, TOP, L, R, B = 960, 26, 46, 200, 170, 34
    H = TOP + rowh * len(fl) + B
    xmax = 200
    pw = W - L - R
    sx = lambda v: L + v / xmax * pw
    b = []
    for v in range(0, xmax + 1, 50):
        b.append(f'<line class="gridline" x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{TOP - 4}" y2="{TOP + rowh * len(fl)}"/>')
        b.append(f'<text class="ax" x="{sx(v):.1f}" y="{TOP + rowh * len(fl) + 18}" text-anchor="middle">{v}</text>')
    b.append(f'<text class="lab2" x="{sx(xmax) + 26:.1f}" y="{TOP + rowh * len(fl) + 18}">frames</text>')
    for i, r in enumerate(fl.itertuples()):
        y = TOP + i * rowh + 6
        t = 14
        when = f"{r.t0:%b} {r.t0.day} {r.t0:%H:%M}"
        b.append(f'<text class="lab" x="{L - 12}" y="{y + 11}" text-anchor="end">Flight {r.flight}  <tspan class="lab2">{when}</tspan></text>')
        lf = r.fire / xmax * pw
        ln = r.nofire / xmax * pw
        span = f"{when} to {r.t1:%H:%M}"
        if r.fire:
            b.append(f'<path class="mk-orange" d="{bar_h(sx(0), y, lf - (2 if r.nofire else 0), t, round_end=not r.nofire)}"/>')
            b.append(hit(sx(0), y - 5, max(lf, 6), rowh, f"{r.fire} fire frames|Flight {r.flight}, {span}"))
        if r.nofire:
            b.append(f'<path class="mk-blue" d="{bar_h(sx(0) + lf, y, ln, t)}"/>')
            b.append(hit(sx(0) + lf, y - 5, max(ln, 6), rowh, f"{r.nofire} no-fire frames|Flight {r.flight}, {span}"))
            b.append(f'<text class="note" x="{sx(0) + lf + ln + 8:.1f}" y="{y + 11}">{r.fire} fire, {r.nofire} no fire</text>')
    lx = L
    b.append(f'<rect class="mk-orange" x="{lx}" y="{TOP - 34}" width="12" height="12" rx="2"/><text class="lab" x="{lx + 18}" y="{TOP - 24}">Fire</text>'
             f'<rect class="mk-blue" x="{lx + 70}" y="{TOP - 34}" width="12" height="12" rx="2"/><text class="lab" x="{lx + 88}" y="{TOP - 24}">No fire</text>')
    return svg(W, H, "Frames per flight, split by label", b)


def slice_bars(rows, panels):
    """rows: list of dicts with group, label, sub, and one value per panel key. panels: [(key, title, subtitle)]."""
    W, L, GAP, R = 980, 330, 40, 24
    pw = (W - L - GAP - R) / len(panels)
    TOP, rowh, grph = 66, 22, 28
    ys, y, last = [], TOP, None
    for r in rows:
        if r["group"] != last:
            y += grph if last is not None else grph - 6
            last = r["group"]
        ys.append(y)
        y += rowh
    H = y + 44
    b = []
    x0s = [L + i * (pw + GAP) for i in range(len(panels))]
    for x0, (key, title, sub) in zip(x0s, panels):
        b.append(f'<text class="ttl" x="{x0:.1f}" y="{TOP - 40}">{E(title)}</text><text class="lab2" x="{x0:.1f}" y="{TOP - 24}">{E(sub)}</text>')
        for v in [0, 0.25, 0.5, 0.75, 1.0]:
            xx = x0 + v * pw
            b.append(f'<line class="gridline" x1="{xx:.1f}" x2="{xx:.1f}" y1="{TOP - 12}" y2="{y + 2}"/>')
            b.append(f'<text class="ax" x="{xx:.1f}" y="{y + 20}" text-anchor="middle">{v:g}</text>')
    last = None
    for r, yy in zip(rows, ys):
        if r["group"] != last:
            b.append(f'<text class="grp" x="0" y="{yy - 9}">{E(r["group"].upper())}</text>')
            last = r["group"]
        b.append(f'<text class="lab" x="0" y="{yy + 11}">{E(r["label"])} <tspan class="lab2">{E(r["sub"])}</tspan></text>')
        for x0, (key, title, sub) in zip(x0s, panels):
            v = r.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            b.append(f'<path class="mk-blue" d="{bar_h(x0, yy + 3, v * pw, 12)}"/>')
            b.append(hit(x0, yy, pw, rowh, f"{v:.3f} recall|{r['label']} ({r['sub']})|{title}"))
    b.append(f'<text class="lab2" x="{L:.1f}" y="{H - 6}">Recall at each model\'s default threshold (share of fire frames it flags)</text>')
    return svg(W, H, "Recall on the two hardest slices of fire frames, per model", b)


def speed_dots(rows, budget_s=30):
    """rows: dicts with label, sub, ms. x = seconds per minute of footage at 1 frame per second (log scale)."""
    W, L, R, TOP, rowh = 960, 330, 40, 40, 24
    vals = [60 * r["ms"] / 1000 for r in rows]
    lo = 10 ** math.floor(math.log10(min(vals)))
    hi = 10 ** math.ceil(math.log10(max(max(vals), budget_s * 1.5)))
    pw = W - L - R
    sx = lambda v: L + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * pw
    H = TOP + rowh * len(rows) + 46
    b = []
    k = lo
    while k <= hi * 1.0001:
        lab = f"{k:g} s" if k < 60 else (f"{k / 60:g} min" if k % 60 == 0 else f"{k:g} s")
        b.append(f'<line class="gridline" x1="{sx(k):.1f}" x2="{sx(k):.1f}" y1="{TOP - 8}" y2="{TOP + rowh * len(rows)}"/>')
        b.append(f'<text class="ax" x="{sx(k):.1f}" y="{TOP + rowh * len(rows) + 18}" text-anchor="middle">{lab}</text>')
        k *= 10
    xb = sx(budget_s)
    b.append(f'<line class="refline" x1="{xb:.1f}" x2="{xb:.1f}" y1="{TOP - 20}" y2="{TOP + rowh * len(rows)}"/>')
    b.append(f'<text class="note" x="{xb + 6:.1f}" y="{TOP - 12}">{budget_s} s budget from our proposal</text>')
    for i, (r, v) in enumerate(zip(rows, vals)):
        yy = TOP + i * rowh + 12
        b.append(f'<text class="lab" x="0" y="{yy + 4}">{E(r["label"])} <tspan class="lab2">{E(r["sub"])}</tspan></text>')
        b.append(f'<line class="gridline" x1="{L}" x2="{sx(v):.1f}" y1="{yy}" y2="{yy}"/>')
        b.append(f'<circle class="dot" cx="{sx(v):.1f}" cy="{yy}" r="5"/>')
        b.append(hit(L, yy - rowh / 2, pw, rowh, f"{v:.1f} s per minute of footage|{r['ms']:.0f} ms per frame|{r['label']} ({r['sub']})"))
    b.append(f'<text class="lab2" x="{L:.1f}" y="{H - 6}">Seconds of compute per minute of footage, sampling 1 frame per second (log scale)</text>')
    return svg(W, H, "Inference cost per minute of footage for each model", b)
