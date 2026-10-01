"""Data loading and HTML helpers for build_html.py."""
import base64
import html
import io
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results"
FIGS = HERE / "figures"
LOGS = HERE / "logs"
E = lambda s: html.escape(str(s), quote=True)


def rd(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def f3(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def pct(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.0f}%"


def table(headers, rows, num=(), groups=None):
    """headers: list of labels; rows: list of lists (strings already escaped or numbers); num: column indexes
    right-aligned; groups: optional list of group labels per row (a header row is inserted when it changes)."""
    out = ['<div class="table-wrap"><table class="data"><thead><tr>']
    out += [f'<th class="{"n" if i in num else ""}">{E(h)}</th>' for i, h in enumerate(headers)]
    out.append("</tr></thead><tbody>")
    last = None
    for k, r in enumerate(rows):
        if groups is not None and groups[k] != last:
            out.append(f'<tr class="grp"><td colspan="{len(headers)}">{E(groups[k])}</td></tr>')
            last = groups[k]
        cells = []
        for i, c in enumerate(r):
            v = f3(c) if isinstance(c, float) else c
            cells.append(f'<td class="{"n" if i in num else ""}">{v}</td>')
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def chip(text, kind):
    return f'<span class="chip {kind}">{E(text)}</span>'


def img_data_uri(path, max_w=1100, q=74):
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def vlm_wallclock():
    """Full 738-frame wall-clock per VLM, from the run logs (shared-GPU conditions, not a clean benchmark)."""
    out = {}
    for p in LOGS.glob("vlm_*.log"):
        m = re.search(r"(V\d_\w+?) done (\d+) imgs in (\d+)s", p.read_text(errors="ignore"))
        if m:
            out[m.group(1)] = (int(m.group(3)), int(m.group(2)))
    p = LOGS / "vlm_V2_molmo2_8b.log"
    if p.exists():
        m = re.search(r"done (\d+) (\d+)s", p.read_text(errors="ignore"))
        if m:
            out["V2_molmo2_8b"] = (int(m.group(2)), int(m.group(1)))
    man = rd("wallclock_manual.csv")
    for r in man.itertuples():
        out[r.key] = (int(r.seconds), int(r.frames))
    return out


def train_logs():
    rows = []
    for p in sorted(RES.glob("trainlog_*.csv")):
        t = pd.read_csv(p)
        rows.append(dict(run=p.stem.replace("trainlog_", ""), folds=len(t), train_s=float(t.train_s.sum()),
                         per_fold=float(t.train_s.mean()), params=int(t.trainable_params.iloc[0]),
                         n_train=int(t.n_train.mean())))
    return pd.DataFrame(rows)


def mmss(s):
    s = int(round(s))
    return f"{s // 60} min {s % 60:02d} s" if s >= 60 else f"{s} s"
