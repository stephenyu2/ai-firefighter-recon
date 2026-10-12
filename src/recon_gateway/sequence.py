"""Load a frame sequence: a folder of images (optionally with matching Celsius TIFFs), or a FLAME 3 time window."""
import base64
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image

EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class Frame:
    id: str
    rgb_path: Path
    tiff_path: Path | None = None
    t: float = 0.0
    label: int | None = None
    src_id: str | None = None  # dataset id kept for bookkeeping; never shown to a model

    def rgb_b64(self, max_side=None):
        if max_side is None:
            return base64.b64encode(self.rgb_path.read_bytes()).decode()
        im = Image.open(self.rgb_path).convert("RGB")
        im.thumbnail((max_side, max_side))
        import io
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    def tiff_b64(self):
        return base64.b64encode(self.tiff_path.read_bytes()).decode() if self.tiff_path else None


@dataclass
class Sequence:
    id: str
    frames: list = field(default_factory=list)

    @property
    def span_s(self):
        return round(self.frames[-1].t - self.frames[0].t, 1) if len(self.frames) > 1 else 0.0


def alias(seq):
    """Copy of the sequence with neutral ids (S, f01..fNN). Dataset ids can encode the label or burn phase
    ('Fire_121', 'plot1_preburn_0040'), so nothing a model reads may contain them. Returns (aliased, id_map)."""
    frames = [Frame(id=f"f{i + 1:02d}", rgb_path=f.rgb_path, tiff_path=f.tiff_path, t=f.t, label=f.label, src_id=f.src_id or f.id)
              for i, f in enumerate(sorted(seq.frames, key=lambda f: f.t))]
    return Sequence(id="S", frames=frames), {f.id: f.src_id for f in frames}


def _exif_time(path):
    try:
        with Image.open(path) as im:
            ex = im.getexif(); v = ex.get_ifd(0x8769).get(36867) or ex.get(306)
        return datetime.strptime(v, "%Y:%m:%d %H:%M:%S").timestamp() if v else None
    except Exception:
        return None


def from_folder(rgb_dir, tiff_dir=None, seq_id=None, interval_s=3.0):
    rgb_dir = Path(rgb_dir)
    paths = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in EXTS)
    frames = []
    for i, p in enumerate(paths):
        tp = None
        if tiff_dir:
            c = [q for q in Path(tiff_dir).glob(p.stem + ".*") if q.suffix.lower() in (".tif", ".tiff")]
            tp = c[0] if c else None
        t = _exif_time(p)
        frames.append(Frame(id=p.stem, rgb_path=p, tiff_path=tp, t=t if t is not None else i * interval_s))
    frames.sort(key=lambda f: f.t)
    return Sequence(id=seq_id or rgb_dir.name, frames=frames)


def from_flame_index(index_csv, flame_root, group=None, flight=None, max_frames=8, stride=1):
    """Build a time-ordered FLAME 3 sequence from the experiment index (capture-time order, never file order)."""
    import pandas as pd
    df = pd.read_csv(index_csv, parse_dates=["ts"]).sort_values("ts")
    if group is not None:
        df = df[df.group == group]
    elif flight is not None:
        df = df[df.flight == flight]
    df = df.iloc[::stride].head(max_frames)
    root = Path(flame_root) / "FLAME 3 CV Dataset (Sycan Marsh)"
    frames = []
    for r in df.itertuples():
        lab = "Fire" if r.y == 1 else "No Fire"
        stem = f"{int(r.uid.split('_')[1]):05d}"
        frames.append(Frame(id=r.uid, rgb_path=root / lab / "RGB/Corrected FOV" / f"{stem}.JPG",
                            tiff_path=root / lab / "Thermal/Celsius TIFF" / f"{stem}.TIFF", t=r.ts.timestamp(), label=int(r.y)))
    return Sequence(id=str(group if group is not None else f"flight{flight}"), frames=frames)
