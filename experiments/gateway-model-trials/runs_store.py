"""Saved brief runs. run_briefs.py writes results/runs/<version>/<label>/<sequence>.json (git-ignored); `python runs_store.py
pack v2` packs them into results/runs_v2.jsonl.gz (one line per label and sequence, planner transcript dropped), which is
the committed copy. Loose files win when present."""
import gzip
import json
import os
from pathlib import Path

RESULTS = Path(os.environ.get("RESULTS_DIR", Path(__file__).parent / "results"))  # RESULTS_DIR: another experiment's results
_packed = {}


def _pack(version):
    if version not in _packed:
        p = RESULTS / f"runs_{version}.jsonl.gz"
        rows = {}
        if p.exists():
            with gzip.open(p, "rt") as fh:
                for line in fh:
                    r = json.loads(line)
                    rows[(r["label"], r["seq"])] = r["run"]
        _packed[version] = rows
    return _packed[version]


def load_run(version, label, seq):
    p = RESULTS / "runs" / version / label / f"{seq}.json"
    if p.exists():
        return json.loads(p.read_text())
    return _pack(version)[(label, seq)]


def pack(version):
    """Merges results/runs/<version>/ into results/runs_<version>.jsonl.gz: loose runs replace packed ones with the same
    label and sequence, every other packed run is kept. Written to a temporary file first, so a failure never truncates
    the committed pack. Returns (runs in the pack, loose runs merged)."""
    loose = RESULTS / "runs" / version
    if not loose.is_dir():
        raise SystemExit(f"nothing to pack: {loose} does not exist (run_briefs.py writes it)")
    rows = dict(_pack(version))
    merged = 0
    for run_dir in sorted(loose.iterdir()):
        for p in sorted(run_dir.glob("*.json")):
            r = json.loads(p.read_text())
            if r.get("planner"):
                r["planner"].pop("transcript", None)
            rows[(run_dir.name, p.stem)] = r
            merged += 1
    target = RESULTS / f"runs_{version}.jsonl.gz"
    tmp = target.with_name(target.name + ".tmp")
    with gzip.open(tmp, "wt") as fh:
        for (label, seq) in sorted(rows):
            fh.write(json.dumps({"label": label, "seq": seq, "run": rows[(label, seq)]}, separators=(",", ":")) + "\n")
    os.replace(tmp, target)
    _packed.pop(version, None)
    return len(rows), merged


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 or sys.argv[1] != "pack":
        sys.exit("usage: python runs_store.py pack VERSION")
    total, merged = pack(sys.argv[2])
    print(f"packed {total} runs ({merged} from loose files)")
