"""Command line: python -m recon_gateway --frames DIR [--tiffs DIR] | --flame-group GROUP --index CSV --flame-root DIR"""
import argparse
import json
import os
import sys
from pathlib import Path

from . import agent
from .brief import to_markdown
from .sequence import from_flame_index, from_folder

REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description="Turn a drone frame sequence into a reconnaissance brief.")
    p.add_argument("--frames", help="folder of RGB frames (ordered by EXIF time, else filename)")
    p.add_argument("--tiffs", help="optional folder of Celsius TIFFs with matching file names")
    p.add_argument("--flame-group", help="FLAME 3 60-second window id from the experiment index, e.g. 2_3")
    p.add_argument("--flame-flight", type=int)
    p.add_argument("--index", default=str(REPO / "training/flame3-lora/index.csv"), help="FLAME 3 experiment index CSV")
    p.add_argument("--flame-root", default=str(Path(os.environ.get("RECON_DATA_ROOT") or REPO / "data") / "flame3_cv"),
                   help="folder containing 'FLAME 3 CV Dataset (Sycan Marsh)' (default: $RECON_DATA_ROOT/flame3_cv)")
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--mode", choices=["agent", "pipeline"], default="agent")
    p.add_argument("--model", default=None, help="LLM route or served model name (default: RECON_LLM_MODEL)")
    p.add_argument("--deciders", default="", help="comma list of decision models to call through the gateway: kai, clef, clef_flash")
    p.add_argument("--guidance", default=None, help="observer guidance: all, or a comma list of planner, decide, writeup")
    p.add_argument("--out", default="brief")
    a = p.parse_args()
    if a.frames:
        seq = from_folder(a.frames, a.tiffs)
        seq.frames = seq.frames[:: a.stride][: a.max_frames]
    elif a.flame_group or a.flame_flight is not None:
        seq = from_flame_index(a.index, a.flame_root, group=a.flame_group, flight=a.flame_flight, max_frames=a.max_frames, stride=a.stride)
    else:
        sys.exit("Give --frames or --flame-group/--flame-flight")
    res = agent.run(seq, mode=a.mode, llm_model=a.model, deciders=[d for d in a.deciders.split(",") if d], guidance=a.guidance)
    Path(a.out + ".json").write_text(json.dumps(res, indent=2))
    Path(a.out + ".md").write_text(to_markdown(res["brief"], res["meta"]))
    print(to_markdown(res["brief"], res["meta"]))
    print("LLM's own call:", res["llm_decision"]["status"], "| measured:", res["measured_decision"]["status"])
    for name, d in res["deciders"].items():
        print(f"{name}:", d["status"], d["status_probabilities"], "| confidence", d["confidence"], "| direction assessable p", d["direction_assessable_p"])
    print("steps:", " -> ".join(f"{t['tool']}({t['seconds']}s)" for t in res["trace"]))
    if res["policy_corrections"]:
        print("policy corrections:", *res["policy_corrections"], sep="\n  - ")


if __name__ == "__main__":
    main()
