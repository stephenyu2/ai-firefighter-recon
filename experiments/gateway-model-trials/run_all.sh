#!/bin/bash
# Reruns the whole v2 trial through the gateway: briefs (3 repeats, unguided then guided), deciders with probes,
# per-frame checks (two samples, guided and not, plus one repeat for run-to-run noise), speed, then packs the saved runs
# into results/runs_v2.jsonl.gz and rebuilds the tables and report.
# Needs the proxy running with every route in ALL and the sequences built by make_sequences.py.
set -e
cd "$(dirname "$0")"
source gateway_env.sh
PY=${PY:-python}
ALL="qwen38-27b-bf16 qwen38-27b-fp8 qwen38-27b-int4 qwen38-27b-nvfp4 qwen3vl-4b clef clef_flash"
ROUTES="qwen38-27b-bf16 qwen38-27b-fp8 qwen38-27b-int4 qwen38-27b-nvfp4 qwen3vl-4b"
echo "== briefs unguided x3 $(date +%T)"; VERSION=v2 REPEATS=3 $PY run_briefs.py $ROUTES
echo "== briefs guided x3 $(date +%T)"; VERSION=v2 REPEATS=3 RECON_GUIDANCE=1 $PY run_briefs.py $ROUTES
echo "== deciders with probes $(date +%T)"; $PY deciders_with_probes.py
echo "== frames seed0 $(date +%T)"; VERSION=v2b $PY frame_eval.py $ALL
echo "== frames seed0 repeat $(date +%T)"; VERSION=v2b REP=2 $PY frame_eval.py $ALL
echo "== frames seed0 guided $(date +%T)"; VERSION=v2b GUIDED=1 $PY frame_eval.py $ALL
echo "== frames seed1 $(date +%T)"; VERSION=v2b SEED=1 $PY frame_eval.py $ALL
echo "== frames seed1 guided $(date +%T)"; VERSION=v2b GUIDED=1 SEED=1 $PY frame_eval.py $ALL
echo "== speed $(date +%T)"; $PY throughput.py $ROUTES
$PY runs_store.py pack v2
$PY summarize_v2.py > /dev/null
$PY build_report.py
echo "ALL_DONE $(date +%T)"
