#!/bin/bash
# Reruns the decision-cascade trials through the gateway, writing only into ./results:
#   single-frame checks for the d3 family, the full Qwen3.8 BF16 pipeline with d3 and d3-mini as in-brief deciders,
#   the sequence variants, the cascade on the 20 test sequences and the full Florida burn; then packs the saved runs
#   and rebuilds results/summary.json.
# Needs the proxy routes for qwen38-27b-bf16, the classifier and d3 / d3-rtx / d3-mini / d3-flash
# (../../gateway/litellm/config.example.yaml) and the FLAME 3 data under RECON_DATA_ROOT.
set -e
cd "$(dirname "$0")"
source ../gateway-model-trials/gateway_env.sh
PY=${PY:-python}
RES="$PWD/results"
D3="d3 d3_rtx d3_mini d3_flash"
cd ../gateway-model-trials
echo "== frames seed0 $(date +%T)"; RESULTS_DIR=$RES VERSION=v2b $PY frame_eval.py $D3
echo "== frames seed0 repeat $(date +%T)"; RESULTS_DIR=$RES VERSION=v2b REP=2 $PY frame_eval.py $D3
echo "== frames seed0 guided $(date +%T)"; RESULTS_DIR=$RES VERSION=v2b GUIDED=1 $PY frame_eval.py $D3
echo "== frames seed1 $(date +%T)"; RESULTS_DIR=$RES VERSION=v2b SEED=1 $PY frame_eval.py $D3
echo "== frames seed1 guided $(date +%T)"; RESULTS_DIR=$RES VERSION=v2b GUIDED=1 SEED=1 $PY frame_eval.py $D3
echo "== full pipeline rerun $(date +%T)"; RESULTS_DIR=$RES VERSION=v3 DECIDERS=d3,d3_mini $PY run_briefs.py qwen38-27b-bf16
cd ../decision-cascade
echo "== sequence variants $(date +%T)"; $PY d3_sequences.py $D3
echo "== cascade $(date +%T)"; $PY cascade_eval.py d3 d3_mini
echo "== full burn $(date +%T)"; $PY fullburn_cascade.py d3
RESULTS_DIR=$RES $PY ../gateway-model-trials/runs_store.py pack v3
$PY summarize.py > /dev/null
echo "ALL_DONE $(date +%T)"
