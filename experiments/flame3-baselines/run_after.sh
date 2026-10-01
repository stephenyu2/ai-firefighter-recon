#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
while pgrep -f "run_finetune_queue2.sh" > /dev/null; do sleep 30; done
echo "FT queue finished $(date +%T)"
source "${MOLMO_VENV:-$HOME/school/5980/.venv-molmo}/bin/activate"
echo "START V2_molmo2_8b $(date +%T)"
MOLMO_ID=allenai/Molmo2-8B MOLMO_OUT=results/vlm_raw_V2_molmo2_8b.csv python run_molmo.py > logs/vlm_V2_molmo2_8b.log 2>&1
source "${VENV:-$HOME/school/5980/.venv}/bin/activate"
python vlm_to_scores.py results/vlm_raw_V2_molmo2_8b.csv V2_molmo2_8b >> logs/vlm_V2_molmo2_8b.log 2>&1
echo "END V2_molmo2_8b $(date +%T) $(tail -1 logs/vlm_V2_molmo2_8b.log)"
for pair in "Qwen/Qwen3-VL-8B-Instruct V7_qwen3vl_8b" "google/gemma-4-E4B-it V8_gemma4_e4b"; do
  set -- $pair
  echo "START $2 $(date +%T)"
  python run_vlm.py "$1" "$2" > "logs/vlm_$2.log" 2>&1 && python vlm_to_scores.py "results/vlm_raw_$2.csv" "$2" >> "logs/vlm_$2.log" 2>&1
  echo "END $2 exit=$? $(date +%T) $(tail -1 logs/vlm_$2.log)"
done
echo "START latency benchmark $(date +%T)"
python bench_latency.py vlm > logs/bench_latency.log 2>&1
source "${MOLMO_VENV:-$HOME/school/5980/.venv-molmo}/bin/activate"
python bench_latency_molmo.py >> logs/bench_latency.log 2>&1
echo "END latency benchmark $(date +%T)"
echo ALL_DONE
