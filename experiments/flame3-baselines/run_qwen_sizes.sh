#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
source "${VENV:-$HOME/school/5980/.venv}/bin/activate"
for pair in "Qwen/Qwen3.5-4B V5_qwen35_4b" "Qwen/Qwen3.5-9B V6_qwen35_9b"; do
  set -- $pair
  echo "START $2 $(date +%T)"
  python run_vlm.py "$1" "$2" > "logs/vlm_$2.log" 2>&1 && python vlm_to_scores.py "results/vlm_raw_$2.csv" "$2" >> "logs/vlm_$2.log" 2>&1
  echo "END $2 exit=$? $(date +%T) $(tail -1 logs/vlm_$2.log)"
done
python bench_latency.py vlm >> logs/bench_latency.log 2>&1
echo "bench done $(date +%T)"
