#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
source "${VENV:-$HOME/school/5980/.venv}/bin/activate"
while pgrep -f "run_finetune.py dinov2_lora rgb grouped" > /dev/null; do sleep 15; done
run() {
  local tag="$1"; shift
  echo "START $tag $(date +%T)"
  python run_finetune.py "$@" > "logs/ft_$tag.log" 2>&1
  echo "END $tag exit=$? $(date +%T) | $(grep -E '^saved|Error' logs/ft_$tag.log | tail -1)"
}
run resnet50_rgb_purged resnet50_full rgb purged 0 8
run firevit_lora_rgb_purged firevit_lora rgb purged 0 8
run dinov2_lora_rgb_purged dinov2_lora rgb purged 0 8
run dinov2_lora_thermal_purged dinov2_lora thermal purged 0 8
run resnet50_rgb_day resnet50_full rgb day 0 8
run firevit_lora_rgb_day firevit_lora rgb day 0 8
run dinov2_lora_rgb_day dinov2_lora rgb day 0 8
run dinov2_lora_thermal_day dinov2_lora thermal day 0 8
run resnet50_rgb_random resnet50_full rgb random 0 8
for s in 0 1 2; do
  run firevit_lora_rgb_16shot_purged_s$s firevit_lora rgb purged 16 40 $s
  run dinov2_lora_rgb_16shot_purged_s$s dinov2_lora rgb purged 16 40 $s
done
echo FT_QUEUE2_DONE
