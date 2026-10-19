#!/usr/bin/env bash
# The v2 rows' GPU work on vy-glm53-eu, submitted in order from a VM at this tree's pushed head (README "GPU batch").  Each job
# waits for its GPUs on the node (gpu-lease, first come first served, never preempting), so the jobs of a window run back to back.
#   gpu_batch.sh a            window A: the bf16 v6 capture (1 GPU), the four rows' Match GPU legs (8), the G2 rows-4k point (8)
#   gpu_batch.sh b ROWS.env   window B: each row's Commit (8), in the file's order; a line is "ROW BUILD_RUN MATCH_RUN"
# env: [ON=vy-glm53-eu], [LOG=/tmp/glm-v2-gpu-batch.log], [RESEARCH="uv run --frozen --no-sync research"].
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
read -r -a R <<<"${RESEARCH:-uv run --frozen --no-sync research}"
ON=${ON:-vy-glm53-eu} LOG=${LOG:-/tmp/glm-v2-gpu-batch.log}
J=integrations/vllm/manifests/glm_v2_tree/jobs
ROWS=(glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager
      glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__greedy__eager
      glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__stoch-t0.8-p0.95__eager
      glm53__fp8__h200__tp8__b32__i256__o32__mixed-arrivals__greedy__eager)
[ "$(git rev-parse --abbrev-ref HEAD)" = cursor/glm-v2-21f3 ] || { echo "not on cursor/glm-v2-21f3"; exit 2; }
git fetch -q origin cursor/glm-v2-21f3
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/cursor/glm-v2-21f3)" ] && git diff --quiet HEAD || { echo "HEAD is not the pushed, clean head"; exit 2; }
sub() {  # sub TAG RESEARCH_RUN_ARGS... -- CMD...
  local tag=$1; shift
  echo "[$(date -u +%FT%TZ)] $tag at $(git rev-parse --short=10 HEAD): $*" >>"$LOG"
  "${R[@]}" run --on "$ON" --project verity --cwd source "$@" 2>&1 | tee -a "$LOG" | grep -E 'research fetch r[0-9]' || true
}
case "${1:-}" in
  a)
    sub bf16-v6 --gpus 1 --mem-gb 64 --timeout 45m -- bash "$J/cap_bf16_v6.sh"
    for row in "${ROWS[@]}"; do
      sub "match-gpu $row" --gpu-host --mem-gb 400 --timeout 3h -- env ROW="$row" bash "$J/match_gpu.sh"
    done
    sub g2-rows-4k --gpus 8 --mem-gb 400 --timeout 75m -- env ARM=rows-4k bash benchmarks/served_commit/tp_decode.sh
    ;;
  b)
    [ -s "${2:-}" ] || { echo "usage: gpu_batch.sh b ROWS.env (lines: ROW BUILD_RUN MATCH_RUN)"; exit 2; }
    while read -r row build match; do
      [ -n "$row" ] && [ "${row#\#}" = "$row" ] || continue
      sub "commit $row" --gpu-host --mem-gb 600 --timeout 10h --custody-ttl 24h -- env ROW="$row" BUILD_RUN="$build" MATCH_RUN="$match" bash "$J/commit.sh"
    done <"$2"
    ;;
  *) sed -n '2,7p' "$0"; exit 2 ;;
esac
