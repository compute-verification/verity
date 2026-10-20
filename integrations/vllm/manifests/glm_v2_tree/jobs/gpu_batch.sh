#!/usr/bin/env bash
# The v2 rows' GPU work on an H200 node, submitted in order from a VM at a pushed commit of this tree (README "GPU batch"): window B
# from the rows' tree (`$SW/<row>.tree`; a worktree at it once the branch has moved).  Each job waits for its GPUs on the node
# (gpu-lease, first come first served, never preempting), so the jobs of a window run back to back.
#   gpu_batch.sh a [NODE]     window A: the four rows' Match GPU legs (8 GPUs each); NODE vy-glm53-eu (default) or vy-glm53-h8
#                             (preemptible: an evicted leg is rerun with `gpu_batch.sh a vy-glm53-h8 ROW`)
#   gpu_batch.sh a NODE ROW   window A for one row
#   gpu_batch.sh b ROWS.env   window B: each row's Commit (8) on vy-glm53-eu only, in the file's order; a line is "ROW BUILD_RUN MATCH_RUN"
#   gpu_batch.sh t1r [NODE]   the bf16 v6 capture (1 GPU) and the G2 rows-4k point (8), only if T1R's runs of them did not hold
# The legs need no Build on the node (GLM-5.3 has no model-level post-load hook: every rank's v2 trace records post_load_tensors
# applies False), so nothing is copied to the node before window A.
# env: [LOG=/tmp/glm-v2-gpu-batch.log], [RESEARCH="uv run --frozen --no-sync research"].
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
read -r -a R <<<"${RESEARCH:-uv run --frozen --no-sync research}"
EU=vy-glm53-eu LOG=${LOG:-/tmp/glm-v2-gpu-batch.log}
J=integrations/vllm/manifests/glm_v2_tree/jobs
ROWS=(glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager
      glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__greedy__eager
      glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__stoch-t0.8-p0.95__eager
      glm53__fp8__h200__tp8__b32__i256__o32__mixed-arrivals__greedy__eager)
git fetch -q origin cursor/glm-v2-21f3
git merge-base --is-ancestor HEAD origin/cursor/glm-v2-21f3 && git diff --quiet HEAD ||
  { echo "HEAD is not a clean, pushed commit of cursor/glm-v2-21f3"; exit 2; }
sub() {  # sub TAG RESEARCH_RUN_ARGS... -- CMD...
  local tag=$1; shift
  echo "[$(date -u +%FT%TZ)] $tag on $ON at $(git rev-parse --short=10 HEAD): $*" >>"$LOG"
  "${R[@]}" run --on "$ON" --project verity --cwd source "$@" 2>&1 | tee -a "$LOG" | grep -E 'research fetch r[0-9]' || true
}
node() {
  case $1 in vy-glm53-eu|vy-glm53-h8) ON=$1 ;; *) echo "node $1: window A and t1r run on vy-glm53-eu or vy-glm53-h8"; exit 2 ;; esac
}
case "${1:-}" in
  a)
    node "${2:-$EU}"
    if [ -n "${3:-}" ]; then
      printf '%s\n' "${ROWS[@]}" | grep -qx "$3" || { echo "$3 is not a v2 row"; exit 2; }
      ROWS=("$3")
    fi
    for row in "${ROWS[@]}"; do
      sub "match-gpu $row" --gpu-host --mem-gb 400 --timeout 3h -- env ROW="$row" bash "$J/match_gpu.sh"
    done
    ;;
  t1r)
    node "${2:-$EU}"
    sub bf16-v6 --gpus 1 --mem-gb 64 --timeout 45m -- bash "$J/cap_bf16_v6.sh"
    sub g2-rows-4k --gpus 8 --mem-gb 400 --timeout 75m -- env ARM=rows-4k bash benchmarks/served_commit/tp_decode.sh
    ;;
  b)
    ON=$EU
    [ -s "${2:-}" ] || { echo "usage: gpu_batch.sh b ROWS.env (lines: ROW BUILD_RUN MATCH_RUN)"; exit 2; }
    while read -r row build match; do
      [ -n "$row" ] && [ "${row#\#}" = "$row" ] || continue
      sub "commit $row" --gpu-host --mem-gb 600 --timeout 10h --custody-ttl 24h -- env ROW="$row" BUILD_RUN="$build" MATCH_RUN="$match" bash "$J/commit.sh"
    done <"$2"
    ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
