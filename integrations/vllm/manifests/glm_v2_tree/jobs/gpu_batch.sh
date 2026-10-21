#!/usr/bin/env bash
# The v2 rows' GPU work on an 8x H200 node, submitted from a VM at a pushed commit of this tree (README "GPU batch"): window B
# from the rows' tree (`$SW/<row>.tree`; a worktree at it once the branch has moved).  Each job waits for its GPUs on the node
# (gpu-lease, first come first served, never preempting), so the jobs of a window run back to back, in the rows' value order
# (B1 greedy, B8 greedy, B8 stochastic, B32) whatever order ROWS.env lists them in, so that a short window lands the most.
# NODE is any vy-glm53-* GPU node (vy-glm53-eu, the default; vy-glm53-h8, preemptible; a RunPod node registered as vy-glm53-rp1).
#   gpu_batch.sh a [NODE] [ROW]          window A: the rows' Match GPU legs (8 GPUs each); an evicted leg reruns with ROW
#   gpu_batch.sh b ROWS.env [NODE] [ROW] window B: each row's Commit (8); a line is "ROW BUILD_RUN MATCH_RUN".  An evicted Commit
#                                        reruns whole with ROW: a new commitment and new draws (commit.sh moves the unfinished
#                                        commit/ aside), its attempt counted in the row's commit_attempts.jsonl
#   gpu_batch.sh t1r [NODE]              the bf16 v6 capture (1 GPU) and the G2 rows-4k point (8), only if T1R's runs of them did not hold
# MODEL picks the rows and their tree: glm53 (the default; the four rows, cursor/glm-v2-21f3) or dsv4-pro (B1,
# cursor/dsv4-v2-21f3).  GLM-5.3's legs need no Build on the node (no model-level post-load hook: every rank's v2 trace records
# post_load_tensors applies False), so nothing is copied to the node before window A.  DeepSeek-V4-Pro's do (its mHC broadcast,
# model.sh POST_LOAD): its Build of record goes to the node before window A (copy_rowdir.sh with build_record.sh's rowdir.sha256),
# and match_gpu.sh refuses the legs without it.
# env: [MODEL=glm53], [LOG=/tmp/$MODEL-v2-gpu-batch.log], [RESEARCH="uv run --frozen --no-sync research"],
#      [ROW_ON=ROOT: window A's row dirs kept on another disk at their paths (match_gpu.sh)].
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
read -r -a R <<<"${RESEARCH:-uv run --frozen --no-sync research}"
MODEL=${MODEL:-glm53}
EU=vy-glm53-eu LOG=${LOG:-/tmp/$MODEL-v2-gpu-batch.log}
J=integrations/vllm/manifests/glm_v2_tree/jobs
case $MODEL in
  glm53) BRANCH=cursor/glm-v2-21f3
         ROWS=(glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager
               glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__greedy__eager
               glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__stoch-t0.8-p0.95__eager
               glm53__fp8__h200__tp8__b32__i256__o32__mixed-arrivals__greedy__eager) ;;
  dsv4-pro) BRANCH=cursor/dsv4-v2-21f3 ROWS=(dsv4-pro__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager) ;;
  *) echo "MODEL $MODEL: glm53 or dsv4-pro"; exit 2 ;;
esac
git fetch -q origin "$BRANCH"
git merge-base --is-ancestor HEAD "origin/$BRANCH" && git diff --quiet HEAD ||
  { echo "HEAD is not a clean, pushed commit of $BRANCH"; exit 2; }
sub() {  # sub TAG RESEARCH_RUN_ARGS... -- CMD...: launched and detached (`research run --on` returns once the run starts)
  local tag=$1 out; shift
  echo "[$(date -u +%FT%TZ)] $tag on $ON at $(git rev-parse --short=10 HEAD): $*" >>"$LOG"
  out=$("${R[@]}" run --on "$ON" --project verity --source . --cwd source "$@" 2>&1) || true
  echo "$out" >>"$LOG"
  grep -E 'research fetch r[0-9]' <<<"$out" ||
    { echo "$tag: not launched, so the batch stops here (later rows would run out of order):"; tail -5 <<<"$out"; exit 1; }
}
ALL=("${ROWS[@]}")
node() {
  case $1 in
    vy-glm53-cpu*) echo "node $1: the GPU windows run on a vy-glm53-* GPU node"; exit 2 ;;
    vy-glm53-?*) ON=$1 ;;
    *) echo "node $1: the GPU windows run on a vy-glm53-* GPU node (vy-glm53-eu, vy-glm53-h8, vy-glm53-rp1)"; exit 2 ;;
  esac
}
v2_row() { printf '%s\n' "${ALL[@]}" | grep -qx "$1" || { echo "$1 is not a v2 row"; exit 2; }; }
case "${1:-}" in
  a)
    node "${2:-$EU}"
    [ -z "${3:-}" ] || { v2_row "$3"; ROWS=("$3"); }
    for row in "${ROWS[@]}"; do
      sub "match-gpu $row" --mem-gb 400 --timeout 3h -- env ROW="$row" ${ROW_ON:+ROW_ON="$ROW_ON"} bash "$J/match_gpu.sh"
    done
    ;;
  t1r)
    node "${2:-$EU}"
    sub bf16-v6 --gpus 1 --mem-gb 64 --timeout 45m -- bash "$J/cap_bf16_v6.sh"
    sub g2-rows-4k --gpus 8 --mem-gb 400 --timeout 75m -- env ARM=rows-4k bash benchmarks/served_commit/tp_decode.sh
    ;;
  b)
    [ -s "${2:-}" ] || { echo "usage: gpu_batch.sh b ROWS.env [NODE] [ROW] (lines: ROW BUILD_RUN MATCH_RUN)"; exit 2; }
    node "${3:-$EU}"
    [ -z "${4:-}" ] || { v2_row "$4"; ROWS=("$4"); }
    declare -A BUILD MATCH
    while read -r row build match; do
      [ -n "$row" ] && [ "${row#\#}" = "$row" ] || continue
      v2_row "$row"
      [ -n "$build" ] && [ -n "$match" ] || { echo "$2: $row needs a BUILD_RUN and a MATCH_RUN"; exit 2; }
      [ -z "${BUILD[$row]:-}" ] || { echo "$2 lists $row twice"; exit 2; }
      BUILD[$row]=$build MATCH[$row]=$match
    done <"$2"
    [ -z "${4:-}" ] || [ -n "${BUILD[$4]:-}" ] || { echo "$2 does not list $4"; exit 2; }
    for row in "${ROWS[@]}"; do
      [ -n "${BUILD[$row]:-}" ] || continue
      sub "commit $row" --mem-gb 600 --timeout 10h --custody-ttl 24h -- \
        env ROW="$row" BUILD_RUN="${BUILD[$row]}" MATCH_RUN="${MATCH[$row]}" NODE="$ON" bash "$J/commit.sh"
    done
    ;;
  *) sed -n '2,/^# env:/p' "$0"; exit 2 ;;
esac
