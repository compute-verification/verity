#!/usr/bin/env bash
# Measurement only: served_util.sh's h1 arm on this tree and on BASE's, one after the other in one run, so no arm's capture
# overlaps another's Lean build or serving. ARMS (default "base new noprover newspy"): base, BASE's tree with --h1-prover;
# new, this tree with it; noprover, this tree without it; newspy, new under py-spy; nodump, new without its live dump. GPU_TESTS runs only in new's lease.
# The other env (T, N, PROMPT_IDS, SLOTS, MAX_MODEL_LEN, H1_SEAL, GPU_LEASE_WHO, ...) is served_util.sh's. cwd: a clone.
set -uo pipefail
OUT0=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT0"
NEW=$PWD; ID=${RESEARCH_RUN_ID:-local}
B=$(mktemp -d "$(dirname "$NEW")/ab-base.XXXXXX")
trap 'git -C "$NEW" worktree remove --force "$B/tree" 2>/dev/null; rm -rf "$B"' EXIT
git -C "$NEW" worktree add -q --detach "$B/tree" "${BASE:?BASE is the commit without the change}" || exit 2
disk() {
  echo "$(date -u +%H:%M:%SZ) $1: vm.dirty ratio $(cat /proc/sys/vm/dirty_ratio) background $(cat /proc/sys/vm/dirty_background_ratio)" \
    "bytes $(cat /proc/sys/vm/dirty_bytes) $(grep -E '^(Dirty|Writeback):' /proc/meminfo | tr -s ' \n' ' ')"
  df -h "${UTIL_ROOT:-/workspace/research/vllm-util80}" | tail -1; free -g | sed -n 2p
}
{ echo "new $(git -C "$NEW" rev-parse HEAD) base $(git -C "$B/tree" rev-parse HEAD)"; disk start; } > "$OUT0/ab_env.txt" 2>&1
for arm in ${ARMS:-base new noprover newspy}; do
  tree=$NEW
  case $arm in
    # the collector holds every package's tree to RESEARCH_SOURCE_SHA, the shipped commit's (this tree's) unless declared
    base) tree=$B/tree; set -- H1_PROVER=1 PYSPY=0 GPU_TESTS= RESEARCH_SOURCE_SHA="$(git -C "$B/tree" rev-parse HEAD)" ;;
    new) set -- H1_PROVER=1 PYSPY=0 ;;
    noprover) set -- H1_PROVER=0 PYSPY=0 GPU_TESTS= ;;
    newspy) set -- H1_PROVER=1 PYSPY=1 GPU_TESTS= ;;
    nodump) set -- H1_PROVER=1 PYSPY=0 GPU_TESTS= H1_NODUMP=1 ;;
    *) echo "arm $arm: not base, new, noprover, newspy or nodump" | tee -a "$OUT0/ab.txt"; exit 2 ;;
  esac
  (cd "$tree" && env "$@" OUT="$OUT0/$arm" TREE="$tree" RESEARCH_RUN_ID="$ID-$arm" bash "$tree/benchmarks/one_stage/served_util.sh")
  echo "$(date -u +%H:%M:%SZ) $arm rc=$?" | tee -a "$OUT0/ab.txt"
  disk "after $arm" >> "$OUT0/ab_env.txt" 2>&1
done
