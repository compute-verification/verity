#!/usr/bin/env bash
# What retaining every pass costs a served Pearl-C run on an H100 or H200, and where its time goes (retain_profile.py):
#   research run --on NODE --source . --cwd source -- bash benchmarks/pouw/pearl_c_vllm/retain_profile.sh
# 1. benchmarks/harness/setup/pearl_c_sm90.sh builds pearl_c_sm90's cubin from this tree, with its SASS gates;
# 2. under `gpu-lease 1`, `retain_profile.py probe`: the node's copy, pinned-allocation, page-cache and disk rates;
# 3. for each load NAME:R:P:O:T of LOADS (default: the Pearl+ piece's prefill, 32 requests of 1,024 tokens at 8,192 tokens
#    a step; the same at 2,048 a step; decode, 32 requests of 64 tokens and 64 outputs; a mixed load), each under its own
#    `gpu-lease 1` (LEASE_MIN, default 20): `served_passes.py serve --no-retain`, then `retain_profile.py serve` of the same
#    load with every pass retained, the passes' bytes counted and then deleted.
# Everything kept goes to $OUT (served JSON, profiles, logs); the passes go in a mktemp directory under PASSES_ROOT (default
# SCRATCH_ROOT, default /workspace/tmp; /dev/shm serves into tmpfs), removed at exit.  env: SPEC, LOADS, LEASE_MIN,
# VERITY_BENCH_VENV; PROBE=0 skips the probe and UNRETAINED=0 the unretained serves; PLAIN=1 adds each load's retained
# `served_passes.py serve`, without the profile's timers (PYSPY=1: under `py-spy record --gil`, the stacks holding the GIL
# to $OUT/NAME-plain-gil.txt; COPY=1: the passes then copied to SCRATCH_ROOT with `cp -a`, hard links kept,
# synced and timed), and PROFILE_UNRETAINED=1 a profiled serve of each load with no pass
# retained; WAIT_ENDS=1 serves with
# `retain_profile.py serve --wait-ends`; SEED=S serves seeded (`--seed S`) and writes each load's file digests; with it,
# OLD_SRC=DIR (another commit's source tree on the node) serves each load again with DIR's verity_vllm and compares the
# two serves' files.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
OUT=${OUT:-${RESEARCH_RUN_DIR:-$PWD}/out}
mkdir -p "$OUT" "${SCRATCH_ROOT:=/workspace/tmp}" || exit 1
S=$(umask 077; mktemp -d -p "$SCRATCH_ROOT" retain-profile.XXXXXX) || exit 1
R=$(umask 077; mktemp -d -p "${PASSES_ROOT:-$SCRATCH_ROOT}" retain-passes.XXXXXX) || { rm -rf "$S"; exit 1; }
trap 'rm -rf "$S" "$R"' EXIT
SPEC=${SPEC:-benchmarks/harness/specs/llama32-1b-1gpu-pouw-protocol-served-h200.toml}
LOADS=${LOADS:-prefill:32:1024:1:8192 prefill2k:32:1024:1:2048 decode:32:64:64:8192 mixed:64:256:16:4096}
export VERITY_BENCH_VENV=${VERITY_BENCH_VENV:-/workspace/cp/venv312} HF_HUB_OFFLINE=1 GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-retain}
# shellcheck source=/dev/null
. benchmarks/harness/env.sh
unset VLLM_BATCH_INVARIANT
( export RESEARCH_RUN_DIR=$S
  # shellcheck source=/dev/null
  . benchmarks/harness/setup/pearl_c_sm90.sh && printf '%s\n' "$PEARLC_KERNEL" "$PEARLC_CUBIN" > "$S/setup.paths" ) \
  > "$OUT/setup.log" 2>&1 || { echo "setup failed: $(tail -1 "$OUT/setup.log")"; exit 4; }
{ read -r PEARLC_KERNEL; read -r PEARLC_CUBIN; } < "$S/setup.paths"
export PEARLC_KERNEL PEARLC_CUBIN
lease() { gpu-lease 1 --wait --max-min "${LEASE_MIN:-20}" -- "$@"; }
P="$VENV/bin/python benchmarks/pouw/pearl_c_vllm"
df -h "$SCRATCH_ROOT" "$R" | tail -n +2
if [ "${PROBE:-1}" != 0 ]; then
  lease $P/retain_profile.py probe --scratch "$S/probe" --out "$OUT/probe.json" > "$OUT/probe.log" 2>&1
  echo "probe exit $?: $(tail -1 "$OUT/probe.log" | cut -c1-600)"
fi
seed=() extra=(); [ -n "${SEED:-}" ] && seed=(--seed "$SEED")
[ "${WAIT_ENDS:-0}" != 0 ] && seed+=(--wait-ends)
retained() {  # NAME TAG [env...]: one profiled retained serve of load NAME into $R/NAME-TAG, its digests with SEED
  local name=$1 tag=$2; shift 2
  env "$@" gpu-lease 1 --wait --max-min "${LEASE_MIN:-20}" -- $P/retain_profile.py serve --profile "$OUT/$name-$tag-profile.json" \
    "${seed[@]}" "${args[@]}" "${extra[@]}" --passes "$R/$name-$tag" --out "$OUT/$name-$tag.json" > "$OUT/$name-$tag.log" 2>&1
  echo "$name $tag exit $?: $(grep -m1 '^served ' "$OUT/$name-$tag.log" | cut -c1-400)"
  echo "$name $tag bytes: $(du -sb "$R/$name-$tag" | cut -f1)"
  [ -n "${SEED:-}" ] && $P/retain_profile.py digests "$R/$name-$tag" --out "$OUT/$name-$tag-digests.json"
  rm -rf "${R:?}/$name-$tag"
}
for load in $LOADS; do
  IFS=: read -r name r p o t <<< "$load"
  args=(--spec "$SPEC" --requests "$r" --prompt-tokens "$p" --output-tokens "$o" --max-num-batched-tokens "$t")
  if [ "${UNRETAINED:-1}" != 0 ]; then
    lease $P/served_passes.py serve "${args[@]}" --no-retain --passes "$R/$name-unretained" \
      --out "$OUT/$name-unretained.json" > "$OUT/$name-unretained.log" 2>&1
    echo "$name unretained exit $?: $(grep -m1 '^served ' "$OUT/$name-unretained.log" | cut -c1-400)"
    rm -rf "${R:?}/$name-unretained"
  fi
  if [ "${PLAIN:-0}" != 0 ]; then
    spy=(); [ "${PYSPY:-0}" != 0 ] && spy=(py-spy record --gil --threads -r 200 -f raw -o "$OUT/$name-plain-gil.txt" --)
    lease "${spy[@]}" $P/served_passes.py serve "${args[@]}" --passes "$R/$name-plain" \
      --out "$OUT/$name-plain.json" > "$OUT/$name-plain.log" 2>&1
    echo "$name plain exit $?: $(grep -m1 '^served ' "$OUT/$name-plain.log" | cut -c1-400)"
    echo "$name plain bytes: $(du -sb "$R/$name-plain" | cut -f1)"
    if [ "${COPY:-0}" != 0 ]; then
      t0=$(date +%s.%N)
      cp -a "$R/$name-plain" "$S/copy" && sync -f "$S/copy" &&
        echo "$name plain copied to $SCRATCH_ROOT in $(awk "BEGIN{print $(date +%s.%N) - $t0}") s: $(du -sb "$S/copy" | cut -f1) bytes, modes $(find "$S/copy" -type f -printf '%m\n' | sort | uniq -c | tr '\n' ' ')"
      rm -rf "${S:?}/copy"
    fi
    rm -rf "${R:?}/$name-plain"
  fi
  retained "$name" retained
  if [ "${PROFILE_UNRETAINED:-0}" != 0 ]; then
    extra=(--no-retain); retained "$name" unretained-profiled; extra=()
  fi
  if [ -n "${SEED:-}" ] && [ -n "${OLD_SRC:-}" ]; then
    retained "$name" old "PYTHONPATH=$OLD_SRC/adapters/vllm:$PYTHONPATH"
    "$VENV/bin/python" -c 'import json, sys; a, b = (json.load(open(f)) for f in sys.argv[1:]); d = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k)); print(f"{sys.argv[1]} vs old: {len(a)} and {len(b)} files, {len(d)} differ", d[:8])' \
      "$OUT/$name-retained-digests.json" "$OUT/$name-old-digests.json"
  fi
done
