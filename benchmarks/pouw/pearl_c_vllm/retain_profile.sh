#!/usr/bin/env bash
# What retaining every pass costs a served Pearl-C run on an H100 or H200, and where its time goes (retain_profile.py):
#   research run --on NODE --source . --cwd source -- bash benchmarks/pouw/pearl_c_vllm/retain_profile.sh
# 1. benchmarks/harness/setup/pearl_c_sm90.sh builds pearl_c_sm90's cubin from this tree, with its SASS gates;
# 2. under `gpu-lease 1`, `retain_profile.py probe`: the node's copy, pinned-allocation, page-cache and disk rates;
# 3. for each load NAME:R:P:O:T of LOADS (default: the Pearl+ piece's prefill, 32 requests of 1,024 tokens at 8,192 tokens
#    a step; the same at 2,048 a step; decode, 32 requests of 64 tokens and 64 outputs; a mixed load), each under its own
#    `gpu-lease 1` (LEASE_MIN, default 20): `served_passes.py serve --no-retain`, then `retain_profile.py serve` of the same
#    load with every pass retained, the passes' bytes counted and then deleted with the job's scratch.
# Everything kept goes to $OUT (served JSON, profiles, logs); the passes stay in a mktemp directory under SCRATCH_ROOT
# (default /workspace/tmp, the passes' own file system), removed at exit. env: SPEC, LOADS, LEASE_MIN, VERITY_BENCH_VENV.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
OUT=${OUT:-${RESEARCH_RUN_DIR:-$PWD}/out}
mkdir -p "$OUT" "${SCRATCH_ROOT:=/workspace/tmp}" || exit 1
S=$(umask 077; mktemp -d -p "$SCRATCH_ROOT" retain-profile.XXXXXX) || exit 1
trap 'rm -rf "$S"' EXIT
SPEC=${SPEC:-benchmarks/specs/llama32-1b-1gpu-pouw-protocol-served-h200.toml}
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
df -h "$SCRATCH_ROOT" | tail -1
lease $P/retain_profile.py probe --scratch "$S/probe" --out "$OUT/probe.json" > "$OUT/probe.log" 2>&1
echo "probe exit $?: $(tail -1 "$OUT/probe.log" | cut -c1-600)"
for load in $LOADS; do
  IFS=: read -r name r p o t <<< "$load"
  args=(--spec "$SPEC" --requests "$r" --prompt-tokens "$p" --output-tokens "$o" --max-num-batched-tokens "$t")
  lease $P/served_passes.py serve "${args[@]}" --no-retain --passes "$S/$name-unretained" \
    --out "$OUT/$name-unretained.json" > "$OUT/$name-unretained.log" 2>&1
  echo "$name unretained exit $?: $(grep -m1 '^served ' "$OUT/$name-unretained.log" | cut -c1-400)"
  lease $P/retain_profile.py serve --profile "$OUT/$name-profile.json" "${args[@]}" --passes "$S/$name" \
    --out "$OUT/$name-retained.json" > "$OUT/$name-retained.log" 2>&1
  echo "$name retained exit $?: $(grep -m1 '^served ' "$OUT/$name-retained.log" | cut -c1-400)"
  echo "$name retained bytes: $(du -sb "$S/$name" | cut -f1)"
  rm -rf "${S:?}/$name" "${S:?}/$name-unretained"
done
