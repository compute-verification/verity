#!/usr/bin/env bash
# libvc on one leased GPU (cwd = the source tree's root). Outside the lease: one library for sm_90a, sm_100a and sm_120a (so
# every node shows all three build), and the workspace's Python. Under one lease: the conformance (conformance.py) on this GPU,
# then (BENCH=1) the performance gate (bench/bench.sh, its own lease).
#
#   OUT=DIR GPU_LEASE_WHO=WHO bash benchmarks/hash/lib/gpu.sh      (on each of an H200, a B200 and an RTX PRO 6000)
# LEASE_ARGS (gpu-lease's flags, e.g. --preemptible), QUICK=1 (conformance --quick), BENCH=1
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
export D=benchmarks/hash/lib
W=$(mktemp -d)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvcc --version > "$O/nvcc.txt" 2>&1
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv -i 0 > "$O/gpu-name.txt" 2>&1
ARCHS="90 100 120" bash $D/build.sh "$W/build" > "$O/build.log" 2>&1
echo "build rc $?" > "$O/rc.txt"
grep -E "Compiling entry|registers|spill" "$O/build.log" > "$O/ptxas.txt" 2>/dev/null
cuobjdump --list-elf "$W/build/libvc.so" > "$O/elf.txt" 2>&1
uv run python -c 'import verity, sys; print(sys.version)' > "$O/python.txt" 2>&1
echo "python rc $?" >> "$O/rc.txt"
[ -f "$W/build/libvc.so" ] || { cat "$O/rc.txt"; tail -40 "$O/build.log"; exit 1; }
export VC_LIB="$W/build/libvc.so" QUICK=${QUICK:-0}
# shellcheck disable=SC2016,SC2086
gpu-lease 1 --wait --max-min 14 ${LEASE_ARGS:-} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  q=""; [ "$QUICK" = 1 ] && q=--quick
  uv run python $D/conformance.py --lib "$VC_LIB" $q > "$O/conformance.jsonl" 2> "$O/conformance.err"
  echo "conformance rc $?" >> "$O/rc.txt"'
echo "lease rc $?" >> "$O/rc.txt"
if [ "${BENCH:-0}" = 1 ]; then
  bash $D/bench/bench.sh > "$O/bench.log" 2>&1
  echo "bench rc $?" >> "$O/rc.txt"
fi
cat "$O/rc.txt"
grep -c '"ok": true' "$O/conformance.jsonl" 2>/dev/null
grep '"ok": false' "$O/conformance.jsonl" 2>/dev/null | head -20
tail -1 "$O/conformance.jsonl" 2>/dev/null
tail -5 "$O/conformance.err" 2>/dev/null
grep -q '"summary": true, .*"ok": true' "$O/conformance.jsonl"
