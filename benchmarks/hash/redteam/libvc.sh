#!/usr/bin/env bash
# libvc (#2617) red team on one GPU, cwd = the audited tree: their gpu.sh unedited (the full conformance, then their bench at
# 142 MB only, REPS interleaved reps), then libvc_sabotage.py (sent beside this script) over a libvc.so built by their
# build.sh, under its own lease.  LEASE_ARGS as gpu.sh takes them (--preemptible on vy-nebius-2).
set -uo pipefail
: "${OUT:?}" "${RESEARCH_RUN_DIR:?}"
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
D=benchmarks/hash/lib
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
mkdir -p "$OUT/gpu"
OUT="$OUT/gpu" BENCH=1 DECODE_N=34737 MID_N=" " GIB_N=" " REPS=${REPS:-15} bash $D/gpu.sh > "$OUT/gpu.log" 2>&1
echo "gpu.sh rc $?" > "$OUT/rc.txt"
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d .)
ARCHS=$cc bash $D/build.sh "$S/build" > "$OUT/sabotage-build.log" 2>&1
echo "sabotage build rc $?" >> "$OUT/rc.txt"
export S D
# shellcheck disable=SC2016,SC2086
gpu-lease 1 --wait --max-min 12 ${LEASE_ARGS:-} -- bash -c '
  uv run python "$RESEARCH_RUN_DIR/inputs/libvc_sabotage.py" "$D" "$S/build/libvc.so" > "$OUT/sabotage.jsonl" 2> "$OUT/sabotage.err"
  echo "sabotage rc $?" >> "$OUT/rc.txt"'
cat "$OUT/rc.txt"
tail -1 "$OUT/gpu/conformance.jsonl" 2>/dev/null
tail -1 "$OUT/sabotage.jsonl" 2>/dev/null
grep -q '"summary": true, .*"ok": true' "$OUT/gpu/conformance.jsonl" && grep -q '"summary": true, "ok": true' "$OUT/sabotage.jsonl"
