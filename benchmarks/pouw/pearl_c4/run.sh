#!/usr/bin/env bash
# Pearl-C4 on one leased GPU, from the shipped tree (build.sh): the tree against MANIFEST.sha256, the one visible GPU's identity
# (a single index in CUDA_VISIBLE_DEVICES, as `gpu-lease 1` sets it; its UUID in NODE_UUIDS when that is set; cc 12.0), an
# nvidia-smi trace of that GPU only, then
#   bash run.sh [check]            dev.py check: every kernel on both fixtures, every buffer byte for byte, both formats and CTA heights
#   bash run.sh time [MxNxK ...]   the check, then dev.py's self-timing per format (default the two headline shapes; not a panel number)
# Nothing is built here.  Python: $PY, else python3 when it has numpy, else `uv run --with numpy`.
# Exit: 3 identity refused; 5 the tree does not match its manifest; 4 a buffer differs (nothing timed); else dev.py's.
# Outputs under $RESEARCH_RUN_DIR/out (or ./out): identity.txt, env.txt, inputs.check, toolchain.txt (nvcc and its flags),
# smi_trace.csv, check.jsonl, time.jsonl.
set -uo pipefail
cd "$(dirname "$0")"
MODE=${1:-check}; shift || true
OUT=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$OUT"
IDX=${CUDA_VISIBLE_DEVICES:-}
case $IDX in ''|*[!0-9]*) echo "REFUSED: CUDA_VISIBLE_DEVICES='$IDX' is not one index (run under gpu-lease 1)"; exit 3 ;; esac
nvidia-smi -i "$IDX" --query-gpu=index,uuid,name,pci.bus_id,driver_version,clocks.sm,clocks.mem,clocks.max.sm \
  --format=csv,noheader > "$OUT/identity.txt" || { echo "REFUSED: nvidia-smi -i $IDX failed"; exit 3; }
UUID=$(cut -d, -f2 "$OUT/identity.txt" | tr -d ' ')
if [ -n "${NODE_UUIDS:-}" ] && ! tr ',' '\n' <<< "$NODE_UUIDS" | grep -qx "$UUID"; then
  echo "REFUSED: GPU $IDX ($UUID) is not one of NODE_UUIDS"; exit 3
fi
{ nvidia-smi -i "$IDX" -q -d CLOCK,PERFORMANCE,POWER; python3 --version; lscpu | grep -E "Model name|^CPU\(s\)"
  grep PRETTY /etc/os-release; } > "$OUT/env.txt" 2>&1
if ! sha256sum -c MANIFEST.sha256 > "$OUT/inputs.check" 2>&1; then echo "REFUSED: the tree does not match MANIFEST.sha256"; exit 5; fi
cp toolchain.txt "$OUT/toolchain.txt"
PY=${PY:-python3}
if ! $PY -c "import numpy" 2> /dev/null; then PY="uv run --no-project --with numpy python3"; fi
nvidia-smi -i "$IDX" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active,utilization.gpu \
  --format=csv,nounits -lms 100 > "$OUT/smi_trace.csv" &
SMI=$!
trap 'kill $SMI 2> /dev/null' EXIT
RC=0
for d in check check2; do
  [ -d "$d" ] || continue
  $PY dev.py check "$d" --expect-uuid "$UUID" >> "$OUT/check.jsonl"; RC=$?
  [ "$RC" -eq 0 ] || break
done
grep -h '"check"\|"refused"' "$OUT/check.jsonl"
if [ "$RC" -eq 0 ] && [ "$MODE" = time ]; then
  SHAPES=("$@"); [ ${#SHAPES[@]} -gt 0 ] || SHAPES=(8192x8192x8192 64x8192x8192)
  for fmt in nvfp4 mxfp4; do
    $PY dev.py time "${SHAPES[@]}" --fmt "$fmt" --expect-uuid "$UUID" >> "$OUT/time.jsonl"; RC=$?
    [ "$RC" -eq 0 ] || break
  done
  grep -h '_ms' "$OUT/time.jsonl"
fi
echo "gpu $IDX $UUID: $MODE exit $RC"
exit $RC
