#!/usr/bin/env bash
# The open word-writer routes on the H200, through opcode_probe's gate: tld4.cu (TLD of a 16-byte texel) and, built for
# sm_90a, opcode_probe's LDSM/STSM/LDS.128 (lm.cu), side units alone (side.cu's sa_*: TLD, TEX, UBLKRED, UBLKCP, REDG),
# REDUX and VOTE (collective.cu), MATCH.ANY (match.cu) and the f64 pipe's DFMA and conversions (pipes.cu). Each source
# is built on its own (one that doesn't build for sm_90a is logged and skipped) and gated outside the lease; under one
# gpu-lease, probe.py times every probe whose verdict is the declared one, with the SM clock sampled.
#   research run --on vy-dc-h200-5 --project verity --campaign pouw-night --source <tree> --cwd source \
#     --env GPU_LEASE_WHO=n-floor -- bash benchmarks/pouw/np_floor/probes.sh [SOURCE.cu...]
# SOURCE.cu... replaces the list (every probe of each is timed): race.cu, the honest wgmma against the word programs.
# Writes $OUT/build.log, gate-build.{txt,json}, sass/, smi.csv and one probe.py run directory a binary.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OP="$HERE/../opcode_probe"
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "probes.sh: no nvcc" >&2; exit 2; }
export NVCC PATH="$(dirname "$NVCC"):$PATH"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/sass"
# source, and the probes of it to time (probe.py run --only)
declare -A ONLY=(
  ["$HERE/tld4.cu"]='.'
  ["$OP/probes/lm.cu"]='^(ldsm|stsm|lds128|sts128)'
  ["$OP/probes/side.cu"]='^sa_(tld1c?d?|texc|blkred512|blkred4k|blkcp4k|red32|redv4)$'
  ["$OP/probes/collective.cu"]='^(redux_|vote_)'
  ["$OP/probes/match.cu"]='^match_any'
  ["$OP/probes/pipes.cu"]='^(dfma|i2f64|f2f64|s64|cvt_u64|cvt_frnd64|cvt_f64_u32|cvt_f64_s64)$'
)
srcs=("$HERE/tld4.cu" "$OP/probes/lm.cu" "$OP/probes/side.cu" "$OP/probes/collective.cu" "$OP/probes/match.cu"
  "$OP/probes/pipes.cu")
if [ $# -gt 0 ]; then
  srcs=()
  for s in "$@"; do s=$(cd "$(dirname "$s")" && pwd)/$(basename "$s"); srcs+=("$s"); ONLY[$s]=${ONLY[$s]:-.}; done
fi
: >"$WORK/built"
for s in "${srcs[@]}"; do
  if python3 "$OP/probe.py" build --arch sm_90a --out "$WORK" "$s" >"$WORK/one" 2>>"$OUT/build.log"; then
    cat "$WORK/one" >>"$WORK/built"
    echo "$(basename "$s" .cu) ${ONLY[$s]}" >>"$WORK/only"
  else
    echo "probes.sh: $(basename "$s") did not build for sm_90a" | tee -a "$OUT/build.log" >&2
  fi
done
mapfile -t bins <"$WORK/built"
[ ${#bins[@]} -gt 0 ] || exit 1
for b in "${bins[@]}"; do cuobjdump -sass "$b" >"$OUT/sass/$(basename "$b").sass" 2>&1; done
python3 "$OP/probe.py" gate "${bins[@]}" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
grep -c '^PASS' "$OUT/gate-build.txt" | xargs echo "probes.sh: gate-build PASS"
grep -A2 '^REFUSED' "$OUT/gate-build.txt" >&2 || true

gpu-lease 1 --wait --max-min 20 -- bash -c '
  nvidia-smi -i "$GPU_LEASE_UUID" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,clocks_throttle_reasons.active \
    --format=csv -lms 200 >"$0/smi.csv" &
  p=$!
  while read -r stem only; do
    python3 "$1" run "$2/$stem" --out "$0/$stem" --reps 5 --passed-only --only "$only" \
      >"$0/$stem.txt" 2>&1 || echo "probes.sh: probe.py run $stem failed" >&2
    tail -n 40 "$0/$stem.txt"
  done <"$2/only"
  kill $p' "$OUT" "$OP/probe.py" "$WORK"
