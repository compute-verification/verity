#!/usr/bin/env bash
# RT-HOPPER-1's H200 job: sm90_rs.cu built and SASS-gated outside the lease, then timed under one gpu-lease (10 probes,
# a few minutes), then rs.py. From a pushed commit:
#
#   research run --on vy-dc-h200 --project verity --campaign eps1 --source <tree> --cwd source --custody-ttl 12h \
#     --timeout 30m --env GPU_LEASE_WHO=rt-hopper -- bash benchmarks/pouw/rt_hopper/rs.sh
#
# Outputs in $OUT: build.log, sass/, gate-build.{txt,json}, env/ (nvcc, nvidia-smi, smi.csv), probes/ (rows.jsonl,
# gate.json, times.jsonl), rs.json. Exit: 0 done, 1 build or gate refused, 2 wrong device or no nvcc.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="$HERE/../opcode_probe/probe.py"
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "rs.sh: no nvcc" >&2; exit 2; }
export NVCC PATH="$(dirname "$NVCC"):$PATH"

if [ "${RS_LEASED:-}" != 1 ]; then
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  mkdir -p "$OUT/sass" "$OUT/env" "$OUT/probes"
  "$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
  nvidia-smi -q >"$OUT/env/nvidia-smi.txt" 2>&1
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  [ "${cap%%.*}" = 9 ] || { echo "rs.sh: needs compute capability 9.x, the device reports '$cap'" >&2; exit 2; }
  python3 "$PROBE" build --arch sm_90a --out "$WORK" "$HERE/sm90_rs.cu" >"$WORK/built" 2>"$OUT/build.log" \
    || { cat "$OUT/build.log" >&2; exit 1; }
  mapfile -t bins <"$WORK/built"
  for b in "${bins[@]}"; do cuobjdump -sass "$b" >"$OUT/sass/$(basename "$b").sass" 2>&1; done
  python3 "$PROBE" gate "${bins[@]}" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
  rc=$?
  grep -c '^PASS' "$OUT/gate-build.txt" | xargs echo "rs.sh: gate-build PASS"
  [ $rc -eq 0 ] || { grep -A2 '^REFUSED' "$OUT/gate-build.txt" >&2; exit 1; }
  lease=()
  command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 15 --)
  [ ${#lease[@]} -eq 0 ] && echo "rs.sh: no gpu-lease on this node; running on its GPU directly" >&2
  "${lease[@]}" env RS_LEASED=1 WORK="$WORK" bash "${BASH_SOURCE[0]}"
  rc=$?
  python3 "$HERE/rs.py" "$OUT" --json "$OUT/rs.json"
  exit $rc
fi

nvidia-smi --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv -lms 200 >"$OUT/env/smi.csv" 2>&1 &
smi_pid=$!
trap 'kill "$smi_pid" 2>/dev/null' EXIT
mapfile -t bins <"$WORK/built"
for b in "${bins[@]}"; do python3 "$PROBE" run "$b" --out "$OUT/probes" --reps 10; done
