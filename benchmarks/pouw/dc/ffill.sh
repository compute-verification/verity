#!/usr/bin/env bash
# ffill on the H200: the FFMA filler's conformance (ffill.cu's dumps against ffill_ref.py, bit for bit) and its cost
# beside the LI-CP-W GEMM (probes/sm90_wgmma.cu's wgli_fill against wgli_redux). From a pushed commit, on a 1x H200:
#
#   research run --on <node> --project verity --campaign pouw --source <tree> --cwd source --custody-ttl 12h \
#     --timeout 40m --env GPU_LEASE_WHO=dc -- bash benchmarks/pouw/dc/ffill.sh
#
# Outside the lease: build ffill and the sm90 wgmma probes; SASS-check ffill (the fma kernel one FFMA; the chain kernel an
# IMAD with BIAS and two FFMA a step, no FMUL or FADD; no .FTZ anywhere: sass/ffill-check.txt) and gate the probes
# (gate-build.txt). A refusal stops the run. Under the lease: ffill fma and ffill chain (conformance/), then probe.py run
# (probes/: rows.jsonl, gate.json, times.jsonl) with nvidia-smi sampled every 200 ms (env/smi.csv). After it, on the CPU:
# ffill_ref.py check (conformance/check.json). Exit: 0 conformant, 1 build, SASS check or gate refused, 3 not conformant,
# 2 wrong device or no nvcc. DC_BUILD_ONLY=1 stops after the checks (any machine with CUDA 13.0).
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
PROBE="$HERE/../opcode_probe/probe.py"
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "ffill.sh: no nvcc (CUDA 13.0 has sm_90a)" >&2; exit 2; }
export NVCC PATH="$(dirname "$NVCC"):$PATH"

if [ "${DC_LEASED:-}" != 1 ]; then
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  mkdir -p "$OUT/sass" "$OUT/env" "$OUT/probes" "$OUT/conformance"
  "$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
  if [ "${DC_BUILD_ONLY:-}" != 1 ]; then
    nvidia-smi -q >"$OUT/env/nvidia-smi.txt" 2>&1
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    [ "${cap%%.*}" = 9 ] || { echo "ffill.sh: needs compute capability 9.x, the device reports '$cap'" >&2; exit 2; }
  fi
  "$NVCC" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a "$HERE/ffill.cu" -o "$WORK/ffill" >"$OUT/build.log" 2>&1 \
    || { cat "$OUT/build.log" >&2; exit 1; }
  python3 "$PROBE" build --arch sm_90a --out "$WORK" "$HERE/probes/sm90_wgmma.cu" \
    >"$WORK/built" 2>>"$OUT/build.log" || { cat "$OUT/build.log" >&2; exit 1; }

  sass="$OUT/sass/ffill.sass"
  cuobjdump -sass "$WORK/ffill" >"$sass" 2>&1
  fn() { awk -v f="$1" '/Function :/{p = index($0, f) > 0; next} p' "$sass"; }
  fma_ffma=$(fn fma_kernel | grep -c ' FFMA ')
  ch_ffma=$(fn chain_kernel | grep -c ' FFMA ')
  ch_imad=$(fn chain_kernel | grep -cE ' IMAD R[^;]*, 0x4b400000 ;')
  ch_fp=$(fn chain_kernel | grep -cE ' (FMUL|FADD)')
  ftz=$(grep -c '\.FTZ' "$sass")
  printf 'fma_kernel FFMA %s\nchain_kernel FFMA %s IMAD(BIAS) %s FMUL/FADD %s\n.FTZ %s\n' \
    "$fma_ffma" "$ch_ffma" "$ch_imad" "$ch_fp" "$ftz" | tee "$OUT/sass/ffill-check.txt"
  if [ "$fma_ffma" != 1 ] || [ "$ch_ffma" -eq 0 ] || [ "$ch_ffma" -ne $((2 * ch_imad)) ] || [ "$ch_fp" != 0 ] \
    || [ "$ftz" != 0 ]; then
    echo "ffill.sh: ffill's SASS is not the filler's (sass/ffill-check.txt)" >&2
    exit 1
  fi

  mapfile -t bins <"$WORK/built"
  for b in "${bins[@]}"; do cuobjdump -sass "$b" >"$OUT/sass/$(basename "$b").sass" 2>&1; done
  python3 "$PROBE" gate "${bins[@]}" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
  rc=$?
  grep -c '^PASS' "$OUT/gate-build.txt" | xargs echo "ffill.sh: gate-build PASS"
  [ $rc -eq 0 ] || { grep -A2 '^REFUSED' "$OUT/gate-build.txt" >&2; exit 1; }
  [ "${DC_BUILD_ONLY:-}" = 1 ] && exit 0

  lease=()
  command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 30 --)
  [ ${#lease[@]} -eq 0 ] && echo "ffill.sh: no gpu-lease on this node; running on its one GPU directly" >&2
  "${lease[@]}" env DC_LEASED=1 WORK="$WORK" bash "${BASH_SOURCE[0]}"
  lrc=$?
  [ -s "$OUT/conformance/fma.bin" ] && [ -s "$OUT/conformance/chain.bin" ] || { echo "ffill.sh: no dumps" >&2; exit 2; }
  PYTHONPATH="$ROOT" python3 "$HERE/ffill_ref.py" check "$OUT/conformance" >"$OUT/conformance/check.txt" 2>&1
  crc=$?
  tail -1 "$OUT/conformance/check.txt"
  rm -f "$OUT/conformance/fma.bin" "$OUT/conformance/chain.bin"
  [ $crc -eq 0 ] || { echo "ffill.sh: the GPU's filler is not ffill_ref's (conformance/check.json)" >&2; exit 3; }
  exit $lrc
fi

smi_pid=
nvidia-smi --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv -lms 200 >"$OUT/env/smi.csv" 2>&1 &
smi_pid=$!
trap '[ -n "$smi_pid" ] && kill "$smi_pid" 2>/dev/null' EXIT
"$WORK/ffill" fma "$OUT/conformance" "${FFILL_CASES:-1000000}" 20261018 | tee "$OUT/conformance/fma.json" || exit 2
"$WORK/ffill" chain "$OUT/conformance" 2048 256 20261018 | tee "$OUT/conformance/chain.json" || exit 2
mapfile -t bins <"$WORK/built"
python3 "$PROBE" run "${bins[@]}" --out "$OUT/probes" --reps "${REPS:-5}" >"$OUT/probes/run.txt" 2>&1
rc=$?
[ $rc -eq 0 ] || { tail -5 "$OUT/probes/run.txt" >&2; exit 1; }
echo "ffill.sh: done"
