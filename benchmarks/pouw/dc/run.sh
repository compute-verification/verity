#!/usr/bin/env bash
# The data-centre probes on one GPU: the probes (M3 on sm_90a, M4 on sm_100a), the scalar pipes, and the Dither GEMM
# gate and bench. From a pushed commit, on a 1x H200 or 1x B200 node:
#
#   research run --on <node> --project verity --campaign pouw --source <tree> --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=dc -- bash benchmarks/pouw/dc/run.sh sm90|sm100
#
# Outside the lease: build every binary and SASS-gate it (gate-build.txt; a refusal stops the run). Under one lease:
# dgemm gate on three small shapes, each dump rechecked by dgemm_ref.py (dgemm/gate-<shape>/result.json, ref.json); dgemm
# bench only if every gate is exact (dgemm/bench.jsonl); on sm100, fp8acc gate (fp8acc/: is kind::f8f6f4 exact in emu8's
# binades, the B300 question; its CPU selftest runs at build); probe.py run on the family's probes and scalar.cu
# (probes/, scalar/: rows.jsonl, gate.json, times.jsonl) with nvidia-smi sampled every 200 ms (env/smi.csv); then
# analyze.py (analysis.json, verdict.json).
# Exit: 0 all done, 1 build or gate refused, 4 a dgemm gate not exact (probes still ran), 2 wrong device or no nvcc.
# DC_BUILD_ONLY=1 stops after the build and SASS gate and needs no GPU (any machine with CUDA 13.0).
# sm90jb is the stress of the H200 rows (probes/sm90_jbreak.cu): probes only, no dgemm. DC_ONLY=REGEX times only
# the probes it matches and skips dgemm and fp8acc; DC_PASSED_ONLY=1 times what passes when some probes are refused.
set -uo pipefail
FAM=${1:?usage: run.sh sm90|sm100}
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="$HERE/../opcode_probe/probe.py"
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "run.sh: no nvcc (CUDA 13.0 has sm_90a and sm_100a)" >&2; exit 2; }
export NVCC PATH="$(dirname "$NVCC"):$PATH"
case "$FAM" in
  sm90) ARCH=sm_90a; CC=9; SRC=sm90_wgmma; GATES=("128 256 256 7" "256 512 1024 11" "192 384 2048 13") ;;
  sm100) ARCH=sm_100a; CC=10; SRC=sm100_tcgen05; GATES=("128 256 256 7" "256 512 1024 11" "384 768 2048 13") ;;
  sm90jb) ARCH=sm_90a; CC=9; SRC=sm90_jbreak; GATES=(); DC_ONLY=${DC_ONLY:-^jb} ;;
  *) echo "run.sh: family sm90, sm90jb or sm100, not $FAM" >&2; exit 2 ;;
esac
BENCH=("4096 4096 4096" "8192 8192 8192" "512 8192 8192")

if [ "${DC_LEASED:-}" != 1 ]; then
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  mkdir -p "$OUT/sass" "$OUT/env" "$OUT/dgemm" "$OUT/probes" "$OUT/scalar"
  "$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
  if [ "${DC_BUILD_ONLY:-}" != 1 ]; then
    nvidia-smi -q >"$OUT/env/nvidia-smi.txt" 2>&1
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    [ "${cap%%.*}" = "$CC" ] || { echo "run.sh: $FAM needs compute capability $CC.x, the device reports '$cap'" >&2; exit 2; }
  fi

  python3 "$PROBE" build --arch "$ARCH" --out "$WORK" "$HERE/probes/$SRC.cu" "$HERE/probes/scalar.cu" \
    >"$WORK/built" 2>"$OUT/build.log" || { cat "$OUT/build.log" >&2; exit 1; }
  lt=(-DDG_CUBLASLT -lcublasLt -Xlinker "-rpath=$(cd "$(dirname "$NVCC")/../lib64" && pwd -P)")
  "$NVCC" -O3 -std=c++17 -gencode "arch=compute_${ARCH#sm_},code=$ARCH" "${lt[@]}" "$HERE/dgemm.cu" -o "$WORK/dgemm" \
    >>"$OUT/build.log" 2>&1 \
    || "$NVCC" -O3 -std=c++17 -gencode "arch=compute_${ARCH#sm_},code=$ARCH" "$HERE/dgemm.cu" -o "$WORK/dgemm" \
    >>"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
  extra=("$WORK/dgemm")
  if [ "$FAM" = sm100 ]; then
    "$NVCC" -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a "$HERE/fp8acc.cu" -o "$WORK/fp8acc" \
      >>"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
    mkdir -p "$OUT/fp8acc-selftest"
    "$WORK/fp8acc" selftest "$OUT/fp8acc-selftest" | tail -1 || { echo "run.sh: fp8acc selftest failed" >&2; exit 1; }
    extra+=("$WORK/fp8acc")
  fi
  mapfile -t bins <"$WORK/built"
  for b in "${bins[@]}" "${extra[@]}"; do cuobjdump -sass "$b" >"$OUT/sass/$(basename "$b").sass" 2>&1; done
  python3 "$PROBE" gate "${bins[@]}" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
  rc=$?
  grep -c '^PASS' "$OUT/gate-build.txt" | xargs echo "run.sh: gate-build PASS"
  if [ $rc -ne 0 ]; then
    grep -A2 '^REFUSED' "$OUT/gate-build.txt" >&2
    [ "${DC_PASSED_ONLY:-}" = 1 ] || exit 1
  fi
  for b in IGMMA UTCIMMA LDTM; do grep -c "$b" "$OUT/sass/dgemm.sass" | xargs echo "run.sh: dgemm SASS $b"; done
  [ "${DC_BUILD_ONLY:-}" = 1 ] && exit 0

  lease=()
  command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 40 --)
  [ ${#lease[@]} -eq 0 ] && echo "run.sh: no gpu-lease on this node; running on its one GPU directly" >&2
  "${lease[@]}" env DC_LEASED=1 WORK="$WORK" bash "${BASH_SOURCE[0]}" "$FAM"
  exit $?
fi

# ---- leased ------------------------------------------------------------------------------------------------------------
smi_pid=
nvidia-smi --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv -lms 200 >"$OUT/env/smi.csv" 2>&1 &
smi_pid=$!
trap '[ -n "$smi_pid" ] && kill "$smi_pid" 2>/dev/null' EXIT

exact=1
[ -n "${DC_ONLY:-}" ] && GATES=()
for g in "${GATES[@]}"; do
  read -r m n k seed <<<"$g"
  d="$OUT/dgemm/gate-${m}x${n}x${k}"
  mkdir -p "$d"
  "$WORK/dgemm" gate "$d" "$m" "$n" "$k" "$seed" | tee "$d/result.json" || exact=0
  python3 "$HERE/dgemm_ref.py" "$d" | tee "$d/ref.json" || exact=0
  rm -f "$d/X.bin" "$d/W.bin"
done
if [ -n "${DC_ONLY:-}" ]; then
  :
elif [ $exact = 1 ]; then
  for s in "${BENCH[@]}"; do "$WORK/dgemm" bench 20 $s | tee -a "$OUT/dgemm/bench.jsonl"; done
else
  echo "run.sh: a dgemm gate is not exact; no bench (result.json's lbo_sbo_swapped says whether the descriptor fields are)" >&2
fi

if [ "$FAM" = sm100 ] && [ -z "${DC_ONLY:-}" ]; then
  mkdir -p "$OUT/fp8acc"
  "$WORK/fp8acc" gate "$OUT/fp8acc" | tee "$OUT/fp8acc/result.jsonl"
  frc=${PIPESTATUS[0]}
  case "$frc" in
    0) ;;
    4) echo "run.sh: kind::f8f6f4 is not exact in emu8's binades (a finding: fp8acc/summary.json)" >&2 ;;
    *) echo "run.sh: fp8acc gate did not run (exit $frc; see above)" >&2 ;;
  esac
fi

mapfile -t bins <"$WORK/built"
for b in "${bins[@]}"; do
  case "$(basename "$b")" in scalar) o="$OUT/scalar" ;; *) o="$OUT/probes" ;; esac
  sel=()
  [ -n "${DC_ONLY:-}" ] && sel+=(--only "$DC_ONLY")
  [ "${DC_PASSED_ONLY:-}" = 1 ] && sel+=(--passed-only)
  python3 "$PROBE" run "$b" --out "$o" --reps 10 "${sel[@]}" || echo "run.sh: probe.py run $(basename "$b") failed" >&2
done
[ "$FAM" = sm90jb ] || python3 "$HERE/analyze.py" "$FAM" "$OUT" --json "$OUT/analysis.json" | tee "$OUT/verdict.json"
[ $exact = 1 ] || exit 4
