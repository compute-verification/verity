#!/usr/bin/env bash
# lowbit_dc's job on one GPU (B200 sm_100a, H200 sm_90a, or fpx alone on B300 sm_103a; the family is read from the
# device).
#   run.sh       fpx: build fpx.cu and dump its SASS; under one gpu-lease `fpx run` (records.jsonl, windows.jsonl,
#                summary.json); then analyze.py's tables (analysis.md, analysis.json).
#   run.sh gemm  egemm: build egemm.cu and ../dc/dgemm.cu (with cuBLASLt) and dump their SASS; egemm selftest; under one
#                gpu-lease: egemm gate on three shapes x two distributions (gemm/gate.jsonl), then, only if every gate is
#                exact, egemm bench and dgemm bench on the same shapes (gemm/bench.jsonl, gemm/dgemm-bench.jsonl);
#                analyze.py gemm's ratios (gemm/summary.json, gemm/summary.md).
#   run.sh pgemm pgemm (B200 only): build pgemm.cu with cuBLASLt and dump its SASS and registers; pgemm selftest; under
#                one gpu-lease: pgemm gate on three shapes x three distributions (pgemm/gate.jsonl), then, only if every
#                gate is exact, pgemm bench (pgemm/bench.jsonl) and, when the run was sent DC's pipelined int8 kernel
#                (dc/pgemm.cu and its dc.cuh from cursor/eps1-dc-pipe-ea24, sent as dc-pgemm.cu and dc.cuh), its
#                `rate 7 2048`
#                (pgemm/dc-rate.jsonl); analyze.py pgemm's γ, time split and issue cost (pgemm/summary.json,
#                pgemm/summary.md).
#
#   research run --on vy-dc-b200 --project verity --campaign cmp-13 --source <tree> --cwd source \
#     --env GPU_LEASE_WHO=lowbit-dc [--env DC_PGEMM_COMMIT=SHA --send dc-pgemm.cu --send dc.cuh] \
#     -- bash benchmarks/pouw/lowbit_dc/run.sh [gemm|pgemm]
# Exit: 0 every verdict as stated, 4 not, 1 build failed, 2 no device or no nvcc.
set -uo pipefail
OUT=${OUT:?run under research run}
MODE=${1:-fpx}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "run.sh: no nvcc" >&2; exit 2; }
export PATH="$(dirname "$NVCC"):$PATH"
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
case "$cap" in
  9.0) ARCH=90a ;;
  10.0) ARCH=100a ;;
  10.3) ARCH=103a ;;
  *) echo "run.sh: needs sm_90a, sm_100a or sm_103a, the device reports '$cap'" >&2; exit 2 ;;
esac
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
"$NVCC" --version >"$OUT/nvcc.txt" 2>&1
GEN=(-gencode "arch=compute_$ARCH,code=sm_$ARCH")
MMA_RE="UTC[A-Z]*MMA[A-Z0-9_.]*|[A-Z]GMMA\.[0-9x]*\.[A-Z0-9_.]*"
lease=()
command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 15 --)
GPU_CSV='nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit --format=csv -i "${GPU_LEASE_UUID:-0}"'

if [ "$MODE" = fpx ]; then
  "$NVCC" -O3 -std=c++17 -Xptxas -v "${GEN[@]}" "$HERE/fpx.cu" -o "$WORK/fpx" >"$OUT/build.log" 2>&1 \
    || { cat "$OUT/build.log" >&2; exit 1; }
  cuobjdump -sass "$WORK/fpx" >"$OUT/fpx.sass"
  grep -oE "$MMA_RE" "$OUT/fpx.sass" | sort | uniq -c | tee "$OUT/mma-opcodes.txt"
  "${lease[@]}" bash -c "$GPU_CSV"' >"$2/gpu.csv" 2>&1; "$1/fpx" run "$2"' _ "$WORK" "$OUT"
  rc=$?
  python3 "$HERE/analyze.py" "$OUT" --json "$OUT/analysis.json" >"$OUT/analysis.md" || rc=${rc:-1}
  cat "$OUT/summary.json" 2>/dev/null
  exit $rc
fi
lt=(-lcublasLt -Xlinker "-rpath=$(cd "$(dirname "$NVCC")/../lib64" && pwd -P)")
if [ "$MODE" = pgemm ]; then
  [ "$ARCH" = 100a ] || { echo "run.sh: pgemm runs on sm_100a (B200), the device is sm_$ARCH" >&2; exit 2; }
  G="$OUT/pgemm"
  mkdir -p "$G"
  "$NVCC" -O3 -std=c++17 -Xptxas -v "${GEN[@]}" -DPG_CUBLASLT "$HERE/pgemm.cu" "${lt[@]}" -o "$WORK/pgemm" \
    >"$G/build.log" 2>&1 || { cat "$G/build.log" >&2; exit 1; }
  cuobjdump -sass "$WORK/pgemm" >"$G/pgemm.sass"
  grep -A2 "Compiling entry" "$G/build.log" | grep -E "entry|spill|registers" >"$G/registers.txt" || true
  "$WORK/pgemm" selftest | tee "$G/selftest.json" || { echo "run.sh: pgemm selftest failed" >&2; exit 4; }
  DCI="${RESEARCH_RUN_DIR:-/nonexistent}/inputs" dc=0
  if [ -f "$DCI/dc-pgemm.cu" ] && [ -f "$DCI/dc.cuh" ]; then
    echo "${DC_PGEMM_COMMIT:-unknown}" >"$G/dc-pgemm-commit.txt"
    "$NVCC" -O3 -std=c++17 "${GEN[@]}" "$DCI/dc-pgemm.cu" -o "$WORK/dc-pgemm" >"$G/dc-build.log" 2>&1 && dc=1 \
      || { cat "$G/dc-build.log" >&2; exit 1; }
  fi
  "${lease[@]}" bash -c "$GPU_CSV"' >"$2/gpu.csv" 2>&1
    exact=1
    for s in "128 128 128 7" "256 256 1024 11" "384 512 2048 13"; do
      for d in uniform edge max; do "$1/pgemm" gate $s $d | tee -a "$2/gate.jsonl" || exact=0; done
    done
    [ $exact = 1 ] || { echo "run.sh: a pgemm gate is not exact; no bench" >&2; exit 4; }
    for s in "4096 4096 4096" "8192 8192 8192" "4096 14336 4096" "4096 4096 14336"; do
      "$1/pgemm" bench 20 $s | tee -a "$2/bench.jsonl"
    done
    [ "$3" = 0 ] || "$1/dc-pgemm" rate 7 2048 | tee "$2/dc-rate.jsonl"' _ "$WORK" "$G" "$dc"
  rc=$?
  python3 "$HERE/analyze.py" pgemm "$G" --json "$G/summary.json" >"$G/summary.md" || rc=${rc:-1}
  cat "$G/summary.md" 2>/dev/null
  exit $rc
fi
[ "$MODE" = gemm ] || { echo "run.sh: mode fpx, gemm or pgemm, not $MODE" >&2; exit 2; }
[ "$ARCH" != 103a ] || { echo "run.sh: egemm's native int8 arms need kind::i8, which sm_103a lacks; run fpx" >&2; exit 2; }

G="$OUT/gemm"
mkdir -p "$G"
"$NVCC" -O3 -std=c++17 -Xptxas -v "${GEN[@]}" "$HERE/egemm.cu" -o "$WORK/egemm" >"$G/build.log" 2>&1 \
  || { cat "$G/build.log" >&2; exit 1; }
"$NVCC" -O3 -std=c++17 "${GEN[@]}" -DDG_CUBLASLT "$HERE/../dc/dgemm.cu" "${lt[@]}" -o "$WORK/dgemm" >>"$G/build.log" 2>&1 \
  || { cat "$G/build.log" >&2; exit 1; }
for b in egemm dgemm; do cuobjdump -sass "$WORK/$b" >"$G/$b.sass"; done
grep -oE "$MMA_RE|LDTM[A-Z0-9_.]*" "$G/egemm.sass" | sort | uniq -c | tee "$G/mma-opcodes.txt"
grep -B1 -A1 "spill" "$G/build.log" | grep -E "entry|spill" >"$G/registers.txt" || true
"$WORK/egemm" selftest | tee "$G/selftest.json" || { echo "run.sh: egemm selftest failed" >&2; exit 4; }

"${lease[@]}" bash -c "$GPU_CSV"' >"$2/gpu.csv" 2>&1
  exact=1
  for s in "128 256 256 7" "256 512 1024 11" "384 768 2048 13"; do
    for d in uniform edge; do "$1/egemm" gate $s $d | tee -a "$2/gate.jsonl" || exact=0; done
  done
  [ $exact = 1 ] || { echo "run.sh: an egemm gate is not exact; no bench" >&2; exit 4; }
  for s in "4096 4096 4096" "8192 8192 8192" "512 8192 8192" "4096 14336 4096"; do
    "$1/egemm" bench 20 $s | tee -a "$2/bench.jsonl"
    "$1/dgemm" bench 20 $s | tee -a "$2/dgemm-bench.jsonl"
  done' _ "$WORK" "$G"
rc=$?
python3 "$HERE/analyze.py" gemm "$G" --json "$G/summary.json" >"$G/summary.md" || rc=${rc:-1}
cat "$G/summary.md" 2>/dev/null
exit $rc
