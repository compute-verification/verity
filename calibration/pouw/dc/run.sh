#!/usr/bin/env bash
# The data-centre probes on one GPU: the probes (M3 on sm_90a, M4 on sm_100a), the scalar pipes, and the Dither GEMM
# gate and bench. From a pushed commit, on a 1x H200 or 1x B200 node:
#
#   research run --on <node> --project verity --campaign pouw --source <tree> --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=dc -- bash calibration/pouw/dc/run.sh
#
# The card is the one the run opened: its card record's `arch` picks the family (sm_90a: sm90, sm_100a: sm100; another
# is refused before anything is built) and its architecture file gives nvcc's -gencode (probe.py arch, gencode).
# Outside the lease: build every binary and SASS-gate it (gate-build.txt; a refusal stops the run). Under one lease:
# dgemm gate on three small shapes, each dump rechecked by dgemm_ref.py (dgemm/gate-<shape>/result.json, ref.json); dgemm
# bench only if every gate is exact (dgemm/bench.jsonl); on sm100, fp8acc gate (fp8acc/: is kind::f8f6f4 exact in emu8's
# binades, the B300 question; its CPU selftest runs at build); probe.py run on the family's probes and scalar.cu
# (probes/, scalar/: rows.jsonl, gate.json, times.jsonl) with nvidia-smi sampled every 200 ms (env/smi.csv); then
# analyze.py (analysis.json, verdict.json).
# Exit: 0 all done, 1 build or gate refused, 4 a dgemm gate not exact (probes still ran), 2 wrong device or no nvcc.
# DC_BUILD_ONLY=1 stops after the build and SASS gate and leases no GPU; on a node with none, DC_CARD names the card to
# build for (the name nvidia-smi gives it). DC_JBREAK=1 (sm90jb) is the stress of the H200 rows (probes/sm90_jbreak.cu):
# probes only, no dgemm. DC_ONLY=REGEX times only
# the probes it matches and skips dgemm and fp8acc; DC_PASSED_ONLY=1 times what passes when some probes are refused.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="$HERE/../opcode_probe/probe.py"
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "run.sh: no nvcc (CUDA 13.0 has sm_90a and sm_100a)" >&2; exit 2; }
export NVCC PATH="$(dirname "$NVCC"):$PATH"
CARD=(${DC_CARD:+--card "$DC_CARD"})
ARCH=$(python3 "$PROBE" arch "${CARD[@]}") || exit 2
case "$ARCH" in
  sm_90a) FAM=sm90; SRC=sm90_wgmma; GATES=("128 256 256 7" "256 512 1024 11" "192 384 2048 13") ;;
  sm_100a) FAM=sm100; SRC=sm100_tcgen05; GATES=("128 256 256 7" "256 512 1024 11" "384 768 2048 13") ;;
  *) echo "run.sh: the card's target is $ARCH, and dc has probes for sm_90a and sm_100a only: nothing built" >&2; exit 2 ;;
esac
if [ "${DC_JBREAK:-}" = 1 ]; then
  [ "$FAM" = sm90 ] || { echo "run.sh: DC_JBREAK stresses the sm_90a rows, and the card's target is $ARCH" >&2; exit 2; }
  FAM=sm90jb; SRC=sm90_jbreak; GATES=(); DC_ONLY=${DC_ONLY:-^jb}
fi
GENCODE=$(python3 "$PROBE" gencode "${CARD[@]}") || exit 2
BENCH=("4096 4096 4096" "8192 8192 8192" "512 8192 8192")

if [ "${DC_LEASED:-}" != 1 ]; then
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  mkdir -p "$OUT/sass" "$OUT/env" "$OUT/dgemm" "$OUT/probes" "$OUT/scalar"
  "$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
  [ "${DC_BUILD_ONLY:-}" = 1 ] || nvidia-smi -q >"$OUT/env/nvidia-smi.txt" 2>&1

  python3 "$PROBE" build "${CARD[@]}" --out "$WORK" "$HERE/probes/$SRC.cu" "$HERE/probes/scalar.cu" \
    >"$WORK/built" 2>"$OUT/build.log" || { cat "$OUT/build.log" >&2; exit 1; }
  lt=(-DDG_CUBLASLT -lcublasLt -Xlinker "-rpath=$(cd "$(dirname "$NVCC")/../lib64" && pwd -P)")
  "$NVCC" -O3 -std=c++17 -gencode "$GENCODE" "${lt[@]}" "$HERE/dgemm.cu" -o "$WORK/dgemm" \
    >>"$OUT/build.log" 2>&1 \
    || "$NVCC" -O3 -std=c++17 -gencode "$GENCODE" "$HERE/dgemm.cu" -o "$WORK/dgemm" \
    >>"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
  extra=("$WORK/dgemm")
  if [ "$FAM" = sm100 ]; then
    "$NVCC" -O3 -std=c++17 -gencode "$GENCODE" "$HERE/fp8acc.cu" -o "$WORK/fp8acc" \
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
  [ -z "${GPU_LEASE_UUID:-}" ] && command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 40 --)
  [ ${#lease[@]} -eq 0 ] && [ -z "${GPU_LEASE_UUID:-}" ] && echo "run.sh: no gpu-lease on this node; running on its one GPU directly" >&2
  "${lease[@]}" env DC_LEASED=1 WORK="$WORK" bash "${BASH_SOURCE[0]}"
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
prc=0
for b in "${bins[@]}"; do
  case "$(basename "$b")" in scalar) o="$OUT/scalar" ;; *) o="$OUT/probes" ;; esac
  sel=()
  [ -n "${DC_ONLY:-}" ] && sel+=(--only "$DC_ONLY")
  [ "${DC_PASSED_ONLY:-}" = 1 ] && sel+=(--passed-only)
  python3 "$PROBE" run "$b" --out "$o" --reps 10 "${sel[@]}" \
    || { echo "run.sh: probe.py run $(basename "$b") failed" >&2; prc=1; }
done
record=(--record "$OUT")
[ $prc = 0 ] || record+=(--failed "probe.py run refused or failed a binary")
[ $exact = 1 ] || record=(--record "$OUT" --failed "a dgemm gate is not exact")
ROOT=$HERE/../../..
[ "$FAM" = sm90jb ] || PYTHONPATH="$ROOT/benchmarks/harness:$ROOT/tools/config${PYTHONPATH:+:$PYTHONPATH}" \
  python3 "$HERE/analyze.py" "$FAM" "$OUT" --json "$OUT/analysis.json" "${record[@]}" | tee "$OUT/verdict.json"
[ $exact = 1 ] || exit 4
[ $prc = 0 ] || exit 1
