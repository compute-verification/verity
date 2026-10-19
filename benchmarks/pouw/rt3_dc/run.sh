#!/usr/bin/env bash
# RT3-DC's job (sm90_cpatom.cu, sm100_cpatom.cu), split so no CPU phase runs on a shared GPU node. From a pushed commit:
#
#   research run --on auto --project verity --campaign eps1 --source <tree> --cwd source -- \
#     bash benchmarks/pouw/rt3_dc/run.sh build
#       builds both for their arch, writes their SASS (sm90_cpatom.sass, sm100_cpatom.sass), probe.py's gate of every
#       probe (gate-sm90_cpatom.{txt,json}, gate-sm100_cpatom.*) and bins.tgz. Exit 1 when a build or a gate verdict
#       fails (the binaries are packed either way, so a refused probe's counts are in the gate's text).
#   research run --on vy-dc-h200 (or vy-dc-b200) --project verity --campaign eps1 --source <tree> --cwd source \
#     --send bins.tgz --timeout 25m --env GPU_LEASE_WHO=rt3-dc -- bash benchmarks/pouw/rt3_dc/run.sh time
#       under one gpu-lease, on the device's own binary: each form bit for bit against the host's reference
#       (exact.jsonl), then probe.py run (gate, then every probe interleaved: probe/times.jsonl, rows.jsonl, gate.json),
#       and on the B200 the SM clock read in the kernel under each load (clock.jsonl). NVML sampled every 100 ms
#       (smi.csv), its clock, performance and power reports before and after (smi-{before,after}.txt).
#       STEPS (default "exact probe clock") picks which run, REPS (default 5) probe.py's reps.
# Exit: 0 done, 1 build, gate or a probe failed, 2 no nvcc, no inputs or wrong device.
set -uo pipefail
MODE=${1:?usage: run.sh build | time}
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="$HERE/../opcode_probe/probe.py"
export PYTHONDONTWRITEBYTECODE=1

case "$MODE" in
build)
  NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
  [ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "run.sh: no nvcc" >&2; exit 2; }
  export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$NVCC")")}
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  "$NVCC" --version >"$OUT/nvcc.txt" 2>&1
  rc=0
  for pair in sm90_cpatom:sm_90a sm100_cpatom:sm_100a; do
    src=${pair%%:*}
    arch=${pair##*:}
    NVCC="$NVCC" python3 "$PROBE" build --out "$WORK" --arch "$arch" "$HERE/$src.cu" >"$OUT/build-$src.log" 2>&1 \
      || { cat "$OUT/build-$src.log" >&2; exit 1; }
    "$CUDA_HOME/bin/cuobjdump" -sass "$WORK/$src" | "$CUDA_HOME/bin/cu++filt" >"$OUT/$src.sass" 2>&1
    python3 "$PROBE" gate "$WORK/$src" --json "$OUT/gate-$src.json" >"$OUT/gate-$src.txt" 2>&1 || rc=1
    cat "$OUT/gate-$src.txt"
  done
  tar -C "$WORK" -czf "$OUT/bins.tgz" sm90_cpatom sm100_cpatom
  exit $rc
  ;;
time)
  IN=${RESEARCH_RUN_DIR:?run under research run}/inputs
  if [ "${RT3_LEASED:-}" != 1 ]; then
    [ -f "$IN/bins.tgz" ] || { echo "run.sh: no inputs/bins.tgz (pass the build's with --send)" >&2; exit 2; }
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    case "${cap%%.*}" in
    9) B=sm90_cpatom ;;
    10) B=sm100_cpatom ;;
    *) echo "run.sh: needs compute capability 9.x or 10.x, the device reports '$cap'" >&2; exit 2 ;;
    esac
    WORK=$(mktemp -d)
    trap 'rm -rf "$WORK"' EXIT
    tar -C "$WORK" -xzf "$IN/bins.tgz"
    lease=()
    command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 20 --)
    [ ${#lease[@]} -eq 0 ] && echo "run.sh: no gpu-lease on this node; running on its GPU directly" >&2
    "${lease[@]}" env RT3_LEASED=1 WORK="$WORK" B="$B" bash "${BASH_SOURCE[0]}" time
    exit $?
  fi
  BIN="$WORK/$B"
  nvidia-smi -q -d CLOCK,PERFORMANCE,POWER,TEMPERATURE >"$OUT/smi-before.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.draw,enforced.power.limit,temperature.gpu,clocks_event_reasons.active \
    --format=csv -lms 100 >"$OUT/smi.csv" 2>&1 &
  smi=$!
  rc=0
  STEPS=" ${STEPS:-exact probe clock} "
  STEPS=${STEPS//,/ }
  [[ $STEPS == *" exact "* ]] && { "$BIN" exact 64 >"$OUT/exact.jsonl" 2>&1 || rc=1; }
  [[ $STEPS == *" probe "* ]] && { python3 "$PROBE" run "$BIN" --out "$OUT/probe" --reps "${REPS:-5}" \
    >"$OUT/probe.txt" 2>&1 || rc=1; }
  if [[ $STEPS == *" clock "* && $B == sm100_cpatom ]]; then
    : >"$OUT/clock.jsonl"
    # ~0.3-0.5 s a launch, 20 launches a load: the fold alone, the MMA alone, the honest mixed load (♭'s floor beside
    # the MMA), and the MMA alone again after a minute's load
    for spec in "2 150000" "0 1200000" "1 1200000" "0 1200000"; do
      set -- $spec
      "$BIN" clock "$1" "$2" 20 >>"$OUT/clock.jsonl" 2>&1 || rc=1
    done
  fi
  kill "$smi" 2>/dev/null
  wait "$smi" 2>/dev/null
  nvidia-smi -q -d CLOCK,PERFORMANCE,POWER,TEMPERATURE >"$OUT/smi-after.txt" 2>&1
  exit $rc
  ;;
*)
  echo "usage: run.sh build | time" >&2
  exit 2
  ;;
esac
