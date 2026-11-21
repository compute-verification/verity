#!/usr/bin/env bash
# A card's empirical check: build its probes on a CPU node, then time them on the card under one gpu-lease, so no CPU
# phase runs on the shared GPU node; or a named probe set built and timed on the card the run opened.
#
#   research run --on H200 --project verity --campaign pouw --source <tree> --cwd source \
#     --env GPU_LEASE_WHO=LANE -- bash calibration/pouw/dc/emp.sh run SET [REPS]
#       SET h32 (probes/<family>_h32*.cu, analyzed by h32.py into h32.txt, h32.json and h32.lean) or emp
#       (probes/<family>_emp.cu, analyzed by emp.py into emp.txt and emp.json). The card the run opened sets the target
#       (probe.py arch, its family from archs.py, nvcc's -gencode from probe.py build); a target archs.py or the set has
#       no probes for is refused by name before anything is built. Builds and gates as build does, then times as time
#       does, into the same $OUT/sass/ and $OUT/<binary>/, and analyzes outside the lease.
#   research run --on auto --wait -- bash calibration/pouw/dc/emp.sh build SKU SRC...
#       builds each SRC for the card file calibration/hardware/cards/SKU.toml (its `arch`, and that architecture's file
#       for nvcc's -gencode: probe.py build --sku), gates every probe's SASS (probe.py gate) and writes $OUT/bins.tgz,
#       $OUT/sass/, gate-build.txt and gate-build.json. A source that fails to build is named and skipped; the rest still build.
#       bins.tgz carries the build's cuobjdump and nvdisasm in tools/, which time's gate reads SASS with
#       (probe.py run --cuda-bin), so a node with no CUDA toolkit gates the binaries with the tools that built them.
#   research run --on H200 --project verity --campaign pouw --source <tree> --cwd source --send bins.tgz \
#     --env GPU_LEASE_WHO=LANE -- bash calibration/pouw/dc/emp.sh time [REPS]
#       unpacks the build's bins.tgz (from $RESEARCH_RUN_DIR/inputs/), records the clocks (A1: nvidia-smi -q -d CLOCK
#       before and after, clocks sampled every 200 ms under the lease) and runs probe.py run on every binary (gate, the
#       poisoned sink and its control, then the interleaved timing) into $OUT/<binary>/. With EMP_ONLY=REGEX it times
#       only the probes whose name matches (probe.py run --only), and with EMP_PASSED_ONLY=1 it times the probes that
#       pass even where others are refused (probe.py run --passed-only). With EMP_GPU=INDEX (and EMP_CORES=CPUS) it takes
#       only that card (gpu-lease --on, --cores), once and without waiting, and first exits 3, touching no card, if
#       that card holds a compute process; $OUT/env/node-{before,after}.txt hold every card's leases and processes.
# Exit: 0 done, 1 nothing built, a gate verdict not the declared one or a timing failed, 2 no nvcc, no inputs or a
# target refused, 3 the card in use.
set -uo pipefail
MODE=${1:?usage: emp.sh run SET [REPS] | build SKU SRC... | time [REPS]}
shift
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
PROBE="$HERE/../opcode_probe/probe.py"
export PYTHONDONTWRITEBYTECODE=1

# build_all BIN [--sku SKU] -- SRC...: each SRC into BIN, its SASS gated; leaves the gate's exit in GATE_RC.
build_all() {
  local bin=$1 card=() src
  shift
  while [ "$1" != -- ]; do card+=("$1"); shift; done
  shift
  . "$HERE/../nvcc.sh"
  export NVCC PATH="$(dirname "$NVCC"):$PATH"
  mkdir -p "$bin" "$OUT/sass"
  "$NVCC" --version >"$OUT/nvcc.txt" 2>&1
  for src in "$@"; do
    python3 "$PROBE" build "${card[@]}" --out "$bin" "$src" >>"$OUT/build.log" 2>&1 \
      || echo "emp.sh: build failed: $src (build.log)" | tee -a "$OUT/build.log" >&2
  done
  shopt -s nullglob
  local bins=("$bin"/*)
  shopt -u nullglob
  [ ${#bins[@]} -gt 0 ] || { echo "emp.sh: nothing built" >&2; exit 1; }
  for b in "${bins[@]}"; do cuobjdump -sass "$b" >"$OUT/sass/$(basename "$b").sass" 2>&1; done
  python3 "$PROBE" gate "${bins[@]}" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
  GATE_RC=$?
  grep -c '^PASS' "$OUT/gate-build.txt" | xargs echo "emp.sh: gate PASS"
  grep -A3 '^REFUSED' "$OUT/gate-build.txt" | grep -v 'as declared' >&2
  mkdir "$bin/tools"
  cp "$(command -v cuobjdump)" "$(command -v nvdisasm)" "$bin/tools/"
}

# time_leased BIN REPS: time BIN's binaries under one gpu-lease (time's leased half, below); returns its exit.
time_leased() {
  local bin=$1 reps=$2 lease=() rc=0
  mkdir -p "$OUT/env"
  if [ -n "${EMP_GPU:-}" ]; then
    command -v gpu-lease >/dev/null || { echo "emp.sh: EMP_GPU needs gpu-lease" >&2; return 2; }
    node_state() { gpu-lease status; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv; }
    node_state >"$OUT/env/node-before.txt" 2>&1
    local busy
    busy=$(nvidia-smi -i "$EMP_GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)
    [ -z "$busy" ] || { echo "emp.sh: GPU $EMP_GPU holds a compute process ($busy); nothing started" >&2; return 3; }
    lease=(gpu-lease 1 --on "$EMP_GPU" ${EMP_CORES:+--cores "$EMP_CORES"} --max-min 25 --)
  elif command -v gpu-lease >/dev/null; then
    lease=(gpu-lease 1 --wait --max-min 25 --)
  fi
  [ ${#lease[@]} -eq 0 ] && echo "emp.sh: no gpu-lease on this node; running on its GPU directly" >&2
  "${lease[@]}" env EMP_LEASED=1 WORK="$bin" REPS="$reps" bash "${BASH_SOURCE[0]}" time "$reps" || rc=$?
  [ -n "${EMP_GPU:-}" ] && node_state >"$OUT/env/node-after.txt" 2>&1
  [ $rc -ne 75 ] || { echo "emp.sh: GPU $EMP_GPU's lease is held; nothing started" >&2; return 3; }
  return $rc
}

case "$MODE" in
run)
  SET=${1:?usage: emp.sh run SET [REPS], SET h32 or emp}
  REPS=${2:-5}
  case "$SET" in h32|emp) ;; *) echo "emp.sh: run takes the set h32 or emp, not $SET" >&2; exit 2 ;; esac
  ARCH=$(python3 "$PROBE" arch) || exit 2
  FAM=$(python3 "$HERE/archs.py" "$ARCH" family) || exit 2
  srcs=()
  for f in "$HERE/probes/${FAM}_$SET"*.cu; do
    case "$(basename "$f")" in "${FAM}_$SET.cu"|"${FAM}_${SET}_"*.cu) [ -f "$f" ] && srcs+=("$f") ;; esac
  done
  [ ${#srcs[@]} -gt 0 ] || { echo "emp.sh: the card's target is $ARCH, and the $SET probes have no source for it" \
    "(probes/${FAM}_$SET*.cu): nothing built" >&2; exit 2; }
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  build_all "$WORK/bin" -- "${srcs[@]}"
  [ "$GATE_RC" -eq 0 ] || exit 1
  time_leased "$WORK/bin" "$REPS" || exit $?
  if [ "$SET" = h32 ]; then
    python3 "$HERE/h32.py" "$OUT" --json "$OUT/h32.json" --lean "$OUT/h32.lean" >"$OUT/h32.txt" 2>&1
    arc=$?
  else
    PYTHONPATH="$ROOT:$ROOT/tools/config" python3 "$HERE/emp.py" "$OUT" "$OUT/sass" --json "$OUT/emp.json" \
      >"$OUT/emp.txt" 2>&1
    arc=$?
  fi
  echo "emp.sh: $SET timed; its analysis exited $arc ($OUT/$SET.txt)"
  ;;
build)
  SKU=${1:?usage: emp.sh build SKU SRC..., SKU the card file to build for (calibration/hardware/cards/SKU.toml)}
  shift
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  build_all "$WORK/bin" --sku "$SKU" -- "$@"
  tar -C "$WORK/bin" -czf "$OUT/bins.tgz" .
  [ "$GATE_RC" -eq 0 ] || exit 1
  ;;
time)
  REPS=${1:-5}
  if [ "${EMP_LEASED:-}" != 1 ]; then
    IN=${RESEARCH_RUN_DIR:?run under research run}/inputs
    [ -f "$IN/bins.tgz" ] || { echo "emp.sh: no inputs/bins.tgz (pass the build's with --send)" >&2; exit 2; }
    WORK=$(mktemp -d)
    trap 'rm -rf "$WORK"' EXIT
    tar -C "$WORK" -xzf "$IN/bins.tgz"
    time_leased "$WORK" "$REPS"
    exit $?
  fi
  dev=()
  [ -n "${GPU_LEASE_UUID:-}" ] && dev=(-i "$GPU_LEASE_UUID")
  nvidia-smi "${dev[@]}" -q -d CLOCK,PERFORMANCE,POWER >"$OUT/env/clocks-before.txt" 2>&1
  nvidia-smi "${dev[@]}" --query-gpu=name,uuid,compute_cap,clocks.max.sm,clocks.max.mem,driver_version \
    --format=csv >"$OUT/env/device.csv" 2>&1
  nvidia-smi "${dev[@]}" --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv -lms 200 >"$OUT/env/smi.csv" 2>&1 &
  smi_pid=$!
  trap 'kill "$smi_pid" 2>/dev/null' EXIT
  rc=0
  for b in "$WORK"/*; do
    [ -f "$b" ] && [ -x "$b" ] || continue
    python3 "$PROBE" run "$b" --cuda-bin "$WORK/tools" --out "$OUT/$(basename "$b")" --reps "$REPS" \
      ${EMP_ONLY:+--only "$EMP_ONLY"} ${EMP_PASSED_ONLY:+--passed-only} \
      || { echo "emp.sh: probe.py run $(basename "$b") failed" >&2; rc=1; }
  done
  nvidia-smi "${dev[@]}" -q -d CLOCK,PERFORMANCE,POWER >"$OUT/env/clocks-after.txt" 2>&1
  exit $rc
  ;;
*)
  echo "emp.sh: run, build or time, not $MODE" >&2
  exit 2
  ;;
esac
