#!/usr/bin/env bash
# run.py on one leased RTX PRO 6000, from build.sh's OUT sent as a tarball, on the leased GPU's NUMA node:
#   tar -cf ship.tar -C <build.sh's OUT> .
#   research run --on MACHINE --project verity --source <tree> --cwd source --env GPU_LEASE_WHO=LANE \
#     [--env GPUS=2,3] [--env SHAPES="8x4096x4096 ..."] [--env REPS=20] --send ship.tar --send numabind.py \
#     -- bash benchmarks/pouw/pearl_c_sm120/card.sh
# First the check alone on both variants (OUT/check.jsonl), then the check and the timing at SHAPES on VARIANTS (default
# sm120, the served one; OUT/run.jsonl); each run's summary in OUT/<name>.txt, and the job's exit code the first failure's.
# GPUS: the GPUs the lease may take (gpu-lease --on); MAX_MIN the lease (15); GPU_LEASE_TIMED=1 for a row taken inside a
# booked clean window (gpu-lease --timed). numabind.py is benchmarks/hash/dither/numabind.py; without it run.py
# runs unbound. The node's load (loadavg and the leases) before and after each part is OUT/load.txt. Scratch, the unpacked
# ship, is one mktemp -d directory.
set -euo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:?run this under research run}/out}; mkdir -p "$OUT"; export OUT
IN=${RESEARCH_RUN_DIR:-.}/inputs
LEASE=$(command -v gpu-lease || echo /workspace/pouw/infra/bin/gpu-lease)
load() { { echo "== $1 $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg)"; "$LEASE" status 2>&1 || true; } >> "$OUT/load.txt"; }
if [ "${1:-}" != gpu ]; then
  SHIP=$(mktemp -d); export SHIP; trap 'rm -rf "$SHIP"' EXIT
  tar -xf "$IN/ship.tar" -C "$SHIP"
  BIND=(); [ -f "$IN/numabind.py" ] && BIND=(python3 "$IN/numabind.py" "$OUT/numa.json")
  ON=(); [ -n "${GPUS:-}" ] && ON=(--on "$GPUS")
  rc=0
  "$LEASE" 1 --wait "${ON[@]}" --max-min "${MAX_MIN:-15}" --mem-gb 16 -- "${BIND[@]}" bash "$0" gpu || rc=$?
  exit "$rc"
fi
export PEARLC_EXPECT_UUID=${GPU_LEASE_UUID:?run under gpu-lease}; PEARLC_EXPECT_UUID=${PEARLC_EXPECT_UUID%%,*}
cd "$SHIP"
sha256sum pearl_c_sm120.cubin run.py check/check.json > "$OUT/ship.sha256"
rc=0
load check; SHAPES= VARIANTS="sm120 sm120-unpromoted" python3 run.py check/check.json > "$OUT/check.jsonl" 2> "$OUT/check.txt" || rc=$?
load run; r=0; VARIANTS=${VARIANTS:-sm120} python3 run.py check/check.json > "$OUT/run.jsonl" 2> "$OUT/run.txt" || r=$?
load end; [ "$rc" != 0 ] || rc=$r
tail -n 3 "$OUT/check.txt" "$OUT/run.txt"
exit "$rc"
