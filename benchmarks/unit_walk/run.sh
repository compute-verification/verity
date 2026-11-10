#!/usr/bin/env bash
# The device witness's host unit walk (`flock-circuit unit-walk`, `unit_ab_into`) on served proof-block types: each type
# staged once (benchmarks/served_units/stage.py, two units), then its unit walked over WALK_SLOTS seeded slots at each
# width (FC_UNIT_WIDE: lane groups of 64 slots a pass) on each pool size, WALK_REPS rounds, one type at a time after every
# stage has finished. The guard: every WALK line's `same`, each width's packed words the first run's (rc 9 if not). The
# CPU seconds a slot at width 1 are the baseline, and a wider pass's are what it buys.
# Run from the verity tree (a job run from a clone of it):
#   bash benchmarks/unit_walk/run.sh WALK_UNITS="gq:K=2048 fp8-coord:K=6144" WALK_SLOTS=2048 WALK_THREADS=16,4
# env: WALK_UNITS="gq:K=2048" (stage.py's TYPE[:S=V,...], space-separated)  WALK_SLOTS=2048  WALK_WIDE=1,8,4,2
#      WALK_THREADS=$(nproc),4 (pool sizes)  WALK_REPS=2
# Kept in $OUT: each type's stage.json, its WALK lines (walk-N.out) and UNITPROF lines (walk-N.err), the logs; never a
# circuit.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH PYTHONDONTWRITEBYTECODE=1
{ lscpu | grep -E 'Model name|^CPU\(s\)|L3'; nproc; free -g; git rev-parse HEAD; } > "$O/host.txt" 2>&1
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }
PY=$REPO/.venv/bin/python
export PYTHONPATH=$REPO:$REPO/verity/ml/flock/python:$REPO/integrations/vllm:$REPO/tools/verity/circuit_check/src:$REPO/benchmarks/numerical/python
read -ra UNITS <<< "${WALK_UNITS:-gq:K=2048}"
SLOTS=${WALK_SLOTS:-2048} WIDE=${WALK_WIDE:-1,8,4,2} THREADS=${WALK_THREADS:-$(nproc),4} REPS=${WALK_REPS:-2}
log "unit-walk: source $(git rev-parse HEAD), units ${UNITS[*]}, builds in $W"
mkdir -p "$O/m0-build"
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$W MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh > "$O/m0-build/build.log" 2>&1 &
pids=($!)
for i in "${!UNITS[@]}"; do
  "$PY" benchmarks/served_units/stage.py "${UNITS[$i]}" 2 "$W/u$i" > "$O/stage-$i.log" 2>&1 &
  pids+=($!)
done
wait "${pids[0]}" || { tail -40 "$O/m0-build/build.log"; exit 6; }
BIN=$W/flock-gpu-link/flock/target/release/flock-circuit
staged=()
for i in "${!UNITS[@]}"; do
  if wait "${pids[$((i + 1))]}"; then staged+=("$i"); cp "$W/u$i/stage.json" "$O/stage-$i.json"; else log "stage ${UNITS[$i]} failed"; tail -20 "$O/stage-$i.log"; fi
done
rc=0
[ "${#staged[@]}" = "${#UNITS[@]}" ] || rc=7
for i in "${staged[@]}"; do
  log "walk ${UNITS[$i]}: $SLOTS slots, widths $WIDE, threads $THREADS, $REPS rounds"
  (cd "$W/u$i" && FC_UNIT_PROFILE=1 "$BIN" unit-walk --circuit circuit.txt --slots "$SLOTS" --wide "$WIDE" --threads "$THREADS" \
    --reps "$REPS") > "$O/walk-$i.out" 2> "$O/walk-$i.err" || { log "walk ${UNITS[$i]}: rc $?"; tail -5 "$O/walk-$i.err"; rc=8; }
  if grep -q '"same":false' "$O/walk-$i.out"; then log "walk ${UNITS[$i]}: a width's packed words are not the first run's"; rc=9; fi
  rm -rf "$W/u$i"
done
log "unit-walk: done, rc $rc"
exit $rc
