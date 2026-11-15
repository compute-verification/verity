# leg B, one CPU lane on a CPU node: LANE_UNITS's statements past the GPU prover's bound, BATCH at a time in order: each
# batch staged on disk (stage_prove.sh DEV=cpu STEPS=stage), then proved on the CPU with the Lean verifier on its kept
# sessions (STEPS=prove LEAN=1), its FLOCK_WORK removed before the next batch and at any exit. Batch K's files are under
# out/bK/out (steps.txt, kept/, lean.tsv); out/batches.tsv a row a batch. A failed batch is recorded and not retried.
# env: LANE_UNITS (a name sent with the run) BATCH KILL_AT (UTC epoch s) WIN_MEM_GB WIN_DISK_GB; FW_ROOT (/workspace/tmp;
#      /dev/shm with RAM_WORK=1 stages in RAM, its executables on stage_prove.sh's DISK_WORK, removed with it)
#   research run --on NODE --pool proofs --cwd clone --source TREE --send cpu_lane.sh --send LANE.units --env LANE_UNITS=LANE.units \
#     --env BATCH=N --env KILL_AT=EPOCH --env WIN_MEM_GB=G --env WIN_DISK_GB=G --mem-gb G \
#     -- bash -c 'set -euo pipefail; bash "$RESEARCH_RUN_DIR/inputs/cpu_lane.sh"'
set -u
IN=$RESEARCH_RUN_DIR/inputs; OUT=$RESEARCH_RUN_DIR/out; mkdir -p "$OUT"
FW=
gone() { [ -n "$FW" ] || return 0; local ex; ex=$(cat "$FW/.disk_work" 2>/dev/null); [ -z "$ex" ] || rm -rf "$ex"; rm -rf "$FW"; FW=; }
trap 'gone; echo "$(date -u +%H:%M:%SZ) lane ended" >> "$OUT/steps.txt"' EXIT
trap 'exit 143' TERM INT
say() { echo "$(date -u +%H:%M:%SZ) $*" >> "$OUT/steps.txt"; }
uv sync -q --all-packages --extra torch-cpu || exit 3
export DEV=cpu KILL_AT WIN_MEM_GB WIN_DISK_GB WARM=${WARM:-1} RUNS=${RUNS:-4}
grep -v '^#' "$IN/$LANE_UNITS" | split -l "${BATCH:?}" -d -a 2 - "$OUT/b"
printf 'batch\tstatements\tstage_exit\tprove_exit\tseconds\n' > "$OUT/batches.tsv"
for f in "$OUT"/b[0-9][0-9]; do
  k=$(basename "$f"); mv "$f" "$f.units"; mkdir -p "$OUT/$k"; t0=$(date +%s)
  FW=${FW_ROOT:-/workspace/tmp}/proofs-legB-cpu-${RESEARCH_RUN_ID:-$$}-$k
  say "$k: $(wc -l < "$f.units") statements"
  RESEARCH_RUN_DIR=$OUT/$k FLOCK_WORK=$FW STEPS=stage UNITS=$f.units bash benchmarks/served_units/stage_prove.sh
  se=$?; pe=-
  if [ -d "$FW/data" ]; then
    RESEARCH_RUN_DIR=$OUT/$k FLOCK_WORK=$FW STEPS=prove UNITS=$f.units bash benchmarks/served_units/stage_prove.sh
    pe=$?
  fi
  gone
  printf '%s\t%s\t%s\t%s\t%s\n' "$k" "$(wc -l < "$f.units")" "$se" "$pe" "$(( $(date +%s) - t0 ))" >> "$OUT/batches.tsv"
  say "$k: stage exit $se, prove exit $pe"
  [ $(( KILL_AT - $(date +%s) )) -ge 1800 ] || { say "under 1800 s to KILL_AT: no more batches"; break; }
done
