#!/usr/bin/env bash
# Instruction-count profile of the dietmc prover with valgrind --tool=callgrind.
# perf (perf_event_paranoid=4) and ptrace attach (no CAP_SYS_PTRACE) are both unavailable in the vy-cpu
# container, so this is the available profiler: exact instruction counts per function (Ir), not time.
# Instructions are a fair proxy here because the prover is single-threaded scalar integer code; the one
# place they mislead is blocking I/O, which costs no instructions -- the wall-time split in GPU.md s1 comes
# from dietmc's timers, this profile attributes the *compute* within each phase.
#
#   profile.sh <dir-with-relation.txt,instance.txt,witness.txt> [field] [threads]
set -euo pipefail
DIR=$1; FIELD=${2:-F61p}; THREADS=${3:-1}
DIETMC=${DIETMC:-/workspace/target-swanky/release/dietmc}
OUT=${OUT:-$DIR/profile}
PORT=${PORT:-7311}
mkdir -p "$OUT"
printf "lpn = 'medium'\nthreads = %s\nno_batching = false\n" "$THREADS" > "$OUT/dmc.toml"
BASE=("$DIETMC" --text --relation "$DIR/relation.txt" --instance "$DIR/instance.txt" --config "$OUT/dmc.toml" --connection-addr "127.0.0.1:$PORT")
RUST_LOG=info "${BASE[@]}" > "$OUT/verifier.log" 2>&1 &
VPID=$!
sleep 0.5
RUST_LOG=info valgrind --tool=callgrind --callgrind-out-file="$OUT/callgrind.out" --compress-strings=no --compress-pos=no \
  "${BASE[@]}" --witness "$DIR/witness.txt" > "$OUT/prover.log" 2>&1 || true
wait "$VPID" || true
callgrind_annotate --inclusive=no "$OUT/callgrind.out" > "$OUT/annotate_self.txt" 2>/dev/null || true
callgrind_annotate --inclusive=yes "$OUT/callgrind.out" > "$OUT/annotate_incl.txt" 2>/dev/null || true
python3 "$(dirname "$0")/profile_buckets.py" "$OUT/annotate_self.txt"
