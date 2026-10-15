#!/usr/bin/env bash
# Runs every line of a job file (one shell command per line, run from this directory) with up to P at once.
# Each job's stdout goes to $OUT/<NNN>.json, its stderr to $OUT/<NNN>.err; $OUT/summary.txt lists exit codes.
#   OUT=dir bash run_jobs.sh JOBFILE P
set -u
here=$(cd "$(dirname "$0")" && pwd)
jobs=$1
par=${2:-8}
: "${OUT:?set OUT}"
mkdir -p "$OUT"
cd "$here"
export OUT
nl -ba -nrz -w3 "$jobs" | grep -v -E '^\S+\s+(#|$)' | \
  xargs -P "$par" -L 1 bash -c 'n=$0; cmd="$*"; bash -c "$cmd" > "$OUT/$n.json" 2> "$OUT/$n.err"; echo "$n rc=$? $cmd" >> "$OUT/summary.txt"'
sort -o "$OUT/summary.txt" "$OUT/summary.txt"
fails=$(grep -c -v ' rc=0 ' "$OUT/summary.txt")
echo "jobs: $(wc -l < "$OUT/summary.txt"), nonzero: $fails"
[ "$fails" -eq 0 ]
