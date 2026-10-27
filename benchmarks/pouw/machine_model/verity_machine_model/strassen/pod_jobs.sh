#!/usr/bin/env bash
# The note's tables too slow for a VM, on a CPU node: `research run --cwd source/benchmarks/pouw/machine_model -- bash
# verity_machine_model/strassen/pod_jobs.sh crt noise`. Writes into $OUT; scratch (the venv, the compiled splits) in
# a mktemp dir. crt: each instance's residue dump, then its CRT crossover at each leaf size. noise: HYB's noise-stripping
# rows, U at +-80 against X-side noise +-u for every u in 64..128, and both sides at +-u.
set -u
: "${OUT:?}"
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
export TMPDIR=$d UV_PROJECT_ENVIRONMENT=$d/venv
uv sync --frozen --project . -q
PY=$d/venv/bin/python
J=${JOBS:-60}

crt() {  # NAME "B1 B2 ..."
  local i=$1
  $PY -m verity_machine_model.strassen.crt "$i" 12 1,16,32 "$OUT/w-$i.json" > "$OUT/crt-$i.txt" 2>&1
  echo "crt $i $?" >> "$OUT/done.txt"
  for b in $2; do
    ( $PY -m verity_machine_model.strassen.crt crossover "$i" "$OUT/w-$i.json" "$b" > "$OUT/x-$i-b$b.txt" 2>&1
      echo "x $i $b $?" >> "$OUT/done.txt" ) &
  done
  wait
}

noise() {  # NAME U
  local i=$1 u=$2 w=$((2 * $2 + 1))
  $PY -m verity_machine_model.strassen.sched narrow "$i" 13 "161:256,256:$w,161:$w,$w:$w" > "$OUT/noise-$i-u$u.txt" 2>&1
  echo "noise $i $u $?" >> "$OUT/done.txt"
}
export -f noise
export PY OUT

pids=()
if [[ " $* " == *" crt "* ]]; then
  crt h200-sheet "16 32 64" & pids+=($!)
  crt h200-measured "16 32 64" & pids+=($!)
  crt rtx6000-sheet "32" & pids+=($!)
  crt rtx6000-measured "32" & pids+=($!)
  crt rtx6000-sheet-b1 "1 16 32" & pids+=($!)
  crt rtx6000-measured-b1 "1 16 32" & pids+=($!)
fi
if [[ " $* " == *" noise "* ]]; then
  for i in h200-sheet h200-measured rtx6000-sheet-b1 rtx6000-measured-b1; do
    for u in $(seq 64 128); do echo "$i $u"; done
  done | xargs -P "$J" -n 2 bash -c 'noise "$0" "$1"'
fi
wait "${pids[@]}" 2>/dev/null
echo "all done" >> "$OUT/done.txt"
