#!/usr/bin/env bash
# GD-SEARCH's final on GD-MODEL's model: every grid at both points in N shards, P at a time, under $OUT.
#   research run --on vy-nebius-cpu-11 --pool research ... --cwd source -- bash benchmarks/golden_search/final.sh
set -u
cd "$(dirname "$0")"
O=${OUT:?}
P=${P:-$(nproc)} N=${N:-$P}
mkdir -p "$O"
run() {
  name=$1 grid=$2 pt=$3; shift 3
  seq 0 $((N - 1)) | xargs -P "$P" -I{} python3 search.py run --model gdm --grid "$grid-$pt" --point "$pt" \
    --shard "{}/$N" --scenario "$@" --out "$O/$name" > "$O/$name-$pt.log" 2>&1
  echo "DONE $name $pt $(date -u +%H:%M:%S)" >> "$O/progress.log"
}
for pt in t8192 t512; do run core core $pt nominal; done
python3 search.py reduce "$O/core" --out "$O/red-core" > /dev/null
for pt in t8192 t512; do
  python3 gdm.py flips "$O/red-core/search.json" core-$pt $pt "$O/flips-$pt.json" > "$O/flips-$pt.txt" 2>&1
  python3 gdm.py tiles "$O/red-core/search.json" core-$pt $pt "$O/tiles-$pt.json" > "$O/tiles-$pt.txt" 2>&1
done
for pt in t8192 t512; do for g in today today-period period; do run $g $g $pt nominal; done; done
for g in today today-period period; do python3 search.py reduce "$O/$g" --out "$O/red-$g" > /dev/null; done
for pt in t8192 t512; do run oat sweep $pt oat; done
for pt in t8192 t512; do run mc sweep $pt mcs:0-100; done
python3 search.py reduce "$O/oat" "$O/mc" --out "$O/red-sens" > /dev/null
echo ALLDONE >> "$O/progress.log"
