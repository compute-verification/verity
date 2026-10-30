#!/usr/bin/env bash
# The HT-PREMISE v2 sweep: `search.py` jobs in parallel, results under $OUT (one JSON and log a job, trees in trees/).
# Usage: sweep.sh exact|lsb|margin|kcap PARALLEL
set -euo pipefail
group=$1
par=$2
out=${OUT:?}
shapes5="8192,2048,8192 8192,8192,8192 8192,16384,8192 8192,28672,8192 16384,53248,16384"
shapes7="$shapes5 8192,8192,57344 16384,16384,106496"
jobs=$(mktemp)
trap 'rm -f "$jobs"' EXIT
if [ "$group" = exact ]; then
  # P1 at w = 16, R uniform int8: E 2-bit (a = 4) and 3-bit (a = 8)
  for chip in h200 sm120 h200+s4; do for a in 4 8; do for s in $shapes7; do
    echo "--chips $chip --cases $a,256,16 --shapes $s --depths '4 5 6'"
  done; done; done >> "$jobs"
  echo "--chips h200 --cases 4,256,16 --shapes 16384,53248,16384 --depths 7" >> "$jobs"
elif [ "$group" = margin ]; then
  # the margin (`alt`, lower bounds) of P1 at w = 16; A-LSB's c_q at short K, where it would have to reach 0.9
  for chip in h200 sm120; do for a in 4 8; do for s in $shapes7; do
    echo "--chips $chip --cases $a,256,16 --shapes $s --depths '5 6' --grids down"
  done; done; done >> "$jobs"
  for chip in h200 sm120; do for b in 4 8 16 32; do
    w=$(python3 -c "import math; print(int(math.log2($b)))")
    echo "--chips $chip --cases 4,$b,$w --shapes '8192,128,8192 8192,256,8192 8192,512,8192 8192,1024,8192' --depths '4 6'"
  done; done >> "$jobs"
elif [ "$group" = kcap ]; then
  # the margin under a K cap of 4,096 or 8,192, at the widest served outputs
  for chip in h200 sm120; do for a in 4 8; do
    for s in 8192,4096,8192 8192,4096,57344 8192,4096,106496 8192,8192,106496; do
      echo "--chips $chip --cases $a,256,16 --shapes $s --depths '5 6' --grids down"
    done
  done; done >> "$jobs"
else
  # A-LSB and D0: L (or small R) over 2^b values, b = 2..5; output mod 2^b (the congruence) and mod 2^16 (W = L)
  for chip in h200 sm120 h200+s4; do for a in 4 8; do for b in 4 8 16 32; do
    w=$(python3 -c "import math; print(int(math.log2($b)))")
    for s in $shapes5; do
      echo "--chips $chip --cases $a,$b,$w --shapes $s --depths '4 6'"
      echo "--chips $chip --cases $a,$b,16 --shapes $s --depths '4 6'"
    done
  done; done; done >> "$jobs"
fi
mkdir -p "$out/jobs" "$out/trees"
n=0
while read -r line; do
  n=$((n + 1))
  printf '%s\0' "python3 search.py $line --readings fast --trees $out/trees --json $out/jobs/j$n.json > $out/jobs/j$n.log 2>&1; echo \"j$n rc=\$? $line\" >> $out/done.txt"
done < "$jobs" | xargs -0 -P "$par" -I{} bash -c '{}'
echo "jobs: $n"
