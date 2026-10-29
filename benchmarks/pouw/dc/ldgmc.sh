#!/usr/bin/env bash
# The census's LDGMC gaps on the H200 (ldgmc.cu): build and gate on a CPU node, then time on the H200 under one gpu-lease.
#
#   research run --on auto --cwd source --wait -- bash benchmarks/pouw/dc/ldgmc.sh build
#       builds ldgmc.cu for sm_90a, writes its SASS, gates every form's kernel (gpu_kit/sass_gate.py: the loop holds U
#       LDGMC, a live sink, no spill) and checks it lowers to the census's form; writes $OUT/ldgmc.tgz and gate.txt
#   research run --on vy-dc-h200-5 --cwd source --send ldgmc.tgz --env GPU_LEASE_WHO=<lane> -- \
#     bash benchmarks/pouw/dc/ldgmc.sh time [REPS]
#       runs the build's binary under `gpu-lease 1`, with the clocks before and after; writes $OUT/ldgmc.txt. On a
#       node with no multicast object it then runs each form alone on plain device memory (`ldgmc --unicast`), each
#       line saying whether the form faulted (exit 4) or ran, and at what price
# Exit: 0 done, 1 a gate or a check failed, 2 no nvcc or no inputs.
set -uo pipefail
MODE=${1:?usage: ldgmc.sh build | time [REPS]}
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GATE="$HERE/../gpu_kit/sass_gate.py"
U=4
export PYTHONDONTWRITEBYTECODE=1

case "$MODE" in
build)
  NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
  [ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "ldgmc.sh: no nvcc" >&2; exit 2; }
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  "$NVCC" --version >"$OUT/nvcc.txt" 2>&1
  "$NVCC" -std=c++17 -O3 -gencode arch=compute_90a,code=sm_90a -o "$WORK/ldgmc" "$HERE/ldgmc.cu" -lcuda \
    >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
  "$(dirname "$NVCC")/cuobjdump" -sass "$WORK/ldgmc" >"$OUT/ldgmc.sass"
  # --list runs before cuInit, so on a node without a driver the toolkit's stub stands in for libcuda.so.1
  STUB=$(ls "$(dirname "$NVCC")"/../lib64/stubs/libcuda.so "$(dirname "$NVCC")"/../targets/*/lib/stubs/libcuda.so \
    2>/dev/null | head -n 1)
  mkdir -p "$WORK/stub" && [ -n "$STUB" ] && ln -s "$(realpath "$STUB")" "$WORK/stub/libcuda.so.1"
  LD_LIBRARY_PATH="$WORK/stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$WORK/ldgmc" --list >"$WORK/forms.txt" \
    || { echo "ldgmc.sh: ldgmc --list failed" >&2; exit 1; }
  [ -s "$WORK/forms.txt" ] || { echo "ldgmc.sh: ldgmc --list listed no forms" >&2; exit 1; }
  rc=0
  : >"$OUT/gate.txt"
  while read -r name form _ _; do
    python3 "$GATE" "$WORK/ldgmc" --kernels "$name" --expect "LDGMC.*=$U" >>"$OUT/gate.txt" 2>&1 \
      || { echo "ldgmc.sh: gate refused $name" | tee -a "$OUT/gate.txt" >&2; rc=1; }
    got=$(awk -v k="$name" '/Function : /{f = ($3 == k)} f && /LDGMC/{for (i = 1; i <= NF; i++) if ($i ~ /^LDGMC/) print $i}' \
      "$OUT/ldgmc.sass" | sort -u | tr '\n' ' ')
    if [ "$got" = "$form " ]; then echo "lowering $name: $form" >>"$OUT/gate.txt"
    else echo "ldgmc.sh: $name lowers to '$got', not $form" | tee -a "$OUT/gate.txt" >&2; rc=1; fi
  done <"$WORK/forms.txt"
  tar -C "$WORK" -czf "$OUT/ldgmc.tgz" ldgmc
  grep -c '^lowering' "$OUT/gate.txt" | xargs echo "ldgmc.sh: forms lowered as the census names them:"
  exit $rc
  ;;
time)
  REPS=${2:-5}
  IN=${RESEARCH_RUN_DIR:?run under research run}/inputs
  [ -f "$IN/ldgmc.tgz" ] || { echo "ldgmc.sh: no inputs/ldgmc.tgz (pass the build's with --send)" >&2; exit 2; }
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  tar -C "$WORK" -xzf "$IN/ldgmc.tgz"
  lease=()
  command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 20 --)
  [ ${#lease[@]} -eq 0 ] && { echo "ldgmc.sh: no gpu-lease on this node" >&2; exit 2; }
  "${lease[@]}" bash -c 'nvidia-smi -q -d CLOCK >"$0/clocks-before.txt"; "$1/ldgmc" --reps "$2"; rc=$?
    if [ $rc -eq 3 ]; then
      rc=0
      while read -r name _; do
        "$1/ldgmc" --unicast --form "$name" --reps "$2"; r=$?
        echo "unicast $name: exit $r"; [ $r -eq 0 ] || [ $r -eq 4 ] || rc=1
      done < <("$1/ldgmc" --list)
    fi
    nvidia-smi -q -d CLOCK >"$0/clocks-after.txt"; nvidia-smi topo -m >"$0/topo.txt" 2>&1; exit $rc' \
    "$OUT" "$WORK" "$REPS" | tee "$OUT/ldgmc.txt"
  exit "${PIPESTATUS[0]}"
  ;;
*)
  echo "usage: ldgmc.sh build | time [REPS]" >&2
  exit 2
  ;;
esac
