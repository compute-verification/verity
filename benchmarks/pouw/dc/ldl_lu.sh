#!/usr/bin/env bash
# LDL.LU.128 and LDL.LU.64 on the H200 (sm_90a), the census gaps PTX can't reach: build probes/sm90_w1_lu.cu as emp.sh
# builds a source, set the last-use hint on its kernels' local loads (ldl_lu.py), then gate the patched binary and pack it
# for emp.sh's time mode.
#
#   research run --on auto --cwd source --wait -- bash benchmarks/pouw/dc/ldl_lu.sh
#       writes $OUT/bins.tgz, $OUT/sass/, patch.txt, gate-build.txt and gate-build.json, as `emp.sh build` does; time the
#       bins.tgz with `emp.sh time`
# Exit: 0 every gate verdict the declared one, 1 a build, the patch or a gate failed, 2 no nvcc.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="$HERE/../opcode_probe/probe.py"
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "ldl_lu.sh: no nvcc (CUDA 13.0 builds sm_90a)" >&2; exit 2; }
export NVCC PATH="$(dirname "$NVCC"):$PATH" PYTHONDONTWRITEBYTECODE=1
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/bin" "$OUT/sass"
"$NVCC" --version >"$OUT/nvcc.txt" 2>&1
python3 "$PROBE" build --arch sm_90a --out "$WORK/bin" "$HERE/probes/sm90_w1_lu.cu" >"$OUT/build.log" 2>&1 \
  || { cat "$OUT/build.log" >&2; exit 1; }
b="$WORK/bin/sm90_w1_lu"
python3 "$HERE/ldl_lu.py" "$b" l_ldllu4 l_ldllu2 >"$OUT/patch.txt" || { cat "$OUT/patch.txt"; exit 1; }
cuobjdump -sass "$b" >"$OUT/sass/sm90_w1_lu.sass" 2>&1
python3 "$PROBE" gate "$b" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
rc=$?
grep -c '^PASS' "$OUT/gate-build.txt" | xargs echo "ldl_lu.sh: gate PASS"
tar -C "$WORK/bin" -czf "$OUT/bins.tgz" .
[ $rc -eq 0 ] || exit 1
