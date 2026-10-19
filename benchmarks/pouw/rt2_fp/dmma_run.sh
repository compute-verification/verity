#!/usr/bin/env bash
# RT2-FP-12's sm_120 joint (dmma_imma120.cu): build and gate, then time on one GPU and read (M)'s sum (dmma_joint.py).
# Through the queue, whose wrapper already holds the GPU (GPU_LEASE_UUID set):
#
#   research run --queue --source <tree> --project verity --gpus 1 --mem-gb 4 --max-min 30 \
#     -- bash benchmarks/pouw/rt2_fp/dmma_run.sh
#
# A direct run on a gpu-lease node leases its own (opcode_probe/run.sh's form).
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="$HERE/../opcode_probe/probe.py"
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "dmma_run.sh: no nvcc" >&2; exit 2; }
export NVCC PATH="$(dirname "$NVCC"):$PATH"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/sass"

python3 "$PROBE" build --out "$WORK" "$HERE/dmma_imma120.cu" >"$WORK/built" 2>"$OUT/build.log" \
  || { cat "$OUT/build.log" >&2; exit 1; }
BIN=$(head -1 "$WORK/built")
cuobjdump -sass "$BIN" >"$OUT/sass/dmma_imma120.sass" 2>&1
python3 "$PROBE" gate "$BIN" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
rc=$?
cat "$OUT/gate-build.txt"
[ $rc -eq 0 ] || exit $rc

lease=()
[ -z "${GPU_LEASE_UUID:-}" ] && lease=(gpu-lease 1 --wait --max-min 20 --)
"${lease[@]}" python3 "$PROBE" run "$BIN" --out "$OUT" --reps 10 || exit $?
python3 "$HERE/dmma_joint.py" "$OUT"
