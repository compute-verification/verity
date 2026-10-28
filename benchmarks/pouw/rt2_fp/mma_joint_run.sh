#!/usr/bin/env bash
# The sm_120 MMA joints (mma_joint120.cu): build and gate, then time on one GPU and read (M)'s sum
# (mma_joint.py). Through the queue, whose wrapper already holds the GPU (GPU_LEASE_UUID set):
#
#   research run --queue --source <tree> --project verity --gpus 1 --mem-gb 4 --max-min 30 \
#     -- bash benchmarks/pouw/rt2_fp/mma_joint_run.sh
#
# A direct run on a gpu-lease node leases its own (opcode_probe/run.sh's form). BUILD_ONLY=1 stops after the gate, for
# a node with nvcc and no GPU to spare.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="$HERE/../opcode_probe/probe.py"
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "mma_joint_run.sh: no nvcc" >&2; exit 2; }
export NVCC PATH="$(dirname "$NVCC"):$PATH"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/sass"

python3 "$PROBE" build --out "$WORK" "$HERE/mma_joint120.cu" >"$WORK/built" 2>"$OUT/build.log" \
  || { cat "$OUT/build.log" >&2; exit 1; }
BIN=$(head -1 "$WORK/built")
cuobjdump -sass "$BIN" >"$OUT/sass/mma_joint120.sass" 2>&1
python3 "$PROBE" gate "$BIN" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1
rc=$?
cat "$OUT/gate-build.txt"
[ $rc -eq 0 ] || exit $rc
[ "${BUILD_ONLY:-0}" = 1 ] && exit 0

lease=()
[ -z "${GPU_LEASE_UUID:-}" ] && lease=(gpu-lease 1 --wait --max-min 20 --)
"${lease[@]}" python3 "$PROBE" run "$BIN" --out "$OUT" --reps 10 || exit $?
python3 "$HERE/mma_joint.py" "$OUT"
