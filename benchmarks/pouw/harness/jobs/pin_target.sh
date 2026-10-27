#!/usr/bin/env bash
# A new target's SASS-gate pin, on a card of that target, in a `--cwd clone` run (it rewrites sass_pins.json):
#
#   research run --on vy-dc-h200 --project verity --campaign pouw --source . --cwd clone --env GPU_LEASE_WHO=<you> \
#       -- bash benchmarks/pouw/harness/jobs/pin_target.sh 13.0.1-sm_90a /usr/local/cuda-13.0
#
# Builds ieee_pin.cu's pinned builds for TOOLKIT's target (ieee_pin.arch_of) with CUDA_HOME's nvcc, pins them as TOOLKIT
# (sass_gate.py --pin --toolkit), then runs ieee_pin.py on that entry alone, which gates the builds and the controls and runs
# them on one leased GPU. Leaves the repinned sass_pins.json and ieee-pin.json in $OUT; the entry lands with this run as
# its ``numeric``, and exempts nothing before.
set -euo pipefail
TK=$1; CUDA=$2
H=$(cd "$(dirname "$0")/.." && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
python3 - "$H" "$TK" > "$W/builds" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import ieee_pin as I
for i, flags in enumerate(I.PINNED):
    print(i, " ".join(I.nvcc_args(I.arch_of(sys.argv[2])) + flags))
EOF
while read -r i args; do
  # shellcheck disable=SC2086
  (cd "$W" && "$CUDA/bin/nvcc" $args "$H/ieee_pin.cu" -o "$W/pin$i")
done < "$W/builds"
POUW_CUDA_BIN="$CUDA/bin" python3 "$H/sass_gate.py" "$W"/pin? --pin --toolkit "$TK"
cp "$H/sass_pins.json" "$OUT/sass_pins.json"
python3 "$H/ieee_pin.py" --toolkit "$TK=$CUDA" --out "$OUT/ieee-pin.json"
