#!/usr/bin/env bash
# The wires' check's cost against its wires: wires-time (wires/Main.lean, built in a scratch Lake package against the
# checkout's verifier) times `Spot.Wires.check`, its overlap test (`pairwiseDisjoint`, which compiled code runs as the
# sweep), its order (`orderCode` with `orderedWire`) and the overlap test as a loop over every pair, on N wires each, and
# refuses where the sweep and the loop disagree or the sweep passes a wire twice. The loop runs only up to REF_MAX wires
# (default 20000). Run from a clone (the build writes the verifier's .lake):
#   research run --as proofs --project verity --on auto --cwd clone -- \
#     bash benchmarks/verifier_cost/wires.sh 1000 10000 100000
# $OUT gets wires.jsonl (one row per N and function: wires, ms, peak RSS).
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 0 ] || { echo "usage: wires.sh N..." >&2; exit 2; }
L=$PWD/verity/protocols/proofs/spot/verifier/lean
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
mkdir -p "$S/time"
cp benchmarks/verifier_cost/wires/Main.lean "$L/lean-toolchain" "$S/time/"
cat > "$S/time/lakefile.toml" <<EOF
name = "wires_time"
defaultTargets = ["wires-time"]

[[require]]
name = "core_spot"
path = "$L"

[[lean_exe]]
name = "wires-time"
root = "Main"
EOF
(cd "$S/time" && lake build wires-time) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
for n in "$@"; do
  ref=(); [ "$n" -le "${REF_MAX:-20000}" ] || ref=(--no-ref)
  "$S/time/.lake/build/bin/wires-time" "$n" "${ref[@]}" | grep '^PHASE ' | sed 's/^PHASE //' | tee -a "$O/wires.jsonl"
done
