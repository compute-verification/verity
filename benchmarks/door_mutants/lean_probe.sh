#!/usr/bin/env bash
# What one rebuild of the door's Lean costs: Flock's lowering tests and flock-lower-check cold, then after one edit to
# Lower.lean (bitsOk a no-op) and one to Lift.lean; then Spot's Restrict cold.
set -uo pipefail
R=$PWD
F=verity/protocols/proofs/flock/verifier
S=verity/protocols/proofs/spot/verifier/lean
T="Flock.Test.LowerBad Flock.Test.Lower flock-lower-check"
t() { local t0=$(date +%s); "$@" > "$OUT/last.log" 2>&1; local rc=$?; echo "$(( $(date +%s)-t0 ))s rc=$rc: $*"; tail -5 "$OUT/last.log"; }
t bash tools/verity/lean/setup.sh $F
cd $F
t lake build $T
sed -i 's/^def bitsOk (d : Defn) : V Unit := if onBits d then pure () else throw s!"{d.id}: {WIDE}"/def bitsOk (_d : Defn) : V Unit := pure ()/' Flock/Lower.lean
grep -n '^def bitsOk' Flock/Lower.lean
t lake build $T
git -C "$R" checkout -- $F/Flock/Lower.lean
t lake build $T
sed -i 's/      | some w => .affine p i (b \* w) (c \* w)/      | some w => .affine p i (b * w) c/' Flock/Lift.lean
git -C "$R" diff --stat
t lake build $T
git -C "$R" checkout -- $F/Flock/Lift.lean
t lake build $T
cd "$R"
t bash tools/verity/lean/setup.sh $S
cd $S
t lake build Spot.Restrict Spot.BlockFront
