#!/usr/bin/env bash
# The Lean verifier's read of a real circuit, phase by phase: flock_read.py builds the H1 tile unit's statement circuit at
# each K (`h1_prove/h1unit.Shape(K)`, the 4×8 tile) and one instance on it, and flock-read (flock_read/Main.lean, built in a
# scratch Lake package against the checkout's verifier) times `flock-verify verify --zk`'s read of it. Run from a clone
# (the build writes the verifier's .lake):
#   research run --as proofs --project verity --on vy-nebius-cpu-1 --source . --pool proofs --cwd clone -- \
#     bash benchmarks/verifier_cost/flock_read.sh 256 1024
# $OUT gets, per K, build-K.json (flock_read.py's), phases-K.jsonl (flock-read's PHASE lines) and time-K.txt.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 0 ] || { echo "usage: flock_read.sh K..." >&2; exit 2; }
L=$PWD/verity/core/proof_system/flock
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu
mkdir -p "$S/read"
cp benchmarks/verifier_cost/flock_read/Main.lean "$L/lean-toolchain" "$S/read/"
cat > "$S/read/lakefile.toml" <<EOF
name = "flock_read"
defaultTargets = ["flock-read"]

[[require]]
name = "flock_verifier"
path = "$L"

[[lean_exe]]
name = "flock-read"
root = "Main"
EOF
(cd "$S/read" && lake build flock-read) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
for k in "$@"; do
  w=$S/k$k
  /usr/bin/time -v uv run python benchmarks/verifier_cost/flock_read.py "$k" "$w" > "$O/stage-$k.out" 2> "$O/stage-$k.err" \
    || { tail -40 "$O/stage-$k.err" >&2; exit 1; }
  cp "$w/build.json" "$O/build-$k.json"
  /usr/bin/time -v "$S/read/.lake/build/bin/flock-read" "$w/circuit.txt" "$w/stmt/pub.bin" verity/flock-circuit \
    > "$O/read-$k.out" 2> "$O/time-$k.txt" || { tail -40 "$O/time-$k.txt" >&2; exit 1; }
  grep '^PHASE ' "$O/read-$k.out" | sed 's/^PHASE //' > "$O/phases-$k.jsonl"
  cat "$O/build-$k.json" "$O/phases-$k.jsonl"
  rm -rf "$w"
done
