#!/usr/bin/env bash
# What the templated statement's read costs against the flat one's: per K, the H1 tile unit's statement circuit
# and its templated form as flock_tmpl.sh builds them, then flock-tmpl's `unrollat` (`Tmpl.unrollAt` at MODE, whole and a
# stage at a time), `checks` (the description checks, then the unrolled input-row checks they stand for), `unroll` (the
# logical `Tmpl.unroll`) and `flat` (`Net.parse` on the circuit's flat unit, which the flat
# statement reads in its place). Run from a clone (the build writes the verifier's .lake):
#   research run --as proofs --project verity --on vy-nebius-cpu-3 --source . --pool proofs --cwd clone -- \
#     bash benchmarks/verifier_cost/flock_unroll.sh 448
# MODE (default iv/hybrid/4) is the layout. $OUT gets, per K, build-K.json, unit-K.tmpl, unroll-K.jsonl (every PHASE
# line) and each step's /usr/bin/time -v.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 0 ] || { echo "usage: flock_unroll.sh K..." >&2; exit 2; }
MODE=${MODE:-iv/hybrid/4}
L=$PWD/verity/core/service/flock
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu
mkdir -p "$S/tmpl"
cp benchmarks/verifier_cost/flock_tmpl/Main.lean "$L/lean-toolchain" "$S/tmpl/"
cat > "$S/tmpl/lakefile.toml" <<EOF
name = "flock_tmpl"
defaultTargets = ["flock-tmpl"]

[[require]]
name = "flock_verifier"
path = "$L"

[[lean_exe]]
name = "flock-tmpl"
root = "Main"
EOF
(cd "$S/tmpl" && lake build flock-tmpl) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
X=$S/tmpl/.lake/build/bin/flock-tmpl
step() {  # step NAME K ARGS...: flock-tmpl ARGS, its PHASE lines appended to unroll-K.jsonl
  local name=$1 k=$2; shift 2
  /usr/bin/time -v "$X" "$@" > "$O/$name-$k.out" 2> "$O/$name-time-$k.txt" \
    || { cat "$O/$name-$k.out" "$O/$name-time-$k.txt" >&2; exit 1; }
  grep '^PHASE ' "$O/$name-$k.out" | sed "s/^PHASE {/{\"step\":\"$name\",/" >> "$O/unroll-$k.jsonl"
}
for k in "$@"; do
  w=$S/k$k
  /usr/bin/time -v uv run python benchmarks/verifier_cost/flock_read.py "$k" "$w" --circuit-only > "$O/stage-$k.out" \
    2> "$O/stage-time-$k.txt" || { tail -40 "$O/stage-time-$k.txt" >&2; exit 1; }
  cp "$w/build.json" "$O/build-$k.json"
  uv run python -m verity_flock.tmpl "$w/circuit.txt" "$O/unit-$k.tmpl" > "$O/emit-$k.out" 2> "$O/emit-time-$k.txt" \
    || { tail -40 "$O/emit-time-$k.txt" >&2; exit 1; }
  name=$(head -1 "$O/unit-$k.tmpl" | cut -d' ' -f2)
  step unrollat "$k" unrollat "$O/unit-$k.tmpl" --mode "$MODE"
  step phases "$k" unrollat "$O/unit-$k.tmpl" --mode "$MODE" --phases
  step checks "$k" checks "$O/unit-$k.tmpl"
  step unroll "$k" unroll "$O/unit-$k.tmpl"
  step flat "$k" flat "$w/circuit.txt" "$name"
  cat "$O/build-$k.json" "$O/unroll-$k.jsonl"
  grep -H -E 'Maximum resident|Elapsed' "$O"/*-time-$k.txt
  rm -rf "$w"
done
