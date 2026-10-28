#!/usr/bin/env bash
# The H1 tile unit in flock-ir-tmpl/v2 against its flat form: per K, flock_read.py --circuit-only builds the 4×8
# tile's statement circuit (`h1_prove/h1unit.Shape(K)`), `python -m verity_flock.tmpl` writes its unit's templated form,
# and flock-tmpl (flock_tmpl/Main.lean, built in a scratch Lake package against the checkout's verifier) times the
# templated read (`Tmpl.parse`) and diffs `Tmpl.unroll`'s lines against the circuit's flat unit byte for byte. Run from a
# clone (the build writes the verifier's .lake):
#   research run --as proofs --project verity --on vy-nebius-cpu-1 --source . --pool proofs --cwd clone -- \
#     bash benchmarks/verifier_cost/flock_tmpl.sh 256 4096
# $OUT gets, per K, build-K.json, emit-K.json (the emitter's statistics), unit-K.tmpl, read-K.jsonl and diff-K.jsonl
# (flock-tmpl's PHASE lines) and each step's /usr/bin/time -v. A K whose diff fails stops the run (exit 1).
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 0 ] || { echo "usage: flock_tmpl.sh K..." >&2; exit 2; }
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
for k in "$@"; do
  w=$S/k$k
  /usr/bin/time -v uv run python benchmarks/verifier_cost/flock_read.py "$k" "$w" --circuit-only > "$O/stage-$k.out" \
    2> "$O/stage-time-$k.txt" || { tail -40 "$O/stage-time-$k.txt" >&2; exit 1; }
  cp "$w/build.json" "$O/build-$k.json"
  /usr/bin/time -v uv run python -m verity_flock.tmpl "$w/circuit.txt" "$O/unit-$k.tmpl" > "$O/emit-$k.out" \
    2> "$O/emit-time-$k.txt" || { tail -40 "$O/emit-time-$k.txt" >&2; exit 1; }
  grep '^TMPL ' "$O/emit-$k.out" | sed 's/^TMPL //' > "$O/emit-$k.json"
  /usr/bin/time -v "$X" read "$O/unit-$k.tmpl" > "$O/read-$k.out" 2> "$O/read-time-$k.txt" \
    || { cat "$O/read-$k.out" "$O/read-time-$k.txt" >&2; exit 1; }
  grep '^PHASE ' "$O/read-$k.out" | sed 's/^PHASE //' > "$O/read-$k.jsonl"
  rc=0
  /usr/bin/time -v "$X" diff "$O/unit-$k.tmpl" "$w/circuit.txt" > "$O/diff-$k.out" 2> "$O/diff-time-$k.txt" || rc=$?
  grep '^PHASE ' "$O/diff-$k.out" | sed 's/^PHASE //' > "$O/diff-$k.jsonl"
  cat "$O/build-$k.json" "$O/emit-$k.json" "$O/read-$k.jsonl" "$O/diff-$k.jsonl"
  grep -E 'Maximum resident|Elapsed' "$O"/{stage,emit,read,diff}-time-$k.txt
  rm -rf "$w"
  [ "$rc" = 0 ] || { echo "K=$k: the templated unit's unroll differs from the flat unit (diff-$k.jsonl)" >&2; exit 1; }
done
