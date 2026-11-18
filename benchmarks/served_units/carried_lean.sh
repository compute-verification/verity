#!/usr/bin/env bash
# Flock's Lean verifier (lean_check.sh) on each statement carried_prove.sh proved: KEPT/NAME's last session at the statement
# STMTS/NAME. Run from the tree's root with its workspace synced; OUT/lean.tsv has each verdict, and 0 means all accepted.
# usage: carried_lean.sh KEPT STMTS OUT
set -uo pipefail
K0=$1 ST=$2 OUT=$3; mkdir -p "$OUT/lean"; S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
. benchmarks/served_units/lean_check.sh
LEAN_PY=(uv run --no-sync python); LEAN_BENCH=(uv run --no-sync verity bench); LEAN_TMP=$S; V=verity/protocols/proofs/flock/verifier
exe=$(lean_exes "$V" | cut -d' ' -f1)
[ -n "$exe" ] && (cd "$V" && lake build "$exe") > "$OUT/lean-build.log" 2>&1 \
  || { echo "the Lean verifier ${exe:-of $V/lakefile.toml} did not build: $OUT/lean-build.log"; exit 1; }
EXE=$PWD/$V/.lake/build/bin/$exe; sha256sum "$EXE"
printf 'name\tsession\taccepted\tlabel\twall_s\tpeak_gib\tcircuit_bytes\n' > "$OUT/lean.tsv"; rc=0
for k in "$K0"/*/; do
  k=${k%/}; nm=$(basename "$k"); D=$ST/$nm
  sd=$(for d in "$k"/sessions/*/; do [ -s "$d/session.json" ] && echo "${d%/}"; done | sort | tail -1)
  { [ -n "$sd" ] && [ -s "$D/circuit.txt" ]; } || { echo "$nm: no kept session, or no statement at $D"; rc=1; continue; }
  [ -e "$D/program.json" ] || lean_word "$D" > "$OUT/lean/$nm.word.txt" 2>&1 || { echo "$nm: program-word.json not written"; rc=1; continue; }
  lean_check "$nm" "$sd" "$D" "$k" "$OUT/lean/$nm" "$EXE" "$OUT/lean.tsv" || rc=1
done
cat "$OUT/lean.tsv"
exit $rc
