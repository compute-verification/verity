#!/usr/bin/env bash
# The H1 tile unit's layouts measured on its mined template (PRF-133): per K, flock_read.py --circuit-only builds the
# tile's circuit, `python -m verity_flock.tmpl` mines its template, and tmpl_layouts.py sizes today's, the power-of-two
# and the interleaved layouts and checks each one's order on the flat unit. `tmpl_layouts.sh --tmpl FILE` measures a
# template already mined (no order check). Run with research run --cwd source; $OUT gets layouts-K.jsonl (LAYOUT lines),
# emit-K.json, unit-K.tmpl and each step's /usr/bin/time -v.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 0 ] || { echo "usage: tmpl_layouts.sh K... | --tmpl FILE" >&2; exit 2; }
uv sync -q --all-packages --extra torch-cpu
P=benchmarks/verifier_cost/tmpl_layouts.py
if [ "$1" = --tmpl ]; then
  /usr/bin/time -v uv run python "$P" "$2" > "$O/layouts.out" 2> "$O/layouts-time.txt" \
    || { tail -40 "$O/layouts-time.txt" >&2; exit 1; }
  grep '^LAYOUT ' "$O/layouts.out" | sed 's/^LAYOUT //' | tee "$O/layouts.jsonl"
  exit 0
fi
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
for k in "$@"; do
  w=$S/k$k
  /usr/bin/time -v uv run python benchmarks/verifier_cost/flock_read.py "$k" "$w" --circuit-only > "$O/stage-$k.out" \
    2> "$O/stage-time-$k.txt" || { tail -40 "$O/stage-time-$k.txt" >&2; exit 1; }
  /usr/bin/time -v uv run python -m verity_flock.tmpl "$w/circuit.txt" "$O/unit-$k.tmpl" > "$O/emit-$k.out" \
    2> "$O/emit-time-$k.txt" || { tail -40 "$O/emit-time-$k.txt" >&2; exit 1; }
  grep '^TMPL ' "$O/emit-$k.out" | sed 's/^TMPL //' > "$O/emit-$k.json"
  rc=0
  /usr/bin/time -v uv run python "$P" "$O/unit-$k.tmpl" --flat "$w/circuit.txt" > "$O/layouts-$k.out" \
    2> "$O/layouts-time-$k.txt" || rc=$?
  grep '^LAYOUT ' "$O/layouts-$k.out" | sed 's/^LAYOUT //' | tee "$O/layouts-$k.jsonl"
  grep -E 'Maximum resident|Elapsed' "$O"/{stage,emit,layouts}-time-$k.txt
  rm -rf "$w"
  [ "$rc" = 0 ] || { tail -40 "$O/layouts-time-$k.txt" >&2; echo "K=$k: a layout check failed (layouts-$k.jsonl)" >&2; exit 1; }
done
