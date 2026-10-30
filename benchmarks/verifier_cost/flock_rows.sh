#!/usr/bin/env bash
# flock_rows.py on a CPU node: build flock-rows from BEFORE ("before": a tar of the Lean package before a change, `git
# archive BASE verity/core/proof_system/flock`, sent as an input; `-` for none) and at the checkout ("after"), then time them on
# flock_rows.py's units. Run from a clone (the builds write .lake):
#   git archive -o /tmp/before.tar BASE verity/core/proof_system/flock
#   research run --as proofs --project verity --on vy-nebius-cpu-2 --source . --pool proofs --cwd clone --send /tmp/before.tar -- \
#     bash benchmarks/verifier_cost/flock_rows.sh before.tar --fanout 125,250,500,1000 --cap-bin before=1500
# Arguments after BEFORE go to flock_rows.py; its flock_rows.json lands in $OUT.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
before=${1:?usage: flock_rows.sh BEFORE.tar|- [ARGS...]}
shift
L=verity/core/proof_system/flock
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu
bins=(--bin "after=$PWD/$L/.lake/build/bin/flock-rows")
if [ "$before" != - ]; then
  tar -xf "${RESEARCH_RUN_DIR:?}/inputs/$before" -C "$S"
  (cd "$S/$L" && lake build flock-rows) > "$O/build-before.log" 2>&1 || { tail -40 "$O/build-before.log" >&2; exit 1; }
  bins=(--bin "before=$S/$L/.lake/build/bin/flock-rows" "${bins[@]}")
fi
(cd "$L" && lake build flock-rows) > "$O/build-after.log" 2>&1 || { tail -40 "$O/build-after.log" >&2; exit 1; }
uv run python benchmarks/verifier_cost/flock_rows.py --out "$O" "${bins[@]}" "$@"
