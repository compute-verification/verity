#!/usr/bin/env bash
# CIR-263: the suites named in SUITES (space-separated), or those the change from CHANGED touches, to $OUT; with EMPTY_RECORDS set, circuit-check's
# catalog records start empty and no cached pass is reused.  Leaves TMPDIR as the run gives it.
set -uo pipefail
: "${SUITES:=}"
[ -n "$SUITES" ] || [ -n "${CHANGED:-}" ] || { echo "SUITES or CHANGED"; exit 2; }
date -u +"sync %FT%TZ"; git log -1 --format="head %H"
uv sync -q --all-packages --extra torch-cpu || exit 2
extra=()
if [ -n "${CHANGED:-}" ]; then
  git cat-file -e "$CHANGED^{commit}" 2>/dev/null || git fetch -q origin main
  extra+=(--changed "$CHANGED")
fi
if [ -n "${EMPTY_RECORDS:-}" ]; then
  CIRCUIT_CHECK_RECORDS=$(mktemp -d); export CIRCUIT_CHECK_RECORDS; extra+=(--fresh)
  echo "circuit-check records: an empty directory, and --fresh"
fi
date -u +"suites %FT%TZ"
# shellcheck disable=SC2086
uv run tools/verity/check/suites.py $SUITES "${extra[@]}" --mem-gb 32 --json "$OUT/suites.json" --logs "$OUT/logs" > "$OUT/suites.log" 2>&1
s=$?
echo "suites rc=$s"; grep -vE "^ +ran because" "$OUT/suites.log" | tail -30
date -u +"end %FT%TZ"
exit $s
