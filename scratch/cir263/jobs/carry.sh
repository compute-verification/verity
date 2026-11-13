#!/usr/bin/env bash
# CIR-263: `research.carry OLD NEW` for each NEW in $NEWS, its JSON to $OUT/carry-<NEW>.json and its exit code printed, with
# --landed BASE (the main the branch merged).  Runs in a clone (--cwd clone): the moves rerun in scratch worktrees.
set -uo pipefail
: "${OLD:?OLD}" "${NEWS:?NEWS}" "${BASE:?BASE}"
date -u +"start %FT%TZ"; git log -1 --format="head %H"
for c in "$OLD" "$BASE" $NEWS; do
  git cat-file -e "$c^{commit}" 2>/dev/null || git fetch -q origin "$c" 2>/dev/null || git fetch -q origin main 2>/dev/null || true
  git cat-file -e "$c^{commit}" 2>/dev/null || { echo "no commit $c here"; exit 2; }
done
uv sync -q --project tools/research || exit 2
rc_all=0
for new in $NEWS; do
  date -u +"carry %FT%TZ $OLD $new"
  uv run -q --project tools/research python -m research.carry "$OLD" "$new" --landed "$BASE" > "$OUT/carry-$new.json" 2> "$OUT/carry-$new.err"
  rc=$?
  echo "carry $new exit $rc"
  cat "$OUT/carry-$new.json"; echo
  tail -20 "$OUT/carry-$new.err"
  (( rc )) && rc_all=$rc
done
date -u +"end %FT%TZ"
exit $rc_all
