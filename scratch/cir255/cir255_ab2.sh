#!/bin/bash
# CIR-255 Part C, A/B with no records: the circuit_check suite, --fresh, alone, at --mem-gb 120, on the cut reverted (base)
# and on this tree (branch), each with an empty HOME, so the session's `_from_records` finds no record and builds the catalog
# apart, as every pytest shard of the last ten candidates did.  uv's cache and Pythons stay where they are.
set -u
IN="$RESEARCH_RUN_DIR/inputs"
uv sync -q --all-packages --extra torch-cpu || exit 1
export UV_CACHE_DIR="$(uv cache dir)" UV_PYTHON_INSTALL_DIR="$(uv python dir)"
half() {
  local tag=$1 home
  home=$(mktemp -d)
  HOME="$home" .venv/bin/python -c 'import pathlib; p = pathlib.Path.home() / ".cache/verity-check/circuit-check"; print("records", p, "exists" if p.exists() else "absent")'
  date -u +"$tag suite start %FT%TZ"
  HOME="$home" uv run tools/verity/check/suites.py tools/verity/circuit_check --fresh --mem-gb 120 \
    --json "$OUT/$tag.json" --logs "$OUT/$tag-logs"; echo "$tag suite rc=$?"
  date -u +"$tag suite end %FT%TZ"
  rm -rf "$home"
}
git apply -R "$IN/cir255_cut.patch" && echo "reverted the cut" || { echo "revert failed"; exit 3; }
half base
git apply "$IN/cir255_cut.patch" && echo "reapplied the cut" || { echo "reapply failed"; exit 3; }
git diff --stat
half branch
date -u +"end %FT%TZ"
