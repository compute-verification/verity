#!/bin/bash
# CIR-255 Part C: the served DsaPositionsMerge_v1 out of --all.
# 1. circuit-check --ranges DsaPositionsMerge_v1 on this tree: the served merge (M=8) is still checked, at its spot.
# 2. the circuit_check and definitions suites, --fresh, on this tree.
# 3. the same circuit_check suite, --fresh, with the cut reverted (cir255_cut.patch -R): the baseline on the same node and cores.
set -u
IN="$RESEARCH_RUN_DIR/inputs"
uv sync -q --all-packages --extra torch-cpu || exit 1
date -u +"ranges start %FT%TZ"
/usr/bin/time -v uv run circuit-check --no-cache --out "$OUT/ranges.json" --ranges DsaPositionsMerge_v1 \
  > "$OUT/ranges.log" 2> "$OUT/ranges.time"; echo "ranges rc=$?"
tail -5 "$OUT/ranges.log"
date -u +"branch suites start %FT%TZ"
uv run tools/verity/check/suites.py tools/verity/circuit_check verity/definitions --fresh --mem-gb 120 \
  --json "$OUT/branch.json" --logs "$OUT/branch-logs"; echo "branch suites rc=$?"
git apply -R "$IN/cir255_cut.patch" && echo "reverted the cut" || { echo "revert failed"; exit 3; }
date -u +"base suite start %FT%TZ"
uv run tools/verity/check/suites.py tools/verity/circuit_check --fresh --mem-gb 120 \
  --json "$OUT/base.json" --logs "$OUT/base-logs"; echo "base suite rc=$?"
date -u +"end %FT%TZ"
