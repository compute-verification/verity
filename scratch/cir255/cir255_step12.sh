#!/bin/bash
# CIR-255, #9390 at its head (the run's source):
# 1. circuit-check of the two sidecars whose by-calls roots call DsaPositionsMerge_v1{M=8,K=2048} (ours and vllm's long_row),
#    on the head merged into main 30d6fcc92f (cir255_to_merged.patch; vllm's sidecar is on main only), then --update-pins
#    there and the pin files' diff, then on main itself (cir255_branch_vs_main.patch reversed); each with a cache of its own.
# 2. back at the head (checked clean), the suites that read what the change does, --fresh.
set -u
IN="$RESEARCH_RUN_DIR/inputs"
uv sync -q --all-packages --extra torch-cpu || exit 1
echo "head $(git rev-parse HEAD 2>/dev/null)"
SC="verity/definitions/registry/dsa_topk_reads.circuit-check.py adapters/vllm/verity_vllm/pipeline/long_row.circuit-check.py"
CC=$(mktemp -d)
cc() {  # tag, extra args
  local tag=$1; shift
  date -u +"$tag start %FT%TZ"
  /usr/bin/time -v uv run circuit-check $SC --cache "$CC/$CACHE" --out "$OUT/$tag.json" "$@" > "$OUT/$tag.log" 2> "$OUT/$tag.time"
  echo "$tag rc=$?"; tail -6 "$OUT/$tag.log"
}
git apply --binary "$IN/cir255_to_merged.patch" && echo "applied: head merged into main" || exit 3
git add -A    # the merged tree staged, so the pin diff below is --update-pins' alone
uv sync -q --all-packages --extra torch-cpu || exit 1
CACHE=merged cc merged
CACHE=merged cc merged-pins --update-pins
git diff > "$OUT/pins.diff"
echo "files --update-pins moved:"; git diff --stat
git checkout -- .
git apply -R "$IN/cir255_branch_vs_main.patch" && echo "reverted to main 30d6fcc92f's tree" || exit 3
CACHE=main cc main
git apply "$IN/cir255_branch_vs_main.patch" && git apply -R --binary "$IN/cir255_to_merged.patch" && echo "back at the head" || exit 3
git reset -q
echo "status at the head (empty is clean):"; git status --porcelain | head
uv sync -q --all-packages --extra torch-cpu || exit 1
date -u +"suites start %FT%TZ"
uv run tools/verity/check/suites.py repository verity-config tools/verity/circuit_check verity/definitions verity-cli --fresh \
  --mem-gb 120 --json "$OUT/suites.json" --logs "$OUT/suites-logs"; echo "suites rc=$?"
date -u +"vllm start %FT%TZ"
uv run tools/verity/check/suites.py adapters/vllm --fresh --mem-gb 120 --json "$OUT/suites-vllm.json" --logs "$OUT/suites-vllm-logs" \
  -- -k "settings or long_row"; echo "vllm rc=$?"
date -u +"end %FT%TZ"
