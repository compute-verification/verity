#!/usr/bin/env bash
# A row's traced Build preserved as a `vllm-build/v1` art, the form a switched row's `aot.trace` takes (`records switch --trace`,
# fetched by its AOT Build): `research-outputs write` stages it (build_summary.json, every rank's derive dirs less instances.json.gz)
# and `research data put --preserve` stores it.  Inside trace_compare.sh (PY, RESEARCH_BIN and the custody env set), or alone on the
# node that holds the row dir:
#   research run --on vy-glm53-cpu-2 --mem-gb 8 --timeout 1h --cwd source --project verity -- env ROW=... D=<row dir> bash integrations/vllm/manifests/glm_v2_tree/jobs/trace_art.sh
# env: ROW, D (the traced Build's row dir, `build PASS` in its stages.txt).  output: $OUT/trace_art.txt.
set -uo pipefail
: "${ROW:?ROW}" "${D:?D}"
grep -q '^build PASS' "$D/stages.txt" 2>/dev/null || { echo "$D: no passing Build"; exit 3; }
if [ -z "${RESEARCH_BIN:-}" ]; then
  W0=/workspace/cp/circuits TREE=$PWD
  C=$(dirname "$(dirname "$RESEARCH_RUN_DIR")")/requests/$RESEARCH_RUN_ID/custody
  [ -f "$C/cred.json" ] || { echo "no custody key at $C"; exit 4; }
  export PY=$W0/venv312/bin/python PYTHONPATH=$TREE:$TREE/tools/research/src:$TREE/integrations/vllm RESEARCH_STORE_CONFIG=$C/store.toml
  eval "$($PY -m verity_vllm.pipeline.cli epoch custody-env "$C")"
  RESEARCH_BIN=$(mktemp -d)/research
  printf '#!/bin/sh\nexec %s -m research "$@"\n' "$PY" >"$RESEARCH_BIN" && chmod +x "$RESEARCH_BIN"
fi
S=$(mktemp -d /workspace/research/scratch/trace-art.XXXXXX)
"$PY" -m verity_vllm.pipeline.cli research-outputs write "$D" "$ROW" build --run-dir "$S" || exit 5
meta=$(python3 -c '
import json, sys
e = [o for o in json.load(open(sys.argv[1]))["outputs"] if o["name"] == "build"][0]
print(json.dumps({**e["meta"], "source_run": sys.argv[2]}))' "$S/outputs.json" "$RESEARCH_RUN_ID") || { echo "no build entry in $S/outputs.json"; exit 5; }
art=$("$RESEARCH_BIN" data put --kind vllm-build/v1 --tree "$S/outputs/build" --meta "$meta" --preserve | grep -oE 'art:[0-9a-f]{64}' | tail -1)
rm -rf "$S"
[ -n "$art" ] || { echo "research data put failed"; exit 6; }
echo "$art" | tee "$OUT/trace_art.txt"
