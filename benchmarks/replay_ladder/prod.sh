#!/usr/bin/env bash
# The ladder on a production-like window (prod_plan.py), under research run, with the C4 shard sent (--send):
#   prod.sh MODEL     takes vLLM's benchmark_serving ShareGPT file at its pin (the serve tax's chat load's file: the node's
#                     copy, downloaded where there is none, its SHA-512 checked), builds the plan over it, the C4 shard and
#                     the model's chat template, serves and replays it with run.sh (PLAN, MAX_MODEL_LEN), and breaks each
#                     replay's differing requests down by kind (kinds.json).
# run.sh's variables pass through (ARMS, RBLOCK0_ARMS, REPLAYS, FRESH, ALONE, UNPINNED_REPLAYS, LOGPROBS, SPAN_S, OVER, BURST,
# and FLASHINFER, here 1 unless set: vLLM's defaults with FlashInfer available); N (8000) and SEED (0) set the plan,
# MAX_MODEL_LEN (40960) the servers' context limit. LADDER_ROUTE_CAPTURE=DIR (route_capture_plugin.py and its settings.py,
# sent) has both servers capture their routes too, under capacity/route_capture and record/route_capture.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=${VENV:-/workspace/venv312}; PY=$VENV/bin/python
export HF_HOME=${HF_HOME:-/workspace/hf}
[ $# -eq 1 ] || { echo "prod: name one model (ladder.py's LADDER)"; exit 2; }
C4=${C4:-$(ls "$D"/inputs/*.json.gz 2>/dev/null | head -1)}
[ -f "$C4" ] || { echo "prod: no C4 shard (--send it, or C4=PATH)"; exit 2; }
mkdir -p "$OUT/plan"
sg=$("$PY" "$here/../serve_tax/serve_tax.py" dataset chat --fetch) || exit 2
ck=$("$PY" -c "import sys; sys.path.insert(0, '$here'); import ladder; print(ladder.entry('$1')['local_path'])") || exit 2
HF_HUB_OFFLINE=1 "$PY" "$here/prod_plan.py" build --checkpoint "$ck" --sharegpt "$sg" --c4 "$C4" --out "$OUT/plan/plan.jsonl" \
  --n "${N:-8000}" --seed "${SEED:-0}" || exit 2
PLAN=$OUT/plan/plan.jsonl MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960} FLASHINFER=${FLASHINFER:-1} C4=$C4 bash "$here/run.sh" "$1"
rc=$?
for d in "$OUT"/*/"$1"/; do
  [ -f "$d/record/meta.json" ] && "$PY" "$here/prod_plan.py" kinds "$d" | tee "$d/kinds.line" || rc=1
done
exit $rc
