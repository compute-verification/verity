#!/usr/bin/env bash
# The ladder on a production-like window (prod_plan.py), under research run, with the C4 shard sent (--send):
#   prod.sh MODEL     fetches vLLM's benchmark_serving ShareGPT file at its pin, builds the plan over it, the C4 shard and the
#                     model's chat template, serves and replays it with run.sh (PLAN, MAX_MODEL_LEN), and breaks each replay's
#                     differing requests down by kind (kinds.json).
# run.sh's variables pass through (ARMS, RBLOCK0_ARMS, REPLAYS, FRESH, ALONE, UNPINNED_REPLAYS, LOGPROBS, SPAN_S, OVER, BURST); N (8000)
# and SEED (0) set the plan, MAX_MODEL_LEN (40960) the servers' context limit.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=${VENV:-/workspace/venv312}; PY=$VENV/bin/python
export HF_HOME=${HF_HOME:-/workspace/hf}
SHAREGPT_REPO=anon8231489123/ShareGPT_Vicuna_unfiltered
SHAREGPT_REV=192ab2185289094fc556ec8ce5ce1e8e587154ca
SHAREGPT_FILE=ShareGPT_V3_unfiltered_cleaned_split.json
SHAREGPT_SHA256=35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4
[ $# -eq 1 ] || { echo "prod: name one model (ladder.py's LADDER)"; exit 2; }
C4=${C4:-$(ls "$D"/inputs/*.json.gz 2>/dev/null | head -1)}
[ -f "$C4" ] || { echo "prod: no C4 shard (--send it, or C4=PATH)"; exit 2; }
mkdir -p "$OUT/plan"
sg=$(HF_HUB_OFFLINE=0 "$PY" -c "from huggingface_hub import hf_hub_download as d; print(d('$SHAREGPT_REPO', '$SHAREGPT_FILE', repo_type='dataset', revision='$SHAREGPT_REV'))") || exit 2
echo "$SHAREGPT_SHA256  $sg" | sha256sum -c --quiet || { echo "prod: $sg is not the pinned $SHAREGPT_FILE"; exit 2; }
ck=$("$PY" -c "import sys; sys.path.insert(0, '$here'); import ladder; print(ladder.entry('$1')['local_path'])") || exit 2
HF_HUB_OFFLINE=1 "$PY" "$here/prod_plan.py" build --checkpoint "$ck" --sharegpt "$sg" --c4 "$C4" --out "$OUT/plan/plan.jsonl" \
  --n "${N:-8000}" --seed "${SEED:-0}" || exit 2
PLAN=$OUT/plan/plan.jsonl MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960} C4=$C4 bash "$here/run.sh" "$1"
rc=$?
for d in "$OUT"/*/"$1"/; do
  [ -f "$d/record/meta.json" ] && "$PY" "$here/prod_plan.py" kinds "$d" | tee "$d/kinds.line" || rc=1
done
exit $rc
