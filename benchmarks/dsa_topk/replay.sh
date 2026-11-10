#!/usr/bin/env bash
# Replays of a served replay-ladder window with the canonical top-k in every engine process: under research run with
# --cwd source at a tree holding benchmarks/replay_ladder, with this script, dsa_topk.py and dsa_topk_kernels.py sent:
#   research run --on NODE --project verity --source LADDER_TREE --cwd source \
#     --send benchmarks/dsa_topk/replay.sh --send benchmarks/dsa_topk/dsa_topk.py \
#     --send benchmarks/dsa_topk/dsa_topk_kernels.py \
#     -- bash -c 'SRC=<served run dir>/out/pinned/glm53 bash "$RESEARCH_RUN_DIR/inputs/replay.sh"'
# SRC is the served run's model directory (record/ and its plan); MODEL (glm53) names it in ladder.py; REPLAYS (3).
# Each replay compares itself with the served record and, from the second on, with the first (ladder.py replay);
# out/pinned/MODEL/replay-I/dsa_topk/ holds each engine process's counters (dsa_topk.log_to), and topk-I.json their sum.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out
SRC=${SRC:?the model directory of the served run}
MODEL=${MODEL:-glm53}
VENV=${VENV:-/workspace/venv312}; PY=$VENV/bin/python
export PATH="$VENV/bin:$PATH"
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export PATH="$CUDA_HOME/bin:$PATH"
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM
export LADDER_ARM=pinned TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0
d=$OUT/pinned/$MODEL; mkdir -p "$d"
cp -a "$SRC/plan.jsonl" "$SRC/serve.json" "$SRC/arm_env.txt" "$d/" && cp -a "$SRC/record" "$d/record" || exit 2
rm -rf "$d/record/route_capture"
cache=$(mktemp -d); site=$(mktemp -d)
trap 'rm -rf "$cache" "$site"' EXIT
export VLLM_CACHE_ROOT=$cache/vllm TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton
"$PY" "$D/inputs/dsa_topk.py" site "$site" > /dev/null || exit 2
export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}" VERITY_DSA_TOPK=canonical
sha256sum "$D/inputs/dsa_topk.py" "$D/inputs/dsa_topk_kernels.py" > "$d/dsa_topk.sha256"
"$PY" -c 'from vllm.utils.flashinfer import has_flashinfer; print(f"has_flashinfer={has_flashinfer()}")' > "$d/replay_env.txt"
rc=0
for i in $(seq 1 "${REPLAYS:-3}"); do
  export VERITY_DSA_TOPK_LOG=$d/replay-$i/dsa_topk
  echo "== replay $i $(date -u +%FT%TZ)"
  "$PY" benchmarks/replay_ladder/ladder.py replay --model "$MODEL" --out "$d" --index "$i" --logprobs > "$d/replay-$i.log" 2>&1 \
    || { tail -n 40 "$d/replay-$i.log"; rc=1; }
  grep '^{"mode"' "$d/replay-$i.log" | tail -n 1
  "$PY" - "$VERITY_DSA_TOPK_LOG" <<'PY' | tee "$d/topk-$i.json"
import json, sys
from pathlib import Path
recs = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob("dsa_topk.*.json"))]
tot = {"processes": len(recs), "calls": {}, "launches": {}, "rows": {}}
for r in recs:
    for k in ("calls", "launches"):
        for op, v in r[k].items():
            tot[k][op] = tot[k].get(op, 0) + v
    for dev in r["rows"].values():
        for s, v in dev.items():
            tot["rows"][s] = tot["rows"].get(s, 0) + v
print(json.dumps(tot))
PY
done
"$PY" benchmarks/replay_ladder/prod_plan.py kinds "$d" | tee "$d/kinds.line" || rc=1
exit $rc
