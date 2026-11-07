#!/usr/bin/env bash
# OVERHEADDEF's canonical-top-k capture on vy-spot-glm53, under research run --cwd source from a clean checkout of
# cursor/proofs-replaycov-glm-643a (benchmarks/replay_ladder/run.sh and benchmarks/serve_tax/replay.py), with these sent:
# dsa_topk.py and dsa_topk_kernels.py (cursor/proofs-topk-7adf), capture.py, stage.py and spot_job.sh
# (cursor/proofs-overheaddef-1d19), spot_tp8.sh, gpu-lease and sitecustomize.py (the TP8 shims), and this file.
#   [HOLD=/workspace/research/holds/glm53-sessions] bash "$RESEARCH_RUN_DIR/inputs/spot_canonical.sh"
# 0. HOLD=FILE: vllm's session job (glm53_sessions.sh) waits while FILE exists. It is written first, then the job waits up to
#    IDLE_MIN (40) minutes for every GPU to be under 1 GiB used, and FILE is removed when this script exits.
# 1. serve: run.sh glm53's pinned arm as fe57 served it (fe57's plan and C4 shard, MAX_MODEL_LEN 40960, FLASHINFER=1,
#    SPAN_S 120), with no replays, and with VERITY_DSA_TOPK=canonical in every engine process (OUT/topk-serve). A window
#    whose servers rewrote no long top-k row is refused (exit 4).
# 2. capture: spot_job.sh on that record arm (its own first-stage coin, its replay from an empty engine with capture.py,
#    the commitment, the coins and the stage), under the same plugin (OUT/topk-capture) and the serve's engine env.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out; IN=$D/inputs; mkdir -p "$OUT"
FE57=${FE57:-/workspace/research/runs/r20261107-090959-fe57}
VENV=${VENV:-/workspace/venv312}; PY=$VENV/bin/python
SENT="dsa_topk.py dsa_topk_kernels.py capture.py stage.py spot_job.sh spot_tp8.sh gpu-lease sitecustomize.py spot_canonical.sh"
for f in $SENT; do [ -f "$IN/$f" ] || { echo "canonical: --send $f"; exit 2; }; done
[ -f benchmarks/replay_ladder/run.sh ] && [ -f benchmarks/serve_tax/replay.py ] || { echo "canonical: run from the replay ladder's tree (--cwd source)"; exit 2; }
PLAN=${PLAN:-$FE57/out/plan/plan.jsonl}; C4=${C4:-$(ls "$FE57"/inputs/*.json.gz 2>/dev/null | head -1)}
[ -f "$PLAN" ] && [ -f "$C4" ] || { echo "canonical: no plan ($PLAN) or C4 shard ($C4)"; exit 2; }
{ echo "tree $(git rev-parse HEAD 2>/dev/null)"; (cd "$IN" && sha256sum $SENT); sha256sum "$PLAN" "$C4"; } | tee "$OUT/inputs.sha256"
T=$(mktemp -d); trap 'rm -rf "$T"; [ -n "${HOLD:-}" ] && rm -f "$HOLD" && echo "canonical: removed $HOLD"' EXIT
trap 'exit 143' TERM INT HUP
if [ -n "${HOLD:-}" ]; then
  mkdir -p "$(dirname "$HOLD")" && echo "OVERHEADDEF slot (proofs, $(basename "$D"), canonical-top-k capture): sessions wait" > "$HOLD" || exit 2
  echo "canonical: wrote $HOLD $(date -u +%FT%TZ)"
  for i in $(seq 0 $((${IDLE_MIN:-40} * 2))); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)
    [ "$busy" = 0 ] && break
    [ "$i" = $((${IDLE_MIN:-40} * 2)) ] && { echo "canonical: $busy GPUs still in use after ${IDLE_MIN:-40} min"; nvidia-smi; exit 5; }
    sleep 30
  done
  echo "canonical: GPUs idle $(date -u +%FT%TZ)"
fi
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | tee "$OUT/gpus_before.csv"
"$PY" "$IN/dsa_topk.py" site "$T/topk" >/dev/null || exit 2
unset VLLM_PLUGINS
export PYTHONPATH="$T/topk${PYTHONPATH:+:$PYTHONPATH}" VERITY_DSA_TOPK=canonical
live() {  # the long top-k rows the plugin rewrote, summed over the processes that logged under $1
  "$PY" - "$1" <<'EOF'
import json, sys
from pathlib import Path
logs = sorted(Path(sys.argv[1]).glob("dsa_topk.*.json"))
recs = [json.loads(p.read_text()) for p in logs]
per = [sum(r["rows_long"] for r in rec["rows"].values()) for rec in recs]
print(json.dumps({"dir": sys.argv[1], "processes": len(per), "processes_rewriting": sum(n > 0 for n in per), "rows_long": sum(per)}))
sys.exit(0 if sum(per) > 0 else 1)
EOF
}
echo "== serve $(date -u +%FT%TZ)"
VERITY_DSA_TOPK_LOG=$OUT/topk-serve PLAN=$PLAN C4=$C4 MAX_MODEL_LEN=40960 FLASHINFER=1 CUDA_HOME=/usr/local/cuda-12.9 \
  ARMS=pinned RBLOCK0_ARMS=pinned REPLAYS=0 FRESH=0 ALONE=0 LOGPROBS=1 SPAN_S=120 VENV=$VENV \
  bash benchmarks/replay_ladder/run.sh glm53
echo "canonical: run.sh rc=$? $(date -u +%FT%TZ)"
R=$OUT/pinned/glm53/record
[ -f "$R/meta.json" ] && [ -f "$R/saturated/rep-1/client.jsonl" ] || { echo "canonical: no record arm at $R"; exit 3; }
live "$OUT/topk-serve" | tee "$OUT/topk-serve.json" || { echo "canonical: the servers rewrote no long top-k row"; exit 4; }
cp "$OUT/engine_env.txt" "$OUT/serve_engine_env.txt"
echo "== capture $(date -u +%FT%TZ)"
VERITY_DSA_TOPK_LOG=$OUT/topk-capture ARM=$R LOAD=saturated ENGINE_ENV=$OUT/serve_engine_env.txt HF_HOME=/workspace/hf \
  VENV=$VENV bash "$IN/spot_tp8.sh"
rc=$?
live "$OUT/topk-capture" | tee "$OUT/topk-capture.json" || echo "canonical: the capture's replay rewrote no long top-k row"
echo "canonical: done rc=$rc $(date -u +%FT%TZ)"
exit $rc
