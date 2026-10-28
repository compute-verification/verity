#!/usr/bin/env bash
# PoUW's piece of a served H1 window with its drawn units proved (served_verdicts.sh's POUW_CMD): drawn.py stages each
# drawn call's J10-R units on the window's own registered values, each staged call is proved by one loopback C-Flock --zk
# session drawing all its instances (hidden_zk/job.sh's `session`), drawn.py writes the statements file and checks it as
# credit does, and then `served_verdicts.py pouw` judges the window with it as --statements.
#   POUW_CMD='bash benchmarks/pouw/h1_prove/drawn.sh "$1" "$2" "$3"'      (served_verdicts.sh runs it in the tree)
#   bash benchmarks/pouw/h1_prove/drawn.sh SERVE_DIR WINDOW PIECE_DIR
# env: H1_CIRCUITS (pouw_h1_circuits' --out, with the drawn K's k<K> file: --strata K)  H1_CHECKPOINT (the served model's
#      safetensors dir, b/<K>'s rows) or H1_WEIGHTS (an npz of w<i>, s<i>)  H1_GATE and H1_MODEL (the auditor's model-key
#      for b/<K>, the server's --h1-model; or H1_WEIGHTS_KEY, a key file)  H1_DRAW_K (else period.json's)
#      FLOCK_PROVE, FLOCK_SELFTEST, FLOCK_VERIFY (prebuilt; else job.sh builds them under HZ_CACHE)  H1_DRAWN_MEM_GB (the
#      prover's budget, default 90% of the host's memory)  PY (served_verdicts.sh's venv python, default .venv's)
#      H1_DRAWN_ONLY=1: write and check the statements, no verdict
# Out: PIECE_DIR/drawn/ (stage/, the plan and each call's statement files and session; out/ job.sh's logs;
# statements.json), then served_verdicts.py pouw's files in PIECE_DIR.
set -uo pipefail
SD=${1:?SERVE_DIR} WIN=${2:?WINDOW} P=${3:?PIECE_DIR}
cd "$(dirname "$0")/../../.." || exit 1
VPY=${PY:-$(pwd)/.venv/bin/python}
mkdir -p "$P/drawn"
P=$(cd "$P" && pwd)
export RESEARCH_RUN_DIR=$P/drawn HZ_CONFIGS=drawn HZ_RUNS=1 HZ_WARM=0 HZ_PYTHON=$VPY
export HZ_MEM_GB=${H1_DRAWN_MEM_GB:-$(awk '/MemTotal/ {print int($2 * 0.9 / 1048576)}' /proc/meminfo)}
# shellcheck source=../hidden_zk/job.sh
. benchmarks/pouw/hidden_zk/job.sh
setup
export PYTHONPATH=$SRC:$SRC/integrations/vllm:$SRC/verity/ml/flock/python${PYTHONPATH:+:$PYTHONPATH}
WK=$P/drawn/stage ST=$P/drawn/statements.json
args=(--circuits "${H1_CIRCUITS:?H1_CIRCUITS: pouw_h1_circuits --out, with the drawn K file}")
[ -n "${H1_DRAW_K:-}" ] && args+=(--k "$H1_DRAW_K")
st=("${args[@]}" --work "$WK")
[ -n "${H1_CHECKPOINT:-}" ] && st+=(--checkpoint "$H1_CHECKPOINT")
[ -n "${H1_WEIGHTS:-}" ] && st+=(--weights "$H1_WEIGHTS")
if [ -n "${H1_WEIGHTS_KEY:-}" ]; then st+=(--weights-key "$H1_WEIGHTS_KEY")
elif [ -n "${H1_GATE:-}" ] && [ -n "${H1_MODEL:-}" ]; then st+=(--gate "$H1_GATE" --model "$H1_MODEL"); fi
log "drawn: window $WIN of $SD, staging"
$TIME "$OUT/stage.time" bash -c "$(vlimit "$STAGE_GB") exec $PY $J/../h1_prove/drawn.py stage $(printf '%q ' "$SD" "$WIN" "${st[@]}")" \
  2>&1 | tee -a "$OUT/job.log"
[ "${PIPESTATUS[0]}" = 0 ] || { log "drawn: staging REFUSED"; exit 1; }
for d in $("$PY" -c 'import json, sys; print(" ".join(f"{c["dir"]}:{c["instances"]}" for c in json.load(open(sys.argv[1]))["calls"]))' "$WK/plan.json"); do
  D=${d%:*} N=${d##*:} O=$OUT/$(basename "${d%:*}")
  mkdir -p "$O" && rm -rf "$D/sessions"
  log "drawn: $(basename "$D"): --zk session over its $N drawn units"
  session "$D" "$O" && grep -q '"accepted":true' "$O/prove.out" \
    || { log "drawn: $(basename "$D"): --zk session FAILED: $(tail -1 "$O/prove.out") $(tail -1 "$O/serve.log")"; exit 1; }
done
"$PY" "$J/../h1_prove/drawn.py" statements --work "$WK" --out "$ST" 2>&1 | tee -a "$OUT/job.log"
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
"$PY" "$J/../h1_prove/drawn.py" check "$SD" "$WIN" "${args[@]}" --statements "$ST" --lean "$LEAN" 2>&1 | tee -a "$OUT/job.log"
[ "${PIPESTATUS[0]}" = 0 ] || { log "drawn: credit refuses the statements"; exit 1; }
[ "${H1_DRAWN_ONLY:-0}" = 1 ] && exit 0
exec "$VPY" benchmarks/one_stage/served_verdicts.py pouw "$SD" "$WIN" "$P" --statements "$ST" \
  ${H1_LIVE_EVIDENCE:+--live-evidence "$H1_LIVE_EVIDENCE"}
