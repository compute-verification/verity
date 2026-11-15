#!/usr/bin/env bash
# The serving half of `verity bench compute`: Llama 3.2 1B served under Pearl+ on one H100 or H200, every pass retained,
# then Pearl+'s verdict on the passes in Python and in Lean, written as the record `run.py` reads:
#
#   bash benchmarks/compute/serve.sh                      on a machine with the card, from any directory
#   research run --on NODE --project verity --cwd clone --timeout 2h -- bash benchmarks/compute/serve.sh
#
# Steps, each one's log in $OUT/logs/STEP.log and its exit status a line of $OUT/steps.txt:
#   1. cubin: Pearl-C's sm_90a cubin built from this tree (benchmarks/harness/setup/pearl_c_sm90.sh: nvcc, its SASS gates);
#   2. serve: `served_passes.py serve` of the spec's Pearl-C arm (SPEC) on one card, under `gpu-lease 1` where the machine
#      has it (GPU picks the card, default 0) and directly otherwise: a warm-up, then CLASS's fixed load, SLOTS requests of
#      PROMPT seeded random tokens and OUTPUTS outputs each, every engine step one pass retained under PASSES;
#   3. retain: Pearl+'s window record W over the timed call's passes (`served_passes.py retain --window-record`), the
#      warm-up's dropped;
#   4. verify: Pearl+'s verdict in Python (`served_passes.py verify --tie W --class CLASS`), the passes held to the class's
#      fixed load, DRAWS work-weighted tiles at DELTA, on fast_tile's batch forms (FAST, default on);
#   5. lean: the Lean verdict (`served_passes.py lean`, pouw-pearlplus built here unless PEARLPLUS_LEAN_EXE names one),
#      written only where it is the Python's;
#   6. policy: the inference-only policy at CLASS, from this card's serve-phi cut at Pearl+'s linears
#      (`benchmarks/one_stage/inference_only.py policy`), which names the class and the model `run.py` counts MACs at;
#   7. circuit: the class's circuit record cut at Pearl+'s linears (`prove_path.py class-circuit`), into out/record/.
#   8. verdicts: record/verdicts.json, PoUW's row: `missing`, as the window's compose (which composes it of record)
#      isn't run here, so the credit is the Pearl+ piece's alone.
# Coins (COINS): `local` (default), a local challenger: infra's gate code (`auditor.py`) on a scratch auditor host under
# PASSES with an LMS test key (W 4, H 6), each session's challenger its own serve.py on 127.0.0.1, whose coins are the
# Lean spot-verifier's (`benchmarks/one_stage/pearlplus_local_e2e.py init|gate|stop`): the salt coin a salt-only session's
# and the sample a withheld sample session's on the run's commitment, so the Lean verdict is taken of record. `in-process`:
# the salt coin the in-process challenger's and the sample the node's own draw (`verify_run.node_draw`), the Lean's verdict
# a development one. Both are development parameters, which $OUT/compute-run.json and the COINS line say.
# Writes under OUT ($RESEARCH_RUN_DIR/out under research run, else ./compute-out): pieces/pouw/pearlplus-answer.json,
# pearlplus-lean.json, pearlplus-lean-check.json, verify.json, served.json and coin.json; inference-only-policy.json;
# record/circuit-record.json, record/window-record.json and record/verdicts.json; auditor/ (the local gate's ledger,
# pins, public key and sessions' public files); compute-run.json and coins.txt. The record writes no Program: the tied
# path's window record W stands in its place. The passes stay under PASSES (default a directory under SCRATCH, else
# $TMPDIR), outside OUT: they hold the run's activations and salt key.
# Env: VERITY_BENCH_VENV (the pinned-vLLM Python's venv, else benchmarks/harness/env.sh's search), HF_HOME, CUDA_HOME,
# CLASS (4,32,4), GPU (0), DRAWS (459), DELTA (1/100), FAST (1; empty: the scalar reference alone), JOBS (nproc),
# LEASE_MIN (30). Exit 0 only where Python and Lean both accept.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 3
TREE=$PWD
OUT=${OUT:-${RESEARCH_RUN_DIR:+$RESEARCH_RUN_DIR/out}}
OUT=${OUT:-$TREE/compute-out}
COINS=${COINS:-local} CLASS=${CLASS:-4,32,4} GPU=${GPU-0} DRAWS=${DRAWS:-459} DELTA=${DELTA:-1/100} FAST=${FAST-1} JOBS=${JOBS:-$(nproc)}
IFS=, read -r SLOTS PROMPT OUTPUTS <<< "$CLASS"
REPO=unsloth/Llama-3.2-1B REV=9535bd9b1d1dea6acafbdc4813b728796aeb28da SCHEME=pearl-c-h100-v1-h3
SPEC=benchmarks/harness/specs/llama32-1b-1gpu-pouw-protocol-served-h200.toml
SP=benchmarks/pouw/pearl_c_vllm/served_passes.py
PIECE=$OUT/pieces/pouw REC=$OUT/record
mkdir -p "$OUT/logs" "$PIECE" "$REC" || exit 3
# shellcheck source=/dev/null
. benchmarks/harness/env.sh
PY=$VENV/bin/python
export PATH=$HOME/.elan/bin:$PATH HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
W=${PASSES:-$(mktemp -d -p "${SCRATCH:-${TMPDIR:-/tmp}}" pearlplus-compute.XXXXXX)}
P=$W/passes WR=$W/window-record.json
mkdir -p "$W" && chmod 700 "$W" || exit 3
umask 077
failed=0
step() {  # NAME CMD...: CMD's output to its log, its status a line of steps.txt
  local name=$1 t rc
  shift
  t=$(date +%s)
  "$@" > "$OUT/logs/$name.log" 2>&1
  rc=$?
  printf '%s rc=%s %ss: %s\n' "$name" "$rc" "$(( $(date +%s) - t ))" "$(tail -1 "$OUT/logs/$name.log" | cut -c1-400)" \
    | tee -a "$OUT/steps.txt"
  return "$rc"
}
lease() {  # CMD...: on one card, under gpu-lease where the machine has it
  if command -v gpu-lease > /dev/null; then
    GPU_LEASE_WHO=${GPU_LEASE_WHO:-compute-serve} gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${LEASE_MIN:-30}" -- "$@"
  else
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU} "$@"
  fi
}
cubin() {
  # shellcheck source=/dev/null
  . benchmarks/harness/setup/pearl_c_sm90.sh && printf '%s\n' "$PEARLC_KERNEL" "$PEARLC_CUBIN" > "$W/setup.paths"
}
coins() {  # the coins and parameters the run used, as a line and as compute-run.json
  "$PY" - "$OUT/compute-run.json" "$COINS" "$CLASS" "$DRAWS" "$DELTA" "$SCHEME" "$REPO" "$REV" \
    "$(git -C "$TREE" rev-parse HEAD 2> /dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}")" "$OUT/auditor-gate.json" <<'EOF'
import json, os, sys
out, mode, cls, draws, delta, scheme, repo, rev, commit, gate = sys.argv[1:11]
if mode == "local":
    g = json.load(open(gate)) if os.path.exists(gate) else {}
    w, h = g.get("params") or (None, None)
    held = {"challenger": "local", "gate": "verity/protocols/compose/device/live_challenger/auditor.py on a scratch host "
            "(benchmarks/one_stage/pearlplus_local_e2e.py), each session's challenger serve.py on 127.0.0.1",
            "key": f"LMS test key (W {w}, H {h}), made for this run", "key_sha512": g.get("key_sha512"),
            "lane": g.get("lane"), "salt_coin": "a salt-only session's (live)",
            "sample_coin": "a withheld sample session's on the run's commitment (live)", "lean": "of record (--of-record)"}
    line = f"local challenger (scratch auditor gate, serve.py on 127.0.0.1, LMS test key {w},{h}, Lean spot-verifier's coins)"
else:
    held = {"challenger": "in-process", "salt_coin": "in-process challenger",
            "sample_coin": "node draw (verify_run.node_draw)", "lean": "development (no --of-record)"}
    line = "in-process challenger (salt) and the node's draw (sample)"
doc = {"schema": "verity-bench/compute-run/v1", **held, "development": True, "draws": int(draws), "delta": delta,
       "over_cap": "uncredited", "class": cls, "scheme": scheme, "model": repo, "revision": rev, "commit": commit}
open(out, "w").write(json.dumps(doc, indent=1) + "\n")
print(f"COINS {line}; parameters: development ({draws} draws at delta {delta}, over-cap uncredited, {scheme}, class "
      f"{cls}), {repo}@{rev[:10]}, commit {commit[:12]}")
EOF
}
gate_up() {  # the local challenger's scratch auditor host, its gate command and key into auditor-gate.json
  "$PY" benchmarks/one_stage/pearlplus_local_e2e.py init --root "$W/auditor" && tail -1 > "$OUT/auditor-gate.json" \
    < "$OUT/logs/gate.log"
}
gate_down() {
  [ -d "$W/auditor" ] && "$PY" benchmarks/one_stage/pearlplus_local_e2e.py stop --root "$W/auditor" --out "$OUT" \
    > "$OUT/logs/gate-stop.log" 2>&1
  [ -s "$P/coin.json" ] && cp "$P/coin.json" "$PIECE/coin.json"
  return 0
}
verdicts() {  # PoUW's row as run.py reads it: missing, the window's compose not run here
  "$PY" - "$REC/verdicts.json" "$PIECE/pearlplus-answer.json" "$OUT/compute-run.json" <<'EOF'
import json, os, sys
out, answer, run = sys.argv[1:4]
a = json.load(open(answer)) if os.path.exists(answer) else {}
row = {"verdict": "missing", "owner": "compute-accounting",
       "why": "PoUW's row of record is the window's compose (V3c), not run here: the credit is the Pearl+ piece's alone "
              f"(its verdict {a.get('verdict')}, development coins and parameters)"}
doc = {"format": "verity-bench/compute-verdicts/v1", "development": True, "run": json.load(open(run)),
       "protocols": {"work": row}}
open(out, "w").write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
print(f"work: {row['verdict']}: {row['why']}")
EOF
}
[ -n "${RESEARCH_RUN_ID:-}" ] && "$PY" -m research keep "$W" --owner @compute-accounting --priority normal \
  --expires "$(date -u -d "+${KEEP_DAYS:-7} days" +%F)" --why "the retained passes of compute serve $RESEARCH_RUN_ID" \
  > "$OUT/logs/keep.log" 2>&1
echo "passes: $P" | tee -a "$OUT/steps.txt"

step cubin cubin || exit 1
{ read -r PEARLC_KERNEL; read -r PEARLC_CUBIN; } < "$W/setup.paths"
export PEARLC_KERNEL PEARLC_CUBIN PEARLC_BF16_ROWS=${PEARLC_BF16_ROWS:-0}
unset PEARLC_CHALLENGER_GATE PEARLC_CHALLENGER_PEER PEARLC_CHALLENGER_LANE VLLM_BATCH_INVARIANT
case $COINS in
  local)
    trap gate_down EXIT
    step gate gate_up || exit 1
    { read -r PEARLC_CHALLENGER_GATE; read -r PEARLC_CHALLENGER_PEER; read -r PEARLC_CHALLENGER_LANE; } < <("$PY" -c '
import json, sys
g = json.load(open(sys.argv[1]))
print(g["gate"], g["peer"], g["lane"], sep="\n")' "$OUT/auditor-gate.json") || exit 1
    export PEARLC_CHALLENGER_GATE PEARLC_CHALLENGER_PEER PEARLC_CHALLENGER_LANE ;;
  in-process) ;;
  *) echo "COINS is local or in-process, not $COINS" >&2; exit 3 ;;
esac
coins | tee "$OUT/coins.txt"
step serve lease "$PY" "$SP" serve --spec "$SPEC" --requests "$SLOTS" --prompt-tokens "$PROMPT" \
  --output-tokens "$OUTPUTS" --max-num-batched-tokens "${T:-8192}" --passes "$P" --out "$PIECE/served.json" || failed=1
if [ "$failed" = 0 ]; then
  step volunteer-needed "$PY" "$SP" volunteer-needed
  grep -q '^no' "$OUT/logs/volunteer-needed.log" || { echo "the verdict needs a give-up before its coin; this runs none"; exit 1; }
  step retain "$PY" "$SP" retain --passes "$P" --window-record "$WR" || exit 1
  cp "$WR" "$REC/window-record.json"
  step verify "$PY" "$SP" verify --passes "$P" --tie "$WR" --class "$CLASS" --out "$PIECE/verify.json" \
    --answer "$PIECE/pearlplus-answer.json" --draws "$DRAWS" --delta "$DELTA" --jobs "$JOBS" ${FAST:+--fast} || failed=1
  [ -s "$PIECE/verify.json" ] && step lean "$PY" "$SP" lean --passes "$P" --verify "$PIECE/verify.json" \
    --exe "${PEARLPLUS_LEAN_EXE:-build}" --out "$PIECE/pearlplus-lean.json" --timeout "${LEAN_TIMEOUT:-3600}" || failed=1
fi
phi() {
  CUDA_VISIBLE_DEVICES='' "$PY" -c '
import json, sys
from verity_vllm import TargetProfile
from verity_vllm.engine.served_seed import DEFAULT_SEED
doc = {"model": sys.argv[1], "revision": sys.argv[2], "default_seed": DEFAULT_SEED,
       "target": TargetProfile.default().with_overrides(compute_capability=(9, 0), num_sms=132).to_json(),
       "execution": "bi-eager"}
open(sys.argv[3], "w").write(json.dumps(doc, indent=1) + "\n")
print(json.dumps(doc, sort_keys=True))
' "$REPO" "$REV" "$REC/serve-phi.json"
}
step serve-phi phi \
  && step policy env CUDA_VISIBLE_DEVICES='' "$PY" benchmarks/one_stage/inference_only.py policy --class "$CLASS" \
    --model "$REPO" --revision "$REV" --lease-min "${LEASE_MIN:-30}" --tree "$(git rev-parse HEAD 2> /dev/null)" \
    --serve "$REC/serve-phi.json" --pearl-tables "$SCHEME" --out "$OUT/inference-only-policy.json" || failed=1
step circuit env CUDA_VISIBLE_DEVICES='' "$PY" benchmarks/prove_path/prove_path.py class-circuit --serve "$REC/serve-phi.json" \
  --class "$CLASS" --pearl-tables "$SCHEME" --out "$REC/circuit-record.json"
step verdicts verdicts
gate_down
trap - EXIT
coins
exit "$failed"
