#!/usr/bin/env bash
# The lead served by serve-openai on one H200 beside stock, in one gpu-lease (serve_lead.py). Before the lease: serve.cu
# built into a library with the served committer's leaf kernels (as run.sh builds ssn), SASS-gated (sass_gate.json), its
# mainloop checked (IGMMA in every ssn_k, no local memory) and the recorded gate held to the tree (gate.json, gate.py
# --recorded). Under it: the library's exact check at each of MODES (check_m<M>.jsonl), then each arm served in turn on
# served_util.sh's mix (N requests of exactly PROMPT_IDS prompt ids, max_tokens 1, greedy, all queued at once, SLOTS
# requests a step and a window at T tokens a step, one step a window, no prefix cache, the int8 Llama-3.2-1B under
# verity_int8, VLLM_BATCH_INVARIANT=1, eager, synchronous scheduling), WARM requests first; the SM clock sampled beside it
# all. The defaults are V3's class: T 512 with SLOTS 4 of PROMPT_IDS 16 (64 tokens a step) and MAX_MODEL_LEN 64;
# served_util.sh's T 8,192 prefill mix is T=8192 PROMPT_IDS=1024 SLOTS=8 MAX_MODEL_LEN=1088. After it: report.json and
# report.txt, each arm's steady step wall (its full steps of SLOTS·PROMPT_IDS tokens) and its tokens against the first
# arm's. Measurement only: what it serves credits nothing. cwd: the source tree's root; the node holds the venv and the
# model.
#   ARMS (stock lead2 lead1 lead0: stock, then the lead at serve.cu's modes 2, 1 and 0; the first is the report's base),
#   T (512), PROMPT_IDS (16), SLOTS (4), MAX_MODEL_LEN (64), N (96), WARM (2·SLOTS), AUDIT (128: lead2's first calls
#   compared with the reference and stock, the warm load's), BUDGET_GB (8: the exported words a hash chunk holds), NBUF (4:
#   mode 2's buffers), MODES (2 1 0: the checks; empty skips them), LEASE_MIN (90), CUDA_HOME (/usr/local/cuda-13.0),
#   VENV (/workspace/cp/venv312), HF_HOME (/workspace/hf), CACHE (/workspace/research/cache/served-lead: vLLM's, Triton's
#   and torch's), PORT (18191), BUILD_ONLY (0; 1 stops after the build, the SASS checks and the gate, on any machine with
#   a CUDA toolkit and the venv).
# Published ($OUT): steps.txt, the build's logs and checks, check_m<M>.jsonl, smi.csv, gpu.txt, and per arm <arm>/: its
# serve.log's protocol and engine lines, serve.json, summary.json, advice_trace.jsonl, the loads' timings, main.jsonl (the
# requests' output ids) and, for a lead arm, lead.json (serve_lead.py's --stats).
set -uo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/steps.txt"; }
export TREE=${TREE:-$PWD} HERE=benchmarks/pouw/np_kernel
export T=${T:-512} PROMPT_IDS=${PROMPT_IDS:-16} SLOTS=${SLOTS:-4} MAX_MODEL_LEN=${MAX_MODEL_LEN:-64} N=${N:-96}
export WARM=${WARM:-$(( 2 * SLOTS ))} STEP_TOKENS=$(( SLOTS * PROMPT_IDS )) ARMS=${ARMS:-stock lead2 lead1 lead0} MODES=${MODES-2 1 0}
export AUDIT=${AUDIT:-128} BUDGET_GB=${BUDGET_GB:-8} NBUF=${NBUF:-4} PORT=${PORT:-18191}
export VENV=${VENV:-/workspace/cp/venv312} CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PY=$VENV/bin/python REPO=verity-int8/Llama-3.2-1B-INT8 REV=15b0976d400d796ee589e7737d03a919526bf800
export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src

if [ "${1:-}" != leased ]; then
  for arm in $ARMS; do
    case $arm in stock|lead0|lead1|lead2) ;; *) say "ARMS: $arm is not stock or lead0, lead1, lead2"; exit 2 ;; esac
  done
  [ "$SLOTS" -ge 1 ] && [ "$STEP_TOKENS" -le "$T" ] && [ "$PROMPT_IDS" -lt "$MAX_MODEL_LEN" ] \
    || { say "SLOTS $SLOTS of PROMPT_IDS $PROMPT_IDS: not 1 to T $T tokens a step, each prompt under MAX_MODEL_LEN $MAX_MODEL_LEN"; exit 2; }
  SCR=$(mktemp -d /tmp/served-lead-XXXXXX); export SCR
  trap 'rm -rf "$SCR"' EXIT
  say "build $(git -C "$TREE" rev-parse --short=10 HEAD 2>/dev/null) into $SCR: ARMS $ARMS T $T SLOTS $SLOTS PROMPT_IDS" \
    "$PROMPT_IDS MAX_MODEL_LEN $MAX_MODEL_LEN N $N WARM $WARM"
  COMMITTER=integrations/vllm/verity_vllm/commit/committer
  python3 "$HERE/plain_rows.py" "$COMMITTER/native_rows.cu" "$SCR/native_rows_p.cu" || { say "plain_rows.py failed"; exit 3; }
  "$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v --no-compress -Xcompiler -fPIC \
    -shared "$HERE/serve.cu" "$COMMITTER/native_rows.cu" "$SCR/native_rows_p.cu" -I "$COMMITTER" -I "$HERE" -lcuda \
    -o "$SCR/libssn_lead.so" > "$OUT/build.log" 2>&1
  rc=$?; say "build libssn_lead.so rc $rc"; [ "$rc" = 0 ] || { tail -40 "$OUT/build.log"; exit 3; }
  POUW_CUDA_BIN="$CUDA_HOME/bin" POUW_SASS_GATE_CACHE=none python3 benchmarks/pouw/harness/sass_gate.py \
    "$SCR/libssn_lead.so" --json "$OUT/sass_gate.json" 2>&1 | tee -a "$OUT/steps.txt"
  rc=${PIPESTATUS[0]}; say "SASS gate rc $rc"; [ "$rc" = 0 ] || exit 4
  "$CUDA_HOME/bin/cuobjdump" -sass "$SCR/libssn_lead.so" > "$SCR/lib.sass"
  python3 - "$SCR/lib.sass" > "$OUT/mainloop.json" <<'EOF'
import json, re, sys
funcs, cur = {}, None
for line in open(sys.argv[1]):
    m = re.search(r"Function : (\S+)", line)
    if m:
        cur = m.group(1); funcs[cur] = {"IGMMA": 0, "local": 0}
    elif cur:
        funcs[cur]["IGMMA"] += "IGMMA" in line
        funcs[cur]["local"] += bool(re.search(r"\b(LDL|STL)\b", line))
gemms = {k: v for k, v in funcs.items() if "ssn_k" in k}
ok = len(gemms) == 4 and all(v["IGMMA"] > 0 and v["local"] == 0 for v in gemms.values())
print(json.dumps({"ok": ok, "functions": funcs}))
sys.exit(0 if ok else 1)
EOF
  rc=$?; say "mainloop check rc $rc"; [ "$rc" = 0 ] || { cat "$OUT/mainloop.json"; exit 4; }
  "$PY" "$HERE/gate.py" --recorded --json > "$OUT/gate_recorded.json" 2> "$OUT/gate.err"
  rc=$?; say "gate --recorded rc $rc: $(tr -d '\n ' < "$OUT/gate_recorded.json" | cut -c1-240)"; [ "$rc" = 0 ] || { cat "$OUT/gate.err"; exit 4; }
  [ "${BUILD_ONLY:-0}" = 1 ] && { say "BUILD_ONLY: no lease, no GPU work"; exit 0; }
  gpu-lease status 2>&1 | tee -a "$OUT/steps.txt"
  say "lease requested"
  GPU_LEASE_WHO=vllm-lead gpu-lease 1 --wait --max-min "${LEASE_MIN:-90}" -- bash "$TREE/$HERE/served_lead.sh" leased
  rc=$?; say "LEASED PART rc $rc"
  arms=(); for arm in $ARMS; do [ -f "$OUT/$arm/advice_trace.jsonl" ] && arms+=(--arm "$arm=$OUT/$arm"); done
  if [ ${#arms[@]} -gt 0 ]; then
    "$PY" "$TREE/$HERE/serve_lead.py" report "${arms[@]}" --T "$STEP_TOKENS" --out "$OUT/report.json" > "$OUT/report.txt" 2>&1
    say "REPORT rc $?"; cat "$OUT/report.txt"
  fi
  exit "$rc"
fi

say "leased CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
ulimit -n "$(ulimit -Hn)"
CACHE=${CACHE:-/workspace/research/cache/served-lead}; mkdir -p "$CACHE"
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1 PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export TORCH_EXTENSIONS_DIR=$CACHE/torch-extensions TRITON_CACHE_DIR=$CACHE/triton VLLM_CACHE_ROOT=$CACHE/vllm
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit,memory.used \
  --format=csv > "$OUT/gpu.txt" 2>&1
cat "$OUT/gpu.txt" | tee -a "$OUT/steps.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,temperature.gpu,memory.used --format=csv -lms 200 \
  -i "${CUDA_VISIBLE_DEVICES:-0}" > "$OUT/smi.csv" 2>&1 &
smi=$!
srv=
trap 'kill $smi 2>/dev/null; [ -n "$srv" ] && kill -TERM "$srv" 2>/dev/null' EXIT
LIB=$SCR/libssn_lead.so
worst=0
for m in $MODES; do
  (cd "$TREE/integrations/vllm" && "$PY" "$TREE/$HERE/serve_lead.py" check --lib "$LIB" --mode "$m" --budget-gb "$BUDGET_GB" \
    --nbuf "$NBUF" --out "$OUT/check_m$m.jsonl") > "$OUT/check_m$m.log" 2>&1
  rc=$?; say "check mode $m rc $rc: $(tail -1 "$OUT/check_m$m.log" | cut -c1-200)"
  [ "$rc" = 0 ] || { worst=$rc; tail -20 "$OUT/check_m$m.log"; }
done
[ "$worst" = 0 ] || { say "the library's check failed: nothing served"; exit "$worst"; }

alive() { case $(ps -o stat= -p "$1" 2>/dev/null) in ''|Z*) return 1 ;; esac; }
load() {  # $1 requests, $2 seed, $3 the arm's scratch, $4 name
  "$PY" "$TREE/examples/vllm/serve_load.py" --port "$PORT" --model "$REPO" --varied "$1" --prompt-range "$PROMPT_IDS:$PROMPT_IDS" \
    --max-tokens 1 --rate 1000 --seed "$2" --out "$3/$4.jsonl" > "$3/$4.log" 2>&1
}
for arm in $ARMS; do
  D=$SCR/$arm; A=$OUT/$arm; mkdir -p "$D" "$A"
  args=(--model "$REPO" --revision "$REV" --out "$D/serve" --port "$PORT" --precision verity_int8
        --run-id "${RESEARCH_RUN_ID:-local}-$arm" --admit arrival --window-requests "$SLOTS" --max-num-seqs "$SLOTS"
        --max-num-batched-tokens "$T" --max-model-len "$MAX_MODEL_LEN" --window-steps 1 --no-prefix-caching --no-commit)
  case $arm in
    stock) lead=(--stock) ;;
    lead*) m=${arm#lead}; lead=(--lib "$LIB" --mode "$m" --budget-gb "$BUDGET_GB" --nbuf "$NBUF" --max-m "$T" --stats "$A/lead.json")
           [ "$m" = 2 ] && lead+=(--audit "$AUDIT") ;;
  esac
  (cd "$TREE/integrations/vllm" && exec "$PY" "$TREE/$HERE/serve_lead.py" serve "${lead[@]}" -- "${args[@]}") > "$D/serve.log" 2>&1 &
  srv=$!
  up=0
  for _ in $(seq 1 900); do
    grep -q "SERVE-OPENAI listening" "$D/serve.log" && { up=1; break; }
    alive "$srv" || break
    sleep 2
  done
  grep -aE "SERVE-OPENAI|LEAD|engine resolved|Error|error|refus|Traceback" "$D/serve.log" | grep -av prompt | tail -60 > "$A/serve.log.txt"
  if [ "$up" != 1 ]; then
    say "ARM $arm: not listening: $(grep -am1 -E 'Error|refus' "$D/serve.log" | cut -c1-400)"
    tail -40 "$D/serve.log" > "$A/serve.tail.txt"
    kill -TERM "$srv" 2>/dev/null; wait "$srv"; srv=; worst=10; continue
  fi
  say "ARM $arm $(grep -am1 'SERVE-OPENAI listening' "$D/serve.log" | cut -c1-160) $(grep -am1 'LEAD ' "$D/serve.log" | cut -c1-200)"
  load "$WARM" 7 "$D" warm; say "ARM $arm WARM $WARM requests (exit $?)"
  load "$N" 1 "$D" main; lrc=$?
  say "ARM $arm MAIN $N requests (exit $lrc): $(tail -1 "$D/main.log" | cut -c1-200)"
  kill -TERM "$srv"
  for _ in $(seq 1 300); do alive "$srv" || break; sleep 2; done
  if alive "$srv"; then say "ARM $arm: pid $srv still up 600 s after its stop: killed"; kill -KILL "$srv"; fi
  wait "$srv"; src=$?; srv=
  say "ARM $arm stopped (exit $src): $(grep -a 'SERVE-OPENAI stopped' "$D/serve.log" | tail -1 | cut -c1-200)"
  grep -aE "SERVE-OPENAI|LEAD|engine resolved|Error|error|refus|Traceback" "$D/serve.log" | grep -av prompt | tail -60 > "$A/serve.log.txt"
  for f in serve/advice_trace.jsonl serve/serve.json serve/summary.json main.jsonl main.timings.json warm.timings.json; do
    [ -f "$D/$f" ] && cp "$D/$f" "$A/"
  done
  [ "$lrc" = 0 ] && [ "$src" = 0 ] || worst=11
done
exit "$worst"
