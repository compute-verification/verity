#!/usr/bin/env bash
# serve-openai --retain device --seal --release-after pouw on one H200, against --retain host of the same load: Llama-3.2-1B
# int8 (the B4 row's manifest and taps served_verdicts.sh defaults to), ROUNDS rounds of SLOTS fixed requests of
# PROMPT tokens and WINDOW_STEPS steps at T (--max-num-batched-tokens, default 8192), each round one Commit window.
#  1. host: --retain host --seal; every window sealed; then `retain_device_gpu.py reads` picks each window's draw by hand from
#     its sealed store and sizes the device pool at POOL_X (default 1.6) of the largest window, so a window's steps can't all fit
#     while the window before still holds the pool.
#  2. device: --retain device --retain-device-gb that --seal --release-after pouw --draws-dir; the drawer (`draw`) writes each
#     window's draw DELAY_S after its windows.jsonl line, so the next round's steps wait on the pool for that window's release.
#  3. `retain_device_gpu.py compare`: each device-sealed window holds exactly its draw's leaves, byte-equal to the host bundle's,
#     its drawn reads verify against its own run root, its undrawn read is refused by name; the pool's peak and waits.
#   research run --on NODE --gpus 1 --project verity --source TREE --cwd source --timeout 90m -- \
#     bash integrations/vllm/tools/retain_device_gpu.sh
# The serve dirs (sealed bundles: private values) stay on the node under `research keep`; $OUT keeps the records and logs, and
# summary.txt, whose RETAIN-DEVICE line is the verdict.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
TREE=$PWD
H=/workspace/research/vllm-composed-h200int8
ROW=${ROW:-$H/r20261026-185144-3ccb/llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager}
HIDDEN_SO=${HIDDEN_SO:-$H/fa2/build/fa3_matReqG/verity_fa3_matReqG.so} NORM_TAP_SO=${NORM_TAP_SO:-$H/norm_tap/build/verity_norm_tap.so}
VENV=${VENV:-/workspace/cp/venv312}
REPO=verity-int8/Llama-3.2-1B-INT8 REV=15b0976d400d796ee589e7737d03a919526bf800
T=${T:-8192} SLOTS=${SLOTS:-8} PROMPT=${PROMPT:-448} WINDOW_STEPS=${WINDOW_STEPS:-16} ROUNDS=${ROUNDS:-3}
POOL_X=${POOL_X:-1.6} DELAY_S=${DELAY_S:-20} PORT=${PORT:-18210}
ROOT=${RETAIN_ROOT:-/workspace/research/vllm-retain-device}
BASE=$ROOT/${RESEARCH_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}
export PY=$VENV/bin/python HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
if [ -z "${CUDA_HOME:-}" ]; then
  for c in /usr/local/cuda-13.0 /usr/local/cuda-12.9 /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && { CUDA_HOME=$c; break; }; done
fi
export CUDA_HOME PATH=$VENV/bin:$CUDA_HOME/bin:$HOME/.local/bin:$PATH
export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src
export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch-extensions TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
export HIDDEN_GPU_BUILD=$ROOT/cache/jit/hidden_gpu NATIVE_COLLECT_BUILD=$ROOT/cache/jit/nc
export VLLM_BATCH_INVARIANT=1 VERITY_WEIGHTS_HASH=${VERITY_WEIGHTS_HASH:-device-checked}
# a prefill of SLOTS x PROMPT tokens is a learning step past the default 4 GiB of transient host copies (no warm-up learns its plan)
export VERITY_LEARN_HOST_BUDGET_MB=${VERITY_LEARN_HOST_BUDGET_MB:-24576}
HELP=$TREE/integrations/vllm/tools/retain_device_gpu.py
say() { echo "$*" | tee -a "$OUT/summary.txt"; }
count() { cat "$1" 2>/dev/null | grep -c .; }
for f in "$ROW/manifest.json" "$HIDDEN_SO" "$NORM_TAP_SO" "$PY"; do [ -e "$f" ] || { say "PREFLIGHT: no $f"; exit 3; }; done
mkdir -p "$BASE" "$ROOT/cache"
{ id -un; hostname; git -C "$TREE" rev-parse HEAD 2>/dev/null; nproc; free -g | head -2; nvidia-smi -L; df -h /workspace | tail -1
  echo "base $BASE CUDA_HOME $CUDA_HOME T $T SLOTS $SLOTS PROMPT $PROMPT WINDOW_STEPS $WINDOW_STEPS ROUNDS $ROUNDS"; } | tee "$OUT/machine.txt"
{ "$PY" -m research keep "$ROOT" --owner @compute-accounting --expires 7d --priority normal \
    --why "retain_device_gpu.sh: --retain device vs --retain host serve dirs (sealed bundles)" \
  || research keep "$ROOT" --owner @compute-accounting --expires 7d --priority normal \
    --why "retain_device_gpu.sh: --retain device vs --retain host serve dirs (sealed bundles)"; } 2>&1 | tail -2

serve() {  # NAME RETAIN-ARGS...: one endpoint, ROUNDS rounds of the fixed load, every closed window sealed, then SIGTERM
  local name=$1; shift
  local D=$BASE/$name log=$OUT/$name/serve.log srv up=0 c=0 s=0 r rc=0
  mkdir -p "$D" "$OUT/$name"
  (cd "$TREE/integrations/vllm" && exec "$PY" -m verity_vllm.pipeline.cli serve-openai --model "$REPO" --revision "$REV" --out "$D" \
     --port "$PORT" --precision verity_int8 --run-id "${RESEARCH_RUN_ID:-local}-$name" --scheme rows-sha512 --seal --gpu-tree \
     --window-steps "$WINDOW_STEPS" --max-num-batched-tokens "$T" --max-num-seqs "$SLOTS" --max-model-len 512 \
     --required-manifest "$ROW/manifest.json" --hidden-so "$HIDDEN_SO" --norm-tap-so "$NORM_TAP_SO" --tap-cap-mb 2048 \
     --admit release-order --window-requests "$SLOTS" --disk-floor-gb 50 --seal-hold-gb 32 --seal-warm-gb 0 "$@") > "$log" 2>&1 &
  srv=$!
  echo "$srv" > "$OUT/$name/pid"
  for _ in $(seq 1 900); do
    grep -q "SERVE-OPENAI listening" "$log" && { up=1; break; }
    kill -0 "$srv" 2>/dev/null || break
    sleep 2
  done
  if [ "$up" != 1 ]; then
    tail -30 "$log"
    say "SERVE $name: not listening: $(grep -m1 -E 'UsageError|Error|refus' "$log" | cut -c1-500)"
    kill -TERM "$srv" 2>/dev/null; wait "$srv"; return 10
  fi
  say "SERVE $name $(grep -m1 'SERVE-OPENAI listening' "$log" | cut -c1-200)"
  for r in $(seq 1 "$ROUNDS"); do
    "$PY" "$TREE/examples/vllm/serve_load.py" --port "$PORT" --model "$REPO" --fixed "$SLOTS" --window-steps "$WINDOW_STEPS" \
      --prompt-range "$PROMPT:$PROMPT" --release "$name-r$r" --seed "$r" --out "$D/load-r$r.jsonl" > "$OUT/$name/load-r$r.log" 2>&1 || rc=$?
    say "REQUESTS $name round $r (exit $rc): $(tail -1 "$OUT/$name/load-r$r.log" | cut -c1-200)"
  done
  for _ in $(seq 1 600); do
    c=$(count "$D/windows.jsonl"); s=$(count "$D/seals.jsonl")
    [ "$c" -gt 0 ] && [ "$s" -ge "$c" ] && break
    kill -0 "$srv" 2>/dev/null || break
    sleep 2
  done
  kill -TERM "$srv" 2>/dev/null; wait "$srv"; local ex=$?
  touch "$D/.stop"                                  # the drawer drew every window the stop closed: the stop's seals waited on it
  c=$(count "$D/windows.jsonl"); s=$(count "$D/seals.jsonl")
  for f in serve.json summary.json windows.jsonl seals.jsonl requests.jsonl step_trace.jsonl; do cp "$D/$f" "$OUT/$name/" 2>/dev/null; done
  say "SERVE $name stopped (exit $ex): $c window(s) closed, $s sealed; $(grep -o '"release": {[^}]*' "$D/summary.json" 2>/dev/null | cut -c1-400)"
  [ "$rc" = 0 ] && [ "$c" -gt 0 ] && [ "$s" -ge "$c" ]
}

serve host --retain host || { say "STOP at the host serve"; exit 11; }
"$PY" "$HELP" reads "$BASE/host" "$OUT/reads" > "$OUT/reads.json" 2> "$OUT/reads.err" || { cat "$OUT/reads.err"; say "STOP: no draws from the host serve"; exit 12; }
GB=$("$PY" -c 'import json, math, sys; r = json.load(open(sys.argv[1])); print(f"{math.ceil(r["max_window_bytes"] * float(sys.argv[2]) / 2**30 * 100) / 100:.2f}")' \
  "$OUT/reads.json" "$POOL_X")
WGB=$("$PY" -c 'import sys; print(f"{float(sys.argv[1]) * 0.95:.2f}")' "$GB")
say "HOST windows and largest window's committed bytes: $(cat "$OUT/reads.json"); device pool $GB GB (x$POOL_X), --window-gb $WGB"

DD=$BASE/device
mkdir -p "$DD"
( "$PY" "$HELP" draw "$DD" "$OUT/reads" --protocol pouw --delay-s "$DELAY_S" --draws-dir "$DD/draws" --stop "$DD/.stop" ) > "$OUT/drawer.log" 2>&1 &
DRAWER=$!
serve device --retain device --retain-device-gb "$GB" --window-gb "$WGB" --release-after pouw --draws-dir "$DD/draws"; sr=$?
kill "$DRAWER" 2>/dev/null; wait "$DRAWER" 2>/dev/null
cp -r "$DD/draws" "$OUT/device/" 2>/dev/null
"$PY" "$HELP" compare "$DD" "$BASE/host" "$OUT/reads" > "$OUT/compare.json" 2> "$OUT/compare.err"; cr=$?
tail -1 "$OUT/compare.json" | tee -a "$OUT/summary.txt"
[ -s "$OUT/compare.err" ] && tail -5 "$OUT/compare.err"
grep -h '"release"' "$DD/seals.jsonl" 2>/dev/null | "$PY" -c 'import json, sys
for l in sys.stdin:
    s = json.loads(l); r = s.get("release") or {}
    print("SEAL w%s: release %s" % (s["window"], json.dumps({k: r.get(k) for k in ("union_leaves", "kept_bytes", "dropped_bytes", "device_freed_bytes", "draw_wait_s", "discarded")})))' \
  | tee -a "$OUT/summary.txt"
du -sh "$BASE"/host "$BASE"/device 2>/dev/null | tee -a "$OUT/summary.txt"
say "device serve rc=$sr compare rc=$cr"
exit $(( sr != 0 ? 13 : cr != 0 ? 14 : 0 ))
