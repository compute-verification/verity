#!/usr/bin/env bash
# The whole-step tensor utilization of serve-openai on the int8 Llama-3.2-1B prefill mix (VLLM-33), one arm a run, measured
# on the GPU by Nsight Systems' GPU metrics (Tensor Active, 10 kHz) over the engine's steady state; `served_util.py` reads
# the profile and the server's step trace.  cwd: the source tree's root; the node holds the venv, the B4 row's Build and the
# taps under UTIL_ROOT.
#
# The mix: N requests of exactly 1,024 prompt ids, max_tokens 1, greedy, all queued at once (`serve_load.py --varied N
# --prompt-range 1024:1024 --max-tokens 1 --rate 1000`), served at T tokens a step with T/1024 requests a step and a window
# (--max-num-seqs, --window-requests), --window-steps 1, --admit arrival, no prefix cache. WARM requests are served first,
# outside the capture. Arms (ARM):
#   stock   --no-commit, no protocol: the ceiling
#   noseal  the rows-sha512 Commit, --no-seal: every value is still copied to host and committed, then dropped at the
#           window's close, so none of its blocks can be proved later
#   seal    the Commit with --seal: each closed window sealed to UTIL_ROOT's scratch (deleted after the run)
#   h1      the Commit (sealed iff H1_SEAL=1) and PoUW H1 at T and S 1 with the epoch's in-process coins (no challenger
#           gate: it credits nothing; the gate's own costs are not in this run)
# Env: ARM, T (8192), N (64), WARM (2·T/1024), GPU_HZ (10000), LEASE_MIN (90), EXTRA (more serve-openai arguments).
# Published ($OUT): util.json and util.txt, the profile (prof.nsys-rep), serve.json, summary.json, windows.jsonl's timings
# and sizes (in util.json), the step trace, the loads' timings files (times and token counts), steps.txt. Request
# bodies, outputs, sealed windows (deleted once measured) and the H1 transcripts stay in UTIL_ROOT/scratch/<run id>.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
TREE=${TREE:-$PWD}
ARM=${ARM:?ARM is stock, noseal, seal or h1}; T=${T:-8192}; N=${N:-64}; SLOTS=$(( T / 1024 )); WARM=${WARM:-$(( 2 * SLOTS ))}
ROOT=${UTIL_ROOT:-/workspace/research/vllm-util80}
VENV=${VENV:-/workspace/cp/venv312}; PY=$VENV/bin/python
REPO=verity-int8/Llama-3.2-1B-INT8 REV=15b0976d400d796ee589e7737d03a919526bf800
ROW=${ROW:-$(dirname "$(find "$ROOT" -maxdepth 3 -path '*__b4__i256__*' -name manifest.json | sort | tail -1)")}
HIDDEN_SO=${HIDDEN_SO:-$ROOT/fa2/build/fa3_matReqG/verity_fa3_matReqG.so}
NORM_TAP_SO=${NORM_TAP_SO:-$ROOT/norm_tap/build/verity_norm_tap.so}
NSYS=${NSYS:-/usr/local/cuda-13.0/bin/nsys}
SCR=$ROOT/scratch/${RESEARCH_RUN_ID:-local}; D=$SCR/serve
PORT=${PORT:-18190}
export TREE ARM T N SLOTS WARM ROOT VENV PY REPO REV ROW HIDDEN_SO NORM_TAP_SO NSYS SCR D PORT OUT
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/steps.txt"; }

if [ "${1:-}" != leased ]; then
  case $ARM in stock|noseal|seal|h1) ;; *) say "ARM $ARM: not stock, noseal, seal or h1"; exit 2 ;; esac
  [ $(( T % 1024 )) = 0 ] || { say "T $T: not a multiple of 1,024"; exit 2; }
  [ -f "$ROW/manifest.json" ] || { say "no B4 row Build under $ROOT"; exit 3; }
  if [ "$ARM" != stock ] && { [ ! -f "$HIDDEN_SO" ] || [ ! -f "$NORM_TAP_SO" ]; }; then
    say "no taps: $HIDDEN_SO, $NORM_TAP_SO"; exit 3
  fi
  mkdir -p "$D"
  research keep "$ROOT/scratch" --owner @vllm --expires 3d --priority normal \
    --why "vllm-util80 (VLLM-33): serve dirs of the utilization runs (request bodies, outputs, H1 transcripts)" > /dev/null 2>&1
  say "ARM $ARM T $T N $N WARM $WARM row $(basename "$ROW") tree $(git -C "$TREE" rev-parse --short HEAD 2>/dev/null)"
  GPU_LEASE_WHO=vllm-util80 gpu-lease 1 --wait --max-min "${LEASE_MIN:-90}" -- bash "$TREE/benchmarks/one_stage/served_util.sh" leased
  rc=$?
  say "LEASED PART rc=$rc"
  "$PY" "$TREE/benchmarks/one_stage/served_util.py" --sqlite "$SCR/prof.sqlite" --trace "$D/advice_trace.jsonl" \
    --serve-dir "$D" --timings "$SCR/main.timings.json" --arm "$ARM" --T "$T" --N "$N" --out "$OUT/util.json" > "$OUT/util.txt" 2>&1
  say "ANALYSIS rc=$?"; cat "$OUT/util.txt"
  if [ -d "$D/windows" ]; then
    say "SEALED $(du -sh "$D/windows" | cut -f1), deleted"; rm -rf "$D/windows"
  fi
  exit "$rc"
fi

# under the lease
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-13.0
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src
export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch-extensions TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
export VLLM_BATCH_INVARIANT=1 VERITY_RETAIN=host HIDDEN_GPU_BUILD=$ROOT/cache/jit/hidden_gpu NATIVE_COLLECT_BUILD=$ROOT/cache/jit/nc
nvidia-smi --query-gpu=name,uuid,clocks.max.sm,power.limit,memory.used --format=csv > "$OUT/gpu.txt" 2>&1
args=(--model "$REPO" --revision "$REV" --out "$D" --port "$PORT" --precision verity_int8 --run-id "${RESEARCH_RUN_ID:-local}"
      --required-manifest "$ROW/manifest.json" --admit arrival --window-requests "$SLOTS" --max-num-seqs "$SLOTS"
      --max-num-batched-tokens "$T" --max-model-len 1088 --window-steps 1 --no-prefix-caching)
commit=(--scheme rows-sha512 --gpu-tree --hidden-so "$HIDDEN_SO" --norm-tap-so "$NORM_TAP_SO" --tap-cap-mb "${TAP_CAP_MB:-2048}"
        --window-gb 32 --seal-hold-gb 32 --seal-warm-gb 0 --disk-floor-gb "${DISK_FLOOR_GB:-20}")
case $ARM in
  stock) args+=(--no-commit) ;;
  noseal) args+=("${commit[@]}" --no-seal) ;;
  seal) args+=("${commit[@]}" --seal) ;;
  h1) args+=("${commit[@]}" "$([ "${H1_SEAL:-0}" = 1 ] && echo --seal || echo --no-seal)" --protocols pouw:eps1-sm120-h1fx1) ;;
esac
SESS=util80-${RESEARCH_RUN_ID:-local}
cd "$TREE/integrations/vllm" || exit 9
# shellcheck disable=SC2086
"$NSYS" launch --session-new="$SESS" --trace=cuda,nvtx --cuda-graph-trace=node \
  "$PY" -m verity_vllm.pipeline.cli serve-openai "${args[@]}" ${EXTRA:-} > "$SCR/serve.log" 2>&1 &
srv=$!
trap '"$NSYS" shutdown --session="$SESS" --kill=sigterm >/dev/null 2>&1; wait "$srv"; exit 143' TERM INT
up=0
for _ in $(seq 1 1500); do
  grep -q "SERVE-OPENAI listening" "$SCR/serve.log" && { up=1; break; }
  kill -0 "$srv" 2>/dev/null || break
  sleep 2
done
log() { grep -aE "SERVE-OPENAI|H1|declare|Error|error|refus|Traceback" "$SCR/serve.log" | grep -av prompt | tail -${1:-40} > "$OUT/serve.log.txt"; }
if [ "$up" != 1 ]; then
  log 80; say "SERVE: not listening: $(grep -am1 -E 'UsageError|Error|refus' "$SCR/serve.log" | cut -c1-400)"
  "$NSYS" shutdown --session="$SESS" --kill=sigterm > /dev/null 2>&1; wait "$srv"; exit 10
fi
say "SERVE $(grep -am1 'SERVE-OPENAI listening' "$SCR/serve.log" | cut -c1-200)"
load() {  # $1 requests, $2 seed, $3 name
  "$PY" "$TREE/examples/vllm/serve_load.py" --port "$PORT" --model "$REPO" --varied "$1" --prompt-range 1024:1024 \
    --max-tokens 1 --rate 1000 --seed "$2" --out "$SCR/$3.jsonl" > "$SCR/$3.log" 2>&1
}
load "$WARM" 7 warm; say "WARM $WARM requests (exit $?)"
"$NSYS" start --session="$SESS" -o "$SCR/prof" --force-overwrite=true --sample=none --cpuctxsw=none \
  --gpu-metrics-devices=cuda-visible --gpu-metrics-frequency="${GPU_HZ:-10000}" --gpu-metrics-set=gh100 > "$OUT/nsys_start.txt" 2>&1
say "CAPTURE start (exit $?)"
load "$N" 1 main; lrc=$?
say "MAIN $N requests (exit $lrc): $(tail -1 "$SCR/main.log" | cut -c1-200)"
"$NSYS" stop --session="$SESS" > "$OUT/nsys_stop.txt" 2>&1
say "CAPTURE stop (exit $?)"
"$NSYS" shutdown --session="$SESS" --kill=sigterm > /dev/null 2>&1
wait "$srv"; say "SERVE stopped (exit $?)"
trap - TERM INT
log
"$NSYS" export --type sqlite --force-overwrite=true -o "$SCR/prof.sqlite" "$SCR/prof.nsys-rep" > "$OUT/nsys_export.txt" 2>&1
say "EXPORT (exit $?): $(du -sh "$SCR/prof.nsys-rep" "$SCR/prof.sqlite" 2>/dev/null | tr '\n' ' ')"
cp "$SCR/prof.nsys-rep" "$OUT/"
for f in serve.json summary.json advice_trace.jsonl pauses.jsonl; do cp "$D/$f" "$OUT/" 2>/dev/null; done
cp "$SCR/warm.timings.json" "$SCR/main.timings.json" "$OUT/" 2>/dev/null
[ "$lrc" = 0 ]
