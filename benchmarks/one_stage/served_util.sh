#!/usr/bin/env bash
# The whole-step tensor utilization of serve-openai on the int8 Llama-3.2-1B prefill mix, one arm a run, measured
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
#           gate: it credits nothing; the gate's own costs are not in this run). H1's verifier of record is the Lean
#           `pouw-check-difftest`, built here with lake (elan under $HOME), so run this arm with `--cwd clone`
#   graphs  stock with the engine's options overridden by UTIL_ENGINE_JSON (default: enforce_eager off, so torch.compile
#           with CUDA graphs captured at every power of two up to T, and async scheduling on), without the bench row. Not
#           our served stack: serve-openai serves eager, and the row's Program declares async scheduling off. Under async
#           scheduling a step's slots stay held until the step before it is processed, so SEQS defaults to 2·T/1024
#   async   stock eager with async scheduling on (UTIL_ENGINE_JSON's default {"async_scheduling": true}), without the bench
#           row, as graphs: not the served stack either (the B4 row's Build declares synchronous scheduling), so util.json's
#           served_stack is false for both
# ASYNC=1 serves a Commit arm (noseal, seal, h1) with --async-scheduling, in a tree whose serve-openai takes it with the
# Commit on (its `serve_row` module). No bench row's Build declares async scheduling yet, so serve-openai's row check (`serve_row.of`) takes
# the row whichever scheduler it declares, as `tests/pipeline/serve_async_commit_gpu.py` does: its records attest the row's
# LAG 0 Programs, so the arm is <arm>-async and not the served stack. SEQS defaults to 2·T/1024, as for async.
# Env: ARM, T (8192), N (64), WARM (2·T/1024), GPU_HZ (10000), LEASE_MIN (90), EXTRA (more serve-openai arguments, split
# on spaces), PROTOCOLS_CONFIG (one --protocols-config argument), PYSAMPLE=1 (Nsight Systems' Python backtrace sampling at
# PYHZ, 1000, for the host gaps; its sqlite export keeps none of the Python frames), PYSPY=1 (py-spy 0.4.2, installed under
# UTIL_ROOT/tools/pyspy, launches the server under the session, since the node's ptrace_scope 1 lets it read only its own
# children, and records it and its subprocesses at PYSPY_HZ, 20, as a Chrome trace in the scratch, which `pyspy_steps.py`
# cuts to the full steps into pyspy_steps.json), SEAL_SHM=1 (the serve dir's windows/ on /dev/shm, so a seal writes to memory
# rather than the node's network disk: what the step is when the disk doesn't bound the seal), SEQS (--max-num-seqs and --window-requests, T/1024, 2·T/1024 for graphs and async; the
# window's request cap also bounds the requests in flight), HEADS (the merged heads, written to heads.txt with the tree's
# commit and parents), RELEASE_AFTER (a Commit arm's values held on the device, --retain device --retain-device-gb
# RETAIN_DEVICE_GB (32) --release-after RELEASE_AFTER), STANDIN_DRAWS (protocols of RELEASE_AFTER no drawer serves yet: the
# harness draws each window for each once its windows.jsonl line is out, reading only the step-input port's ids and
# positions at each of its steps, 16T bytes a step, which the seal's positions check reads and the release keeps only when a
# draw names them; standin_draws.txt counts them).
# A Commit arm at T 8,192 needs VERITY_LEARN_HOST_BUDGET_MB above its step's committed bytes (the default 4,096 MiB fails
# every step closed: "bounded staging: learning step 0 exceeds the transient host budget").
# Published ($OUT): util.json and util.txt, the profile (prof.nsys-rep), serve.json, summary.json, windows.jsonl's timings
# and sizes (in util.json), the step trace, the loads' timings files (times and token counts), steps.txt. Request
# bodies, outputs, sealed windows (deleted once measured) and the H1 transcripts stay in UTIL_ROOT/scratch/<run id>.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
TREE=${TREE:-$PWD}
ARM=${ARM:?ARM is stock, noseal, seal, h1, graphs or async}; T=${T:-8192}; N=${N:-64}; SLOTS=$(( T / 1024 )); WARM=${WARM:-$(( 2 * SLOTS ))}
case $ARM:${ASYNC:-0} in graphs:*|async:*|*:1) SEQS=${SEQS:-$(( 2 * SLOTS ))} ;; *) SEQS=${SEQS:-$SLOTS} ;; esac
LABEL=$ARM; [ "${ASYNC:-0}" = 1 ] && LABEL=$ARM-async
ROOT=${UTIL_ROOT:-/workspace/research/vllm-util80}
VENV=${VENV:-/workspace/cp/venv312}; PY=$VENV/bin/python
REPO=verity-int8/Llama-3.2-1B-INT8 REV=15b0976d400d796ee589e7737d03a919526bf800
ROW=${ROW:-$(dirname "$(find "$ROOT" -maxdepth 3 -path '*__b4__i256__*' -name manifest.json | sort | tail -1)")}
HIDDEN_SO=${HIDDEN_SO:-$ROOT/fa2/build/fa3_matReqG/verity_fa3_matReqG.so}
NORM_TAP_SO=${NORM_TAP_SO:-$ROOT/norm_tap/build/verity_norm_tap.so}
NSYS=${NSYS:-/usr/local/cuda-13.0/bin/nsys}
SCR=$ROOT/scratch/${RESEARCH_RUN_ID:-local}; D=$SCR/serve
PORT=${PORT:-18190}
export TREE ARM LABEL T N SLOTS SEQS WARM ROOT VENV PY REPO REV ROW HIDDEN_SO NORM_TAP_SO NSYS SCR D PORT OUT
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/steps.txt"; }

if [ "${1:-}" != leased ]; then
  case $ARM in stock|noseal|seal|h1|graphs|async) ;; *) say "ARM $ARM: not stock, noseal, seal, h1, graphs or async"; exit 2 ;; esac
  case $ARM:${ASYNC:-0} in *:0|noseal:1|seal:1|h1:1) ;; *) say "ASYNC=1 is for the Commit's arms (noseal, seal, h1): stock's is ARM=async"; exit 2 ;; esac
  { echo "tree $(git -C "$TREE" log -1 --format='%H parents %P' 2>/dev/null)"; for h in ${HEADS:-}; do echo "$h"; done; } > "$OUT/heads.txt"
  [ $(( T % 1024 )) = 0 ] || { say "T $T: not a multiple of 1,024"; exit 2; }
  [ -f "$ROW/manifest.json" ] || { say "no B4 row Build under $ROOT"; exit 3; }
  if [ "$ARM" != stock ] && [ "$ARM" != graphs ] && [ "$ARM" != async ] && { [ ! -f "$HIDDEN_SO" ] || [ ! -f "$NORM_TAP_SO" ]; }; then
    say "no taps: $HIDDEN_SO, $NORM_TAP_SO"; exit 3
  fi
  mkdir -p "$D"
  if [ "$ARM" = h1 ]; then
    export PATH=$HOME/.elan/bin:$PATH
    command -v lake > /dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
      | sh -s -- -y --default-toolchain none > "$OUT/lean.txt" 2>&1
    (cd "$TREE/verity/core/protocols/pouw/lean/check" && lake build pouw-check-difftest) >> "$OUT/lean.txt" 2>&1
    lrc=$?; say "LEAN pouw-check-difftest built (exit $lrc): $(tail -1 "$OUT/lean.txt" | cut -c1-200)"
    [ "$lrc" = 0 ] || exit 4
  fi
  if [ "${PYSPY:-0}" = 1 ] && [ ! -x "$ROOT/tools/pyspy/bin/py-spy" ]; then
    uv pip install -q --target "$ROOT/tools/pyspy" --python "$PY" py-spy==0.4.2 >> "$OUT/pyspy.txt" 2>&1 || { say "PYSPY: install failed"; exit 3; }
  fi
  research keep "$ROOT/scratch" --owner @vllm --expires 3d --priority normal \
    --why "vllm-util80: serve dirs of the utilization runs (request bodies, outputs, H1 transcripts)" > /dev/null 2>&1
  say "ARM $LABEL T $T N $N WARM $WARM SEQS $SEQS row $(basename "$ROW") tree $(git -C "$TREE" rev-parse --short HEAD 2>/dev/null)" \
    "learn budget ${VERITY_LEARN_HOST_BUDGET_MB:-4096} MiB heads ${HEADS:-}"
  GPU_LEASE_WHO=vllm-util80 gpu-lease 1 --wait --max-min "${LEASE_MIN:-90}" -- bash "$TREE/benchmarks/one_stage/served_util.sh" leased
  rc=$?
  say "LEASED PART rc=$rc"
  "$PY" "$TREE/benchmarks/one_stage/served_util.py" --sqlite "$SCR/prof.sqlite" --trace "$D/advice_trace.jsonl" \
    --serve-dir "$D" --timings "$SCR/main.timings.json" --arm "$LABEL" --T "$T" --N "$N" --out "$OUT/util.json" > "$OUT/util.txt" 2>&1
  say "ANALYSIS rc=$?"; cat "$OUT/util.txt"
  if [ "${PYSPY:-0}" = 1 ]; then
    "$PY" "$TREE/benchmarks/one_stage/pyspy_steps.py" --trace-json "$SCR/pyspy.json" --t0 "$SCR/pyspy.t0" \
      --steps "$D/advice_trace.jsonl" --timings "$SCR/main.timings.json" --T "$T" --out "$OUT/pyspy_steps.json" > "$OUT/pyspy_steps.txt" 2>&1
    say "PYSPY ANALYSIS rc=$?"; cat "$OUT/pyspy_steps.txt"
  fi
  if [ -d "$D/windows" ]; then
    say "SEALED $(du -sh "$D/windows/" | cut -f1)$([ -L "$D/windows" ] && echo ' on /dev/shm'), deleted"
    [ -L "$D/windows" ] && rm -rf "$(readlink -f "$D/windows")"
    rm -rf "$D/windows"
  fi
  exit "$rc"
fi

# under the lease; the server holds one socket a request in flight, so N past the default 1,024 files needs the hard limit
ulimit -n "$(ulimit -Hn)"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-13.0
export PATH=$VENV/bin:$CUDA_HOME/bin:$HOME/.elan/bin:$PATH
export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src
export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch-extensions TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
export VLLM_BATCH_INVARIANT=1 VERITY_RETAIN=${VERITY_RETAIN:-host} HIDDEN_GPU_BUILD=$ROOT/cache/jit/hidden_gpu NATIVE_COLLECT_BUILD=$ROOT/cache/jit/nc
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit,memory.used --format=csv > "$OUT/gpu.txt" 2>&1
args=(--model "$REPO" --revision "$REV" --out "$D" --port "$PORT" --precision verity_int8 --run-id "${RESEARCH_RUN_ID:-local}"
      --admit arrival --window-requests "$SEQS" --max-num-seqs "$SEQS"
      --max-num-batched-tokens "$T" --max-model-len 1088 --window-steps 1 --no-prefix-caching)
[ "$ARM" = graphs ] || [ "$ARM" = async ] || args+=(--required-manifest "$ROW/manifest.json")
[ -n "${PROTOCOLS_CONFIG:-}" ] && args+=(--protocols-config "$PROTOCOLS_CONFIG")
[ -n "${RELEASE_AFTER:-}" ] && args+=(--retain device --retain-device-gb "${RETAIN_DEVICE_GB:-32}" --release-after "$RELEASE_AFTER")
commit=(--scheme rows-sha512 --gpu-tree --hidden-so "$HIDDEN_SO" --norm-tap-so "$NORM_TAP_SO" --tap-cap-mb "${TAP_CAP_MB:-2048}"
        --window-gb 32 --seal-hold-gb 32 --seal-warm-gb 0 --disk-floor-gb "${DISK_FLOOR_GB:-20}")
serve=("$PY" -m verity_vllm.pipeline.cli serve-openai)
case $ARM in
  stock) args+=(--no-commit) ;;
  noseal) args+=("${commit[@]}" --no-seal) ;;
  seal) args+=("${commit[@]}" --seal) ;;
  h1) args+=("${commit[@]}" "$([ "${H1_SEAL:-0}" = 1 ] && echo --seal || echo --no-seal)" --protocols pouw:eps1-sm120-h1fx1) ;;
  graphs|async)
    args+=(--no-commit)
    sizes=1; s=2; while [ "$s" -le "$T" ]; do sizes+=",$s"; s=$(( 2 * s )); done
    [ "$ARM" = async ] && export UTIL_ENGINE_JSON=${UTIL_ENGINE_JSON:-'{"async_scheduling": true}'}
    export UTIL_ENGINE_JSON=${UTIL_ENGINE_JSON:-"{\"enforce_eager\": false, \"async_scheduling\": true, \"compilation_config\": {\"cudagraph_capture_sizes\": [$sizes]}}"}
    serve=("$PY" -c 'import json, os, sys
from verity_vllm.pipeline import cli, serve_openai as so
over, base = json.loads(os.environ["UTIL_ENGINE_JSON"]), so.engine_options
so.engine_options = lambda opts: base(opts) | over
print("UTIL engine options overridden (not the served stack):", json.dumps(over), flush=True)
sys.exit(cli.main())' serve-openai)
    say "ENGINE $UTIL_ENGINE_JSON" ;;
esac
if [ "${ASYNC:-0}" = 1 ]; then
  args+=(--async-scheduling)
  serve=("$PY" -c 'import json, sys
from pathlib import Path
from verity_vllm.pipeline import cli, serve_row as RW, serve_seal as SS
def of(opts):
    if not opts.required_manifest:
        return None
    path = Path(SS.programs_root_of(opts)) / RW.WORKLOAD_PROGRAM
    got = json.loads(path.read_text())["applicability"]["engine"]["async_scheduling"]
    print(f"UTIL row scheduler check bypassed (not the served stack): the row declares async_scheduling {got}", flush=True)
    return {"async_scheduling": got, "source": str(path)}
RW.of = of
sys.exit(cli.main())' serve-openai)
fi
if [ "${SEAL_SHM:-0}" = 1 ]; then
  mkdir -p "/dev/shm/vllm-util80/${RESEARCH_RUN_ID:-local}" && rm -rf "$D/windows" && ln -sfn "/dev/shm/vllm-util80/${RESEARCH_RUN_ID:-local}" "$D/windows"
fi
if [ "${PYSPY:-0}" = 1 ]; then
  serve=("$ROOT/tools/pyspy/bin/py-spy" record --rate "${PYSPY_HZ:-20}" --idle --threads --subprocesses --nolineno
         --format chrometrace -o "$SCR/pyspy.json" -- "${serve[@]}")
  date -u +%s.%N > "$SCR/pyspy.t0"
fi
SESS=util80-${RESEARCH_RUN_ID:-local}
pysample=(); sample=none
[ "${PYSAMPLE:-0}" = 1 ] && { pysample=(--python-sampling=true --python-sampling-frequency="${PYHZ:-1000}"); sample=process-tree; }
cd "$TREE/integrations/vllm" || exit 9
# shellcheck disable=SC2086
"$NSYS" launch --session-new="$SESS" --trace=cuda,nvtx --cuda-graph-trace=node "${pysample[@]}" \
  "${serve[@]}" "${args[@]}" ${EXTRA:-} > "$SCR/serve.log" 2>&1 &
srv=$!
drawer=
if [ -n "${STANDIN_DRAWS:-}" ]; then
  "$PY" -c 'import json, os, sys, time
d, protos, done = sys.argv[1], [p for p in sys.argv[2].split(",") if p], set()
while True:
    try:
        lines = open(f"{d}/windows.jsonl").read().split("\n")[:-1]
    except FileNotFoundError:
        lines = []
    for x in lines:
        row = json.loads(x)
        k = row.get("window")
        port = [[s, n, 0, 1 << 40] for s in range(int(row.get("steps") or 0)) for n in ("step.input/ids", "step.input/positions")]
        for p in protos if k is not None else ():
            if (k, p) not in done:
                os.makedirs(f"{d}/draws", exist_ok=True)
                tmp = f"{d}/draws/.w{k}.{p}.tmp"
                with open(tmp, "w") as f:
                    json.dump({"window": k, "protocol": p, "reads": port}, f)
                os.rename(tmp, f"{d}/draws/w{k}.{p}.json")
                done.add((k, p))
                print(f"w{k} {p}", flush=True)
    time.sleep(0.2)' "$D" "$STANDIN_DRAWS" > "$OUT/standin_draws.txt" 2>&1 &
  drawer=$!
fi
spid=
alive() { case $(ps -o stat= -p "$1" 2>/dev/null) in ''|Z*) return 1 ;; esac; }
# `wait "$srv"` waits on the `nsys launch` client, which returns at the session's shutdown even when the server itself
# hangs in its stop (a sealer that died leaves it waiting on seals), and a server left up keeps the GPU lease's lock
reap() {
  [ -n "$spid" ] || return 0
  for _ in $(seq 1 $(( ${SERVE_STOP_S:-600} / 2 ))); do alive "$spid" || return 0; sleep 2; done
  local kids; kids=$(ps -o pid= --ppid "$spid" 2>/dev/null | tr -s ' \n' ' ')
  say "SERVE: pid $spid still up ${SERVE_STOP_S:-600} s after the shutdown ($(grep -a 'SERVE-OPENAI stopping' "$SCR/serve.log" | tail -1 | cut -c1-120)); killed with its children$kids"
  # shellcheck disable=SC2086
  kill -KILL "$spid" $kids 2>/dev/null
}
trap '"$NSYS" shutdown --session="$SESS" --kill=sigterm >/dev/null 2>&1; wait "$srv"; reap; [ -n "$drawer" ] && kill "$drawer"; exit 143' TERM INT
up=0
for _ in $(seq 1 1500); do
  grep -q "SERVE-OPENAI listening" "$SCR/serve.log" && { up=1; break; }
  kill -0 "$srv" 2>/dev/null || break
  sleep 2
done
log() { grep -aE "SERVE-OPENAI|UTIL (engine|row)|\[engine\]|H1|declare|Error|error|refus|Traceback" "$SCR/serve.log" | grep -av prompt | tail -${1:-40} > "$OUT/serve.log.txt"; }
if [ "$up" != 1 ]; then
  log 80; say "SERVE: not listening: $(grep -am1 -E 'UsageError|Error|refus' "$SCR/serve.log" | cut -c1-400)"
  "$NSYS" shutdown --session="$SESS" --kill=sigterm > /dev/null 2>&1; wait "$srv"; [ -n "$drawer" ] && kill "$drawer"; exit 10
fi
spid=$(ss -Hltnp "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
say "SERVE $(grep -am1 'SERVE-OPENAI listening' "$SCR/serve.log" | cut -c1-200) (pid ${spid:-unknown})"
load() {  # $1 requests, $2 seed, $3 name
  "$PY" "$TREE/examples/vllm/serve_load.py" --port "$PORT" --model "$REPO" --varied "$1" --prompt-range 1024:1024 \
    --max-tokens 1 --rate 1000 --seed "$2" --out "$SCR/$3.jsonl" > "$SCR/$3.log" 2>&1
}
load "$WARM" 7 warm; say "WARM $WARM requests (exit $?)"
"$NSYS" start --session="$SESS" -o "$SCR/prof" --force-overwrite=true --sample="$sample" --cpuctxsw=none \
  --gpu-metrics-devices=cuda-visible --gpu-metrics-frequency="${GPU_HZ:-10000}" --gpu-metrics-set=gh100 > "$OUT/nsys_start.txt" 2>&1
say "CAPTURE start (exit $?)"
load "$N" 1 main; lrc=$?
say "MAIN $N requests (exit $lrc): $(tail -1 "$SCR/main.log" | cut -c1-200)"
"$NSYS" stop --session="$SESS" > "$OUT/nsys_stop.txt" 2>&1
say "CAPTURE stop (exit $?)"
# the server's own stop drains its seals; nsys's shutdown signals the whole tree, the spawned sealer with it
pspid=; [ "${PYSPY:-0}" = 1 ] && [ -n "$spid" ] && pspid=$(ps -o ppid= -p "$spid" | tr -d ' ')
[ -n "$spid" ] && { kill -TERM "$spid"; reap; say "SERVE: its own stop $(grep -a 'SERVE-OPENAI stopping' "$SCR/serve.log" | tail -1 | cut -c1-120)"; }
# py-spy writes its trace once its children are gone, or at a SIGINT; nsys's shutdown would SIGTERM it unwritten
if [ -n "$pspid" ]; then
  for i in $(seq 1 300); do alive "$pspid" || break; [ "$i" = 60 ] && kill -INT "$pspid"; sleep 1; done
  say "PYSPY $(du -sh "$SCR/pyspy.json" 2>/dev/null | cut -f1) ($(alive "$pspid" && echo 'still up' || echo done))"
fi
"$NSYS" shutdown --session="$SESS" --kill=sigterm > /dev/null 2>&1
wait "$srv"; say "SERVE stopped (exit $?)"
reap
[ -n "$drawer" ] && { kill "$drawer"; say "STANDIN draws $(wc -l < "$OUT/standin_draws.txt") ($STANDIN_DRAWS, the step-input port only)"; }
trap - TERM INT
log
"$NSYS" export --type sqlite --force-overwrite=true -o "$SCR/prof.sqlite" "$SCR/prof.nsys-rep" > "$OUT/nsys_export.txt" 2>&1
say "EXPORT (exit $?): $(du -sh "$SCR/prof.nsys-rep" "$SCR/prof.sqlite" 2>/dev/null | tr '\n' ' ')"
cp "$SCR/prof.nsys-rep" "$OUT/"
for f in serve.json summary.json advice_trace.jsonl pauses.jsonl; do cp "$D/$f" "$OUT/" 2>/dev/null; done
cp "$SCR/warm.timings.json" "$SCR/main.timings.json" "$OUT/" 2>/dev/null
[ "$lrc" = 0 ]
