#!/usr/bin/env bash
# What each part of the served step costs, arm by arm on one card: served_verdicts.sh's endpoint, engine and load at the run
# of record's class (r20261106-165116-614b: Llama 3.2 1B's BF16 checkpoint, SLOTS 4 requests of PROMPTS 32:32 tokens each
# running WINDOW_STEPS 4 steps, so 128 prompt and 16 output tokens a load, LOAD=fixed's one full window), served with one more
# part turned on in each arm. Every arm is one `verity-vllm serve-openai` lifetime under the same engine settings and
# environment, so two neighbouring arms differ by one part:
#   stock   --no-commit: no Commit, no protocol (served_stock.sh's arm)
#   commit  the Commit alone: --scheme rows-sha512 --seal with the row's required manifest and both taps (served_verdicts.sh's
#           serve line with no protocol)
#   pearl   the Commit and Pearl+ under it (--protocols pouw:pearl-c-h100-v1-h3 on Pearl-C's sm_90 cubin), each window's
#           salt coin an in-process challenger's (served_verdicts.sh's PEARLPLUS_TIE=1 with PEARLC_CHALLENGER_GATE empty)
#   live    pearl with each window's salt coin a live session's on the auditor host (--challenger-gate GATE --challenger-peer
#           PEER --challenger-lane GATE_LANE): the remote demo's serve. It needs the lane's gate key on this node, this node
#           among the gate's peers, and the lane's Pearl-C period open (served_verdicts.sh's pearl_period), which this
#           script doesn't open; without GATE the preflight refuses the arm. Not run yet on any node from this script.
# Pearl+ refuses --no-commit and a live session is Pearl+'s, so neither is served alone: the Commit's cost is commit less
# stock, Pearl+'s pearl less commit, the live session's live less pearl.
#
# Rounds are interleaved: WARM warm rounds, then REPS rounds, each round one lifetime of every arm in ARMS, the order rotated
# by one a round, under one lease of the card (LEASE_MIN minutes at most; the builds below hold no card). A lifetime takes
# LOADS loads of `examples/vllm/serve_load.py --fixed`, each under a release run of its own; in a Commit arm the next load
# waits for the window's seal. The first load after the endpoint listens is what the run of record's one load was; later
# ones are warm. Once the endpoint stops, `served_ab.py life` reads the serve dir on the node and writes
# $OUT/<round>/<arm>.json (served_overhead.py's aggregate, each load's step times, each window's timed parts, and the
# seconds from the server's start to its endpoint listening: the load is synthetic and its class public), and the serve
# dir is deleted. `served_ab.py table $OUT` then prints the table.
#
# Before the first lease, on CORES alone (the whole script runs under `taskset -c CORES`; the card is CARD): `verity-vllm
# predict row` for the row's Build of record; the two taps (pod_fa3_tap.sh's guarded build at TAP_HDIMS, pod_norm_tap.sh)
# with the venv's CUDA (TAP_CUDA); Pearl-C's sm_90 cubin (pearl_c_sm90.sh, CUBIN_CUDA); the Lean verifier of record the
# in-process challenger reads (`lake build pouw-check-difftest`). Taps, caches, the cubin and the predicted Build are under
# STAGE (default /workspace/compute-accounting-served-ab, under 1 GB, kept between runs so a second run rebuilds nothing:
# a filesystem that maps shared objects, as the node's /dev/shm, mounted noexec, doesn't); the serve dirs, a few GB a
# Commit window, under SERVE_ROOT (default /dev/shm/compute-accounting-served-ab: the node's / is nearly full) until
# their lifetime's record is written. Neither is in the run dir.
# The card's start check is gpu_lease.sh's, taken without --wait: a card whose lock is held, or with another's compute
# process on it, ends the run (exit 75) instead of waiting. The container's CPU throttle count is written at each round's
# start and end (served_ab.txt), since another lane's timed row on this node is void when it moves. With NOT_AFTER (a UTC
# time, `date -d` syntax) no round starts after it, so the run stays inside its booked hours.
#   research run --on vy-spot-glm53 --project verity --source TREE --cwd clone --timeout 240m -- \
#     env CARD=1 CORES=0-11 NOT_AFTER=2026-11-11T07:05Z bash benchmarks/one_stage/served_ab.sh
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
TREE=$PWD
CARD=${CARD:-1} CORES=${CORES:-0-11}
if [ -z "${SERVED_AB_PINNED:-}" ]; then
  exec env SERVED_AB_PINNED=1 OUT="$OUT" taskset -c "$CORES" bash "$TREE/benchmarks/one_stage/served_ab.sh" "$@"
fi
STAGE=${STAGE:-/workspace/compute-accounting-served-ab} SERVE_ROOT=${SERVE_ROOT:-/dev/shm/compute-accounting-served-ab}
ROW_ID=${ROW_ID:-llama32-1b__bf16__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager}
REPO=${REPO:-unsloth/Llama-3.2-1B} REV=${REV:-9535bd9b1d1dea6acafbdc4813b728796aeb28da} PRECISION=${PRECISION:-bfloat16}
VENV=${VENV:-/workspace/venv312}
PORT=${PORT:-18195} SLOTS=${SLOTS:-4} PROMPTS=${PROMPTS:-32:32} WINDOW_STEPS=${WINDOW_STEPS:-4} SEED=${SEED:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-512} T=${T:-512} DISK_FLOOR_GB=${DISK_FLOOR_GB:-100}
ARMS=${ARMS:-stock commit pearl} REPS=${REPS:-5} LOADS=${LOADS:-3} WARM=${WARM:-1}
LEASE_MIN=${LEASE_MIN:-25} LEASE_MEM_GB=${LEASE_MEM_GB:-128} TAP_HDIMS=${TAP_HDIMS:-64}
PEARL_ARM=pouw:pearl-c-h100-v1-h3
first_dir() { local d; for d in "$@"; do [ -x "$d/bin/nvcc" ] && [ -x "$d/bin/cuobjdump" ] && { echo "$d"; return; }; done; }
TAP_CUDA=${TAP_CUDA:-$(first_dir /usr/local/cuda-12.9 /usr/local/cuda)}
CUBIN_CUDA=${CUBIN_CUDA:-$(first_dir /usr/local/cuda-13.0 /usr/local/cuda-13.3 /usr/local/cuda-12.9)}
if [ ! -f "$TAP_CUDA/include/cusparse.h" ]; then  # a toolkit without the math libraries' headers, which torch's ATen/cuda
  # headers include (node A's 12.9): the venv's wheels' copies, for the norm tap's build and the Commit's JIT builds
  for lib in cusparse cusolver cufft curand; do
    for d in "$VENV"/lib/python*/site-packages/nvidia/"$lib"/include; do [ -d "$d" ] && CPATH=${CPATH:+$CPATH:}$d; done
  done
  export CPATH
fi
HIDDEN_SO=${HIDDEN_SO:-$STAGE/fa2/build/fa3_matReqG/verity_fa3_matReqG.so}
NORM_TAP_SO=${NORM_TAP_SO:-$STAGE/norm_tap/build/verity_norm_tap.so}
PRED=${PRED:-$STAGE/predicted/$ROW_ID}
RUNS=$SERVE_ROOT/runs/${RESEARCH_RUN_ID:-local}
export PY=$VENV/bin/python HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 CUDA_HOME=$TAP_CUDA
export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src
export TORCH_EXTENSIONS_DIR=$STAGE/cache/torch-extensions TRITON_CACHE_DIR=$STAGE/cache/triton VLLM_CACHE_ROOT=$STAGE/cache/vllm
export VLLM_BATCH_INVARIANT=1 VERITY_RETAIN=host HIDDEN_GPU_BUILD=$STAGE/cache/jit/hidden_gpu NATIVE_COLLECT_BUILD=$STAGE/cache/jit/nc
export PEARLC_BF16_ROWS=${PEARLC_BF16_ROWS:-0}   # served_verdicts.sh's under PEARLPLUS_TIE: sm_90's BF16-rows path off
export PATH=$VENV/bin:$CUDA_HOME/bin:$HOME/.local/bin:$HOME/.elan/bin:$PATH
PEARLC_KERNEL=$TREE/benchmarks/pouw/pearl_c_sm90 PEARLC_CUBIN=$STAGE/pearlplus-cubin/out/setup/pearl_c_sm90/pearl_c_sm90.cubin
AB=$TREE/benchmarks/one_stage/served_ab.py
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/served_ab.txt"; }
count() { cat "$1" 2>/dev/null | grep -c .; }
throttled() { awk '$1 == "nr_throttled" { print $2 }' /sys/fs/cgroup/cpu.stat 2>/dev/null; }
has() { case " $ARMS " in *" $1 "*) return 0 ;; esac; return 1; }

serve_args() {  # arm $1's serve-openai arguments beyond the engine's, one a line
  local arm=$1
  if [ "$arm" = stock ]; then printf '%s\n' --no-commit --required-manifest "$PRED/manifest.json"; return; fi
  printf '%s\n' --scheme rows-sha512 --seal --window-steps "$WINDOW_STEPS" --required-manifest "$PRED/manifest.json" \
    --hidden-so "$HIDDEN_SO" --norm-tap-so "$NORM_TAP_SO" --tap-cap-mb 2048 --window-gb 32 \
    --disk-floor-gb "$DISK_FLOOR_GB" --seal-hold-gb 32 --seal-warm-gb 0
  [ "$arm" = commit ] && return
  printf '%s\n' --protocols "$PEARL_ARM" --protocols-config \
    "$(python3 -c 'import json, sys; print(json.dumps({"pouw": {"device": "cuda", "kernel": sys.argv[1], "cubin": sys.argv[2]}}))' "$PEARLC_KERNEL" "$PEARLC_CUBIN")"
  [ "$arm" = live ] && printf '%s\n' --challenger-gate "$GATE" --challenger-peer "$PEER" --challenger-lane "$GATE_LANE"
}

life() {  # under the lease: round $1's lifetime of arm $2: the endpoint, its LOADS loads, the endpoint stopped, its record
  local name=$1 arm=$2 D=$RUNS/$1/$2 srv up=0 rc=0 j release args w s t0=$SECONDS started
  mkdir -p "$D" "$OUT/$name"
  mapfile -t args < <(serve_args "$arm")
  cd "$TREE/integrations/vllm" || return 9
  started=$(date +%s.%N)   # the record's listen_s runs from here to the server's own time as it begins to listen
  "$PY" -m verity_vllm.pipeline.cli serve-openai --model "$REPO" --revision "$REV" --out "$D" --port "$PORT" \
    --precision "$PRECISION" --run-id "${RESEARCH_RUN_ID:-local}-$name-$arm" --max-num-batched-tokens "$T" \
    --admit release-order --window-requests "$SLOTS" --max-model-len "$MAX_MODEL_LEN" "${args[@]}" \
    > "$D/serve.log" 2>&1 &
  srv=$!
  trap 'kill -TERM "$srv" 2>/dev/null; wait "$srv"; exit 143' TERM INT
  for _ in $(seq 1 900); do
    grep -q "SERVE-OPENAI listening" "$D/serve.log" && { up=1; break; }
    kill -0 "$srv" 2>/dev/null || break
    sleep 2
  done
  if [ "$up" != 1 ]; then
    tail -60 "$D/serve.log" > "$OUT/$name/$arm.serve.tail"
    say "$name $arm: not listening: $(grep -m1 -E 'UsageError|Error|refus' "$D/serve.log" | cut -c1-400)"
    kill -TERM "$srv" 2>/dev/null; wait "$srv"; trap - TERM INT; return 10
  fi
  local listening=$((SECONDS - t0))
  for j in $(seq 1 "$LOADS"); do
    release=$("$PY" -c 'import uuid; print(uuid.uuid4().hex)')
    "$PY" "$TREE/examples/vllm/serve_load.py" --port "$PORT" --model "$REPO" --fixed "$SLOTS" --window-steps "$WINDOW_STEPS" \
      --prompt-range "$PROMPTS" --release "$release" --seed "$SEED" --out "$D/load-$j.jsonl" > "$D/load-$j.log" 2>&1 || rc=$?
    [ "$arm" = stock ] && continue
    for _ in $(seq 1 600); do   # the load's window closed and sealed, before the next load opens the next one
      w=$(count "$D/windows.jsonl"); s=$(count "$D/seals.jsonl")
      [ "$w" -ge "$j" ] && [ "$s" -ge "$w" ] && break
      kill -0 "$srv" 2>/dev/null || break
      sleep 0.5
    done
  done
  trap - TERM INT
  kill -TERM "$srv" 2>/dev/null; wait "$srv"
  local stopped=$?
  "$PY" "$AB" life "$D" --arm "$arm" --round "$name" --started "$started" > "$OUT/$name/$arm.json" 2> "$D/life.err" \
    || { rm -f "$OUT/$name/$arm.json"; tail -60 "$D/serve.log" > "$OUT/$name/$arm.serve.tail"; rc=12; }
  say "$name $arm: listening after $listening s, loads exit $rc, stopped (exit $stopped), $(count "$D/windows.jsonl") window(s)" \
    "closed, $(count "$D/seals.jsonl") sealed$([ "$rc" = 12 ] && echo "; no record: $(tail -1 "$D/life.err" | cut -c1-400)")"
  [ -n "${KEEP_SERVE:-}" ] || rm -rf "$D"
  return "$rc"
}

round() {  # under the lease: round $1 (warm-<i> or rep-<i>), one lifetime of every arm, the order rotated by $2
  local name=$1 k=$2 arms n i arm bad=0
  read -r -a arms <<< "$ARMS"; n=${#arms[@]}
  say "$name: card ${GPU_LEASE_UUID:-?} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}), cores $(taskset -cp $$ | sed 's/.*: //')," \
    "throttle count $(throttled)"
  for i in $(seq 0 $((n - 1))); do
    arm=${arms[$(((i + k) % n))]}
    life "$name" "$arm" || bad=$?
  done
  say "$name: ended, throttle count $(throttled)"
  return "$bad"
}

prep() {  # on CORES, no card held: the Build of record, the taps, the cubin, the Lean verifier
  local rc jobs
  jobs=$(nproc)
  mkdir -p "$STAGE/cache" "$PRED" "$OUT/prep"
  if [ ! -f "$PRED/predicted.json" ]; then
    (cd "$TREE/integrations/vllm" && "$PY" -m verity_vllm.pipeline.cli predict row --row "$ROW_ID" --out "$PRED") \
      > "$OUT/prep/predict_row.json" 2> "$OUT/prep/predict_row.err"
    rc=$?
    [ "$rc" = 0 ] && [ -f "$PRED/predicted.json" ] \
      || { say "PREP predict row $ROW_ID refused (exit $rc): $(grep -v '^\[predict\]' "$OUT/prep/predict_row.err" | tail -2 | cut -c1-600)"; return 3; }
  fi
  say "PREP predict row: $PRED/manifest.json"
  has stock && [ "$ARMS" = stock ] && return 0
  (cd "$TREE/integrations/vllm" && VENV=$VENV CUDA_HOME=$TAP_CUDA FA2=$STAGE/fa2 FA3_TAP_ROW_GUARD=1 FA3_TAP_HDIMS=$TAP_HDIMS \
    MAX_JOBS=$jobs bash verity_vllm/ops/pod_fa3_tap.sh) > "$OUT/prep/fa3_tap.log" 2>&1
  rc=$?
  say "PREP hidden tap (exit $rc, $jobs jobs, $TAP_CUDA): $(tail -1 "$OUT/prep/fa3_tap.log" | cut -c1-300)"
  [ "$rc" = 0 ] && [ -f "$HIDDEN_SO" ] || { cp "$STAGE/fa2/logs/build_fa3_matReqG.log" "$OUT/prep/" 2>/dev/null; return 4; }
  (cd "$TREE/integrations/vllm" && VENV=$VENV CUDA_HOME=$TAP_CUDA NT=$STAGE/norm_tap bash verity_vllm/ops/pod_norm_tap.sh) \
    > "$OUT/prep/norm_tap.log" 2>&1
  rc=$?
  say "PREP norm tap (exit $rc): $(tail -1 "$OUT/prep/norm_tap.log" | cut -c1-300)"
  [ "$rc" = 0 ] && [ -f "$NORM_TAP_SO" ] || { cp "$STAGE/norm_tap/logs/build.log" "$OUT/prep/norm_tap_build.log" 2>/dev/null; return 5; }
  has pearl || has live || return 0
  ( export RESEARCH_RUN_DIR=$STAGE/pearlplus-cubin VERITY_BENCH_VENV=$VENV CUDA_HOME=$CUBIN_CUDA
    if [ ! -x "$CUBIN_CUDA/bin/nvdisasm" ]; then  # cuobjdump -sass runs nvdisasm, which node A's 12.9 lacks: the venv's Triton's
      for d in "$VENV"/lib/python*/site-packages/triton/backends/nvidia/bin; do [ -x "$d/nvdisasm" ] && PATH=$PATH:$d; done
    fi
    mkdir -p "$RESEARCH_RUN_DIR" && cd "$TREE" || exit 1
    # shellcheck source=/dev/null
    . benchmarks/harness/env.sh && . benchmarks/harness/setup/pearl_c_sm90.sh ) > "$OUT/prep/pearlplus_cubin.log" 2>&1
  rc=$?
  [ "$rc" = 0 ] && [ -s "$PEARLC_CUBIN" ] \
    || { say "PREP Pearl-C cubin: pearl_c_sm90.sh failed (exit $rc, $CUBIN_CUDA): $(tail -1 "$OUT/prep/pearlplus_cubin.log" | cut -c1-400)"; return 6; }
  say "PREP Pearl-C cubin ($CUBIN_CUDA): sha256 $(sha256sum "$PEARLC_CUBIN" | cut -c1-16)"
  has pearl || return 0   # the in-process challenger's entropy; a live session's coins need none
  (cd "$TREE/verity/core/protocols/pouw/lean/check" && lake build pouw-check-difftest) > "$OUT/prep/lean.log" 2>&1
  rc=$?
  say "PREP Lean verifier of record (exit $rc): $(tail -1 "$OUT/prep/lean.log" | cut -c1-300)"
  [ "$rc" = 0 ] || return 7
}

if [ "${1:-}" = round ]; then round "$2" "$3"; exit $?; fi
{ id -un; hostname; git -C "$TREE" rev-parse HEAD 2>/dev/null; echo "cores $CORES ($(nproc)) card $CARD"; nvidia-smi -L
  cat /sys/fs/cgroup/cpu.max 2>/dev/null; echo "stage $STAGE serve dirs $SERVE_ROOT arms $ARMS reps $REPS loads $LOADS warm $WARM"; } > "$OUT/machine.txt"
for f in "$PY" "$TAP_CUDA/bin/nvcc" "$HF_HOME"; do [ -e "$f" ] || { say "PREFLIGHT: no $f"; exit 3; }; done
for a in $ARMS; do
  case $a in stock | commit | pearl) ;;
    live) [ -n "${GATE:-}" ] && [ -n "${PEER:-}" ] && [ -n "${GATE_LANE:-}" ] \
            || { say "PREFLIGHT: the live arm needs GATE (the lane's gate command), PEER (this node's name among the gate's peers) and GATE_LANE, with the lane's Pearl-C period open"; exit 3; } ;;
    *) say "PREFLIGHT: an arm is stock, commit, pearl or live, not $a"; exit 3 ;;
  esac
done
LEASE=$TREE/tools/research/src/research/pods/sh/gpu_lease.sh
command -v gpu-lease > /dev/null && LEASE=$(command -v gpu-lease)
t0=$(throttled)
say "start: arms $ARMS, $REPS rounds after $WARM warm, $LOADS loads a lifetime; throttle count $t0"
prep || { rc=$?; say "STOP at prep (exit $rc); throttle count $(throttled), $t0 at the start"; exit "$rc"; }
say "prep done; throttle count $(throttled), $t0 at the start"
export OUT
k=0 bad=0
for name in $(seq -f 'warm-%g' 1 "$WARM") $(seq -f 'rep-%g' 1 "$REPS"); do
  if [ -n "${NOT_AFTER:-}" ] && [ "$(date -u +%s)" -gt "$(date -u -d "$NOT_AFTER" +%s)" ]; then
    say "$name: not started, past NOT_AFTER $NOT_AFTER"; bad=11; break
  fi
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-compute-accounting-served-ab} bash "$LEASE" 1 --on "$CARD" --max-min "$LEASE_MIN" \
    --mem-gb "$LEASE_MEM_GB" -- bash "$TREE/benchmarks/one_stage/served_ab.sh" round "$name" "$k"
  rc=$?
  [ "$rc" = 0 ] || { bad=$rc; say "$name: exit $rc$([ "$rc" = 75 ] && echo ": card $CARD is held or has another's process on it, so nothing started")"; }
  [ "$rc" = 75 ] || [ "$rc" = 10 ] && break   # a card in use, or an arm that doesn't come up: no later round would differ
  k=$((k + 1))
done
rm -rf "$RUNS"
python3 "$AB" table "$OUT" --out "$OUT" | tee -a "$OUT/served_ab.txt"
say "end (exit $bad); throttle count $(throttled), $t0 at the start"
exit "$bad"
