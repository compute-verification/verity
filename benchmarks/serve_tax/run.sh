#!/usr/bin/env bash
# The serve tax on one node (README.md), from the shipped tree's root under `research run`, the C4 shard sent with
# --send (or C4=PATH) unless a ready plan is:
#   ARMS="compiled eager ..." RATES=light=R,medium=R,saturated=R [MODEL_ID=moonlight] [REPS=3] [SPAN_S=60]
#     [REQUESTS=N] [WINDOW_S=600] [KV_WINDOW_S=WINDOW_S] [TICK_MS=SAT,FREE | TICK_FROM=RECORD_ARM_DIR]
#     [REPLAY=LOAD [REPLAY_FROM=RECORD_ARM_DIR] [REPLAY_MODES="compiled eager"] [REPLAY_LOGPROBS=1]] [MAX_MIN=240]
#     bash benchmarks/serve_tax/run.sh
#   CAPACITY=1 [REQUESTS=800] bash benchmarks/serve_tax/run.sh   # stock compiled under one burst: its capacity
#   REPLAY=LOAD REPLAY_FROM=RECORD_ARM_DIR [REPLAY_MODES=...] [REPLAY_CAPTURE=ID,...] bash benchmarks/serve_tax/run.sh
#     # only replay an earlier window, at the model's GPUs as tensor-parallel width, taking REPLAY_CAPTURE's requests'
#     # logits rows and ids at the sampler into OUT/replay-<mode>/capture (replay.py --capture)
#   ARMS="restore-base restore-pause restore" RATES=medium=R REPS=1 SPAN_S=720 EDGES_S=180,360,540 [SAVE_KV=1]
#     [RESTORE=1:2 [RESTORE_ARM="restore ..."] [RESTORE_WARM=N]] [NUM_GPU_BLOCKS=N] [KV_DIR=DIR]
#     bash benchmarks/serve_tax/run.sh   # boundary states at the edges, one restored
#   ARMS="compiled record tokens-only boundary" RATES=... EDGES_EVERY_S=30 [KV_COPY=all|first|none] [PLAN_FILE=PLAN.jsonl]
#     [MAX_MODEL_LEN=N] [FLASHINFER=1] [LDCONFIG=1] [BUDGET_MIN=M] bash benchmarks/serve_tax/run.sh   # the production tax
#   ARMS=two-stage RATES=... [PLAN_FILE=PLAN.jsonl] PRIVATE=DIR [BOUNDARY_ROWS='{"check":T,...}']
#     bash benchmarks/serve_tax/run.sh   # the two-stage serve: its launch line is the arm (serve_tax.ARMS), its boundary
#                                        # committed under PRIVATE/two-stage/boundary
#   bash benchmarks/serve_tax/run.sh --model glm53 [--arms "ARM ..."|all] [--loads NAME[=RUNG+RUNG...],...]
#     [--dataset NAME=FILE] [--reps 6] [--target PERCENT] [--json]   # the serving cost, by vLLM's own client
# The last form is the serving cost (README.md): each flag sets the variable of its name (MODEL_ID, ARMS, LOADS, DATASETS,
# REPS, TARGET, JSON) and any of them makes vLLM's own client the default (CLIENT=vllm-bench: `vllm bench serve`, no C4
# shard and no plan). Its arms are by default stock, stock again and two-stage, with --arms all one arm for each change
# in between and the canonical top-k's own; its loads chat and long. Chat's dataset file is the node's copy of the pinned
# one, downloaded once into HF_HOME before the GPUs are held and checked by its SHA-512 (serve_tax.py passes --fetch);
# --dataset chat=FILE names another copy of it, and no run sends or keeps the file. --dataset prefix=FILE serves a table
# of planned requests (arrival times and lengths; prefix_plan.jsonl is the full-size one) in place of load prefix's
# four, each sent as it arrives: the row then says whether stock kept up, and which of its figures is the cost.
# PASS_PROMPTS=NAME=N,... makes a pass of load NAME send N prompts at every rung in place of the load's own, and
# NAME=Kx K prompts for each request its rung holds in flight (serve_tax.parse_pass_prompts): for a ladder past the
# rungs the load's own prompts can hold, and the same for every arm of a cost. After the table it prints the cost of
# ARMS's last arm over compiled (serve_tax.py cost) and writes OUT/serving_cost.{json,txt} and the run's result.json (each
# load's cost a measurement, and the number one under the model's metric id once every load has run); --json prints the
# JSON object last. In this form REPS is 6 and WINDOW_S 30 by default, and OUT/gpus.csv names the cards.
# WITH="OUT_DIR ..." adds earlier jobs' arms to the cost: a load they served and this job didn't joins the number.
# The number is not given without the row on a repeated prefix under it, which is load prefix's, served in a job of its
# own (--loads prefix --arms "compiled no-prefix-cache two-stage": its cost is the row alone) and joined with WITH.
# An arm behind the network certifier (tick-certifier) is served by two commands, and this script is the node's:
#   ARMS=tick-certifier SIDE=server LOADS=chat=RUNG TICK_MS=SAT,FREE [FRONT=PORT] [CELLS=NAME=N,...] [STOP_AT=TIME]
#     bash benchmarks/serve_tax/run.sh --model glm53
# starts the arm's engine and the id front before it on loopback port FRONT and holds them until the certifier's run
# has ended or STOP_AT (serve_tax.certified_server); OUT/tick-certifier/server.json is its record, and it has no table
# and no cost. The other command runs where the certifier does, a host that reaches FRONT and is never this node:
# serve_tax.py run --arm tick-certifier --side client, with the same LOADS and CELLS. Its directory is the arm's.
# PRIVATE is where the arms keep what never leaves the node, each in PRIVATE/<arm>: a boundary commitment's classes, whose
# files hold the requests' tokens and the salts (OUT keeps each class's public commit.json, a line of
# <arm>/server/boundary.PID.jsonl), the history tables and the top-k's counters. It is a directory outside every run's.
# A serve's later stages read it, so without it such an arm is refused; in the serving cost's form alone, where nothing
# reads it afterwards, each such arm gets a fresh one, removed when it stops. BOUNDARY_ROWS replaces the object of an
# arm's boundary commitment (a model with no check template of its own needs it).
# EDGES_EVERY_S puts the boundary arm's edge each that many seconds of a load (its replay blocks' length); at TP > 1 each
# rank keeps its KV copy and rows in host memory, and KV_COPY=first copies rank 0's pool alone (for ranks that hold the
# same KV bytes, as GLM-5.3's replicated MLA cache does). PLAN_FILE (else a sent inputs/plan.jsonl) is a ready plan, served
# as it is (the replay ladder's production plan: each request's own sampling, stopping at its end of text), copied to
# OUT/plan.jsonl; MAX_MODEL_LEN the engines' context limit, a number of tokens or auto (vLLM's word for the longest the
# cards' KV cache holds); PUBLISHED_FRONTEND=rust starts the published arm behind vLLM's Rust front end, on a build
# that has it (python, the default, is the front end of every other arm), and PUBLISHED_ARGS=--FLAG=VALUE,... adds flags
# of vLLM's own to that arm's line alone. FLASHINFER=1 puts CUDA_HOME's bin first on PATH (CUDA_HOME, else
# the newest toolkit under /usr/local), so vLLM runs FlashInfer's sampler, autotune and fused all-reduce, and runs a sent
# inputs/fi_smoke.py before the first engine so that its JIT lands outside the served traffic; OUT/engine_env.txt records
# the toolkit, has_flashinfer and FlashInfer's caches' digests (fi_cache) before the arms and after each. LDCONFIG=1 refreshes the
# linker cache first. BUDGET_MIN bounds the arms' minutes: each arm after the first serves the reps of REPS that fit its
# share of the minutes left, equal among the arms still to serve (serve_tax.py fit, from the arm before it), and one if
# none does, since reps are cut and never an arm (OUT/arms.txt says what was cut). STOP_AT=TIME (with its zone, as
# 2030-01-01T00:00:00Z) is the time the job's GPUs are another's: an arm starts no pass it can't expect to finish and stop
# its server by then (serve_tax.stop_before), keeps the passes it served, and its meta.json says where it stopped
# (stopped); an arm whose turn comes after TIME is not served. The job ends between two passes and exits as if whole.
# Each restore arm keeps the KV bytes and states of SAVE_KV's edges in KV_DIR/<arm> (KV_DIR by default a fresh directory
# under the node's scratch, removed at the end); RESTORE=FROM:TO then restores each RESTORE_ARM's edge FROM into a fresh
# engine (warmed first on RESTORE_WARM of the replay block's prompts) and replays to edge TO (restore.py) into
# OUT/<arm>-replay. RESTORE_FROM=OUT_DIR with KV_DIR and no ARMS restores an earlier job's arms alone.
# One plan (OUT/plan.jsonl, from the shard's first DOCS documents) is made before the GPUs are held; then each arm is a
# server lifetime of its own (serve_tax.py run) into OUT/<arm>, in ARMS's order, all under one gpu-lease; the tick arms
# take τ from TICK_MS, else from TICK_FROM, else from this job's own record arm, calibrated on its saturated load.
# REPLAY then replays LOAD's first rep of the record arm (by default this job's) in each of REPLAY_MODES (replay.py)
# into OUT/replay-<mode>; a mode listed again replays into OUT/replay-<mode>-<k>, its tokens compared with the first's too
# (REPLAY_MODES="compiled compiled": does the replay agree with itself?). Then OUT/table.{json,md}. MODEL_ID moonlight serves the replay-determinism benchmark's kept
# checkpoint on one GPU; glm53 the GLM-5.3 TP drivers' (benchmarks/served_commit/tp_decode.sh) on eight; llama8b and
# qwen3-30b their pinned snapshots under HF_HOME on one. VENV's vLLM must be SERVE_VLLM: d9105ea80, the integration's
# pin, unless set.
set -uo pipefail
while [ $# -gt 0 ] && [ "$1" != leased ]; do
  flag=$1; shift
  case $flag in
    --json) JSON=1 ;;
    --model|--arms|--loads|--dataset|--reps|--target)
      [ $# -gt 0 ] || { echo "serve-tax: $flag takes a value"; exit 2; }
      case $flag in
        --model) MODEL_ID=$1 ;; --arms) ARMS=$1 ;; --loads) LOADS=$1 ;; --reps) REPS=$1 ;; --target) TARGET=$1 ;;
        --dataset) DATASETS="${DATASETS:+$DATASETS }$1" ;;
      esac
      shift ;;
    *) echo "serve-tax: unknown flag $flag (--model, --arms, --loads, --dataset NAME=FILE, --reps, --target, --json)"; exit 2 ;;
  esac
  CLIENT=${CLIENT:-vllm-bench}
done
CLIENT=${CLIENT:-corpus}
if [ "$CLIENT" = vllm-bench ]; then
  ARMS=${ARMS:-compiled compiled-again two-stage}; LOADS=${LOADS:-chat,long}
  [ "$ARMS" = all ] && ARMS="compiled compiled-again record no-prefix-cache boundary-rows canonical-topk two-stage"
  # six reps: the fewest whose ratios bound their median at 95% (the cost's interval). A Commit period shorter than a
  # pass: a pass then holds the commits of the requests it finishes. At a period longer than a pass they would fall
  # between passes, where no request waits on them, and the cost would leave the commitment's work out; a short period
  # counts a period's fixed work more often than a long one, so the cost errs high.
  REPS=${REPS:-6}; WINDOW_S=${WINDOW_S:-30}
  export ARMS LOADS
fi
export CLIENT ${MODEL_ID:+MODEL_ID} ${REPS:+REPS} ${DATASETS:+DATASETS}
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# SERVE_VLLM: the vLLM build to serve on, one of the adapter's engine.stock.BUILDS. The pin's environment is the node's
# own; another build's is ops/vllm_env.sh's, which installs it beside the pin's the first time, before any GPU is held
SERVE_VLLM=${SERVE_VLLM:-d9105ea80}
if [ "$SERVE_VLLM" = d9105ea80 ]; then
  [ -d /workspace/cp/venv312 ] && venv=/workspace/cp/venv312 || venv=/workspace/venv312   # a dc node's, else a pod's
else
  venv=$(SERVE_VLLM=$SERVE_VLLM OUT= bash "$here/../../adapters/vllm/verity_vllm/ops/vllm_env.sh" | tail -n 1)
  [ -n "$venv" ] || { echo "serve-tax: no environment for SERVE_VLLM=$SERVE_VLLM"; exit 2; }
  # what this build compiles (vLLM's graphs, DeepGEMM and autotune files, FlashInfer's JIT, Triton) goes to caches of its
  # own under its environment: the node's, which the pin's engines read, get no file from it
  export VLLM_CACHE_ROOT=$venv/cache/vllm FLASHINFER_WORKSPACE_BASE=$venv/cache TRTLLM_DG_CACHE_DIR=$venv/cache/tensorrt_llm \
    TRITON_CACHE_DIR=$venv/cache/triton
fi
VENV=${VENV:-$venv}; PY=$VENV/bin/python
export PATH="$VENV/bin:$PATH"   # the venv's ninja, which FlashInfer's JIT (vLLM's default sampler) runs
[ -d /workspace/cp/hf ] && hf=/workspace/cp/hf || hf=/workspace/hf
export HF_HOME=${HF_HOME:-$hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
MODEL_ID=${MODEL_ID:-moonlight}
case $MODEL_ID in
  moonlight) CK=${CHECKPOINT:-/workspace/research/scratch/replay-determinism/moonlight}; GPUS=1 ;;
  glm53)
    REV=aca966e4e02791568aa6a4ced368624b3d897f42
    if [ -e "/workspace/cp/glm53-ckpt/$REV/.verified" ]; then CK=/workspace/cp/glm53-ckpt/$REV; else CK=/workspace/fs/glm53/$REV; fi
    CK=${CHECKPOINT:-$CK}; GPUS=8 ;;
  llama8b|qwen3-30b) CK=${CHECKPOINT:-$("$PY" "$here/serve_tax.py" checkpoint --model-id "$MODEL_ID")}; GPUS=1 ;;
  *) echo "serve-tax: MODEL_ID is moonlight, glm53, llama8b or qwen3-30b, not $MODEL_ID"; exit 2 ;;
esac
# nvcc, which the kv and boundary arms build the row-table kernels with: CUDA_HOME's, else the newest toolkit under
# /usr/local; with FLASHINFER=1 first on PATH, for FlashInfer's JIT too, which finds CUDA_HOME's headers and libraries
# itself (an inherited CPATH or LIBRARY_PATH could mix toolkits)
if [ "${FLASHINFER:-0}" = 1 ]; then
  CUDA_HOME=${CUDA_HOME:-$(ls -d /usr/local/cuda-[0-9]*.[0-9]* 2>/dev/null | sort -V | tail -1)}
  export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
  unset CPATH LIBRARY_PATH
else
  command -v nvcc >/dev/null || for d in ${CUDA_HOME:+"$CUDA_HOME/bin"} $(ls -d /usr/local/cuda*/bin 2>/dev/null | sort -rV); do
    [ -x "$d/nvcc" ] && { export PATH="$PATH:$d"; break; }
  done
fi
C4=${C4:-$(ls "$D"/inputs/*.json.gz 2>/dev/null | head -1)}
REPS=${REPS:-3}; SPAN_S=${SPAN_S:-60}; MAX_MIN=${MAX_MIN:-240}; DOCS=${DOCS:-45000}
WINDOW_S=${WINDOW_S:-600}; KV_WINDOW_S=${KV_WINDOW_S:-$WINDOW_S}; PLAN=${PLAN:-$OUT/plan.jsonl}
PLAN_FILE=${PLAN_FILE:-$(ls "$D"/inputs/plan.jsonl 2>/dev/null)}
[ -z "$PLAN_FILE" ] || [ -f "$PLAN_FILE" ] || { echo "serve-tax: no plan at PLAN_FILE=$PLAN_FILE"; exit 2; }
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM
# Inductor's runtime R0_BLOCK choice (each process times a halved candidate) changes the reduction order: pin it
export TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0
replay_only=0; [ -z "${ARMS:-}" ] && [ "${CAPACITY:-0}" != 1 ] && [ -n "${REPLAY_FROM:-}${RESTORE_FROM:-}" ] && replay_only=1
[ -f "$C4" ] || [ $replay_only = 1 ] || [ -n "$PLAN_FILE" ] || [ "$CLIENT" = vllm-bench ] || { echo "serve-tax: no C4 shard (--send it, or C4=PATH)"; exit 2; }
[ -d "$CK" ] || { echo "serve-tax: no checkpoint at $CK"; exit 2; }
v=$("$PY" -c 'from vllm._version import __commit_id__ as c; print(c.removeprefix("g"))') || exit 2
[ "$v" = "$SERVE_VLLM" ] || { echo "serve-tax: $VENV holds vLLM $v, not SERVE_VLLM=$SERVE_VLLM"; exit 2; }
mkdir -p "$OUT"
[ ! -f "$VENV/freeze.txt" ] || cp "$VENV/freeze.txt" "$OUT/vllm_env.freeze.txt"   # ops/vllm_env.sh's: what it installed

# FlashInfer's caches every engine reads, by their files' digests: its JIT GEMMs (TRTLLM_DG_CACHE_DIR, else
# ~/.tensorrt_llm/cache) and vLLM's autotune file per config hash (VLLM_CACHE_ROOT's flashinfer_autotune_cache), which the
# arms share, as they share the hash
fi_cache() {
  local kv c
  for kv in "tensorrt_llm_cache:${TRTLLM_DG_CACHE_DIR:-$HOME/.tensorrt_llm/cache}" \
            "autotune_cache:${VLLM_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/vllm}/flashinfer_autotune_cache"; do
    c=${kv#*:}
    if [ -d "$c" ]; then
      printf '%s=%s files=%s ' "${kv%%:*}" "$(cd "$c" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | cut -c1-64)" \
        "$(find "$c" -type f | wc -l)"
    else
      printf '%s=none ' "${kv%%:*}"
    fi
  done
  echo
}

if [ "${1:-}" = leased ]; then
  t_leased=$(date +%s)
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,clocks.max.sm --format=csv
  # the cards the job's figures were measured on, kept with them
  nvidia-smi --query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem --format=csv > "$OUT/gpus.csv" 2>&1 || true
  smoke=$(ls "$D"/inputs/fi_smoke.py 2>/dev/null)
  if [ -n "$smoke" ] && [ "${FLASHINFER:-0}" = 1 ]; then
    echo "== fi_smoke $(date -u +%FT%TZ)"
    GPUS=$GPUS "$PY" "$smoke" "$OUT/fi_smoke.json" > "$OUT/fi_smoke.log" 2>&1 || { tail -n 20 "$OUT/fi_smoke.log"; exit 1; }
  fi
  echo "before the arms: $(fi_cache)" >> "$OUT/engine_env.txt"
  common=(--model-id "$MODEL_ID" --checkpoint "$CK" ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"}
    ${PUBLISHED_FRONTEND:+--published-frontend "$PUBLISHED_FRONTEND"}
    ${PUBLISHED_ARGS:+"--published-args=$PUBLISHED_ARGS"})
  if [ "$CLIENT" = vllm-bench ]; then
    load=(--client vllm-bench --loads "$LOADS" ${PASS_PROMPTS:+--pass-prompts "$PASS_PROMPTS"})
    for f in ${DATASETS:-}; do load+=(--dataset "$f"); done
  else
    common+=(--c4 "$C4")
    load=(--rates "${RATES:-}" --span-s "$SPAN_S" --plan "$PLAN" --docs "$DOCS")
  fi
  if [ "${CAPACITY:-0}" = 1 ]; then
    "$PY" "$here/serve_tax.py" run --arm compiled "${common[@]}" --rates burst=1000 --requests "${REQUESTS:-800}" --reps 1 \
      --warm-s 10 --out "$OUT/capacity"
    exit $?
  fi
  rc=0
  if [ -n "${TICK_MS:-}" ]; then tick=(--tick-ms "$TICK_MS"); else tick=(--tick-from "${TICK_FROM:-$OUT/record}"); fi
  count=(); [ -n "${REQUESTS:-}" ] && count=(--requests "$REQUESTS")
  [ -n "${NUM_GPU_BLOCKS:-}" ] && count+=(--num-gpu-blocks "$NUM_GPU_BLOCKS")
  [ -n "${EDGES_S:-}" ] && count+=(--edges-s "$EDGES_S" --save-kv "${SAVE_KV:-1}" --kv-dir "$KV_DIR")
  [ -n "${EDGES_EVERY_S:-}" ] && count+=(--edges-every-s "$EDGES_EVERY_S" ${SAVE_KV:+--save-kv "$SAVE_KV"} --kv-copy "${KV_COPY:-all}")
  last=; to_serve=$(wc -w <<< "${ARMS:-}")
  [ -z "${STOP_AT:-}" ] || stop_at=$(date -d "$STOP_AT" +%s) || { echo "serve-tax: STOP_AT=$STOP_AT is not a time"; exit 2; }
  for arm in ${ARMS:-}; do
    reps=$REPS
    if [ -n "${BUDGET_MIN:-}" ] && [ -n "$last" ]; then
      # this arm's share of the time left, the arms still to serve sharing it equally: a cost compares the reps both
      # of its arms have, so the time must not go to the arms served first
      left=$(((BUDGET_MIN * 60 - ($(date +%s) - t_leased)) / to_serve))
      reps=$("$PY" "$here/serve_tax.py" fit "$OUT/$last" --budget-s "$left" --reps "$REPS") || reps=$REPS
      [ "$reps" -lt "$REPS" ] && echo "== $arm: $reps of $REPS reps fit its $((left / 60)) min, an equal share for each of the $to_serve arms left of BUDGET_MIN=$BUDGET_MIN" | tee -a "$OUT/arms.txt"
      [ "$reps" -lt 1 ] && { reps=1; echo "== $arm: serving 1 rep anyway (reps are cut, never an arm)" | tee -a "$OUT/arms.txt"; }
    fi
    if [ -n "${STOP_AT:-}" ] && [ "$(date +%s)" -ge "$stop_at" ]; then
      echo "== $arm and the arms after it: not served, STOP_AT=$STOP_AT has passed" | tee -a "$OUT/arms.txt"
      break
    fi
    echo "== $arm $(date -u +%FT%TZ)"
    "$PY" "$here/serve_tax.py" run --arm "$arm" "${common[@]}" "${load[@]}" --reps "$reps" ${STOP_AT:+--stop-at "$STOP_AT"} \
      --window-s "$WINDOW_S" --kv-window-s "$KV_WINDOW_S" "${tick[@]}" "${count[@]}" \
      ${BOUNDARY_ROWS:+--boundary-rows "$BOUNDARY_ROWS"} ${PRIVATE:+--private "$PRIVATE/$arm"} \
      ${SIDE:+--side "$SIDE"} ${FRONT:+--front "$FRONT"} ${CELLS:+--cells "$CELLS"} --out "$OUT/$arm" || rc=1
    echo "after $arm: $(fi_cache)" >> "$OUT/engine_env.txt"
    echo "== $arm done $(date -u +%FT%TZ): $reps reps, $((($(date +%s) - t_leased) / 60)) min since the GPUs were held" | tee -a "$OUT/arms.txt"
    last=$arm; to_serve=$((to_serve - 1))
  done
  for ra in ${RESTORE:+${RESTORE_ARM:-restore}}; do
    echo "== restore $ra ${RESTORE} $(date -u +%FT%TZ)"
    "$PY" "$here/restore.py" --arm-dir "${RESTORE_FROM:-$OUT}/$ra" --from-edge "${RESTORE%%:*}" --to-edge "${RESTORE##*:}" \
      --kv-dir "$KV_DIR/$ra" --checkpoint "$CK" --warm "${RESTORE_WARM:-0}" --out "$OUT/$ra-replay" \
      > "$OUT/$ra-replay.log" 2>&1 || { tail -n 60 "$OUT/$ra-replay.log"; rc=1; }
    tail -n 1 "$OUT/$ra-replay.log"
  done
  if [ -n "${REPLAY:-}" ]; then
    declare -A times=()
    for mode in ${REPLAY_MODES:-compiled eager}; do
      flag=(); [ "$mode" = eager ] && flag=(--eager)
      [ "${REPLAY_LOGPROBS:-0}" = 1 ] && flag+=(--logprobs)
      n=$(( ${times[$mode]:-0} + 1 )); times[$mode]=$n
      dir="replay-$mode"; [ "$n" -gt 1 ] && { dir="replay-$mode-$n"; flag+=(--against "$OUT/replay-$mode"); }
      echo "== replay $REPLAY $mode #$n $(date -u +%FT%TZ)"
      "$PY" "$here/replay.py" --arm-dir "${REPLAY_FROM:-$OUT/record}" --load "$REPLAY" --checkpoint "$CK" --tp "$GPUS" \
        "${flag[@]}" ${REPLAY_CAPTURE:+--capture "$REPLAY_CAPTURE"} --out "$OUT/$dir" > "$OUT/$dir.log" 2>&1 \
        || { tail -n 40 "$OUT/$dir.log"; rc=1; }
      tail -n 1 "$OUT/$dir.log"
    done
  fi
  exit $rc
fi

if [ "${CAPACITY:-0}" != 1 ] && [ -z "${ARMS:-}" ]; then
  [ $replay_only = 1 ] || { echo "serve-tax: ARMS: the arms to serve, in order (or only REPLAY with REPLAY_FROM, or RESTORE with RESTORE_FROM)"; exit 2; }
  [ -z "${RESTORE_FROM:-}" ] || [ -n "${KV_DIR:-}" ] || { echo "serve-tax: RESTORE_FROM reads the edges' KV bytes in KV_DIR/<arm>: set KV_DIR"; exit 2; }
elif [ "$CLIENT" = vllm-bench ]; then
  # before the GPUs are held: the arms and the passes each serves checked, each load's pinned dataset file downloaded
  # where the node has none, and its SHA-512 checked
  ds=(); for f in ${DATASETS:-}; do ds+=(--dataset "$f"); done
  "$PY" "$here/serve_tax.py" passes --loads "$LOADS" "${ds[@]}" --fetch --arms "$ARMS" --model-id "$MODEL_ID" \
    ${PASS_PROMPTS:+--pass-prompts "$PASS_PROMPTS"} \
    ${BOUNDARY_ROWS:+--boundary-rows "$BOUNDARY_ROWS"} ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"} || exit 2
elif [ "${CAPACITY:-0}" != 1 ]; then
  [ -z "$PLAN_FILE" ] || cp "$PLAN_FILE" "$PLAN" || exit 2
  "$PY" "$here/serve_tax.py" plan --model-id "$MODEL_ID" --checkpoint "$CK" --c4 "$C4" --rates "${RATES:?RATES: NAME=REQ_PER_S,...}" \
    --span-s "$SPAN_S" ${REQUESTS:+--requests "$REQUESTS"} --docs "$DOCS" --out "$PLAN" || exit 1
fi
fresh_kv=0
if [ -n "${EDGES_S:-}" ] && [ -z "${KV_DIR:-}" ]; then
  KV_DIR=$(mktemp -d "${SCRATCH:-/workspace/research/scratch}/serve-tax-kv.XXXXXX") || exit 2; fresh_kv=1
fi
export PLAN DOCS WINDOW_S KV_WINDOW_S KV_DIR
# LDCONFIG=1: the linker cache refreshed first, so libnvrtc and the rest resolve to the toolkit the node's dev packages
# were installed for, the same for every engine of the job
[ "${LDCONFIG:-0}" = 1 ] && ldconfig
for k in SERVE_VLLM VLLM_CACHE_ROOT FLASHINFER VLLM_USE_FLASHINFER_SAMPLER CUDA_HOME TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK MAX_MODEL_LEN \
    PUBLISHED_FRONTEND PUBLISHED_ARGS PLAN_FILE; do
  echo "$k=${!k-unset}"; done > "$OUT/engine_env.txt"
{ command -v nvcc && nvcc --version | tail -1
  echo "has_flashinfer=$("$PY" -c 'from vllm.utils.flashinfer import has_flashinfer; print(has_flashinfer())')"
  [ -f "$PLAN" ] && echo "plan_sha256=$(sha256sum "$PLAN" | cut -c1-64)"; } >> "$OUT/engine_env.txt" 2>&1
if command -v gpu-lease > /dev/null; then
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-serve-tax} gpu-lease "$GPUS" --wait --max-min "$MAX_MIN" -- bash "$here/run.sh" leased
  rc=$?
else
  # a node without gpu-lease queues its jobs by other means: refuse to start on GPUs another job still holds
  for _ in $(seq 31); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${used:-0}" -lt "${FREE_MIB:-2048}" ] && break
    sleep 20
  done
  if [ "${used:-0}" -ge "${FREE_MIB:-2048}" ]; then
    echo "serve-tax: a GPU still holds $used MiB after 10 min; not starting"; rc=75
  else
    bash "$here/run.sh" leased; rc=$?
  fi
fi
[ $fresh_kv = 1 ] && rm -rf "$KV_DIR"
gzip -f "$OUT"/*/server/step_trace.*.jsonl "$OUT"/*/server/edges/outputs.*.jsonl "$OUT"/*-replay/*.jsonl 2>/dev/null
dirs=()
for d in "$OUT"/*/; do [ -f "$d/meta.json" ] && dirs+=("$d"); done
[ ${#dirs[@]} -gt 0 ] && "$PY" "$here/serve_tax.py" table "${dirs[@]}" --out "$OUT"
# the cost of the last arm over compiled; a job that ends on stock has none: compiled alone (a calibration of its
# rungs), or published, which is stock under other settings and no arm of ours. WITH: the out/ directories of earlier
# jobs on this node whose arms served another load, which joins the number
if [ "$CLIENT" = vllm-bench ] && [ ${#dirs[@]} -gt 0 ] && [ "${ARMS##* }" != compiled ] && [ "${ARMS##* }" != published ]; then
  for o in ${WITH:-}; do for d in "$o"/*/; do [ -f "$d/meta.json" ] && dirs+=("$d"); done; done
  cost=("$PY" "$here/serve_tax.py" cost "${dirs[@]}" --arm "${ARMS##* }" ${TARGET:+--target "$TARGET"})
  "${cost[@]}" --out "$OUT" --result || rc=1
  [ "${JSON:-0}" = 1 ] && "${cost[@]}" --json
fi
exit $rc
