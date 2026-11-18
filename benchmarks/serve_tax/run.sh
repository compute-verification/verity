#!/usr/bin/env bash
# The serve tax on one node (README.md), from the shipped tree's root under `research run`, the C4 shard sent with
# --send (or C4=PATH) unless a ready plan is:
#   ARMS="compiled eager ..." RATES=light=R,medium=R,saturated=R [MODEL_ID=moonlight] [REPS=3] [SPAN_S=60]
#     [REQUESTS=N] [DRAIN_S=600] [WINDOW_S=600] [KV_WINDOW_S=WINDOW_S] [TICK_MS=SAT,FREE | TICK_FROM=RECORD_ARM_DIR] [TICK_RULE=JSON]
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
#     [MAX_MODEL_LEN=N] [FLASHINFER=1] [LDCONFIG=1] [BUDGET_MIN=M] bash benchmarks/serve_tax/run.sh   # the served configuration's tax
#   ARMS=two-stage RATES=... [PLAN_FILE=PLAN.jsonl] PRIVATE=DIR [BOUNDARY_ROWS='{"check":T,...}']
#     bash benchmarks/serve_tax/run.sh   # the two-stage serve: its launch line is the arm (serve_tax.ARMS), its boundary
#                                        # committed under PRIVATE/two-stage/boundary
#   bash benchmarks/serve_tax/run.sh --model glm53 [--arms "ARM ..."|all] [--loads NAME[=RUNG+RUNG...],...]
#     [--dataset NAME=FILE] [--reps 6] [--target PERCENT] [--json]   # the serving cost, by vLLM's own client
#   bash benchmarks/serve_tax/run.sh --model llama32-1b --profile coding-agent|chat|fixed-size --find-load|--feature SCHEME
#     # a serving profile of benchmarks/production.toml: where stock saturates, or one arm of ours beside stock
# The last form is the serving cost (README.md): each flag sets the variable of its name (MODEL_ID, ARMS, LOADS, DATASETS,
# REPS, TARGET, JSON) and any of them makes vLLM's own client the default (CLIENT=vllm-bench: `vllm bench serve`, no C4
# shard and no plan). Its arms are by default stock, stock again and two-stage, with --arms all one arm for each change
# in between and the canonical top-k's own; its loads chat and long. Chat's dataset file is the node's copy of the pinned
# one, downloaded once into HF_HOME before the GPUs are held and checked by its SHA-512 (serve_tax.py passes --fetch);
# --dataset chat=FILE names another copy of it, and no run sends or keeps the file. --dataset prefix=FILE serves a table
# of planned requests (arrival times and lengths; prefix_plan.jsonl is the full-size one) in place of load prefix's
# four, each sent as it arrives: the row then says whether stock kept up, and which of its figures is the cost.
# --profile NAME serves as benchmarks/production.toml's profile does (serve_tax.profile_run): the file's stock launch
# (prod for glm53, stock vLLM at vLLM's own limits for the others), the profile's load at the model's context, no
# sampling flag (the model's generation_config) and each request to its recorded length (fixed-size's 1,024 tokens).
# It fixes ARMS, LOADS, REPS, LIFETIMES, SAMPLED, MAX_MODEL_LEN, PASS_PROMPTS, the published arm's flags, TARGET and
# WITH, and refuses each one given, naming it. --profile, --feature and --find-load are flags alone: PROFILE, FEATURE
# and FIND_LOAD in the environment are refused. --find-load serves stock alone over its load's rungs (serve_tax.LOADS;
# requests in flight up to the profile's prompts) and prints where its output
# tokens a second stop rising: the file's line, the version to raise and the pin test's digest
# (OUT/find_load.json); it edits no file. --feature SCHEME names a registered useful-work scheme (serve_tax.feature),
# whose arm ARM is served with stock in turn at the profile's one load, two lifetimes of three reps each, and writes
# the run's result through the kit's report: its three lines, the first our extra card time a token under the model's
# metric id. Where ARM
# refuses what the profile has on, the result is `cannot run` in the refusal's words and the job exits 1, as for an
# arm whose run refuses it; where a change on our side alone lets the same prompts through, ARM is served with it
# (OURS, its label in the result) as the second line, stock unchanged. The passes are planned within the model's
# cap_minutes (BUDGET_MIN and STOP_AT, the earlier of the given and the cap).
# PASS_PROMPTS=NAME=N,... makes a pass of load NAME send N prompts at every rung in place of the load's own, and
# NAME=Kx K prompts for each request its rung holds in flight (serve_tax.parse_pass_prompts): for a ladder past the
# rungs the load's own prompts can hold, and the same for every arm of a cost. SAMPLED=1 has every request sampled at
# the recipe's temperature and top-p in place of greedy, each naming its user; an arm that keeps the record is sent
# that user's seed too (serve_tax.user_seed, sends_seed), under which its replay samples the request again, and a
# stock arm none, as a served request carries none; under LIFETIMES the seed is another in each lifetime, so that no
# two lifetimes of the arm serve one sample of text. After the table it prints the cost of
# ARMS's last arm over the stock arm it is made from, compiled or, for the arms at vLLM's recipe's settings, prod
# (serve_tax.py cost, serve_tax.stock_of; --arms "prod prod-record prod-again" --loads agentic is that cost on an
# agent's sessions, where a rung is the users and PASS_PROMPTS the turns each sends) and writes
# OUT/serving_cost.{json,txt} and the run's result.json (each
# load's cost a measurement, and the number one under the model's metric id once every load has run); --json prints the
# JSON object last. In this form REPS is 6 and WINDOW_S 30 by default, and OUT/gpus.csv names the cards.
# WITH="OUT_DIR ..." adds earlier jobs' arms to the cost: a load they served and this job didn't joins the number.
# LIFETIMES=L serves the arms L times, each time a server of its own, in ARMS's order and then that order turned round,
# and so on ("prod prod-record" three times: prod, prod-record, prod-record, prod, prod, prod-record), so that a slow
# change of the node over the job falls on both arms alike; lifetime K of an arm is OUT/<arm>.K, and the cost pairs
# pass R of lifetime K of its two arms (serve_tax.lifetimes), its noise floor stock's own lifetimes where no twin of
# stock is served. TURNED=1 begins with ARMS's order turned round ("prod prod-record" twice: prod-record, prod, prod,
# prod-record), so the last arm named is served first. A cost's interval is over the lifetimes an arm, so its verdict
# needs two at least. LEAVE_OUT=N leaves each lifetime's first N passes, its warm-up, out of every figure: they are served
# and recorded like the rest, and the table and the cost say how many were left out. WARM_S is the seconds of the
# warm-up load every lifetime begins with (20; a short pass of vLLM's client under it), and WARM_S=0 sends none: a
# pass LEAVE_OUT leaves out warms the server as it does. A replay names the lifetime it replays
# (REPLAY_FROM=OUT/<arm>.K).
# The number is not given without the row on a repeated prefix under it, which is load prefix's, served in a job of its
# own (--loads prefix --arms "compiled no-prefix-cache two-stage": its cost is the row alone) and joined with WITH.
# An arm behind the network attester (tick-certifier) is served by two commands, and this script is the node's:
#   ARMS="[compiled] tick-certifier" SIDE=server LOADS=chat=RUNG TICK_MS=SAT,FREE [FRONT=PORT] [CELLS=NAME=N,...]
#     [STOP_AT=TIME] bash benchmarks/serve_tax/run.sh --model glm53
# starts the arm's engine and the id front before it on loopback port FRONT and holds them until the network attester's run
# has ended or STOP_AT (serve_tax.certified_server); OUT/tick-certifier/server.json is its record, and it has no table
# and no cost here. The other command runs where the network attester does, a host that reaches FRONT and is never this node:
# serve_tax.py run --arm tick-certifier --side client, with the same LOADS and CELLS. Its directory is the arm's, and
# `serve_tax.py cost` over it and this job's compiled gives its cost. SIDE is its arms' alone: stock named before it
# is served whole, under the same lease.
# PRIVATE is where the arms keep what never leaves the node, each in PRIVATE/<arm>: a boundary commitment's classes, whose
# files hold the requests' tokens and the salts (OUT keeps each class's public commit.json, a line of
# <arm>/server/boundary.PID.jsonl), the history tables, the top-k's counters, and an arm's step traces, served ids and
# edges' outputs and dumps (PRIVATE/<arm>/server/, each gzipped at the job's end), the corpus client's whole records
# (PRIVATE/<arm>/client/; OUT keeps each less its served ids), the plan (PRIVATE/plan.jsonl), and each replay's tokens,
# step records and capture (PRIVATE/replay-<mode>, a restore's in PRIVATE/<arm>-replay).
# It is a directory outside every run's. A serve's later stages read it, so without it an arm that records, a replay,
# or any arm under the corpus client is refused; in the serving cost's form alone, where nothing reads it
# afterwards, an arm that records nothing gets a fresh one, removed when it stops. BOUNDARY_ROWS replaces the object of an
# arm's boundary commitment (a model with no check template of its own needs it).
# EDGES_EVERY_S puts the boundary arm's edge each that many seconds of a load (its replay blocks' length); at TP > 1 each
# rank keeps its KV copy and rows in host memory, and KV_COPY=first copies rank 0's pool alone (for ranks that hold the
# same KV bytes, as GLM-5.3's replicated MLA cache does). PLAN_FILE (else a sent inputs/plan.jsonl) is a ready plan, served
# as it is (the replay ladder's serving plan: each request's own sampling, stopping at its end of text), copied to
# PRIVATE/plan.jsonl; MAX_MODEL_LEN the engines' context limit, a number of tokens or auto (vLLM's word for the longest the
# cards' KV cache holds); PUBLISHED_FRONTEND=rust starts the published arm behind vLLM's Rust front end, on a build
# that has it (python, the default, is the front end of every other arm), and PUBLISHED_ARGS=--FLAG=VALUE,... adds flags
# of vLLM's own to that arm's line alone. FLASHINFER=1 puts CUDA_HOME's bin first on PATH (CUDA_HOME, where unset torch's own toolkit, else
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
# DRAIN_S is how long the trace corpus's client waits, after a load's last arrival, for the answers still out (600 s):
# a request unanswered by then is cut and counted unserved, so a load whose last request is very long (a prompt at
# the model's context limit) names more. It is that client's alone: a pass of vLLM's own client reads none.
# One plan (PRIVATE/plan.jsonl, from the shard's first DOCS documents) is made before the GPUs are held; then each arm is a
# server lifetime of its own (serve_tax.py run) into OUT/<arm>, in ARMS's order, all under one gpu-lease; the tick arms
# take τ from TICK_MS, else from TICK_FROM, else from this job's own record arm, calibrated on its saturated load.
# REPLAY then replays LOAD's first rep of the record arm (by default this job's) in each of REPLAY_MODES (replay.py)
# into OUT/replay-<mode>; a mode listed again replays into OUT/replay-<mode>-<k>, its tokens compared with the first's too
# (REPLAY_MODES="compiled compiled": does the replay agree with itself?). A pass vLLM's client sent is named with its
# rung (REPLAY=agentic@4) and replayed from an arm whose server kept each request's ids (REPLAY_FROM=$OUT/prod-record:
# that client keeps none). The replay of an arm that served on the canonical top-k loads that plugin too, and keeps its
# counters in PRIVATE/replay-<mode> where PRIVATE is set. Then OUT/table.{json,md}. MODEL_ID moonlight serves the replay-determinism benchmark's kept
# checkpoint on one GPU; glm53 the GLM-5.3 TP drivers' (benchmarks/served_commit/tp_decode.sh) on eight; glm53-standin
# its stand-in on one, a directory under /tmp of the cut config and that checkpoint's tokenizer; llama8b,
# qwen3-30b and llama32-1b their pinned snapshots under HF_HOME on one. GPU_LEASE_ON and GPU_LEASE_CORES are the
# lease's --on and --cores. VENV's vLLM must be SERVE_VLLM: d9105ea80, the integration's
# pin, unless set. On a machine that is not ours (a Slurm node, under research run --local-only) there is no gpu-lease:
# the job waits for its own cards (CUDA_VISIBLE_DEVICES, as Slurm sets it) to be free; VENV must be given; HF_HOME is
# the hub's own place under XDG_CACHE_HOME unless given; and a profile's job names its commit from research run's
# --tree-sha (RESEARCH_SOURCE_SHA) or RESEARCH_SOURCE_COMMIT, with no .git, and its card from nvidia-smi's name
# through the card files, never from the machine's name.
set -uo pipefail
[ -z "${PROFILE:-}${FEATURE:-}${FIND_LOAD:-}" ] || {
  echo "serve-tax: PROFILE, FEATURE and FIND_LOAD are not read from the environment: --profile NAME with --feature SCHEME" \
    "or --find-load"; exit 2; }
while [ $# -gt 0 ] && [ "$1" != leased ]; do
  flag=$1; shift
  case $flag in
    --json) JSON=1 ;;
    --find-load) FIND_LOAD=1 ;;
    --model|--arms|--loads|--dataset|--reps|--target|--profile|--feature)
      [ $# -gt 0 ] || { echo "serve-tax: $flag takes a value"; exit 2; }
      case $flag in
        --model) MODEL_ID=$1 ;; --arms) ARMS=$1 ;; --loads) LOADS=$1 ;; --reps) REPS=$1 ;; --target) TARGET=$1 ;;
        --dataset) DATASETS="${DATASETS:+$DATASETS }$1" ;; --profile) PROFILE=$1 ;; --feature) FEATURE=$1 ;;
      esac
      shift ;;
    *) echo "serve-tax: unknown flag $flag (--model, --arms, --loads, --dataset NAME=FILE, --reps, --target, --json," \
         "--profile NAME, --feature SCHEME, --find-load)"; exit 2 ;;
  esac
  CLIENT=${CLIENT:-vllm-bench}
done
CLIENT=${CLIENT:-corpus}
# the second entry, under the lease, takes the profile as the first did: as a flag (leased --profile NAME)
if [ "${1:-}" = leased ] && [ $# -gt 1 ]; then
  { [ $# = 3 ] && [ "$2" = --profile ]; } || { echo "serve-tax: leased takes --profile NAME alone"; exit 2; }
  PROFILE=$3; set -- leased
fi
if [ "${1:-}" != leased ] && [ -n "${PROFILE:-}${FEATURE:-}${FIND_LOAD:-}" ]; then
  [ -n "${PROFILE:-}" ] || { echo "serve-tax: --feature and --find-load are a profile's: --profile NAME"; exit 2; }
  case "${FEATURE:+f}${FIND_LOAD:+l}" in
    f|l) ;;
    *) echo "serve-tax: --profile $PROFILE takes --feature SCHEME or --find-load, one of them"; exit 2 ;;
  esac
  fixes() { echo "serve-tax: --profile $PROFILE fixes $2 (benchmarks/production.toml): drop $1"; exit 2; }
  [ -z "${ARMS:-}" ] || fixes "--arms/ARMS" "the arms: its stock and --feature's arm, or stock alone for --find-load"
  [ -z "${LOADS:-}" ] || fixes "--loads/LOADS" "the load: its dataset, cut and turns at its one load, or --find-load's ladder"
  [ -z "${REPS:-}${LIFETIMES:-}" ] || fixes "--reps/REPS and LIFETIMES" \
    "the passes: two lifetimes of three reps each, six paired, or one of three for --find-load"
  [ "${SAMPLED:-0}" = 0 ] || fixes SAMPLED "the sampling: no flag, so the model's generation_config applies"
  [ -z "${MAX_MODEL_LEN:-}" ] || fixes MAX_MODEL_LEN "the context limit: the model's in the file"
  [ -z "${PASS_PROMPTS:-}" ] || fixes PASS_PROMPTS "the prompts a pass sends"
  [ -z "${PUBLISHED_ARGS:-}${PUBLISHED_FRONTEND:-}" ] || fixes "PUBLISHED_ARGS and PUBLISHED_FRONTEND" "the stock launch"
  [ -z "${TARGET:-}${WITH:-}" ] || fixes "--target/TARGET and WITH" "the result: its three lines, and no cost against a target"
  WINDOW_S=${WINDOW_S:-30}
elif [ "$CLIENT" = vllm-bench ] && [ -z "${PROFILE:-}" ]; then
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
[[ "${LIFETIMES:-1}" =~ ^[1-9][0-9]*$ ]] || { echo "serve-tax: LIFETIMES=$LIFETIMES: how many times the arms are served, 1 or more"; exit 2; }
{ [[ "${LEAVE_OUT:-0}" =~ ^[0-9]+$ ]] && [ "${LEAVE_OUT:-0}" -lt "${REPS:-3}" ]; } \
  || { echo "serve-tax: LEAVE_OUT=$LEAVE_OUT: the passes each lifetime leaves out, fewer than its REPS=${REPS:-3}"; exit 2; }
[[ "${TURNED:-0}" =~ ^[01]$ ]] || { echo "serve-tax: TURNED=$TURNED: 1 begins with the arms' order turned round, 0 with the order named"; exit 2; }
[[ "${WARM_S:-0}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "serve-tax: WARM_S=$WARM_S: the warm-up's seconds, 0 for none"; exit 2; }
[[ "${DRAIN_S:-600}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "serve-tax: DRAIN_S=$DRAIN_S: the seconds the corpus client waits for answers after a load's last arrival"; exit 2; }
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
[ -x "$PY" ] || { echo "serve-tax: no environment at VENV=$VENV: on a machine that is not ours (no /workspace/venv312), set VENV to one holding vLLM $SERVE_VLLM"; exit 2; }
export PATH="$VENV/bin:$PATH"   # the venv's ninja, which FlashInfer's JIT (vLLM's default sampler) runs
[ -d /workspace/cp/hf ] && hf=/workspace/cp/hf || hf=/workspace/hf
[ -d "$hf" ] || hf=${XDG_CACHE_HOME:-$HOME/.cache}/huggingface   # a machine that is not ours: the hub's own place, the job's
export HF_HOME=${HF_HOME:-$hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
MODEL_ID=${MODEL_ID:-moonlight}
case $MODEL_ID in
  moonlight) CK=${CHECKPOINT:-/workspace/research/scratch/replay-determinism/moonlight}; GPUS=1 ;;
  glm53)
    REV=aca966e4e02791568aa6a4ced368624b3d897f42
    if [ -e "/workspace/cp/glm53-ckpt/$REV/.verified" ]; then CK=/workspace/cp/glm53-ckpt/$REV; else CK=/workspace/fs/glm53/$REV; fi
    CK=${CHECKPOINT:-$CK}; GPUS=8 ;;
  glm53-standin)
    REV=aca966e4e02791568aa6a4ced368624b3d897f42
    if [ -n "${TOKENIZER_FROM:-}" ]; then real=$TOKENIZER_FROM
    elif [ -e "/workspace/cp/glm53-ckpt/$REV/.verified" ]; then real=/workspace/cp/glm53-ckpt/$REV; else real=/workspace/fs/glm53/$REV; fi
    CK=${CHECKPOINT:-/tmp/serve-tax-standin.${D##*/}}
    [ -n "${CHECKPOINT:-}" ] || "$PY" "$here/serve_tax.py" standin --model-id "$MODEL_ID" --tokenizer-from "$real" --out "$CK" \
      || exit 2
    GPUS=1 ;;
  llama8b|qwen3-30b|llama32-1b) CK=${CHECKPOINT:-$("$PY" "$here/serve_tax.py" checkpoint --model-id "$MODEL_ID")}; GPUS=1 ;;
  *) echo "serve-tax: MODEL_ID is moonlight, glm53, glm53-standin, llama8b, qwen3-30b or llama32-1b, not $MODEL_ID"; exit 2 ;;
esac
# CUDA_HOME: the toolkit of torch's own CUDA, which FlashInfer's JIT (vLLM's default sampler) builds with (a newer
# toolkit can lack its headers, cuRAND), and whose nvcc the kv and boundary arms build the row-table kernels with; with
# FLASHINFER=1 first on PATH, for FlashInfer's JIT too, which finds CUDA_HOME's headers and libraries itself (an inherited
# CPATH or LIBRARY_PATH could mix toolkits)
[ -n "${CUDA_HOME:-}" ] || { echo "serve-tax: CUDA_HOME is not set: set it to the CUDA toolkit of torch's CUDA version (a node's placement entry names it)"; exit 2; }
export CUDA_HOME
if [ "${FLASHINFER:-0}" = 1 ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  unset CPATH LIBRARY_PATH
else
  command -v nvcc >/dev/null || export PATH="$PATH:$CUDA_HOME/bin"
fi
C4=${C4:-$(ls "$D"/inputs/*.json.gz 2>/dev/null | head -1)}
REPS=${REPS:-3}; SPAN_S=${SPAN_S:-60}; MAX_MIN=${MAX_MIN:-240}; DOCS=${DOCS:-45000}
WINDOW_S=${WINDOW_S:-600}; KV_WINDOW_S=${KV_WINDOW_S:-$WINDOW_S}; PLAN=${PLAN:-${PRIVATE:+$PRIVATE/plan.jsonl}}
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
if [ -n "${PROFILE:-}" ] && [ "${1:-}" != leased ]; then
  # what the profile serves (serve_tax.profile_plan): ARMS, LOADS, REPS, LIFETIMES, the changes on our side (OURS), the
  # stock arm and the model's minutes (CAP_MIN), within which the arms' passes are planned. No arm where the feature
  # cannot run and no change on our side lets the profile through: its result is written now, and no GPU is held
  plan=$("$PY" "$here/serve_tax.py" profile plan --profile "$PROFILE" --model-id "$MODEL_ID" ${FEATURE:+--feature "$FEATURE"} \
    ${FIND_LOAD:+--find-load}) || exit 2
  eval "$plan"
  { [ -n "${BUDGET_MIN:-}" ] && [ "$BUDGET_MIN" -le "$CAP_MIN" ]; } || BUDGET_MIN=$CAP_MIN
  export ARMS LOADS REPS LIFETIMES OURS STOCK CAP_MIN BUDGET_MIN
  if [ -z "$ARMS" ]; then
    "$PY" "$here/serve_tax.py" profile result --profile "$PROFILE" --model-id "$MODEL_ID" --feature "$FEATURE" --out "$OUT"
    exit $?
  fi
fi
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
    [ "${SAMPLED:-0}" = 1 ] && load+=(--sampled)
    for f in ${DATASETS:-}; do load+=(--dataset "$f"); done
  else
    common+=(--c4 "$C4")
    load=(--rates "${RATES:-}" --span-s "$SPAN_S" --plan "$PLAN" --docs "$DOCS")
  fi
  if [ "${CAPACITY:-0}" = 1 ]; then
    "$PY" "$here/serve_tax.py" run --arm compiled "${common[@]}" --rates burst=1000 --requests "${REQUESTS:-800}" --reps 1 \
      --warm-s 10 --private "$PRIVATE/capacity" --out "$OUT/capacity"
    exit $?
  fi
  rc=0
  if [ -n "${TICK_MS:-}" ]; then tick=(--tick-ms "$TICK_MS"); else tick=(--tick-from "${TICK_FROM:-$OUT/record}"); fi
  [ -n "${TICK_RULE:-}" ] && tick+=(--tick-rule "$TICK_RULE")
  count=(); [ -n "${REQUESTS:-}" ] && count=(--requests "$REQUESTS")
  [ -n "${NUM_GPU_BLOCKS:-}" ] && count+=(--num-gpu-blocks "$NUM_GPU_BLOCKS")
  [ -n "${EDGES_S:-}" ] && count+=(--edges-s "$EDGES_S" --save-kv "${SAVE_KV:-1}" --kv-dir "$KV_DIR")
  [ -n "${EDGES_EVERY_S:-}" ] && count+=(--edges-every-s "$EDGES_EVERY_S" ${SAVE_KV:+--save-kv "$SAVE_KV"} --kv-copy "${KV_COPY:-all}")
  [ -n "${WARM_S:-}" ] && count+=(--warm-s "$WARM_S")
  [ -n "${DRAIN_S:-}" ] && count+=(--drain-s "$DRAIN_S")
  # the server lifetimes in order: the arms as named, then turned round, and so on (LIFETIMES), so that two arms'
  # lifetimes of one number are neighbours in time, or the turned order first (TURNED=1); with one lifetime an arm's
  # directory is its name
  order=()
  for k in $(seq "${LIFETIMES:-1}"); do
    names=(${ARMS:-}); [ $(((k + ${TURNED:-0}) % 2)) = 0 ] && names=($(printf '%s\n' "${names[@]}" | tac))
    for arm in "${names[@]}"; do order+=("$arm:$k"); done
  done
  last=; to_serve=${#order[@]}
  if [ -n "${PROFILE:-}" ]; then
    # a profile's job stops by its model's minutes: an arm starts no pass it can't finish by then, and the result says
    # cannot tell where a side's passes were cut
    cap_at=$((t_leased + CAP_MIN * 60))
    { [ -n "${STOP_AT:-}" ] && [ "$(date -d "$STOP_AT" +%s)" -le "$cap_at" ]; } 2>/dev/null \
      || STOP_AT=$(date -u -d "@$cap_at" +%FT%TZ)
  fi
  [ -z "${STOP_AT:-}" ] || stop_at=$(date -d "$STOP_AT" +%s) || { echo "serve-tax: STOP_AT=$STOP_AT is not a time"; exit 2; }
  for item in "${order[@]}"; do
    arm=${item%%:*}; dir=$arm; life=()
    [ "${LIFETIMES:-1}" -gt 1 ] && { dir=$arm.${item##*:}; life=(--lifetime "${item##*:}"); }
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
    echo "== $dir $(date -u +%FT%TZ)"
    profile=(); [ -n "${PROFILE:-}" ] && profile=(--profile "$PROFILE")
    [ -n "${PROFILE:-}" ] && [ "$arm" != "$STOCK" ] && [ -n "${OURS:-}" ] && profile+=(--ours "$OURS")
    "$PY" "$here/serve_tax.py" run --arm "$arm" "${common[@]}" "${load[@]}" "${profile[@]}" --reps "$reps" ${STOP_AT:+--stop-at "$STOP_AT"} \
      --window-s "$WINDOW_S" --kv-window-s "$KV_WINDOW_S" "${tick[@]}" "${count[@]}" "${life[@]}" \
      ${LEAVE_OUT:+--leave-out "$LEAVE_OUT"} ${SIDE:+--side "$SIDE"} ${FRONT:+--front "$FRONT"} ${CELLS:+--cells "$CELLS"} \
      ${BOUNDARY_ROWS:+--boundary-rows "$BOUNDARY_ROWS"} ${PRIVATE:+--private "$PRIVATE/$dir"} --out "$OUT/$dir" || rc=1
    echo "after $dir: $(fi_cache)" >> "$OUT/engine_env.txt"
    echo "== $dir done $(date -u +%FT%TZ): $reps reps, $((($(date +%s) - t_leased) / 60)) min since the GPUs were held" | tee -a "$OUT/arms.txt"
    last=$dir; to_serve=$((to_serve - 1))
  done
  for ra in ${RESTORE:+${RESTORE_ARM:-restore}}; do
    echo "== restore $ra ${RESTORE} $(date -u +%FT%TZ)"
    "$PY" "$here/restore.py" --arm-dir "${RESTORE_FROM:-$OUT}/$ra" --from-edge "${RESTORE%%:*}" --to-edge "${RESTORE##*:}" \
      --kv-dir "$KV_DIR/$ra" --checkpoint "$CK" --warm "${RESTORE_WARM:-0}" --out "$OUT/$ra-replay" \
      --private "$PRIVATE/$ra-replay" \
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
        "${flag[@]}" ${REPLAY_CAPTURE:+--capture "$REPLAY_CAPTURE"} ${PRIVATE:+--private "$PRIVATE/$dir"} \
        --out "$OUT/$dir" > "$OUT/$dir.log" 2>&1 \
        || { tail -n 40 "$OUT/$dir.log"; rc=1; }
      tail -n 1 "$OUT/$dir.log"
    done
  fi
  exit $rc
fi

if [ -z "${PRIVATE:-}" ]; then
  recording=$("$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); import serve_tax as S
print(" ".join(a for a in sys.argv[2:] if S.records(S.ARMS[a])))' "$here" ${ARMS:-}) || exit 2
  [ -z "$recording" ] || { echo "serve-tax: arms $recording record their step traces, served ids or edges' outputs, which hold request values: PRIVATE, a directory outside every run's"; exit 2; }
  [ -z "${REPLAY:-}" ] || { echo "serve-tax: REPLAY keeps each replay's tokens on the node: PRIVATE, a directory outside every run's"; exit 2; }
  [ "$CLIENT" = vllm-bench ] || { echo "serve-tax: the corpus client keeps each request's served ids, and the plan its prompts, on the node: PRIVATE, a directory outside every run's (or CLIENT=vllm-bench)"; exit 2; }
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
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-serve-tax} gpu-lease "$GPUS" --wait --max-min "$MAX_MIN" ${GPU_LEASE_ON:+--on "$GPU_LEASE_ON"} \
    ${GPU_LEASE_CORES:+--cores "$GPU_LEASE_CORES"} -- bash "$here/run.sh" leased ${PROFILE:+--profile "$PROFILE"}
  rc=$?
else
  # a node without gpu-lease queues its jobs by other means: refuse to start on GPUs another job still holds, of those
  # this job may use (CUDA_VISIBLE_DEVICES, as a scheduler such as Slurm sets it), else of all
  for _ in $(seq 31); do
    used=$(nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i "$CUDA_VISIBLE_DEVICES"} --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${used:-0}" -lt "${FREE_MIB:-2048}" ] && break
    sleep 20
  done
  if [ "${used:-0}" -ge "${FREE_MIB:-2048}" ]; then
    echo "serve-tax: a GPU still holds $used MiB after 10 min; not starting"; rc=75
  else
    bash "$here/run.sh" leased ${PROFILE:+--profile "$PROFILE"}; rc=$?
  fi
fi
[ $fresh_kv = 1 ] && rm -rf "$KV_DIR"
[ -z "${PRIVATE:-}" ] || gzip -f "$PRIVATE"/*-replay/*.jsonl "$PRIVATE"/replay-*/replay_steps.jsonl \
  "$PRIVATE"/*/server/step_trace.*.jsonl "$PRIVATE"/*/server/ids.*.jsonl \
  "$PRIVATE"/*/server/edges/outputs.*.jsonl 2>/dev/null
dirs=()
for d in "$OUT"/*/; do [ -f "$d/meta.json" ] && dirs+=("$d"); done
[ ${#dirs[@]} -gt 0 ] && "$PY" "$here/serve_tax.py" table "${dirs[@]}" --out "$OUT"
# the cost of the last arm over the stock arm it is made from (compiled, or prod); a job that ends on stock has
# none: compiled alone (a calibration of its rungs), or published or prod, which are stock under other settings and
# no arm of ours; nor has the node's side of an arm behind the network attester, whose passes are in the other
# side's directory. WITH: the out/ directories of earlier jobs on this node whose arms served another load, which
# joins the number
if [ -n "${PROFILE:-}" ]; then
  # a profile's result in place of a cost (serve_tax.profile_result): exit 1 where the feature cannot run, as for an arm
  # whose run refuses it
  "$PY" "$here/serve_tax.py" profile result "${dirs[@]}" --profile "$PROFILE" --model-id "$MODEL_ID" \
    ${FEATURE:+--feature "$FEATURE"} ${FIND_LOAD:+--find-load} --out "$OUT" || rc=1
elif [ "$CLIENT" = vllm-bench ] && [ ${#dirs[@]} -gt 0 ] && [ "${ARMS##* }" != compiled ] && [ "${ARMS##* }" != published ] \
    && [ "${ARMS##* }" != prod ] && [ "${SIDE:-}" != server ]; then
  for o in ${WITH:-}; do for d in "$o"/*/; do [ -f "$d/meta.json" ] && dirs+=("$d"); done; done
  cost=("$PY" "$here/serve_tax.py" cost "${dirs[@]}" --arm "${ARMS##* }" ${TARGET:+--target "$TARGET"})
  "${cost[@]}" --out "$OUT" --result || rc=1
  [ "${JSON:-0}" = 1 ] && "${cost[@]}" --json
fi
exit $rc
