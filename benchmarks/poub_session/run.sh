#!/usr/bin/env bash
# PoUB sessions (cwd = the source tree): the verifier's process on cores of its own, the prover's on GPUS
# GPUs (default 1) whose HBM holds one store, a contiguous shard each, over TCP. Everything runs under one
# gpu-lease $GPUS --wait (with TIMED=1, --timed: the fill queue pauses, for a whole-node session): build session.cu and
# kv.cu (p1024, masked), start the verifier, then the prover, which gates its devices against the driver before
# anything is timed. Results go to $RESEARCH_RUN_DIR/out (verifier.json, prover.json and their logs).
#
#   RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=ra-first-session PLAN=honest GIB=16 taskset -c 48-95 \
#     bash benchmarks/poub_session/run.sh      # on an RTX PRO 6000 node, 96 GB of host memory
#
# PLAN is the prover's sessions (honest, skip:F; comma-separated, one session each); GIB the store across every GPU
# (whole 1 GiB chunks); WINDOWS, WINDOW_MS, SAMPLES, JITTER_MS the sessions'; CHECK_WORKERS the verifier's processes
# computing each window's exact bound (default a core less than it may use); VERIFIER_CPUS and PROVER_CPUS the cores
# each process is pinned to; MAXMIN the lease's minutes; PY the Python (default the fill's torch venv; a verifier needs
# only numpy, so on a CPU node any venv with it).
#
# With PROVER_SLICE the prover runs at nice 0 in a scope of that top-level slice (root's systemd-run, then this user
# again), on PROVER_CPUS and in at most PROVER_MEM_GB. The node's user slice holds only 0-119, where nice 19 doesn't make
# another session's CPU jobs yield to the prover (cgroups share CPU by weight, a session's as much as every lease's
# together): on 48-95 beside them its host saw roots up to 35 ms late, 1.34x the median pass.
# The timing slice's CPUs 128-191 are fill's timing jobs', each of which holds a GPU, so none runs in a whole-node lease.
#
# The whole node's HBM (752 GB, 700 GiB on eight GPUs; its pass handles go to host memory, so 260 GB of it):
#
#   GPUS=8 TIMED=1 GIB=700 ROLE=prover PROVER_SLICE=<the timing slice> PROVER_CPUS=128-191 ... \
#     bash benchmarks/poub_session/run.sh
#
# On two machines, two runs with the same PORT and SESSIONS: ROLE=verifier on the verifier's machine (no GPU), which
# opens `ssh -R PORT` to PEER (the developer's machine, a name in /etc/vy/peers) and listens behind it, so the prover's
# connections to its own 127.0.0.1:PORT cross the network; then ROLE=prover on PEER, which connects there, and which
# takes its lease only once that tunnel listens (up to TUNNEL_WAIT_S, default 7200).
#
# Against the auditor's verifier host (`verity/protocols/accounting/memory/poub/attester/live_verifier/host.py`), the lane starts the run there with its
# gate key (its gate's `run --port PORT ...`) and ROLE=prover gets VERIFIER_SSH, ssh's `-i KEY prover@HOST`: the
# prover opens `ssh -L PORT` to the host's tunnel account, which forwards only to the run ports, and takes its lease once
# that account says the run's verifier listens.
#
#   ROLE=verifier PEER=PROVER_NODE PORT=7791 SESSIONS=2 taskset -c 96-127 bash benchmarks/poub_session/run.sh
#   ROLE=prover PORT=7791 PLAN=honest,skip:0.0625 taskset -c 48-95 bash benchmarks/poub_session/run.sh
#
# ROLE has no default: the verifier runs on a machine the developer doesn't control (AGENTS.md's auditor-trusted
# devices), never on PEER, so a run says which side it is. ROLE=both runs the verifier beside the prover on this
# machine, a test of these scripts that audits nothing.
# ROLE=verifier is the audit's verifier as a service. It keeps the registrations in REGISTRY (default
# ~/.local/state/remote-session/registry.json on its own machine, across its runs: a node with a live registration there
# is refused), and leaves out/verifier.json, out/registry.json (REGISTRY when it ends) and out/address.json, the address
# the prover connects to: PEER's 127.0.0.1:PORT, a Commit's REMOTE_VERIFIER. VERIFIER_ARGS go to verifier.py as shell
# words, split and never evaluated (quotes group one, as `--challenger-ssh '-i KEY HOST'` needs; --lean-exe refused),
# under ROLE=both too: with `--challenger serve
# --challenger-key DIR` each session's coins come from the challenger's own process,
# `verity/protocols/run/verifier/live_challenger/serve.py` on this machine (`challengers.py`), its out/ under out/challenger; with `--challenger gate --challenger-ssh WORDS --challenger-peer NAME --challenger-pin SHA512
# --challenger-period P --challenger-main FILE` (FILE: `git rev-list --first-parent origin/main`, the auditor's own),
# from the auditor device's challenger through its gate, its fetched sessions, ledger and pins log there
# (PORT then outside the gate's 7811-7899). A registration left live by a run that ended early refuses the node until
# CLOSE=NODE_HEX closes it (the same ROLE=verifier run, which closes it in REGISTRY and exits). The verifier builds
# the Lean verifier with elan's `lake` ($HOME/.elan/bin, `tools/verity/lean/setup.sh`).
#
# CPU=BLOCKS makes the prover the driver's reference over a store of BLOCKS blocks under CODEC (default p512/k2): no
# GPU, no lease, no build, and kv:P and kvdrop:P sessions (KV pages appended and released mid-session, `prover.py`)
# with PROVER_ARGS such as --kv-every and --kv-pool. Its passes take the CPU's time, so JITTER_MS must cover one.
# On a card (no CPU, SERVE unset or node) CODEC defaults to the deployed codec (`codec.DEFAULT`), and p512/kK is the other
# codec kv.cu is built for (KV_DEFS).
# LIFETIME (windows) sets the verifier's cap L on an appended block, after which the prover re-keys it (default: the
# windows of WINDOW_MS in 30 s, or a paced session's groups, `audit.lifetime_windows`).
# REFRESH sets refresh for the sessions (the verifier's --refresh, `audit.Params.refresh`): unset, off, and nothing
# differs; 1, every session; one word a session, as 0,1 with PLAN=P,P, the same plan twice on one registration, first
# under today's rule and then under refresh, which refresh_result.py writes side by side as one result.
#
# SERVE makes the prover vLLM serving under the audit, on one leased GPU in vLLM's venv (VLLM_PY, default the first
# there of NODE_WORKSPACE/jobs/venv312/bin/python and NODE_WORKSPACE/venv312/bin/python, NODE_WORKSPACE default
# /workspace; its `verity-vllm` launcher beside it), this tree's verity_vllm first on its path; PY then defaults to
# that venv where the fill's is not there. SERVE=tiny serves one session a store, from one line:
#
#   RESEARCH_RUN_DIR=DIR ROLE=both SERVE=tiny bash benchmarks/poub_session/run.sh            # a node with one free GPU
#
# ROLE=both runs the verifier beside the prover; the audit's is ROLE=verifier on its own machine with SESSIONS=1,
# then ROLE=prover PORT=... SERVE=... here. SERVE=tiny runs `serve.py --connect`: the vLLM tests' tiny Llama, PLAN serve
# (the default) or servedrop:K. SERVE=row, a row's Commit under the audit, is refused by name: the row is deleted.
#
# A served line, in order, with everything before its lease done without a GPU:
#   1. refuses (exit 3) what cannot run: a PLAN that is not a served one, windows that do not fit the run (below);
#   2. ROLE=both: builds the Lean verifier (LEAN_EXE, naming another, is refused), starts the verifier on VERIFIER_CPUS
#      and goes on once it listens;
#   3. takes the lease and runs the prover on PROVER_CPUS, with OMP_NUM_THREADS and MKL_NUM_THREADS their count
#      (defaults: this process's cores, under ROLE=both the last quarter the verifier's and the rest the prover's);
#   4. says how the session ended in one line, `remote session verdict: ...` (out/verdict.txt), and exits with its
#      code (Exit, below). ROLE=both leaves no verifier behind: it waits VERIFIER_WAIT_S (default 180) for its
#      verifier to end after the prover, then stops it, its process group by its id; after a prover that ended red
#      it first opens and drops one connection to PORT, which a verifier still waiting for its prover ends at.
#      ROLE=prover, whose verifier is not its own, opens and drops that connection too when its prover ends red
#      after taking the lease (KEEP_VERIFIER=1: not), so a verifier started by hand does not listen on.
#
# A served session's windows are the verifier's wall clock and its passes the run's engine steps, so the two must fit:
# windows that outlast the run end the session rejected only when the whole run is over, and a window shorter than a
# pass is rejected with no root. PASS_MS (default 500) states the milliseconds from one pass to the next, an engine
# step and the CPU's read of every registered block; a rejected session's verdict line says what its passes took.
# Without WINDOW_MS the windows are derived (`serve.py --fit`): WINDOWS (default 3) of four passes each. A WINDOW_MS the line states is
# checked by the same sum, and refused where it cannot fit. JITTER_MS defaults to ten passes: a serving prover's
# calibration states no pass time, so the jitter, allowed each way, is the whole allowance for the pass in flight at
# a boundary, and the last window closes twice that long after its boundary. Under ROLE=prover the windows are the
# verifier's line's: state the same WINDOWS and WINDOW_MS here to have them checked, or ask `serve.py --fit` before
# starting the verifier.
#
# The lease is `gpu-lease`: GPU_LEASE names the program, else the one on PATH, else
# NODE_WORKSPACE/pouw/infra/bin/gpu-lease, else this tree's (tools/research/src/research/pods/sh/gpu_lease.sh). GPU
# names the cards it may take (--on), and NOWAIT=1 takes them only if they are free now (exit 75 otherwise) instead
# of queueing; a served line with both also ends 75 where a compute process is on those cards, one that took them
# without the lease, asked before anything is built and again before the lease. A served line declares the cores at
# work while its lease is held to a lease program whose text names --cores: PROVER_CPUS and, under ROLE=both,
# VERIFIER_CPUS, which the lease counts with every other lease's against the node's CPU quota (75 where they pass
# it under NOWAIT=1); to a program that names none it declares none, and says so. A served line's CUDA_HOME defaults
# to the machine's toolkit at the CUDA version V the prover's torch is built for (13.0 where it has no torch), and is
# refused (exit 3) where the machine lists none at V or its nvcc is not V's major version; any other line's to the
# machine's toolkit at 13.0. ROLE=prover takes PORT; ROLE=both and
# ROLE=verifier pick one nothing listens on when it is not stated. Scratch is under TMPDIR (default /tmp), removed
# at exit.
#
# Exit: 0 every session accepted (SERVE=tiny: the prover's record says so and, under ROLE=both, the
#     verifier's too; under ROLE=prover the verdict is the verifier's line's, and this code only the prover's word);
#   3 refused before anything ran: a parameter, a path, windows that cannot fit, no verifier on PORT (and
#     ROLE=verifier's own code for a registration it refused);
#   4 a build failed (nvcc, the Lean verifier);
#   5 SERVE=tiny and ROLE=both SERVE=node: the prover ended 0 beside a verifier that ended red or with no
#     accepted session on record;
#   6 SERVE=tiny and ROLE=both SERVE=node: a session was not accepted: the verifier rejected it, or it
#     failed where the prover's record says (the verdict line has both);
#   75 no lease: gpu-lease's, or NOWAIT=1's for a card with a process on it (124, gpu-lease's too: the lease's MAXMIN
#     ended the prover);
#   1 without SERVE: the prover or, under ROLE=both, the verifier ended red; any other code is the prover's own.
#
# SERVE=node serves the whole node: a vLLM engine a leased GPU (`engine.py`), and the prover (`node.py`) registers
# their weights arenas as the store, where they are served, the engines decoding each weight as they use it
# (`node.sh` starts them; MODEL, ENGINE_ARGS, ENGINE_CPUS, NODE_ARGS). Its cores default to this process's: the first quarter the
# engines' (ENGINE_CPUS), under ROLE=both the last quarter the verifier's, the rest the prover's, all declared to a lease
# program that takes --cores. PACED (K:T_MS:DRAWS:WINDOWS a session of PLAN, `verifier.py --paced`) and PACE_PASS_MS
# make the sessions paced; ROLE=both SERVE=node, the node's session on this machine with its verifier and a local
# challenger beside it, defaults them to 1:1000:2773:7 (seven stretches of a second, a live coin a pass) and 0 (the
# pass its calibration measured), builds the Lean verifier before the lease, as a served line does, and ends by the
# verdict line. The verifier reads the audited node's figures, its HBM and host DRAM among them, from FIGURES, by default out/figures.json, which `sheets.py NODE_KIND`
# writes, NODE_KIND a node file's name (ROLE=both on a GPU: the node file of this machine's card). The north-star run,
# the verifier on a second node's CPUs:
#
#   ROLE=verifier PEER=PROVER_NODE PORT=7799 SESSIONS=2 WINDOWS=7 WINDOW_MS=60000 SAMPLES=33273 QUANTILE=max \
#     JITTER_MS=50 NODE_KIND=rtx-pro-6000-server-x8 \
#     bash benchmarks/poub_session/run.sh                         # 16 CPUs
#   ROLE=prover PORT=7799 SERVE=node PLAN=honest,skip:0.0625 ENGINE_CPUS=0-23 PROVER_CPUS=24-47 \
#     ENGINE_ARGS='--step-hz 2 --align-steps' NODE_ARGS='--decode-s 60 --warm-openings 33273' \
#     bash benchmarks/poub_session/run.sh                         # the whole 8-GPU node, 200 GB of host memory
#
# Start the prover first: its lease waits for all 8 GPUs, and once it holds them it waits for the verifier's tunnel;
# start the verifier then.
case "${1:-}" in -h | --help) sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0 ;; esac
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
export LEAN_ABORT_ON_PANIC=1  # one_stage/settings.py's ABORT_ON_PANIC, for every Lean verifier below and every process it starts
refuse() { echo "run.sh: refused: $*"; exit 3; }
first_of() { # first_of TEST PATH...: the first PATH that `test TEST PATH` holds of
  local t=$1 p; shift
  for p in "$@"; do if test "$t" "$p"; then echo "$p"; return 0; fi; done
  return 1
}
case ${ROLE:-} in
  verifier | prover | both) export ROLE ;;
  *) echo "ROLE is verifier or prover (both only to test these scripts on one machine), not '${ROLE:-}': the verifier" \
       "runs on a machine the developer doesn't control, so there is no default"; exit 3 ;;
esac
# SERVED: tiny, a session this script serves whole (its windows fitted, its verifier under ROLE=both, its verdict)
SERVED=
[ "$ROLE" = verifier ] && unset SERVE
case ${SERVE:-} in
  '' | node) ;;
  tiny) SERVED=$SERVE ;;
  row) refuse "SERVE=row is deleted with the row's Commit (verity-vllm row, epoch resolve): SERVE=tiny or SERVE=node serves under the audit" ;;
  *) refuse "SERVE is tiny, row or node, not $SERVE" ;;
esac
WS=${NODE_WORKSPACE:-/workspace} RS=benchmarks/poub_session
ENV_DIR=${ENV_DIR:-$WS/pouw/fill-out/pous/work/pous-bench}
VPY=
if [ -n "${SERVE:-}" ]; then
  VPY=${VLLM_PY:-$(first_of -x "$WS/jobs/venv312/bin/python" "$WS/venv312/bin/python")} ||
    refuse "no vLLM venv: VLLM_PY is not set, and neither $WS/jobs/venv312/bin/python nor $WS/venv312/bin/python is there"
  [ -x "$VPY" ] || refuse "VLLM_PY=$VPY is not an executable"
fi
PY=${PY:-$(first_of -x "$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python" ${VPY:+"$VPY"})} ||
  refuse "no torch venv: PY is not set, and $ENV_DIR/gpu312-torch2.14.0-cu130/bin/python is not there"
[ -x "$PY" ] || refuse "PY=$PY is not an executable"
export PY PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
if [ -z "${CUDA_HOME:-}" ] && [ -n "$SERVED" ]; then
  # the machine's toolkit at the CUDA version the prover's torch is built for (13.0 where it has no torch to ask)
  TV=$("$VPY" -c 'import torch; print(torch.version.cuda or "")' 2> /dev/null)
  TM=${TV%%.*}
  CUDA_HOME=$(python3 benchmarks/kernels/machine.py toolkit "${TV:-13.0}") ||
    refuse "no CUDA toolkit ${TV:-13.0}: CUDA_HOME is not set, and the machine's record lists none (machine.py, above)"
  # the Commit's collector is built by torch's JIT, which checks no nvcc against the CUDA torch is built for
  NV=$("$CUDA_HOME/bin/nvcc" --version 2> /dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
  [ -z "$TM" ] || [ "$NV" = "$TM" ] || refuse "the vLLM venv's torch is built for CUDA $TM, and $CUDA_HOME's nvcc is" \
    "CUDA ${NV:-unknown}: install a CUDA $TM toolkit and state CUDA_HOME"
fi
export CUDA_HOME=${CUDA_HOME:-$(python3 benchmarks/kernels/machine.py toolkit 13.0)}
export PATH=$CUDA_HOME/bin:$(dirname "$PY"):$PATH
[ -n "$SERVED" ] && . adapters/vllm/verity_vllm/ops/cuda_headers.sh && cuda_headers "$VPY"
[ -x "$HOME/.elan/bin/lake" ] && export PATH="$HOME/.elan/bin:$PATH"
B=$(mktemp -d "${TMPDIR:-/tmp}/remote-session.XXXXXX") || exit 3
SV= # ROLE=both's verifier of a served session, a process group of its own
trap '[ -n "$SV" ] && kill -- "-$SV" 2> /dev/null; rm -rf "$B"' EXIT
export B RS GIB=${GIB:-16}
# The codec a card's session registers and its kv.so is built for: CODEC, default the deployed one, read here from
# the codec's module and written nowhere in this script (CPU= and SERVE=tiny default to p512/k2 below). kv.cu's build fixes the width and masking, the registration's key the
# rounds. p1024 squares by FOLD 6 as the paged build does (benchmarks/vllm_encoded/stack.sh's KVDEFS); p512 is kv.cu's
# plain build. Each build of the session's kv.so reads KV_DEFS, node.sh's too.
DEPLOYED=$("$PY" -c 'from verity.protocols.accounting.memory.poub.verifier import codec; print(codec.DEFAULT.name)') ||
  refuse "$PY cannot read the deployed codec (verifier/codec.py's DEFAULT)"
export CARD_CODEC=${CODEC:-$DEPLOYED}
case $CARD_CODEC in
  p1024/k[2-8]/masked) KV_DEFS="-DKV_FIELD=1024 -DKV_MASK=1 -DKV_FOLD6=1" ;;
  p512/k[2-8]) KV_DEFS="-DKV_FIELD=512" ;;
  *) KV_DEFS= ;;
esac
export KV_DEFS
if [ "$ROLE" != verifier ] && [ -z "${CPU:-}" ] && [ "${SERVE:-}" != tiny ]; then
  [ -n "$KV_DEFS" ] || refuse "CODEC=$CARD_CODEC: a card's session is built at p1024/kK/masked or p512/kK, K from 2 to 8"
fi
[ -z "${LEAN_EXE:-}" ] || refuse "LEAN_EXE names a Lean verifier of the caller's: the session's is the one" \
  "lean_verifier.build builds in its package, and nothing else; unset it"
lean_exe() { # the Lean verifier built in its package
  [ -n "${LEAN:-}" ] && return 0
  LEAN=$("$PY" -c 'from verity.protocols.accounting.memory.poub.verifier.session import lean_verifier as lv; print(lv.build())' \
    2> "$O/lean-build.log") ||
    { tail -20 "$O/lean-build.log"; echo "run.sh: the Lean verifier's build failed"; exit 4; }
}
# VERIFIER_ARGS as shell words, split and never evaluated, into VA; one naming a Lean verifier is refused
VA=()
if [ -n "${VERIFIER_ARGS:-}" ]; then
  python3 -c 'import shlex, sys; sys.stdout.write("".join(w + "\0" for w in shlex.split(sys.argv[1])))' \
    "$VERIFIER_ARGS" > "$O/verifier-args.bin" || refuse "VERIFIER_ARGS is not shell words"
  mapfile -d '' VA < "$O/verifier-args.bin"
  rm -f "$O/verifier-args.bin"
  for w in "${VA[@]}"; do
    case $w in --lean-exe | --lean-exe=*) refuse "VERIFIER_ARGS names --lean-exe: the session's Lean verifier is the" \
      "one lean_verifier.build builds in its package, and nothing else" ;; esac
  done
fi
if [ -n "${SERVE:-}" ]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=${HF_HOME:-$WS/hf} PATH=$(dirname "$VPY"):$PATH
  export PYTHONPATH=$PWD/adapters/vllm:$PWD/adapters/torch:$PWD/tools/config:$PWD/verity/protocols/proof/flock/prover/python:$PWD${PYTHONPATH:+:$PYTHONPATH}
fi
if [ -n "$SERVED" ]; then
  PLAN=${PLAN:-serve}
  case $SERVED:$PLAN in
    tiny:serve) ;;
    tiny:servedrop:*) [[ ${PLAN#servedrop:} =~ ^[0-9]+$ ]] || refuse "PLAN=$PLAN: servedrop:K withholds the first K blocks appended" ;;
    tiny:*) refuse "PLAN=$PLAN is not a plan of a served session: SERVE=tiny takes serve or servedrop:K" ;;
  esac
  [ "${SESSIONS:-1}" = 1 ] || refuse "SESSIONS=$SESSIONS: a served store serves one session"
  FIT=()
  # the windows: the verifier's under ROLE=prover, so only a pair this line states is checked there
  PASS_MS=${PASS_MS:-500}
  if [ "$ROLE" = both ] || { [ -n "${WINDOWS:-}" ] && [ -n "${WINDOW_MS:-}" ]; }; then
    FIT+=(${WINDOWS:+--windows "$WINDOWS"} ${WINDOW_MS:+--window-ms "$WINDOW_MS"})
  fi
  FITTED=$("$PY" $RS/serve.py --fit --pass-ms "$PASS_MS" "${FIT[@]}") || { echo "run.sh: ${FITTED:-serve.py --fit failed}"; exit 3; }
  [ "$ROLE" = both ] && read -r WINDOWS WINDOW_MS <<< "$FITTED"
  JITTER_MS=${JITTER_MS:-$((10 * PASS_MS))}
  # cores: this process's, the last quarter to ROLE=both's verifier and the rest to the prover; with one core, both on it
  read -r VC PC < <("$PY" -c '
import os, sys
c = sorted(os.sched_getaffinity(0))
v = max(1, len(c) // 4) if sys.argv[1] == "both" else 0
print(",".join(map(str, c[-v:] if v else c)), ",".join(map(str, c[:len(c) - v] or c)))' "$ROLE")
  VERIFIER_CPUS=${VERIFIER_CPUS:-$VC} PROVER_CPUS=${PROVER_CPUS:-$PC}
  THREADS=$(unset OMP_NUM_THREADS OMP_THREAD_LIMIT; taskset -c "$PROVER_CPUS" nproc) ||
    refuse "PROVER_CPUS=$PROVER_CPUS are not cores this process may run on"
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS:-$THREADS}
fi
if [ "${SERVE:-}" = node ]; then
  # cores: this process's, the first quarter to the engines, under ROLE=both the last quarter to the verifier, the rest
  # to the prover; with one core, all on it
  read -r EC VC PC < <("$PY" -c '
import os, sys
c = sorted(os.sched_getaffinity(0))
q = max(1, len(c) // 4)
v = q if sys.argv[1] == "both" else 0
p = c[q:len(c) - v] or c
print(*(",".join(map(str, x)) for x in (c[:q], c[-v:] if v else p, p)))' "$ROLE")
  export ENGINE_CPUS=${ENGINE_CPUS:-$EC}
  VERIFIER_CPUS=${VERIFIER_CPUS:-$VC} PROVER_CPUS=${PROVER_CPUS:-$PC}
  if [ "$ROLE" = both ]; then PACED=${PACED-1:1000:2773:7} PACE_PASS_MS=${PACE_PASS_MS-0}; fi
fi
export PLAN=${PLAN:-honest} WINDOWS=${WINDOWS:-7}
export WINDOW_MS=${WINDOW_MS:-1000} SAMPLES=${SAMPLES:-2773} JITTER_MS=${JITTER_MS:-2}
export VERIFIER_CPUS=${VERIFIER_CPUS:-88-95} PROVER_CPUS=${PROVER_CPUS:-48-87}
port_free() { ! ss -Hltn "sport = :$1" 2> /dev/null | grep -q .; }
if [ -z "${PORT:-}" ]; then
  [ "$ROLE" = prover ] && refuse "ROLE=prover takes PORT, the port of its verifier's line (address.json): none is picked" \
    "here, since no verifier listens on a port this line picks"
  PORT=$((7700 + RANDOM % 200))
  for ((k = 0; k < 200; k++)); do
    port_free "$PORT" && break
    PORT=$((7700 + (PORT - 7699) % 200))
  done
  port_free "$PORT" || refuse "no free port from 7700 to 7899"
fi
export PORT
export SESSIONS=${SESSIONS:-$(($(tr -cd , <<< "$PLAN" | wc -c) + 1))}
FORM=(--windows "$WINDOWS" --window-ms "$WINDOW_MS" --samples "$SAMPLES" --jitter-ms "$JITTER_MS" --sessions "$SESSIONS")
if [ -n "${PACED:-}" ]; then
  read -ra PACE <<< "$PACED"
  [ ${#PACE[@]} = "$SESSIONS" ] || refuse "PACED=$PACED paces ${#PACE[@]} session(s), and PLAN=$PLAN has $SESSIONS"
  FORM=(--paced "${PACE[@]}" ${PACE_PASS_MS:+--pace-pass-ms "$PACE_PASS_MS"} --jitter-ms "$JITTER_MS")
fi
VERIFY=(--port "$PORT" --out "$O" "${FORM[@]}" ${QUANTILE:+--quantile "$QUANTILE"}
  ${CHECK_WORKERS:+--check-workers "$CHECK_WORKERS"} ${LIFETIME:+--lifetime "$LIFETIME"} ${REFRESH:+--refresh "$REFRESH"})
if [ "${SERVE:-}" = node ] && [ "$ROLE" = both ]; then
  lean_exe
  VERIFY+=(--lean-exe "$LEAN")
fi
if [ "$ROLE" != prover ] && [ -z "${CLOSE:-}" ]; then
  if [ -z "${FIGURES:-}" ]; then
    if [ -z "${NODE_KIND:-}" ] && [ "$ROLE" = both ] && [ -z "${CPU:-}" ]; then
      NODE_KIND=$(python3 benchmarks/kernels/machine.py session-node "$(nvidia-smi --query-gpu=name --format=csv,noheader \
        | head -1)" 0) || refuse "no node file for this machine's card: $NODE_KIND"
      NODE_KIND=${NODE_KIND#local/} NODE_KIND=${NODE_KIND%/*}
    fi
    [ -n "${NODE_KIND:-}" ] || refuse "the verifier reads the audited node's figures: set NODE_KIND (a node file under" \
      "calibration/hardware/nodes) or FIGURES (their JSON, benchmarks/poub_session/sheets.py NODE)"
    FIGURES=$O/figures.json
    "$PY" benchmarks/poub_session/sheets.py "$NODE_KIND" > "$FIGURES" || refuse "no figures for NODE_KIND=$NODE_KIND"
  fi
  VERIFY+=(--figures "$FIGURES")
fi
for ((k = 0; k < ${#VA[@]}; k++)); do
  case ${VA[k]} in
    --challenger-key=*) KEYDIR=${VA[k]#*=} ;;
    --challenger-key) KEYDIR=${VA[k + 1]:-} ;;
    *) continue ;;
  esac
  case $(realpath -m "$KEYDIR")/ in
    "$(realpath -m "$RESEARCH_RUN_DIR")"/*) refuse "--challenger-key $KEYDIR is inside the run's directory, which custody" \
      "uploads: the challenger's key directory holds its seed; give one outside it" ;;
  esac
done
if [ "$ROLE" = verifier ]; then
  REGISTRY=${REGISTRY:-$HOME/.local/state/remote-session/registry.json}
  mkdir -p "$(dirname "$REGISTRY")" || exit 3
  if [ -n "${CLOSE:-}" ]; then
    "$PY" verity/protocols/accounting/memory/poub/verifier/poub_session/verifier_process.py --registry "$REGISTRY" --close "$CLOSE" \
      > "$O/verifier.log" 2>&1; W=$?
    cat "$O/verifier.log"
    cp "$REGISTRY" "$O/registry.json"
    exit $W
  fi
  read -ra TO <<< "$(grep "^${PEER:?} " /etc/vy/peers | cut -d' ' -f2-)"
  [ ${#TO[@]} -gt 0 ] || { echo "no peer $PEER in /etc/vy/peers"; exit 3; }
  printf '{"verifier": "%s", "listen": "127.0.0.1:%s", "peer": "%s", "remote_verifier": "127.0.0.1:%s"}\n' \
    "$(hostname)" "$PORT" "$PEER" "$PORT" | tee "$O/address.json"
  # A tty makes the session interactive, so ssh and sshd set TCP_NODELAY (under -N roots were held ~30 ms every few
  # seconds, enough to make a boundary pass's root late); and with keystroke-timing obfuscation off, since it sends an
  # interactive session's packets on a 20 ms clock.
  ssh -tt -o ObscureKeystrokeTiming=no -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
    -R "127.0.0.1:$PORT:127.0.0.1:$PORT" "${TO[@]}" "sleep 86400" < /dev/null > "$O/tunnel.log" 2>&1 & T=$!
  # shellcheck disable=SC2086
  "$PY" verity/protocols/accounting/memory/poub/verifier/poub_session/verifier_process.py --host 127.0.0.1 "${VERIFY[@]}" --registry "$REGISTRY" "${VA[@]}" \
    > "$O/verifier.log" 2>&1 & V=$!
  while kill -0 $V 2> /dev/null; do
    kill -0 $T 2> /dev/null || { echo "the tunnel to $PEER ended"; cat "$O/tunnel.log"; kill $V; break; }
    wait -n $V $T
  done
  wait $V; W=$?
  kill $T 2> /dev/null
  cp "$REGISTRY" "$O/registry.json"
  tail -20 "$O/verifier.log"
  exit $W
fi
export VERIFY_ARGS="${VERIFY[*]}"
if [ "$ROLE" = prover ] && [ -n "${VERIFIER_SSH:-}" ]; then
  # The verifier host's run at PORT: this ssh -L listens here at once, so the lease waits for the host's tunnel account
  # to say the run's verifier listens (`listening PORT`).
  read -ra TO <<< "$VERIFIER_SSH"
  ssh -T -o BatchMode=yes -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 -L "127.0.0.1:$PORT:127.0.0.1:$PORT" "${TO[@]}" < /dev/null > "$O/tunnel.log" 2>&1 & T=$!
  trap 'kill $T 2> /dev/null; rm -rf "$B"' EXIT
  for ((k = 0; k < ${TUNNEL_WAIT_S:-7200} / 5; k++)); do
    grep -q "^listening $PORT" "$O/tunnel.log" && break
    kill -0 $T 2> /dev/null || break
    sleep 5
  done
  grep -q "^listening $PORT" "$O/tunnel.log" || { echo "the verifier host's run never listened on $PORT"; cat "$O/tunnel.log"; exit 3; }
elif [ "$ROLE" = prover ]; then
  # Into the lease's queue only once the verifier's tunnel listens here: a prover that takes the node before its verifier
  # is up stops retrying its connect a minute after building the store (`wire.connect`).
  for ((k = 0; k < ${TUNNEL_WAIT_S:-7200} / 5; k++)); do
    ss -Hltn "sport = :$PORT" | grep -q . && break
    sleep 5
  done
  ss -Hltn "sport = :$PORT" | grep -q . || { echo "no verifier tunnel on 127.0.0.1:$PORT"; exit 3; }
fi
if [ -n "${GPU_LEASE:-}" ]; then
  LF=$(command -v "$GPU_LEASE") || refuse "GPU_LEASE=$GPU_LEASE is not a program"
  LP=("$GPU_LEASE")
elif LF=$(command -v gpu-lease); then LP=(gpu-lease)
elif [ -x "$WS/pouw/infra/bin/gpu-lease" ]; then
  LF=$WS/pouw/infra/bin/gpu-lease
  LP=("$LF")
else
  LF=$PWD/tools/research/src/research/pods/sh/gpu_lease.sh
  LP=(bash "$LF")
fi
# A served line's lease holds the card: where GPU_LEASE names the tool, a row's Commit would ask for one of its own
# (`verity_vllm.pipeline.gpu_lease`).
[ -n "$SERVED" ] && unset GPU_LEASE
WAIT=(--wait)
[ -n "${NOWAIT:-}" ] && WAIT=()
LEASE=("${LP[@]}" "${GPUS:-1}" "${WAIT[@]}" ${GPU:+--on "$GPU"} ${TIMED:+--timed} --max-min "${MAXMIN:-30}")
[ -n "${GPU_LEASE_UUID:-}" ] || [ -n "${CPU:-}" ] && LEASE=()
if [ -n "${SERVE:-}" ] && [ ${#LEASE[@]} -gt 0 ]; then
  # the cores at work while the lease is held: the prover's, SERVE=node's engines' and, under ROLE=both, its verifier's,
  # which holds no lease
  LEASE_CPUS=$PROVER_CPUS
  [ "$ROLE" = both ] && [ "$VERIFIER_CPUS" != "$PROVER_CPUS" ] && LEASE_CPUS+=,$VERIFIER_CPUS
  [ "$SERVE" = node ] && LEASE_CPUS+=,$ENGINE_CPUS
  if grep -q -e '--cores' "$LF" 2> /dev/null; then LEASE+=(--cores "$LEASE_CPUS")
  else echo "run.sh: the lease program $LF names no --cores: cores $LEASE_CPUS are pinned under it, not declared to it"; fi
fi
[ ${#LEASE[@]} -gt 0 ] && LEASE+=(--)
export CPU=${CPU:-} CODEC=${CODEC:-p512/k2}
card_free() { # NOWAIT=1 with GPU: 75 where a compute process is on those cards, one that took them without the lease
  [ -n "${NOWAIT:-}" ] && [ -n "${GPU:-}" ] && [ ${#LEASE[@]} -gt 0 ] || return 0
  local apps
  apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader) ||
    { echo "run.sh: NOWAIT=1: nvidia-smi can't say what runs on GPU $GPU"; return 75; }
  [ -z "$apps" ] || { echo "run.sh: NOWAIT=1: GPU $GPU is not free: $(wc -l <<< "$apps") compute process(es) on it"; return 75; }
}
if [ -n "$SERVED" ]; then
  card_free || exit 75
  drop_connection() { # a verifier still waiting for its prover ends at a connection opened and dropped
    (exec 3<> "/dev/tcp/127.0.0.1/$PORT") 2> /dev/null
  }
  RUN=("$VPY" $RS/serve.py --connect "127.0.0.1:$PORT" --out "$O" --plan "$PLAN" --codec "$CODEC")
  REC=$O/serve.json
  if [ "$ROLE" = both ]; then
    lean_exe
    set -m # a process group of its own: stopping it stops its Lean verifier and its check workers with it
    taskset -c "$VERIFIER_CPUS" "$PY" verity/protocols/accounting/memory/poub/verifier/poub_session/verifier_process.py --host 127.0.0.1 \
      "${VERIFY[@]}" --lean-exe "$LEAN" "${VA[@]}" < /dev/null > "$O/verifier.log" 2>&1 &
    SV=$!
    set +m
    until grep -q '^verifier: listening' "$O/verifier.log"; do
      if ! kill -0 "$SV" 2> /dev/null; then
        wait "$SV"; W=$?
        SV=
        tail -20 "$O/verifier.log"
        refuse "the verifier ended $W before it listened on $PORT"
      fi
      sleep 0.2
    done
  fi
  P=0
  card_free || P=$?
  if [ $P = 0 ]; then
    "${LEASE[@]}" taskset -c "$PROVER_CPUS" "${RUN[@]}" > "$O/prover.log" 2>&1; P=$?
    tail -20 "$O/prover.log"
  fi
  W=
  if [ "$ROLE" = both ]; then
    [ $P = 0 ] || drop_connection
    for ((k = 0; k < ${VERIFIER_WAIT_S:-180} * 5; k++)); do
      kill -0 "$SV" 2> /dev/null || break
      sleep 0.2
    done
    if kill -0 "$SV" 2> /dev/null; then
      echo "run.sh: the verifier (process group $SV) was still running ${VERIFIER_WAIT_S:-180} s after its prover ended: stopped"
      kill -- "-$SV" 2> /dev/null
    fi
    wait "$SV"; W=$?
    SV=
    tail -20 "$O/verifier.log"
  elif [ $P != 0 ] && [ $P != 75 ] && [ -z "${KEEP_VERIFIER:-}" ]; then
    drop_connection
  fi
  "$PY" $RS/serve.py --verdict --out "$O" --prover-rc "$P" ${W:+--verifier-rc "$W"} --record "$REC" --window-ms "$WINDOW_MS" |
    tee "$O/verdict.txt"
  exit "${PIPESTATUS[0]}"
fi
if [ -n "${SERVE:-}" ]; then
  export VPY
  "${LEASE[@]}" bash $RS/node.sh > "$O/prover.log" 2>&1; P=$?
  tail -20 "$O/prover.log"
  [ "$ROLE" = both ] || exit $P
  W=$(cat "$B/verifier.rc" 2> /dev/null)
  "$PY" $RS/serve.py --verdict --out "$O" --prover-rc "$P" --verifier-rc "${W:-0}" --record "$O/prover.json" |
    tee "$O/verdict.txt"
  exit "${PIPESTATUS[0]}"
fi
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  if [ -n "$CPU" ]; then
    DEV=(--cpu "$CPU" --codec "$CODEC")
    SNODE=$(python3 benchmarks/kernels/machine.py session-node cpu 0)
  else
    SNODE=$(python3 benchmarks/kernels/machine.py session-node "$(nvidia-smi --query-gpu=name --format=csv,noheader \
      -i "${GPU_LEASE_UUID%%,*}" | head -1)" "$(nvidia-smi --query-gpu=index --format=csv,noheader -i "${GPU_LEASE_UUID%%,*}" \
      | head -1)" 2>&1) || { echo "run.sh: no node for the session to register: $SNODE"; exit 3; }
    nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used,memory.total \
      --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
    ARCH=$(python3 benchmarks/kernels/machine.py arch "$(nvidia-smi --query-gpu=name --format=csv,noheader \
      -i "${GPU_LEASE_UUID%%,*}" | head -1)" 2>&1) && GENCODE=$(python3 benchmarks/kernels/machine.py gencode "$ARCH" 2>&1) ||
      { echo "run.sh: no build target for the leased card: ${GENCODE:-$ARCH}"; exit 4; }
    FLAGS=(-O3 -gencode "$GENCODE" -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
    nvcc "${FLAGS[@]}" -o "$B/session.so" verity/protocols/accounting/memory/poub/prover/poub_session/session.cu > "$O/session.ptxas" 2>&1 & S=$!
    nvcc "${FLAGS[@]}" $KV_DEFS -o "$B/kv.so" verity/protocols/accounting/memory/poub/prover/word_codec/kv.cu > "$O/kv.ptxas" 2>&1 & K=$!
    wait $S || { tail -40 "$O/session.ptxas"; exit 4; }
    wait $K || { tail -40 "$O/kv.ptxas"; exit 4; }
    sha256sum "$B"/*.so > "$O/libs.sha256"
    DEV=(--lib "$B/session.so" --kv "$B/kv.so" --gib "$GIB" --codec "$CARD_CODEC")
  fi
  V=
  if [ "$ROLE" = both ]; then
    # shellcheck disable=SC2086
    taskset -c "$VERIFIER_CPUS" "$PY" verity/protocols/accounting/memory/poub/verifier/poub_session/verifier_process.py $VERIFY_ARGS "${VA[@]}" > "$O/verifier.log" 2>&1 & V=$!
  fi
  PIN=(taskset -c "$PROVER_CPUS")
  if [ -n "${PROVER_SLICE:-}" ]; then
    # sudo drops PATH, PYTHONPATH and LD_*, so they are set again
    PIN=(sudo -n --preserve-env env "PATH=$PATH" "PYTHONPATH=${PYTHONPATH:-}" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
      "HOME=$HOME" "USER=$(id -un)" "LOGNAME=$(id -un)" systemd-run --quiet --scope --collect --slice="$PROVER_SLICE"
      --nice=0 -p AllowedCPUs="$PROVER_CPUS" -p MemoryMax="${PROVER_MEM_GB:-300}G" -p MemorySwapMax=0 --
      setpriv --reuid="$(id -u)" --regid="$(id -g)" --init-groups -- "${PIN[@]}")
  fi
  "${PIN[@]}" "$PY" verity/protocols/accounting/memory/poub/prover/poub_session/prover.py --connect "127.0.0.1:$PORT" --out "$O" --plan "$PLAN" \
    "${DEV[@]}" --node "${SESSION_NODE:-$SNODE}" ${PROVER_ARGS:-} > "$O/prover.log" 2>&1
  P=$?
  W=0
  if [ -n "$V" ]; then
    if [ $P != 0 ]; then kill $V 2> /dev/null; fi
    wait $V; W=$?
    tail -20 "$O/verifier.log"
  fi
  tail -20 "$O/prover.log"
  [ $P = 0 ] && [ $W = 0 ]'
