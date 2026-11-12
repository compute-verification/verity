#!/usr/bin/env bash
# PoUB sessions (cwd = the source tree): the verifier's process on cores of its own, the prover's on GPUS
# GPUs (default 1) whose HBM holds one store, a contiguous shard each, over TCP. Everything runs under one
# gpu-lease $GPUS --wait (with TIMED=1, --timed: the fill queue pauses, for a whole-node session): build session.cu and
# kv.cu (p1024, masked), start the verifier, then the prover, which gates its devices against the driver before
# anything is timed. Results go to $RESEARCH_RUN_DIR/out (verifier.json, prover.json and their logs).
#
#   RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=ra-first-session PLAN=honest GIB=16 taskset -c 48-95 \
#     bash benchmarks/remote_session/run.sh      # on an RTX PRO 6000 node, 96 GB of host memory
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
#     bash benchmarks/remote_session/run.sh
#
# On two machines, two runs with the same PORT and SESSIONS: ROLE=verifier on the verifier's machine (no GPU), which
# opens `ssh -R PORT` to PEER (the developer's machine, a name in /etc/vy/peers) and listens behind it, so the prover's
# connections to its own 127.0.0.1:PORT cross the network; then ROLE=prover on PEER, which connects there, and which
# takes its lease only once that tunnel listens (up to TUNNEL_WAIT_S, default 7200).
#
# Against the auditor's verifier host (`verity/protocols/poub/device/live_verifier/host.py`), the lane starts the run there with its
# gate key (its gate's `run --port PORT ...`) and ROLE=prover gets VERIFIER_SSH, ssh's `-i KEY prover@HOST`: the
# prover opens `ssh -L PORT` to the host's tunnel account, which forwards only to the run ports, and takes its lease once
# that account says the run's verifier listens.
#
#   ROLE=verifier PEER=PROVER_NODE PORT=7791 SESSIONS=2 taskset -c 96-127 bash benchmarks/remote_session/run.sh
#   ROLE=prover PORT=7791 PLAN=honest,skip:0.0625 taskset -c 48-95 bash benchmarks/remote_session/run.sh
#
# ROLE has no default: the verifier runs on a machine the developer doesn't control (AGENTS.md's auditor-trusted
# devices), never on PEER, so a run says which side it is. ROLE=both runs the verifier beside the prover on this
# machine, a test of these scripts that audits nothing.
# ROLE=verifier is the audit's verifier as a service. It keeps the registrations in REGISTRY (default
# ~/.local/state/remote-session/registry.json on its own machine, across its runs: a node with a live registration there
# is refused), and leaves out/verifier.json, out/registry.json (REGISTRY when it ends) and out/address.json, the address
# the prover connects to: PEER's 127.0.0.1:PORT, a Commit's REMOTE_VERIFIER. VERIFIER_ARGS go to verifier.py as shell
# words (quotes group one, as `--challenger-ssh '-i KEY HOST'` needs), under ROLE=both too: with `--challenger serve
# --challenger-key DIR` each session's coins come from the challenger's own process,
# `verity/composite/device/live_challenger/serve.py` on this machine (`challengers.py`), its out/ under out/challenger; with `--challenger gate --challenger-ssh WORDS --challenger-peer NAME --challenger-pin SHA512
# --challenger-period P --challenger-main FILE` (FILE: `git rev-list --first-parent origin/main`, the auditor's own),
# from the auditor host's challenger through its gate, its fetched sessions, ledger and pins log there
# (PORT then outside the gate's 7811-7899). A registration left live by a run that ended early refuses the node until
# CLOSE=NODE_HEX closes it (the same ROLE=verifier run, which closes it in REGISTRY and exits). The verifier builds
# the Lean verifier with elan's `lake` ($HOME/.elan/bin, `tools/verity/lean/setup.sh`).
#
# CPU=BLOCKS makes the prover the driver's reference over a store of BLOCKS blocks under CODEC (default p512/k2): no
# GPU, no lease, no build, and kv:P and kvdrop:P sessions (KV pages appended and released mid-session, `prover.py`)
# with PROVER_ARGS such as --kv-every and --kv-pool. Its passes take the CPU's time, so JITTER_MS must cover one.
# LIFETIME (windows) sets the verifier's cap L on an appended block, after which the prover re-keys it (default: the
# windows of WINDOW_MS in 30 s, or a paced session's groups, `audit.lifetime_windows`).
#
# SERVE makes the prover vLLM serving under the audit, on one leased GPU in vLLM's venv (VLLM_PY, default the first
# there of NODE_WORKSPACE/jobs/venv312/bin/python and NODE_WORKSPACE/venv312/bin/python, NODE_WORKSPACE default
# /workspace; its `verity-vllm` launcher beside it), this tree's verity_vllm first on its path; PY then defaults to
# that venv where the fill's is not there. SERVE=tiny and SERVE=row serve one session a store, each from one line:
#
#   RESEARCH_RUN_DIR=DIR ROLE=both SERVE=tiny bash benchmarks/remote_session/run.sh            # a node with one free GPU
#   RESEARCH_RUN_DIR=DIR ROLE=both SERVE=row ROW=ROW_ID bash benchmarks/remote_session/run.sh
#
# ROLE=both runs the verifier beside the prover; the audit's is ROLE=verifier on its own machine with SESSIONS=1,
# then ROLE=prover PORT=... SERVE=... here. SERVE=tiny runs `serve.py --connect`: the vLLM tests' tiny Llama, PLAN serve
# (the default) or servedrop:K. SERVE=row runs row ROW's Commit under the audit, `verity-vllm row run ROW CHECKPOINT
# REPO REV` with the checkpoint `verity-vllm epoch resolve ROW` names and REMOTE_VERIFIER at the verifier: its Commit
# puts the store binder on its engine and proves the store in lockstep with its first instrumented run, a pass an
# engine step (`verity_vllm.protocols.remote_bandwidth.remote.CommitSession`: one GPU, synchronous scheduling declared
# in the row's target, its store the weights and --remote-filler-gib of filler, the KV pool outside it). ROW is a row
# id whose file `verity_vllm.config.row_file` finds (ROW_FILE: that file instead): a stored row whose target declares
# the synchronous scheduling a session needs, such as
# llama32-1b__int8__h200__tp1__b1__i256__o32__mixed__greedy__bi-eager, or a file of your own in a directory
# $VERITY_WORKLOADS names (`verity-vllm workload --row-id ROW ... --target '{..., "async_scheduling": false}' --out
# DIR/ROW.json`).
#
# A served line, in order, with everything before its lease done without a GPU:
#   1. refuses (exit 3) what cannot run: a PLAN that is not a served one, a ROW with no file or whose target does not
#      declare "async_scheduling": false, windows that do not fit the run (below);
#   2. SERVE=row: builds kv.cu for the card (KV_LIB, default build, at KV_ARCH, default the sm_XY of the
#      compute_capability the row's target declares; a path: that build; cpu: none), then runs the row's Build with
#      no GPU visible (STAGES, default
#      build,commit, in a config run, CONFIG_RUN default 1: the Build, the word check and one instrumented Commit,
#      the run the session audits; BUILD_GPU=1 leaves the Build to the lease) in SWEEP_DIR (default: scratch removed
#      at exit, as the native collector's build, NATIVE_COLLECT_BUILD; the row's records under 1 MiB are kept in
#      out/row). Every `row run` gets VLLM_PY as its stages' interpreter (--py: PY may be the fill's venv, which
#      holds no vLLM), CUDA_HOME/bin (--cuda-bin) and ROW_ARGS (such as --norm-tap-so PATH);
#   3. ROLE=both: builds the Lean verifier (LEAN_EXE: that executable instead), starts the verifier on VERIFIER_CPUS
#      and goes on once it listens;
#   4. takes the lease and runs the prover on PROVER_CPUS, with OMP_NUM_THREADS and MKL_NUM_THREADS their count
#      (defaults: this process's cores, under ROLE=both the last quarter the verifier's and the rest the prover's).
#      SERVE=row's is the row's Commit, with --remote-plan PLAN, --remote-kv-pages KV_PAGES (default 24: a count that
#      is none of the cache's other dimensions, which the store binder tells apart by size), --remote-weight-blocks
#      WEIGHT_BLOCKS (default 4096: the Commit's own default registers every weight, and every pass reads every
#      registered block on the CPU) and kv.cu's build, ahead of EXTRA_COMMIT_ARGS. A registration's key answers a
#      hello over the engine's own store, so none is encoded before the engine is on the card: with kv.cu it is
#      encoded there, on the card, and with KV_LIB=cpu on one core while the card waits (state a small WEIGHT_BLOCKS);
#   5. says how the session ended in one line, `remote session verdict: ...` (out/verdict.txt), and exits with its
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
# Without WINDOW_MS the windows are derived (`serve.py --fit`): WINDOWS (default 3) of four passes each, or of half
# the passes the row's expected_engine_steps leave room for where that is fewer. A WINDOW_MS the line states is
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
# to CUDA_ROOT/cuda-V (CUDA_ROOT default /usr/local) for the CUDA version V the prover's torch is built for, else
# cuda-13.0, else cuda; any other line's to /usr/local/cuda-13.0. ROLE=prover takes PORT; ROLE=both and
# ROLE=verifier pick one nothing listens on when it is not stated. Scratch is under TMPDIR (default /tmp), removed
# at exit.
#
# Exit: 0 every session accepted (SERVE=tiny, SERVE=row: the prover's record says so and, under ROLE=both, the
#     verifier's too; under ROLE=prover the verdict is the verifier's line's, and this code only the prover's word);
#   3 refused before anything ran: a parameter, a path, a row, windows that cannot fit, no verifier on PORT (and
#     ROLE=verifier's own code for a registration it refused);
#   4 a build failed (nvcc, the Lean verifier);
#   5 SERVE=tiny, SERVE=row: the prover ended 0 beside a verifier that ended red or with no accepted session on
#     record;
#   6 SERVE=tiny, SERVE=row: the session was not accepted: the verifier rejected it, or it failed where the prover's
#     record says (the verdict line has both);
#   10 to 21 SERVE=row: the row's own (`verity-vllm row`): 10 its Build failed, 12 its Commit failed beside a session
#     that was accepted or never began, 13 no manifest, 15 its plan, 20 the word check, 21 a call boundary;
#   75 no lease: gpu-lease's, or NOWAIT=1's for a card with a process on it (124, gpu-lease's too: the lease's MAXMIN
#     ended the prover);
#   1 without SERVE: the prover or, under ROLE=both, the verifier ended red; any other code is the prover's own.
#
# SERVE=node serves the whole node: a vLLM engine a leased GPU (`engine.py`), and the prover (`node.py`) registers
# their weights arenas as the store, where they are served, the engines decoding each weight as they use it
# (`node.sh` starts them; MODEL, ENGINE_ARGS, ENGINE_CPUS, NODE_ARGS). VERIFIER_ARGS=--host-bytes N puts the serving
# node's host DRAM (its MemTotal) in the memory line. The north-star run, the verifier on a second node's CPUs:
#
#   ROLE=verifier PEER=PROVER_NODE PORT=7799 SESSIONS=2 WINDOWS=7 WINDOW_MS=60000 SAMPLES=33273 QUANTILE=max \
#     JITTER_MS=50 VERIFIER_ARGS='--host-bytes 1843153879040' bash benchmarks/remote_session/run.sh   # 16 CPUs
#   ROLE=prover PORT=7799 SERVE=node PLAN=honest,skip:0.0625 ENGINE_CPUS=0-23 PROVER_CPUS=24-47 \
#     ENGINE_ARGS='--step-hz 2 --align-steps' NODE_ARGS='--decode-s 60 --warm-openings 33273' \
#     bash benchmarks/remote_session/run.sh                         # the whole 8-GPU node, 200 GB of host memory
#
# Start the prover first: its lease waits for all 8 GPUs, and once it holds them it waits for the verifier's tunnel;
# start the verifier then.
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
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
# SERVED: tiny or row, a session this script serves whole (its windows fitted, its verifier under ROLE=both, its verdict)
SERVED=
[ "$ROLE" = verifier ] && unset SERVE
case ${SERVE:-} in
  '' | node) ;;
  tiny | row) SERVED=$SERVE ;;
  *) refuse "SERVE is tiny, row or node, not $SERVE" ;;
esac
WS=${NODE_WORKSPACE:-/workspace} RS=benchmarks/remote_session
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
  # the toolkit of the CUDA version the prover's torch is built for, where the node has it
  CUDA_ROOT=${CUDA_ROOT:-/usr/local}
  TV=$("$VPY" -c 'import torch; print(torch.version.cuda or "")' 2> /dev/null)
  [ "$TV" = 13.0 ] && TV=
  TOOLKITS=(${TV:+"$CUDA_ROOT/cuda-$TV"} "$CUDA_ROOT/cuda-13.0" "$CUDA_ROOT/cuda")
  CUDA_HOME=$(first_of -d "${TOOLKITS[@]}") ||
    refuse "no CUDA toolkit: CUDA_HOME is not set, and none of ${TOOLKITS[*]} is a directory"
fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PATH=$CUDA_HOME/bin:$(dirname "$PY"):$PATH
[ -x "$HOME/.elan/bin/lake" ] && export PATH="$HOME/.elan/bin:$PATH"
B=$(mktemp -d "${TMPDIR:-/tmp}/remote-session.XXXXXX") || exit 3
SV= # ROLE=both's verifier of a served session, a process group of its own
trap '[ -n "$SV" ] && kill -- "-$SV" 2> /dev/null; rm -rf "$B"' EXIT
export B RS GIB=${GIB:-16}
if [ -n "${SERVE:-}" ]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=${HF_HOME:-$WS/hf} PATH=$(dirname "$VPY"):$PATH
  export PYTHONPATH=$PWD/adapters/vllm:$PWD${PYTHONPATH:+:$PYTHONPATH}
fi
if [ -n "$SERVED" ]; then
  PLAN=${PLAN:-serve}
  case $SERVED:$PLAN in
    tiny:serve | row:serve) ;;
    tiny:servedrop:*) [[ ${PLAN#servedrop:} =~ ^[0-9]+$ ]] || refuse "PLAN=$PLAN: servedrop:K withholds the first K blocks appended" ;;
    tiny:*) refuse "PLAN=$PLAN is not a plan of a served session: SERVE=tiny takes serve or servedrop:K" ;;
    *) refuse "PLAN=$PLAN is not a plan of a served session: SERVE=row takes serve" ;;
  esac
  [ "${SESSIONS:-1}" = 1 ] || refuse "SESSIONS=$SESSIONS: a served store serves one session"
  [ "$ROLE" != both ] || [ -z "${LEAN_EXE:-}" ] || [ -x "$LEAN_EXE" ] || refuse "LEAN_EXE=$LEAN_EXE is not an executable"
  FIT=()
  if [ "$SERVED" = row ]; then
    [ -n "${ROW:-}" ] || refuse "SERVE=row takes ROW, a row id"
    VV=$(dirname "$VPY")/verity-vllm
    [ -x "$VV" ] || refuse "no verity-vllm launcher beside $VPY"
    STAGES=${STAGES:-build,commit}
    [[ ,$STAGES, == *,commit,* ]] || refuse "STAGES=$STAGES holds no commit, the stage whose run the session audits"
    ROW_FILE=${ROW_FILE:-$("$VPY" -c 'import sys; from verity_vllm import config; print(config.row_file(sys.argv[1]))' "$ROW" \
      2> "$B/row_file.err")} || refuse "row $ROW: its file can't be found: $(tail -n 1 "$B/row_file.err")"
    FIT=(--row-file "$ROW_FILE")
  fi
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
VERIFY=(--port "$PORT" --out "$O" --windows "$WINDOWS" --window-ms "$WINDOW_MS" --samples "$SAMPLES"
  --jitter-ms "$JITTER_MS" --sessions "$SESSIONS" ${QUANTILE:+--quantile "$QUANTILE"}
  ${CHECK_WORKERS:+--check-workers "$CHECK_WORKERS"} ${LIFETIME:+--lifetime "$LIFETIME"})
if [ "$ROLE" = verifier ]; then
  REGISTRY=${REGISTRY:-$HOME/.local/state/remote-session/registry.json}
  mkdir -p "$(dirname "$REGISTRY")" || exit 3
  if [ -n "${CLOSE:-}" ]; then
    "$PY" verity/protocols/poub/verifier/remote_session/verifier_process.py --registry "$REGISTRY" --close "$CLOSE" \
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
  eval "VA=(${VERIFIER_ARGS:-})"
  # shellcheck disable=SC2086
  "$PY" verity/protocols/poub/verifier/remote_session/verifier_process.py --host 127.0.0.1 "${VERIFY[@]}" --registry "$REGISTRY" "${VA[@]}" \
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
if [ -n "$SERVED" ] && [ ${#LEASE[@]} -gt 0 ]; then
  # the cores at work while the lease is held: the prover's and, under ROLE=both, its verifier's, which holds no lease
  LEASE_CPUS=$PROVER_CPUS
  [ "$ROLE" = both ] && [ "$VERIFIER_CPUS" != "$PROVER_CPUS" ] && LEASE_CPUS+=,$VERIFIER_CPUS
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
  keep_row() { # the row's records under 1 MiB, into out/row
    [ -d "$SWEEP/$ROW" ] || return 0
    mkdir -p "$O/row" && (cd "$SWEEP/$ROW" && find . -type f -size -1024k -print0 | xargs -0 -r cp --parents -t "$O/row")
  }
  drop_connection() { # a verifier still waiting for its prover ends at a connection opened and dropped
    (exec 3<> "/dev/tcp/127.0.0.1/$PORT") 2> /dev/null
  }
  if [ "$SERVED" = row ]; then
    SWEEP=${SWEEP_DIR:-$B/sweep}
    mkdir -p "$SWEEP" || refuse "SWEEP_DIR=$SWEEP can't be made"
    export NATIVE_COLLECT_BUILD=${NATIVE_COLLECT_BUILD:-$B/nc_build} CONFIG_RUN=${CONFIG_RUN:-1}
    read -r CK REPO REV _ < <("$VV" epoch resolve "$ROW")
    [ -n "${REV:-}" ] || refuse "verity-vllm epoch resolve $ROW names no checkpoint, repository and revision"
    # shellcheck disable=SC2086
    ROWRUN=("$VV" row run "$ROW" "$CK" "$REPO" "$REV" --py "$VPY" --sweep-dir "$SWEEP" --cuda-bin "$CUDA_HOME/bin"
      ${ROW_ARGS:-})
    KV=()
    case ${KV_LIB:-build} in
      cpu) ;;
      build)
        KV_ARCH=${KV_ARCH:-$("$PY" -c 'import json, sys
major, minor = json.load(open(sys.argv[1]))["engine"]["target"]["compute_capability"]
print(f"sm_{major}{minor}")' "$ROW_FILE" 2> /dev/null)}
        [[ $KV_ARCH =~ ^sm_[0-9]+a?$ ]] || refuse "KV_ARCH=$KV_ARCH is no card's architecture (sm_90), and the row's target" \
          "declares no compute_capability to build kv.cu for: state KV_ARCH, or KV_LIB=cpu"
        nvcc -O3 -arch="$KV_ARCH" -std=c++17 -Xcompiler -fPIC -shared -DKV_FIELD=1024 -DKV_MASK=1 -o "$B/kv.so" \
          benchmarks/word_codec/kv.cu > "$O/kv.build.log" 2>&1 ||
          { tail -40 "$O/kv.build.log"; echo "run.sh: kv.cu's build for $KV_ARCH failed (KV_LIB=cpu encodes on the CPU)"; exit 4; }
        sha256sum "$B/kv.so" > "$O/libs.sha256"
        KV=(--remote-kv-lib "$B/kv.so") ;;
      *)
        [ -f "$KV_LIB" ] || refuse "KV_LIB=$KV_LIB is not build, cpu or a built kv.cu"
        KV=(--remote-kv-lib "$KV_LIB") ;;
    esac
    # KV_PAGES: the engine's KV pool, a count that is none of its cache's other dimensions (a page's slots, the
    # model's heads), which the store binder tells apart by size
    export EXTRA_COMMIT_ARGS="--remote-plan $PLAN --remote-kv-pages ${KV_PAGES:-24} --remote-weight-blocks ${WEIGHT_BLOCKS:-4096} ${KV[*]} ${EXTRA_COMMIT_ARGS:-}"
    LEASED=$STAGES
    if [ -z "${BUILD_GPU:-}" ] && [[ ,$STAGES, == *,build,* ]]; then
      LEASED=$(tr , '\n' <<< "$STAGES" | grep -vx build | paste -sd, -)
      CUDA_VISIBLE_DEVICES='' taskset -c "$PROVER_CPUS" "${ROWRUN[@]}" --stages build > "$O/build.log" 2>&1; BR=$?
      if [ $BR != 0 ]; then
        tail -20 "$O/build.log"
        keep_row
        echo "run.sh: the row's Build ended $BR with no GPU visible (BUILD_GPU=1 runs it under the lease)"
        exit $BR
      fi
    fi
    RUN=("${ROWRUN[@]}" --stages "$LEASED")
    REC=$SWEEP/$ROW/commit/remote_session.json
    rm -f "$REC"
  else
    RUN=("$VPY" $RS/serve.py --connect "127.0.0.1:$PORT" --out "$O" --plan "$PLAN" --codec "$CODEC")
    REC=$O/serve.json
  fi
  if [ "$ROLE" = both ]; then
    if [ -z "${LEAN_EXE:-}" ]; then
      LEAN_EXE=$("$PY" -c 'from verity.protocols.poub.verifier.session import lean_verifier as lv; print(lv.build())' \
        2> "$O/lean-build.log") ||
        { tail -20 "$O/lean-build.log"; echo "run.sh: the Lean verifier's build failed (LEAN_EXE names a built one)"; exit 4; }
    fi
    eval "VA=(${VERIFIER_ARGS:-})"
    set -m # a process group of its own: stopping it stops its Lean verifier and its check workers with it
    taskset -c "$VERIFIER_CPUS" "$PY" verity/protocols/poub/verifier/remote_session/verifier_process.py --host 127.0.0.1 \
      "${VERIFY[@]}" --lean-exe "$LEAN_EXE" "${VA[@]}" < /dev/null > "$O/verifier.log" 2>&1 &
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
  [ "$SERVED" = row ] && export REMOTE_VERIFIER=127.0.0.1:$PORT # the Commit's session; the Build before it has none
  P=0
  card_free || P=$?
  if [ $P = 0 ]; then
    "${LEASE[@]}" taskset -c "$PROVER_CPUS" "${RUN[@]}" > "$O/prover.log" 2>&1; P=$?
    tail -20 "$O/prover.log"
  fi
  [ "$SERVED" = row ] && keep_row
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
  exit $P
fi
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  if [ -n "$CPU" ]; then
    DEV=(--cpu "$CPU" --codec "$CODEC")
  else
    nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used,memory.total \
      --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
    FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
    nvcc "${FLAGS[@]}" -o "$B/session.so" $RS/session.cu > "$O/session.ptxas" 2>&1 & S=$!
    nvcc "${FLAGS[@]}" -DKV_FIELD=1024 -DKV_MASK=1 -o "$B/kv.so" benchmarks/word_codec/kv.cu > "$O/kv.ptxas" 2>&1 & K=$!
    wait $S || { tail -40 "$O/session.ptxas"; exit 4; }
    wait $K || { tail -40 "$O/kv.ptxas"; exit 4; }
    sha256sum "$B"/*.so > "$O/libs.sha256"
    DEV=(--lib "$B/session.so" --kv "$B/kv.so" --gib "$GIB")
  fi
  V=
  if [ "$ROLE" = both ]; then
    eval "VA=(${VERIFIER_ARGS:-})"
    # shellcheck disable=SC2086
    taskset -c "$VERIFIER_CPUS" "$PY" verity/protocols/poub/verifier/remote_session/verifier_process.py $VERIFY_ARGS "${VA[@]}" > "$O/verifier.log" 2>&1 & V=$!
  fi
  PIN=(taskset -c "$PROVER_CPUS")
  if [ -n "${PROVER_SLICE:-}" ]; then
    # sudo drops PATH, PYTHONPATH and LD_*, so they are set again
    PIN=(sudo -n --preserve-env env "PATH=$PATH" "PYTHONPATH=${PYTHONPATH:-}" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
      "HOME=$HOME" "USER=$(id -un)" "LOGNAME=$(id -un)" systemd-run --quiet --scope --collect --slice="$PROVER_SLICE"
      --nice=0 -p AllowedCPUs="$PROVER_CPUS" -p MemoryMax="${PROVER_MEM_GB:-300}G" -p MemorySwapMax=0 --
      setpriv --reuid="$(id -u)" --regid="$(id -g)" --init-groups -- "${PIN[@]}")
  fi
  "${PIN[@]}" "$PY" $RS/prover.py --connect "127.0.0.1:$PORT" --out "$O" --plan "$PLAN" \
    "${DEV[@]}" ${PROVER_ARGS:-} > "$O/prover.log" 2>&1
  P=$?
  W=0
  if [ -n "$V" ]; then
    if [ $P != 0 ]; then kill $V 2> /dev/null; fi
    wait $V; W=$?
    tail -20 "$O/verifier.log"
  fi
  tail -20 "$O/prover.log"
  [ $P = 0 ] && [ $W = 0 ]'
