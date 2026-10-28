#!/usr/bin/env bash
# Remote bandwidth sessions (cwd = the source tree): the verifier's process on cores of its own, the prover's on GPUS
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
# opens `ssh -R PORT` to PEER (the prover's machine, a name in /etc/vy/peers) and listens behind it, so the prover's
# connections to its own 127.0.0.1:PORT cross the network; then ROLE=prover on PEER, which connects there, and which
# takes its lease only once that tunnel listens (up to TUNNEL_WAIT_S, default 7200).
#
# Against the auditor's verifier host (`benchmarks/live_verifier/host.py`), the lane starts the run there with its
# gate key (its gate's `run --port PORT ...`) and ROLE=prover gets VERIFIER_SSH, ssh's `-i KEY prover@HOST`: the
# prover opens `ssh -L PORT` to the host's tunnel account, which forwards only to the run ports, and takes its lease once
# that account says the run's verifier listens.
#
#   ROLE=verifier PEER=PROVER_NODE PORT=7791 SESSIONS=2 taskset -c 96-127 bash benchmarks/remote_session/run.sh
#   ROLE=prover PORT=7791 PLAN=honest,skip:0.0625 taskset -c 48-95 bash benchmarks/remote_session/run.sh
#
# ROLE has no default: the verifier runs on a machine the prover doesn't control (AGENTS.md's auditor-trusted
# devices), never on PEER, so a run says which side it is. ROLE=both runs the verifier beside the prover on this
# machine, a test of these scripts that audits nothing.
# ROLE=verifier is the audit's verifier as a service. It keeps the registrations in REGISTRY (default
# ~/.local/state/remote-session/registry.json on its own machine, across its runs: a node with a live registration there
# is refused), and leaves out/verifier.json, out/registry.json (REGISTRY when it ends) and out/address.json, the address
# the prover connects to: PEER's 127.0.0.1:PORT, a Commit's REMOTE_VERIFIER. VERIFIER_ARGS go to verifier.py as shell
# words (quotes group one, as `--challenger-ssh '-i KEY HOST'` needs), under ROLE=both too: with `--challenger serve
# --challenger-key DIR` each session's coins come from the challenger's own process,
# `benchmarks/live_challenger/serve.py` on this machine (`challengers.py`), its out/ under out/challenger; with `--challenger gate --challenger-ssh WORDS --challenger-peer NAME --challenger-pin SHA512
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
# SERVE makes the prover vLLM serving under the audit, on one leased GPU in vLLM's venv (VLLM_PY, default
# /workspace/jobs/venv312/bin/python, its `verity-vllm` launcher beside it), this tree's verity_vllm first on its path,
# against ROLE=verifier with SESSIONS=1 (one session a store). SERVE=tiny runs `serve.py --connect` (the vLLM tests'
# tiny Llama, PLAN serve or servedrop:K). SERVE=row runs row ROW, `verity-vllm row run ROW CHECKPOINT REPO REV` with
# the checkpoint `verity-vllm epoch resolve ROW` names, ROW_ARGS (such as --stages commit) and REMOTE_VERIFIER at the
# tunnel; its Commit puts the store binder on its engine and proves the store in lockstep with its first instrumented
# run, a pass an engine step (`verity_vllm.protocols.remote_bandwidth.remote.CommitSession`: one GPU, synchronous scheduling declared in the
# row's target, its store the weights and --remote-filler-gib of filler, the KV pool outside it). Serving int8
# Llama-3.1-8B:
#
#   ROLE=verifier PEER=PROVER_NODE PORT=7797 SESSIONS=1 WINDOWS=7 WINDOW_MS=100 LIFETIME=4 \
#     bash benchmarks/remote_session/run.sh                                           # on the verifier's machine
#   ROLE=prover PORT=7797 SERVE=row ROW=<the int8 Llama-3.1-8B row> bash benchmarks/remote_session/run.sh   # one GPU
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
case ${ROLE:-} in
  verifier | prover | both) export ROLE ;;
  *) echo "ROLE is verifier or prover (both only to test these scripts on one machine), not '${ROLE:-}': the verifier" \
       "runs on a machine the prover doesn't control, so there is no default"; exit 3 ;;
esac
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
export PY=${PY:-$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python}
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
[ -x "$HOME/.elan/bin/lake" ] && export PATH="$HOME/.elan/bin:$PATH"
B=$(mktemp -d /tmp/remote-session.XXXXXX); trap 'rm -rf "$B"' EXIT
export B RS=benchmarks/remote_session PLAN=${PLAN:-honest} GIB=${GIB:-16} WINDOWS=${WINDOWS:-7}
export WINDOW_MS=${WINDOW_MS:-1000} SAMPLES=${SAMPLES:-2773} JITTER_MS=${JITTER_MS:-2}
export VERIFIER_CPUS=${VERIFIER_CPUS:-88-95} PROVER_CPUS=${PROVER_CPUS:-48-87} PORT=${PORT:-$((7700 + RANDOM % 200))}
export SESSIONS=${SESSIONS:-$(($(tr -cd , <<< "$PLAN" | wc -c) + 1))}
VERIFY=(--port "$PORT" --out "$O" --windows "$WINDOWS" --window-ms "$WINDOW_MS" --samples "$SAMPLES"
  --jitter-ms "$JITTER_MS" --sessions "$SESSIONS" ${QUANTILE:+--quantile "$QUANTILE"}
  ${CHECK_WORKERS:+--check-workers "$CHECK_WORKERS"} ${LIFETIME:+--lifetime "$LIFETIME"})
if [ "$ROLE" = verifier ]; then
  REGISTRY=${REGISTRY:-$HOME/.local/state/remote-session/registry.json}
  mkdir -p "$(dirname "$REGISTRY")" || exit 3
  if [ -n "${CLOSE:-}" ]; then
    "$PY" $RS/verifier.py --registry "$REGISTRY" --close "$CLOSE" \
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
  # interactive session's packets on a 20 ms clock (tunnel_rtt.py measures each).
  ssh -tt -o ObscureKeystrokeTiming=no -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
    -R "127.0.0.1:$PORT:127.0.0.1:$PORT" "${TO[@]}" "sleep 86400" < /dev/null > "$O/tunnel.log" 2>&1 & T=$!
  eval "VA=(${VERIFIER_ARGS:-})"
  # shellcheck disable=SC2086
  "$PY" $RS/verifier.py --host 127.0.0.1 "${VERIFY[@]}" --registry "$REGISTRY" "${VA[@]}" \
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
LEASE=(gpu-lease "${GPUS:-1}" --wait ${GPU:+--on "$GPU"} ${TIMED:+--timed} --max-min "${MAXMIN:-30}" --)
[ -n "${GPU_LEASE_UUID:-}" ] || [ -n "${CPU:-}" ] && LEASE=()
export CPU=${CPU:-} CODEC=${CODEC:-p512/k2}
if [ -n "${SERVE:-}" ]; then
  VPY=${VLLM_PY:-/workspace/jobs/venv312/bin/python}
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=${HF_HOME:-/workspace/hf} PATH=$(dirname "$VPY"):$PATH
  export PYTHONPATH=$PWD/integrations/vllm:$PWD${PYTHONPATH:+:$PYTHONPATH}
  case $SERVE in
    tiny) RUN=("$VPY" $RS/serve.py --connect "127.0.0.1:$PORT" --out "$O" --plan "$PLAN" --codec "$CODEC") ;;
    row)
      VV=$(dirname "$VPY")/verity-vllm
      read -r CK REPO REV _ < <("$VV" epoch resolve "${ROW:?}")
      # shellcheck disable=SC2086
      RUN=(env "REMOTE_VERIFIER=127.0.0.1:$PORT" "$VV" row run "$ROW" "${CK:?}" "$REPO" "$REV" ${ROW_ARGS:-}) ;;
    node)
      export VPY
      RUN=(bash $RS/node.sh) ;;
    *) echo "SERVE is tiny, row or node, not $SERVE"; exit 3 ;;
  esac
  "${LEASE[@]}" "${RUN[@]}" > "$O/prover.log" 2>&1; P=$?
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
    taskset -c "$VERIFIER_CPUS" "$PY" $RS/verifier.py $VERIFY_ARGS "${VA[@]}" > "$O/verifier.log" 2>&1 & V=$!
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
