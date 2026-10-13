#!/usr/bin/env bash
# Remote bandwidth sessions on one GPU (cwd = the source tree): the verifier's process on cores of its own, the prover's
# on the GPU, over TCP. Everything runs under one gpu-lease 1 --wait: build session.cu and kv.cu (p1024, masked), start
# the verifier, then the prover, which gates its device against the driver before anything is timed. Results go to
# $RESEARCH_RUN_DIR/out (verifier.json, prover.json and their logs).
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --mem-gb 96 \
#     --campaign resource-accounting --declared-output 'out/*' --env GPU_LEASE_WHO=ra-first-session \
#     --env PLAN=honest --env GIB=16 -- taskset -c 48-95 bash benchmarks/remote_session/run.sh
#
# PLAN is the prover's sessions (honest, skip:F; comma-separated, one session each); GIB the store; WINDOWS, WINDOW_MS,
# SAMPLES, JITTER_MS the sessions'; VERIFIER_CPUS and PROVER_CPUS the cores each process is pinned to.
#
# On two machines, two runs with the same PORT and SESSIONS: ROLE=verifier on the verifier's machine (no GPU), which
# opens `ssh -R PORT` to PEER (the prover's machine, a name in /etc/vy/peers) and listens behind it, so the prover's
# connections to its own 127.0.0.1:PORT cross the network; then ROLE=prover on PEER, which connects there.
#
#   research run --on vy-nebius-1 ... --env ROLE=verifier --env PEER=vy-nebius-2 --env PORT=7791 --env SESSIONS=2 \
#     -- taskset -c 64-127 bash benchmarks/remote_session/run.sh
#   research run --on vy-nebius-2 ... --env ROLE=prover --env PORT=7791 --env PLAN=honest,skip:0.0625 \
#     -- taskset -c 48-95 bash benchmarks/remote_session/run.sh
set -uxo pipefail
export O=${RESEARCH_RUN_DIR:?}/out; mkdir -p "$O"
ENV_DIR=${ENV_DIR:-/workspace/pouw/fill-out/pous/work/pous-bench}
export PY=$ENV_DIR/gpu312-torch2.14.0-cu130/bin/python
[ -x "$PY" ] || { echo "no torch venv at $PY"; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} PATH=/usr/local/cuda-13.0/bin:$(dirname "$PY"):$PATH
B=$(mktemp -d /tmp/remote-session.XXXXXX); trap 'rm -rf "$B"' EXIT
export B RS=benchmarks/remote_session PLAN=${PLAN:-honest} GIB=${GIB:-16} WINDOWS=${WINDOWS:-7}
export WINDOW_MS=${WINDOW_MS:-1000} SAMPLES=${SAMPLES:-2773} JITTER_MS=${JITTER_MS:-2}
export VERIFIER_CPUS=${VERIFIER_CPUS:-88-95} PROVER_CPUS=${PROVER_CPUS:-48-87} PORT=${PORT:-$((7700 + RANDOM % 200))}
export SESSIONS=${SESSIONS:-$(($(tr -cd , <<< "$PLAN" | wc -c) + 1))} ROLE=${ROLE:-both}
VERIFY=(--port "$PORT" --out "$O" --windows "$WINDOWS" --window-ms "$WINDOW_MS" --samples "$SAMPLES"
  --jitter-ms "$JITTER_MS" --sessions "$SESSIONS" ${QUANTILE:+--quantile "$QUANTILE"})
if [ "$ROLE" = verifier ]; then
  read -ra TO <<< "$(grep "^${PEER:?} " /etc/vy/peers | cut -d' ' -f2-)"
  [ ${#TO[@]} -gt 0 ] || { echo "no peer $PEER in /etc/vy/peers"; exit 3; }
  ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -N \
    -R "127.0.0.1:$PORT:127.0.0.1:$PORT" "${TO[@]}" > "$O/tunnel.log" 2>&1 & T=$!
  "$PY" $RS/verifier.py --host 127.0.0.1 "${VERIFY[@]}" > "$O/verifier.log" 2>&1 & V=$!
  while kill -0 $V 2> /dev/null; do
    kill -0 $T 2> /dev/null || { echo "the tunnel to $PEER ended"; cat "$O/tunnel.log"; kill $V; break; }
    wait -n $V $T
  done
  wait $V; W=$?
  kill $T 2> /dev/null
  tail -20 "$O/verifier.log"
  exit $W
fi
export VERIFY_ARGS="${VERIFY[*]}"
LEASE=(gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${MAXMIN:-30}" --)
[ -n "${GPU_LEASE_UUID:-}" ] && LEASE=()
# shellcheck disable=SC2016
"${LEASE[@]}" bash -c '
  nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used,memory.total \
    --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
  nvcc "${FLAGS[@]}" -o "$B/session.so" $RS/session.cu > "$O/session.ptxas" 2>&1 & S=$!
  nvcc "${FLAGS[@]}" -DKV_FIELD=1024 -DKV_MASK=1 -o "$B/kv.so" benchmarks/word_codec/kv.cu > "$O/kv.ptxas" 2>&1 & K=$!
  wait $S || { tail -40 "$O/session.ptxas"; exit 4; }
  wait $K || { tail -40 "$O/kv.ptxas"; exit 4; }
  sha256sum "$B"/*.so > "$O/libs.sha256"
  V=
  if [ "$ROLE" = both ]; then
    # shellcheck disable=SC2086
    taskset -c "$VERIFIER_CPUS" "$PY" $RS/verifier.py $VERIFY_ARGS > "$O/verifier.log" 2>&1 & V=$!
  fi
  taskset -c "$PROVER_CPUS" "$PY" $RS/prover.py --connect "127.0.0.1:$PORT" --out "$O" --plan "$PLAN" \
    --lib "$B/session.so" --kv "$B/kv.so" --gib "$GIB" ${PROVER_ARGS:-} > "$O/prover.log" 2>&1
  P=$?
  W=0
  if [ -n "$V" ]; then
    if [ $P != 0 ]; then kill $V 2> /dev/null; fi
    wait $V; W=$?
    tail -20 "$O/verifier.log"
  fi
  tail -20 "$O/prover.log"
  [ $P = 0 ] && [ $W = 0 ]'
