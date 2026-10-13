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
export SESSIONS=$(($(tr -cd , <<< "$PLAN" | wc -c) + 1))
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
  taskset -c "$VERIFIER_CPUS" "$PY" $RS/verifier.py --port "$PORT" --out "$O" --windows "$WINDOWS" \
    --window-ms "$WINDOW_MS" --samples "$SAMPLES" --jitter-ms "$JITTER_MS" --sessions "$SESSIONS" > "$O/verifier.log" 2>&1 &
  V=$!
  taskset -c "$PROVER_CPUS" "$PY" $RS/prover.py --connect "127.0.0.1:$PORT" --out "$O" --plan "$PLAN" \
    --lib "$B/session.so" --kv "$B/kv.so" --gib "$GIB" > "$O/prover.log" 2>&1
  P=$?
  if [ $P != 0 ]; then kill $V 2> /dev/null; fi
  wait $V; W=$?
  tail -20 "$O/prover.log"; tail -20 "$O/verifier.log"
  [ $P = 0 ] && [ $W = 0 ]'
