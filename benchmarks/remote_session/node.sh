#!/usr/bin/env bash
# run.sh's SERVE=node, inside its lease (cwd = the source tree): one vLLM engine (engine.py, in VPY's venv) on each
# leased GPU, refused before any starts unless CUDA_VISIBLE_DEVICES names exactly the GPUs the lease holds (lease.py),
# then the prover (node.py, in PY's) over their weights arenas, while session.cu and kv.cu build; with
# ROLE=both (a smoke test, not the audit: the verifier must be another machine) the verifier beside them. The
# engines' handshake is in $B; their logs and step records, and node.py's, go to $O. Every engine is stopped by PID
# when the prover ends.
#
# MODEL the checkpoint (default unsloth/Llama-3.1-8B, bf16), ENGINE_ARGS engine.py's (such as --step-hz 2
# --concurrency 64; Qwen/Qwen2.5-32B's 61 GiB of weights take --arena-gib 62 --util 0.8), ENGINE_CPUS the engines'
# cores (default 0-47), NODE_ARGS node.py's (such as --decode-s 60).
set -uxo pipefail
: "${O:?}" "${B:?}" "${RS:?}" "${PY:?}" "${VPY:?}" "${PORT:?}" "${PLAN:?}"
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:?}"
"$PY" $RS/lease.py "${#DEVS[@]}" || exit 3
nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader -i "${GPU_LEASE_UUID:-$CUDA_VISIBLE_DEVICES}" >> "$O/gpu.txt" 2>&1
FLAGS=(-O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
nvcc "${FLAGS[@]}" -o "$B/session.so" $RS/session.cu > "$O/session.ptxas" 2>&1 & S=$!
nvcc "${FLAGS[@]}" -DKV_FIELD=1024 -DKV_MASK=1 -o "$B/kv.so" benchmarks/word_codec/kv.cu > "$O/kv.ptxas" 2>&1 & K=$!
MODEL=${MODEL:-unsloth/Llama-3.1-8B}
# The cache holds snapshots without refs, which an offline lookup by name doesn't resolve.
if [ ! -d "$MODEL" ]; then
  SNAP=$(ls -d "${HF_HOME:-/workspace/hf}/hub/models--${MODEL//\//--}/snapshots/"*/ 2> /dev/null | head -1)
  [ -n "$SNAP" ] && MODEL=$SNAP
fi
PIDS=()
for g in "${!DEVS[@]}"; do
  # shellcheck disable=SC2086
  env -u PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES="${DEVS[$g]}" VLLM_ENABLE_V1_MULTIPROCESSING=0 OMP_NUM_THREADS=4 \
    taskset -c "${ENGINE_CPUS:-0-47}" "$VPY" $RS/engine.py --gpu "$g" --dir "$B" --out "$O" --kv "$B/kv.so" \
    --model "$MODEL" ${ENGINE_ARGS:-} > "$O/engine.$g.log" 2>&1 &
  PIDS+=($!)
  echo $! > "$B/engine.$g.pid"
done
stop() {
  for p in "${PIDS[@]}"; do kill "$p" 2> /dev/null; done
}
wait $S || { tail -40 "$O/session.ptxas"; stop; exit 4; }
wait $K || { tail -40 "$O/kv.ptxas"; stop; exit 4; }
sha256sum "$B"/*.so > "$O/libs.sha256"
V=
if [ "${ROLE:-both}" = both ]; then
  # shellcheck disable=SC2086
  taskset -c "$VERIFIER_CPUS" "$PY" $RS/verifier.py $VERIFY_ARGS ${VERIFIER_ARGS:-} > "$O/verifier.log" 2>&1 & V=$!
fi
# shellcheck disable=SC2086
taskset -c "$PROVER_CPUS" "$PY" $RS/node.py --connect "127.0.0.1:$PORT" --out "$O" --plan "$PLAN" \
  --engines-dir "$B" --engines "${#DEVS[@]}" --lib "$B/session.so" --kv "$B/kv.so" ${NODE_ARGS:-} \
  > "$O/node.log" 2>&1
P=$?
if [ -n "$V" ]; then
  [ $P = 0 ] || kill $V 2> /dev/null
  wait $V; W=$?
  [ $P = 0 ] && [ $W != 0 ] && P=5
  tail -20 "$O/verifier.log"
fi
for p in "${PIDS[@]}"; do
  for ((k = 0; k < 30; k++)); do kill -0 "$p" 2> /dev/null || break; sleep 1; done
done
stop
tail -30 "$O/node.log"
for g in "${!DEVS[@]}"; do tail -3 "$O/engine.$g.log"; done
exit $P
