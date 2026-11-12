#!/usr/bin/env bash
# CIR-261: vLLM d9105ea80's fp8_ds_mla read-back held to Fp8DsMlaUpconvertRow_v1, on ONE card of node A under research run,
# from the shipped tree's root (a tree holding adapters/vllm/tests/properties/fp8_read_back_capture_gpu.py):
#   research run --on vy-spot-glm53 --project verity --cwd source --timeout 50m --mem-gb 32 --send fp8_read_back.sh \
#     --env CARD=3 --env CORES=34-43 --env NOT_BEFORE=2026-11-12T17:30:00Z --env END_UTC=2026-11-12T20:55:00Z \
#     -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/fp8_read_back.sh'
# Before anything touches CARD it exits 76 when nvidia-smi can't be read, a compute process is on CARD, CARD holds 1024 MiB
# or more, or CARD's lease is held; 75 before NOT_BEFORE or with fewer than MAX_MIN minutes left before END_UTC. The capture
# then runs under the tree's gpu_lease.sh on CARD alone for at most MAX_MIN (30) minutes, pinned to CORES; the compare runs
# after the lease, on CORES, without the card. Exit: the larger of the two phases' codes.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=${OUT:-$D/out}; mkdir -p "$OUT"
me=fp8-read-back
CARD=${CARD:?CARD: the index of the one card}
CORES=${CORES:?CORES: the cores of the row}
MAX_MIN=${MAX_MIN:-30}
[ "$MAX_MIN" -le 30 ] || { echo "$me: MAX_MIN=$MAX_MIN: a lease is at most 30 minutes"; exit 2; }
T=adapters/vllm/tests/properties/fp8_read_back_capture_gpu.py
[ -f "$T" ] || { echo "$me: run from the root of a tree holding $T"; exit 2; }
now=$(date -u +%s)
echo "$me: start $(date -u +%FT%TZ) on $(hostname)"
if [ -n "${NOT_BEFORE:-}" ]; then
  nb=$(date -u -d "$NOT_BEFORE" +%s) || { echo "$me: NOT_BEFORE is an ISO time, not $NOT_BEFORE"; exit 2; }
  [ "$now" -ge "$nb" ] || { echo "$me: $(date -u +%FT%TZ) is before NOT_BEFORE=$NOT_BEFORE: nothing done"; exit 75; }
fi
if [ -n "${END_UTC:-}" ]; then
  end=$(date -u -d "$END_UTC" +%s) || { echo "$me: END_UTC is an ISO time, not $END_UTC"; exit 2; }
  [ $(( end - now )) -ge $(( MAX_MIN * 60 )) ] || { echo "$me: fewer than $MAX_MIN minutes before END_UTC=$END_UTC: nothing done"; exit 75; }
fi
[ -d /workspace/venv312 ] && venv=/workspace/venv312 || venv=/workspace/cp/venv312
VENV=${VENV:-$venv}; PY=$VENV/bin/python
v=$("$PY" -c 'import vllm; print(vllm.__version__)' 2> /dev/null) || { echo "$me: $PY imports no vLLM"; exit 2; }
case $v in *+gd9105ea8*) ;; *) echo "$me: vLLM $v is not the pin d9105ea80"; exit 2 ;; esac
echo "$me: vLLM $v"
CUDA=${SERVE_CUDA:-/usr/local/cuda-12.9}

used=$(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null) \
  && uuid=$(nvidia-smi -i "$CARD" --query-gpu=uuid --format=csv,noheader 2> /dev/null) \
  && apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2> /dev/null) \
  || { echo "$me: nvidia-smi failed, so card $CARD is unread: not starting"; exit 76; }
[ -n "$uuid" ] || { echo "$me: no card $CARD"; exit 76; }
[ "$(echo "$apps" | grep -c "$uuid")" = 0 ] || { echo "$me: a compute process is on card $CARD: not starting"; exit 76; }
[ "${used:-0}" -lt 1024 ] || { echo "$me: card $CARD holds $used MiB: not starting"; exit 76; }
LEASE=$PWD/tools/research/src/research/pods/sh/gpu_lease.sh
[ -f "$LEASE" ] || { echo "$me: no gpu_lease.sh at $LEASE"; exit 2; }
lease=$(LC_ALL=C bash "$LEASE" status 2>&1) || { echo "$me: gpu-lease status failed, so the lease is unread: not starting"; exit 76; }
echo "$lease" | grep -Eqx "$CARD free" || { echo "$me: card $CARD's lease is not free: not starting"; echo "$lease" | grep -E "^$CARD " | cut -c1-200; exit 76; }
echo "$me: card $CARD idle ($used MiB, no compute process, lease free) at $(date -u +%FT%TZ)"

export CUDA_HOME=$CUDA PATH="$CUDA/bin:$VENV/bin:$PATH"
unset CPATH LIBRARY_PATH NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM VERITY_DSA_TOPK VERITY_DSA_TOPK_LOG VLLM_PLUGINS
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0
ldconfig
export PYTHONPATH="$PWD:$PWD/adapters/vllm${PYTHONPATH:+:$PYTHONPATH}"
scratch=$(mktemp -d); trap 'rm -rf "$scratch"' EXIT
export TRITON_CACHE_DIR=$scratch/triton TMPDIR=$scratch
echo "$me: capture under the lease at $(date -u +%FT%TZ)"
GPU_LEASE_WHO=circuits-fp8readback bash "$LEASE" 1 --on "$CARD" --max-min "$MAX_MIN" --mem-gb "${MEM_GB:-32}" -- \
  taskset -c "$CORES" "$PY" "$T" --phase capture --out "$OUT"
rc1=$?
echo "$me: capture exit $rc1 at $(date -u +%FT%TZ); card $CARD now holds $(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null) MiB"
WHEEL_SO=6acb4075572bf71c19db6b07c654ed8f488be835f85461baae7ae2f80fa1ac42
taskset -c "$CORES" "$PY" "$T" --phase compare --out "$OUT" --library-sha256 "${LIBRARY_SHA256:-$WHEEL_SO}"
rc2=$?
echo "$me: compare exit $rc2 at $(date -u +%FT%TZ)"
exit $(( rc1 > rc2 ? rc1 : rc2 ))
