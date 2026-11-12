#!/usr/bin/env bash
# circuits-accept (CIR-264): the sampled accept's capture on ONE card of node A under research run, from the shipped tree's
# root (a tree holding adapters/vllm/tests/properties/sampled_accept_capture_gpu.py):
#   research run --on vy-spot-glm53 --project verity --source TREE --cwd source --timeout 30m --mem-gb 16 \
#     --send accept_capture.sh --env CARD=0 --env CORES=0-9 --env NOT_BEFORE=ISO --env END_UTC=ISO [--env MAX_MIN=25] \
#     [--env ARGS='--random 200'] -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/accept_capture.sh'
# It loads no model and no checkpoint: the pin's rejection_sample kernels on hand-built and random logits rows, one request
# of 1 + 5 rows a launch, compared in the job with the tree's reference. Nothing of it is private.
# CARD: the one card, by index. Before anything touches it (the node's calendar, rule 4) it exits 76 when nvidia-smi can't
# be read, a compute process is on CARD, CARD holds CARD_FREE_MIB (1024) MiB or more, or CARD's lease is held. The capture
# then runs under the tree's gpu_lease.sh on CARD alone for at most MAX_MIN (25, never over 30) minutes, without waiting: a
# card taken meanwhile refuses the run (75). NOT_BEFORE: before it, exit 75 having done nothing. END_UTC: start only while
# MAX_MIN minutes remain before it, else exit 75.
# CORES: the lease declares them (--cores: a lease that declares none counts as every CPU of the node against its CPU
# quota, and the other lanes' leases are refused beside it), the capture is pinned to them (taskset), and so is this script.
# A lease refused at the quota exits 75 too: the lease's own lines say each live lease's CPUs, and this prints its status
# after them (who holds what). That is not retried in a loop.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=${OUT:-$D/out}; mkdir -p "$OUT"
me=accept-capture
CARD=${CARD:?CARD: the index of the one card}
CORES=${CORES:?CORES: the cores of the row, e.g. 0-9}
MAX_MIN=${MAX_MIN:-25}
[ "$MAX_MIN" -le 30 ] || { echo "$me: a lease is at most 30 minutes, not MAX_MIN=$MAX_MIN"; exit 2; }
T=adapters/vllm/tests/properties/sampled_accept_capture_gpu.py
[ -f "$T" ] || { echo "$me: run from the root of a tree holding $T"; exit 2; }
taskset -cp "$CORES" $$ > /dev/null || { echo "$me: cannot pin to cores $CORES"; exit 2; }
now=$(date -u +%s)
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
WANT=${VLLM_WANT:-+gd9105ea8}
case $v in *"$WANT"*) ;; *) echo "$me: vLLM $v is not the build asked for (VLLM_WANT=$WANT)"; exit 2 ;; esac
echo "$me: vLLM $v, $PY, cores $CORES, tree $(git rev-parse HEAD 2> /dev/null || basename "$PWD")"

used=$(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null) \
  && uuid=$(nvidia-smi -i "$CARD" --query-gpu=uuid --format=csv,noheader 2> /dev/null) \
  && apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2> /dev/null) \
  || { echo "$me: nvidia-smi failed, so card $CARD is unread: not starting"; exit 76; }
[ -n "$uuid" ] || { echo "$me: no card $CARD"; exit 76; }
[ "$(echo "$apps" | grep -c "$uuid")" = 0 ] || { echo "$me: a compute process is on card $CARD: not starting"; exit 76; }
[ "${used:-0}" -lt "${CARD_FREE_MIB:-1024}" ] || { echo "$me: card $CARD holds $used MiB: not starting"; exit 76; }
LEASE=${GPU_LEASE_SH:-$PWD/tools/research/src/research/pods/sh/gpu_lease.sh}
[ -f "$LEASE" ] || { echo "$me: no gpu_lease.sh at $LEASE"; exit 2; }
mkdir -p "${GPU_LEASE_DIR:-/run/gpu-lease}" 2> /dev/null
lease=$(LC_ALL=C bash "$LEASE" status 2>&1) || { echo "$me: gpu-lease status failed, so the lease is unread: not starting"; exit 76; }
echo "$lease" | grep -Eqx "$CARD free" || { echo "$me: card $CARD's lease is not free: not starting"; echo "$lease" | grep -E "^$CARD " | cut -c1-200; exit 76; }
echo "$me: card $CARD idle ($used MiB, no compute process, lease free) at $(date -u +%FT%TZ)"

export PATH="$VENV/bin:$PATH" PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
unset VLLM_PLUGINS
S=$(mktemp -d)
# shellcheck disable=SC2086
GPU_LEASE_WHO=circuits-accept bash "$LEASE" 1 --on "$CARD" --cores "$CORES" --max-min "$MAX_MIN" --mem-gb "${MEM_GB:-16}" -- \
  taskset -c "$CORES" "$PY" "$T" --out "$OUT/capture" --scratch "$S" ${ARGS:-}
rc=$?
if [ "$rc" = 75 ]; then
  echo "$me: the lease was refused (75), holding nothing; who holds what at $(date -u +%FT%TZ):"
  LC_ALL=C bash "$LEASE" status 2>&1 | cut -c1-240
fi
echo "$me: exit $rc at $(date -u +%FT%TZ); card $CARD now holds $(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null) MiB"
rm -rf "$S"
exit $rc
