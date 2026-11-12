#!/usr/bin/env bash
# circuits-fp8write (CIR-260): the fp8_ds_mla cache-write capture on ONE card of node A under research run, from the shipped
# tree's root:
#   research run --on vy-spot-glm53 --project verity --source TREE --cwd source --timeout 50m --mem-gb 32 \
#     --send ds_mla_capture.sh --env CARD=2 --env CORES=24-33 --env NOT_BEFORE=ISO --env END_UTC=ISO \
#     -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/ds_mla_capture.sh'
# Before NOT_BEFORE, or with fewer than MAX_MIN minutes left before END_UTC, it exits 75 having done nothing. Before anything
# touches the card it exits 76 when nvidia-smi can't be read, a compute process is on CARD, CARD holds CARD_FREE_MIB (1024)
# MiB or more, or CARD's lease is held. The GPU phase runs under gpu_lease.sh on CARD alone for at most MAX_MIN (30) minutes,
# without waiting; the compare phase then runs on CORES with no card. Both phases are pinned to CORES.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=${OUT:-$D/out}; mkdir -p "$OUT"
me=ds-mla-capture
CARD=${CARD:?CARD: the index of the one card}
CORES=${CORES:?CORES: the cores of the row}
MAX_MIN=${MAX_MIN:-30}
T=adapters/vllm/tests/properties/glm53_ds_mla_write_capture_gpu.py
[ -f "$T" ] || { echo "$me: run from the root of a tree holding $T"; exit 2; }
now=$(date -u +%s)
nb=$(date -u -d "${NOT_BEFORE:?NOT_BEFORE: when the row starts}" +%s) || { echo "$me: NOT_BEFORE is an ISO time"; exit 2; }
end=$(date -u -d "${END_UTC:?END_UTC: when the card must be free}" +%s) || { echo "$me: END_UTC is an ISO time"; exit 2; }
[ "$now" -ge "$nb" ] || { echo "$me: $(date -u +%FT%TZ) is before NOT_BEFORE=$NOT_BEFORE: nothing done"; exit 75; }
[ $(( end - now )) -ge $(( MAX_MIN * 60 )) ] || { echo "$me: fewer than $MAX_MIN minutes before END_UTC=$END_UTC: nothing done"; exit 75; }
VENV=${VENV:-/workspace/venv312}; PY=$VENV/bin/python
v=$("$PY" -c 'import vllm; print(vllm.__version__)' 2> /dev/null) || { echo "$me: $PY imports no vLLM"; exit 2; }
case $v in *+gd9105ea8*) ;; *) echo "$me: vLLM $v is not the pin's build (+gd9105ea8)"; exit 2 ;; esac
echo "$me: vLLM $v"

used=$(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null) \
  && uuid=$(nvidia-smi -i "$CARD" --query-gpu=uuid --format=csv,noheader 2> /dev/null) \
  && apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2> /dev/null) \
  || { echo "$me: nvidia-smi failed, so card $CARD is unread: not starting"; exit 76; }
[ -n "$uuid" ] || { echo "$me: no card $CARD"; exit 76; }
[ "$(echo "$apps" | grep -c "$uuid")" = 0 ] || { echo "$me: a compute process is on card $CARD: not starting"; exit 76; }
[ "${used:-0}" -lt "${CARD_FREE_MIB:-1024}" ] || { echo "$me: card $CARD holds $used MiB: not starting"; exit 76; }
LEASE=$PWD/tools/research/src/research/pods/sh/gpu_lease.sh
[ -f "$LEASE" ] || { echo "$me: no gpu_lease.sh at $LEASE"; exit 2; }
lease=$(LC_ALL=C bash "$LEASE" status 2>&1) || { echo "$me: gpu-lease status failed, so the lease is unread: not starting"; exit 76; }
echo "$lease" | grep -Eqx "$CARD free" || { echo "$me: card $CARD's lease is not free: not starting"; echo "$lease" | grep -E "^$CARD " | cut -c1-200; exit 76; }
echo "$me: card $CARD idle ($used MiB, no compute process, lease free) at $(date -u +%FT%TZ)"

TREE=$PWD
T=$TREE/$T
export PYTHONPATH=$TREE/adapters/vllm:$TREE:$TREE/tools/research/src PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset CPATH LIBRARY_PATH VLLM_USE_DEEP_GEMM VERITY_DSA_TOPK VLLM_PLUGINS
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
cd "$TREE/adapters/vllm" || exit 2
echo "$me: gpu phase asks for card $CARD at $(date -u +%FT%TZ)"
GPU_LEASE_WHO=circuits-fp8write bash "$LEASE" 1 --on "$CARD" --cores "$CORES" --max-min "$MAX_MIN" --mem-gb "${MEM_GB:-32}" -- \
  taskset -c "$CORES" "$PY" "$T" --phase gpu --out "$OUT" --scratch "$scratch"
g=$?
echo "$me: gpu phase exit $g at $(date -u +%FT%TZ); card $CARD now holds $(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null) MiB"
[ -f "$OUT/gpu.json" ] || exit "$g"
taskset -c "$CORES" "$PY" "$T" --phase compare --out "$OUT" --workers "$(taskset -c "$CORES" nproc)"
c=$?
echo "$me: compare phase exit $c at $(date -u +%FT%TZ)"
[ "$g" = 0 ] && exit "$c"
exit "$g"
