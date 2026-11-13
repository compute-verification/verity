#!/usr/bin/env bash
# circuits FP8READ: the capture of FlashMLA's V32 sparse FP8 read on ONE card of node A under research run, from the shipped
# tree's root (a tree holding adapters/vllm/tests/properties/glm53_flashmla_fp8_capture_gpu.py):
#   research run --on vy-spot-glm53 --project verity --cwd source --timeout 80m --mem-gb 32 --send fp8read_card.sh \
#     --env CARD=N --env CORES=A-B [--env NOT_BEFORE=ISO] [--env END_UTC=ISO] [--env ARGS=--quick] \
#     -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/fp8read_card.sh'
# It reads no checkpoint: random operands at GLM-5.3's shapes. Three stages, all pinned to CORES: the library's SASS (no
# GPU), the launches under one gpu-lease of at most MAX_MIN (25) minutes on CARD alone, the comparison (no GPU).
# Before anything touches CARD (the node's calendar, rule 4) it exits 76 when nvidia-smi can't be read, a compute process
# is on CARD, CARD holds CARD_FREE_MIB (1024) MiB or more, or CARD's lease is held. NOT_BEFORE: before it, exit 75 having
# done nothing. END_UTC: start only while MAX_MIN + CPU_MIN (35) minutes remain before it, else exit 75.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=${OUT:-$D/out}; mkdir -p "$OUT"
me=fp8read
CARD=${CARD:?CARD: the index of the one card}
CORES=${CORES:?CORES: the cores of the calendar row}
MAX_MIN=${MAX_MIN:-25}; CPU_MIN=${CPU_MIN:-35}
M=tests.properties.glm53_flashmla_fp8_capture_gpu
[ -f adapters/vllm/tests/properties/glm53_flashmla_fp8_capture_gpu.py ] || { echo "$me: run from the root of a tree holding the capture"; exit 2; }
now=$(date -u +%s)
if [ -n "${NOT_BEFORE:-}" ]; then
  nb=$(date -u -d "$NOT_BEFORE" +%s) || { echo "$me: NOT_BEFORE is an ISO time, not $NOT_BEFORE"; exit 2; }
  [ "$now" -ge "$nb" ] || { echo "$me: $(date -u +%FT%TZ) is before NOT_BEFORE=$NOT_BEFORE: nothing done"; exit 75; }
fi
if [ -n "${END_UTC:-}" ]; then
  end=$(date -u -d "$END_UTC" +%s) || { echo "$me: END_UTC is an ISO time, not $END_UTC"; exit 2; }
  [ $(( end - now )) -ge $(( (MAX_MIN + CPU_MIN) * 60 )) ] || { echo "$me: fewer than $((MAX_MIN + CPU_MIN)) minutes before END_UTC=$END_UTC: nothing done"; exit 75; }
fi
[ -d /workspace/venv312 ] && venv=/workspace/venv312 || venv=/workspace/cp/venv312
VENV=${VENV:-$venv}; PY=$VENV/bin/python
v=$("$PY" -c 'import vllm; print(vllm.__version__)' 2> /dev/null) || { echo "$me: $PY imports no vLLM"; exit 2; }
WANT=${VLLM_WANT:-+gd9105ea8}
case $v in *"$WANT"*) ;; *) echo "$me: vLLM $v is not the build asked for (VLLM_WANT=$WANT)"; exit 2 ;; esac
echo "$me: vLLM $v; cores $CORES; card $CARD; $(date -u +%FT%TZ)"
CUDA=${SERVE_CUDA:-/usr/local/cuda-12.9}
[ -x "$CUDA/bin/cuobjdump" ] || { echo "$me: no $CUDA/bin/cuobjdump"; exit 2; }
# cuobjdump -sass runs nvdisasm, which this toolkit directory may not hold: take the first one found
NVD=
for d in "$CUDA/bin" $(ls -d /usr/local/cuda*/bin /workspace/jobs/cuda*/bin /workspace/cuda*/bin /opt/cuda*/bin 2> /dev/null); do
  [ -x "$d/nvdisasm" ] && { NVD=$d; break; }
done
[ -n "$NVD" ] || NVD=$(dirname "$(command -v nvdisasm 2> /dev/null)" 2> /dev/null)
[ -n "$NVD" ] && [ "$NVD" != . ] || NVD=$(dirname "$(timeout 60 find /usr /opt /workspace -maxdepth 7 -name nvdisasm -type f 2> /dev/null | head -1)" 2> /dev/null)
if [ -n "$NVD" ] && [ -x "$NVD/nvdisasm" ]; then export NVDISASM_PATH=$NVD; echo "$me: nvdisasm in $NVD ($("$NVD/nvdisasm" --version 2>&1 | head -1))"
else NVD=; echo "$me: no nvdisasm on this machine; toolkits: $(ls -d /usr/local/cuda* /workspace/jobs/cuda* 2> /dev/null | tr '\n' ' ')"; fi

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
echo "$me: card $CARD idle ($used MiB, no compute process, lease free, uuid $uuid) at $(date -u +%FT%TZ)"

export CUDA_HOME=$CUDA PATH="$CUDA/bin:$VENV/bin:$PATH${NVD:+:$NVD}"
unset CPATH LIBRARY_PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
SCRATCH=$(mktemp -d); export TRITON_CACHE_DIR=$SCRATCH/triton
ROOT=$PWD
cd adapters/vllm || exit 2
# The comparison imports this tree's verity and adapters/vllm: research run's PYTHONPATH names them; where it does not, ask research.
if ! "$PY" -c "import verity.definitions.registry.flashmla_dsv32" 2> /dev/null; then
  pp=$(cd "$ROOT" && PYTHONPATH=tools/research/src "$PY" -m research env --source . 2> /dev/null | sed -n 's/^export PYTHONPATH=//p')
  [ -n "$pp" ] && export PYTHONPATH="$pp"
fi
"$PY" -c "
import verity.definitions.registry.flashmla_dsv32 as m, verity_vllm, os, sys
root = os.path.realpath('$ROOT')
for mod in (m, verity_vllm):
    f = os.path.realpath(mod.__file__)
    print('$me: imports', f)
    if not f.startswith(root + os.sep):
        sys.exit('$me: ' + f + ' is not under the shipped tree ' + root)
" || { echo "$me: the shipped tree's Python is not what this environment imports: nothing done"; rm -rf "$SCRATCH"; exit 2; }
ncores=$(taskset -c "$CORES" nproc)

echo "$me: sass stage at $(date -u +%FT%TZ)"
taskset -c "$CORES" "$PY" -m $M sass --out "$OUT"; rc_sass=$?
echo "$me: sass exit $rc_sass at $(date -u +%FT%TZ)"
if [ "$rc_sass" != 0 ]; then
  # no listing: keep the library's CUDA images, so the listing can be made from them elsewhere
  so=$("$PY" -c "from tests.properties import glm53_flashmla_fp8_capture_gpu as X; print(X.flashmla_library())" 2> /dev/null)
  if [ -n "$so" ]; then
    mkdir -p "$OUT/sass/elf" && ( cd "$OUT/sass/elf" && taskset -c "$CORES" cuobjdump -xelf all "$so" > xelf.log 2>&1; ls | head -40; du -sh . )
  fi
fi
[ "${STAGES:-all}" = sass ] && { rm -rf "$SCRATCH"; exit "$rc_sass"; }

# the card's own report (its clocks' event reasons and thermal slowdown counters) from before and after the launches
smi() {
  nvidia-smi -q -i "$CARD" > "$OUT/nvidia-smi-$1.txt" 2> "$OUT/nvidia-smi-$1.err"; echo "$me: nvidia-smi -q -i $CARD $1 the capture: rc $? ($(wc -l < "$OUT/nvidia-smi-$1.txt") lines) at $(date -u +%FT%TZ)"
  grep -iE "slowdown|power cap|power brak|sync boost|clocks (event|throttle)|counters|GPU Current Temp|GPU T\.?Limit|Power Draw|Performance State" "$OUT/nvidia-smi-$1.txt" | sed "s/^ */$me: smi $1: /; s/  */ /g"
}
smi before
echo "$me: launch stage, asking for the lease at $(date -u +%FT%TZ)"
# shellcheck disable=SC2086
GPU_LEASE_WHO=${GPU_LEASE_WHO:-circuits-fp8read} bash "$LEASE" 1 --on "$CARD" --cores "$CORES" --max-min "$MAX_MIN" --mem-gb "${MEM_GB:-32}" -- \
  taskset -c "$CORES" "$PY" -m $M launch --out "$OUT" ${ARGS:-}
rc_launch=$?
smi after
echo "$me: launch exit $rc_launch at $(date -u +%FT%TZ); card $CARD now holds $(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null) MiB"
# reads only: what the card holds and what its lease says once the launches are over, and again at the run's end
card_now() {
  echo "$me: card $CARD $1: $(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null) MiB used," \
    "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2> /dev/null | grep -c "$uuid") compute process(es)," \
    "lease: $(LC_ALL=C bash "$LEASE" status 2>&1 | grep -E "^$CARD " | cut -c1-200 | tr '\n' ';') at $(date -u +%FT%TZ)"
}
card_now "after the launches"
du -sh "$OUT/capture" "$OUT/sass" 2> /dev/null
[ -f "$OUT/capture/launch.json" ] || { echo "$me: no launch record: nothing to compare"; rm -rf "$SCRATCH"; exit "$rc_launch"; }

echo "$me: compare stage on $ncores cores at $(date -u +%FT%TZ)"
taskset -c "$CORES" "$PY" -m $M compare --out "$OUT" --workers "$(( ncores > 1 ? ncores - 1 : 1 ))" --definition "${DEF_TOKENS:-2}"; rc_cmp=$?
echo "$me: compare exit $rc_cmp at $(date -u +%FT%TZ)"
card_now "at the run's end"
rm -rf "$SCRATCH"
[ "$rc_launch" = 0 ] || exit "$rc_launch"
[ "$rc_sass" = 0 ] || exit "$rc_sass"
exit "$rc_cmp"
