#!/usr/bin/env bash
# One of GLM-5.3's MTP draft-layer captures (glm53_mtp_<NAME>_capture_gpu.py) on one H200 of a shared node, on the card and cores
# the node's booking gives it.  It starts nothing when the card holds a compute process or 1 GiB or more, or its lease is held;
# else it runs the capture under one lease of that card, asked without waiting, of at most MAX_MIN minutes, every process pinned
# to CORES.  The capture's record, PTX and words go to $OUT/<NAME>.
#
#   CARD=0 CORES=0-9 GPU_LEASE_WHO=... research run --on HOST --cwd source -- \
#       bash adapters/vllm/tests/properties/glm53_mtp_capture_card.sh NAME [capture args...]
#
# Env: CARD, CORES (required), OUT (research run's), VENV (/workspace/venv312), GPU_LEASE (gpu-lease, the node's), MAX_MIN (30),
# WORKERS (8).
set -euo pipefail
name=${1:?usage: glm53_mtp_capture_card.sh NAME [args...]}; shift
card=${CARD:?CARD}; cores=${CORES:?CORES}; out=${OUT:?OUT}/$name
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
root=$(cd "$here/../.." && pwd -P); tree=$(cd "$root/../.." && pwd -P)
script=$here/glm53_mtp_${name}_capture_gpu.py
[ -f "$script" ] || { echo "no capture $script"; exit 2; }
py=${VENV:-/workspace/venv312}/bin/python
lease=${GPU_LEASE:-gpu-lease}
command -v "$lease" >/dev/null || { echo "no $lease on this node"; exit 2; }

procs=$(nvidia-smi -i "$card" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
used=$(nvidia-smi -i "$card" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
date -u +"start %FT%TZ card $card: $procs compute processes, $used MiB"
if (( procs > 0 || used >= 1024 )); then echo "card $card is not idle: starting nothing"; exit 75; fi

mkdir -p "$out"
export PYTHONPATH=$root:$tree${PYTHONPATH:+:$PYTHONPATH}
cd "$root"
set +e
taskset -c "$cores" "$lease" 1 --on "$card" --cores "$cores" --max-min "${MAX_MIN:-30}" -- \
    taskset -c "$cores" "$py" "$script" --out "$out" --scratch "${TMPDIR:-/tmp}" --workers "${WORKERS:-8}" "$@"
rc=$?
set -e
date -u +"end %FT%TZ exit $rc"
exit $rc
