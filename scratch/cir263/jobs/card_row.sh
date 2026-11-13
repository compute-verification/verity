#!/usr/bin/env bash
# CIR-263: GLM-5.3's MTP draft-layer captures (glm53_mtp_<NAME>_capture_gpu.py), several under ONE lease of one card, inside one
# calendar row.  It is the pull request's glm53_mtp_capture_card.sh with a start time, an end time and a loop: it starts nothing
# before NOT_BEFORE, nothing when the card holds a compute process or 1 GiB or more, and nothing with under 5 minutes of lease
# left.  The lease (--on CARD --cores CORES --max-min M) ends at END minus 2 minutes at the latest; each capture is ended at END
# minus 3 minutes (timeout, its whole process group), and none starts with under MIN_LEFT seconds to that.  Every process is
# pinned to CORES.  Each capture's record, PTX and words go to $OUT/<NAME>, its output to $OUT/<NAME>.log; scratch is a
# mktemp directory, removed when the loop ends.
#
#   research run --on HOST --project verity --source PR_WORKTREE --cwd source --send card_row.sh \
#       --env NOT_BEFORE=2026-11-13T03:15:00Z --env END=2026-11-13T03:45:00Z --env CARD=0 --env CORES=0-9 \
#       --env GPU_LEASE_WHO=circuits-draftkernels -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/card_row.sh" NAME...'
#
# Env: NOT_BEFORE, END, CARD, CORES, GPU_LEASE_WHO (required), OUT (research run's), VENV (/workspace/venv312), GPU_LEASE
# (gpu-lease, the node's), MIN_LEFT (240), WORKERS (8).
set -uo pipefail
self=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")

if [ "${1:-}" = --inner ]; then
    shift
    tree=$PWD; root=$tree/adapters/vllm
    py=${VENV:-/workspace/venv312}/bin/python
    export PYTHONPATH=$root:$tree${PYTHONPATH:+:$PYTHONPATH}
    cd "$root" || exit 2
    scratch=$(mktemp -d) || exit 2
    trap 'rm -rf "$scratch"' EXIT
    trap 'exit 143' TERM
    worst=0
    for name in "$@"; do
        left=$(( STOP - $(date -u +%s) ))
        if (( left < ${MIN_LEFT:-240} )); then echo "$name: $left s left: not started"; (( worst < 75 )) && worst=75; continue; fi
        mkdir -p "$OUT/$name"
        date -u +"$name start %FT%TZ, $left s left"
        timeout --kill-after=20 "${left}s" "$py" "tests/properties/glm53_mtp_${name}_capture_gpu.py" --out "$OUT/$name" \
            --scratch "$scratch" --workers "${WORKERS:-8}" > "$OUT/$name.log" 2>&1
        rc=$?
        date -u +"$name end %FT%TZ exit $rc"
        tail -n 60 "$OUT/$name.log"
        (( rc > worst )) && worst=$rc
    done
    exit "$worst"
fi

: "${NOT_BEFORE:?NOT_BEFORE}" "${END:?END}" "${CARD:?CARD}" "${CORES:?CORES}" "${OUT:?OUT}" "${GPU_LEASE_WHO:?GPU_LEASE_WHO}"
[ $# -gt 0 ] || { echo "usage: card_row.sh NAME..."; exit 2; }
for name in "$@"; do
    [ -f "adapters/vllm/tests/properties/glm53_mtp_${name}_capture_gpu.py" ] || { echo "no capture $name under $PWD"; exit 2; }
done
now=$(date -u +%s); nb=$(date -u -d "$NOT_BEFORE" +%s) || exit 2; end=$(date -u -d "$END" +%s) || exit 2
if (( now < nb )); then echo "$(date -u +%FT%TZ) is before NOT_BEFORE=$NOT_BEFORE: starting nothing"; exit 75; fi
export STOP=$(( end - 180 ))
max_min=$(( (end - 120 - now) / 60 ))
(( max_min > 30 )) && max_min=30
if (( max_min < 5 )); then echo "$max_min whole minutes of lease left before $END less 2: starting nothing"; exit 75; fi
lease=${GPU_LEASE:-gpu-lease}
command -v "$lease" >/dev/null || lease=/usr/local/bin/gpu-lease
command -v "$lease" >/dev/null || { echo "no gpu-lease on this node"; exit 2; }

procs=$(nvidia-smi -i "$CARD" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
used=$(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
date -u +"start %FT%TZ card $CARD: $procs compute processes, $used MiB; one lease of $max_min min on cores $CORES; captures end by $(date -u -d "@$STOP" +%TZ)"
[[ $used =~ ^[0-9]+$ ]] || { echo "card $CARD: memory.used unreadable ($used): starting nothing"; exit 75; }
if (( procs > 0 || used >= 1024 )); then echo "card $CARD is not idle: starting nothing"; exit 75; fi

taskset -c "$CORES" "$lease" 1 --on "$CARD" --cores "$CORES" --max-min "$max_min" -- taskset -c "$CORES" bash "$self" --inner "$@"
rc=$?
date -u +"end %FT%TZ exit $rc"
exit "$rc"
