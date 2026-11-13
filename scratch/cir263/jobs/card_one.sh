#!/usr/bin/env bash
# CIR-263: ONE capture of record under ONE lease, with the pull request's own card script
# (adapters/vllm/tests/properties/glm53_mtp_capture_card.sh), inside one calendar row.  It adds only the row's clock: it starts
# nothing before NOT_BEFORE and nothing with under 5 whole minutes of lease left before END less 2 minutes (exit 75, the card
# untouched), and it sets MAX_MIN to the whole minutes left before END less 2 minutes, at most 30.  The card script does the
# rest: the idle check of CARD, the lease with --cores CORES, the pinning, the capture and its record under $OUT/<NAME>.
#
#   research run --on vy-spot-glm53 --project verity --source PR_WORKTREE --cwd source --send card_one.sh \
#       --env NOT_BEFORE=2026-11-13T15:45:00Z --env END=2026-11-13T17:45:00Z --env CARD=0 --env CORES=0-9 \
#       --env GPU_LEASE_WHO=circuits-draftkernels -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/card_one.sh" NAME'
set -euo pipefail
name=${1:?usage: card_one.sh NAME [capture args...]}; shift
: "${NOT_BEFORE:?NOT_BEFORE}" "${END:?END}" "${CARD:?CARD}" "${CORES:?CORES}" "${OUT:?OUT}" "${GPU_LEASE_WHO:?GPU_LEASE_WHO}"
card=adapters/vllm/tests/properties/glm53_mtp_capture_card.sh
[ -f "$card" ] || { echo "no $card under $PWD"; exit 2; }
now=$(date -u +%s); nb=$(date -u -d "$NOT_BEFORE" +%s); end=$(date -u -d "$END" +%s)
if (( now < nb )); then echo "$(date -u +%FT%TZ) is before NOT_BEFORE=$NOT_BEFORE: starting nothing"; exit 75; fi
max_min=$(( (end - 120 - now) / 60 ))
(( max_min > 30 )) && max_min=30
if (( max_min < 5 )); then echo "$max_min whole minutes of lease left before $END less 2: starting nothing"; exit 75; fi
command -v gpu-lease >/dev/null || export GPU_LEASE=/usr/local/bin/gpu-lease
export MAX_MIN=$max_min
date -u +"card_one %FT%TZ $name: head $(git rev-parse HEAD 2>/dev/null || echo unknown), MAX_MIN=$MAX_MIN, card $CARD, cores $CORES"
exec bash "$card" "$name" "$@"
