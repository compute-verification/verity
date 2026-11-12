#!/usr/bin/env bash
# usage: launch_card.sh END_UTC CORES [ARGS] [DEF_TOKENS]   (card 1 of node A; prints the run id)
set -euo pipefail
END=${1:?END_UTC}; CORES=${2:?CORES}; ARGS=${3:-}; DEFT=${4:-2}; NB=${NOT_BEFORE:-2026-11-12T18:05:00Z}
cd /agent/repos/verity
[ -z "$(git status --short)" ] || { echo "dirty tree"; exit 2; }
git rev-parse --short=10 HEAD
date -u +%FT%TZ
extra=()
[ -n "$ARGS" ] && extra=(--env "ARGS=$ARGS")
[ -n "${STAGES:-}" ] && extra+=(--env "STAGES=$STAGES")
uv run -q --project tools/research research run --on vy-spot-glm53 --project verity --source /agent/repos/verity --cwd source \
  --timeout 85m --mem-gb 32 --send /tmp/cir262/fp8read_card.sh --env CARD=1 --env "CORES=$CORES" \
  --env "NOT_BEFORE=$NB" --env "END_UTC=$END" "${extra[@]}" --env "DEF_TOKENS=$DEFT" \
  -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/fp8read_card.sh' 2>&1 | tail -4 | cut -c1-260
