#!/usr/bin/env bash
# Carry open branches across the vLLM integration's first layout move with restack.py: merge the move's base into each,
# rerun the map (tools/move/vllm_map.toml) on it, and merge the move. restack.py's docstring has the steps and outcomes.
#
#   tools/move/restack_vllm.sh [--move REF] [--map FILE] BRANCH... [restack.py flags: --dry-run --json --no-pins --onto REF --push-to NEW]
#
# Run it at the root of a checkout that has the move's commit. Before the move lands, --move is the move PR's head (the
# default); after, pass the move's commit on main and `--onto origin/main`. The map defaults to the copy on
# cursor/vllm-arch-map-496a, since main doesn't carry it.
set -euo pipefail
move=origin/cursor/vllm-arch-move1-496a
map=
while [[ $# -gt 0 ]]; do
  case $1 in
    --move) move=$2; shift 2 ;;
    --map) map=$2; shift 2 ;;
    *) break ;;
  esac
done
if [[ -z $map ]]; then
  map=$(mktemp --suffix=.toml)
  trap 'rm -f "$map"' EXIT
  git show origin/cursor/vllm-arch-map-496a:tools/move/vllm_map.toml > "$map"
fi
python3 tools/move/restack.py "$@" --move-commit "$(git rev-parse "$move^{commit}")" \
  --script-arg=--map --script-arg="$map" --script-arg=--no-lock
