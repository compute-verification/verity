#!/usr/bin/env bash
# Carry open branches across the Definitions' move to verity.ml.definitions with restack.py: merge the move's base into each,
# rerun the map (tools/move/defs_map.toml) on it, and merge the move. restack.py's docstring has the steps and outcomes.
#
#   tools/move/restack_defs.sh [--move REF] [--base REF] [--map FILE] BRANCH... [restack.py flags: --dry-run --json --no-pins --onto REF --push-to NEW]
#
# Run it at the root of a checkout that has the move. --move defaults to the move commit on origin/main (found by its subject),
# one commit on the pre-move, and then takes `--onto origin/main`; before the move lands, pass its commit on the move's PR
# branch. Either way the base is the commit's first parent. The map is this checkout's
# tools/move/defs_map.toml. (tools/move/rename_map.toml is the warden rename's, read by rename.py; this map is layout.py's.)
set -euo pipefail
move=
map=tools/move/defs_map.toml
base=
while [[ $# -gt 0 ]]; do
  case $1 in
    --move) move=$2; shift 2 ;;
    --map) map=$2; shift 2 ;;
    --base) base=$2; shift 2 ;;
    *) break ;;
  esac
done
if [[ -z $move ]]; then
  move=$(git log -1 --format=%H --grep='^vllm defs down: move the integration' origin/main)
  [[ -n $move ]] || { echo "restack_defs: the move isn't on origin/main yet; pass --move REF, its commit on the move's PR branch" >&2; exit 2; }
fi
map=$(realpath "$map")
m=$(git rev-parse "$move^{commit}")
if [[ -z $base ]]; then
  base=$(git rev-parse "$m^1")      # the move is one commit on its base, and the lander's merge has the base first
fi
python3 tools/move/restack.py "$@" --move-commit "$m" --base "$base" \
  --script-arg=--map --script-arg="$map" --script-arg=--no-lock
