#!/usr/bin/env bash
# CIR-263: ehproj_words.py on a CPU pod over a card run's kept eh_proj words (inputs/ehproj_words.tar: the run's
# out/eh_proj/words; inputs/routes.json: the route runs it printed), at the checkout's head.  Env: WORKERS (8).
#   research run --on auto --project verity --source PR_WORKTREE --cwd source --cpus 8 --mem-gb 32 --wait \
#       --send ehproj_words.sh --send ehproj_words.py --send ehproj_words.tar --send routes.json \
#       -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/ehproj_words.sh"'
set -uo pipefail
: "${OUT:?OUT}"
ROUTES=$(cat "$RESEARCH_RUN_DIR/inputs/routes.json") || exit 2
date -u +"sync %FT%TZ"
uv sync -q --all-packages --extra torch-cpu || exit 2
tree=$PWD
work=$(mktemp -d) || exit 2
trap 'rm -rf "$work"' EXIT
tar -xf "$RESEARCH_RUN_DIR/inputs/ehproj_words.tar" -C "$work" || exit 2
cd adapters/vllm || exit 2
date -u +"compare %FT%TZ"
PYTHONPATH=$PWD:$tree "$tree/.venv/bin/python" "$RESEARCH_RUN_DIR/inputs/ehproj_words.py" --words "$work/words" --out "$OUT" \
    --routes "$ROUTES" --workers "${WORKERS:-8}"
rc=$?
date -u +"end %FT%TZ exit $rc"
exit $rc
