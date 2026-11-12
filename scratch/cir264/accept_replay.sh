#!/usr/bin/env bash
# The kept capture compared with the reference again off the card, and the registered Definition evaluated on its first N
# cases (the hand vectors come first; N is 12 unless given).  From the root of the tree whose Definition is asked about:
#   research fetch --all RUN --runs-dir DIR --on vy-spot-glm53      # the capture's run
#   tar -czf capture.tgz -C DIR/RUN/out capture/sampled_accept_capture.json capture/cases
#   research run --on auto --project verity --source TREE --cwd source --timeout 30m --mem-gb 16 --send accept_replay.sh \
#     --send capture.tgz [--env N=12] -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/accept_replay.sh'
# A first argument names another tar than inputs/capture.tgz.  The exit status is sampled_accept_capture_gpu.py --replay's:
# 0 when no word differs inside the domain, against the reference and against the Definition.
set -euo pipefail
IN="$RESEARCH_RUN_DIR/inputs"
: "${OUT:=$RESEARCH_RUN_DIR/out}"
tgz=${1:-$IN/capture.tgz}
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
date -u +"start %Y-%m-%dT%H:%M:%SZ on $(hostname), tree $(git rev-parse HEAD 2> /dev/null || basename "$PWD")"
sha256sum "$tgz"
uv venv -q --python 3.12 "$d/venv"
uv pip install -q --python "$d/venv/bin/python" numpy==2.5.4
mkdir "$d/kept"
tar -xzf "$tgz" -C "$d/kept"
record=$(find "$d/kept" -maxdepth 3 -name sampled_accept_capture.json | head -1)
[ -n "$record" ] || { echo "$tgz holds no sampled_accept_capture.json"; exit 2; }
src=$(dirname "$record")
echo "replaying $(find "$src/cases" -name '*.npz' | wc -l) kept cases, the Definition on the first ${N:-12}"
rc=0
"$d/venv/bin/python" adapters/vllm/tests/properties/sampled_accept_capture_gpu.py --replay "$src" --definition "${N:-12}" \
  --out "$OUT/replay" || rc=$?
date -u +"end %Y-%m-%dT%H:%M:%SZ exit $rc"
exit "$rc"
