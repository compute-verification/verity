#!/usr/bin/env bash
# Re-pin a row that this tree moves, on the VM, from its trace_compare.sh run (README "GPU batch", step 1): the row's pins become
# the v2 traced Build's Programs (`verity-vllm neutrality ROW --compare --update`, which drops the old `aot`), and the row is
# switched again to the AOT Build on that trace (`verity-vllm records switch`: the comparison of this tree's predictor with the
# trace, equal on every rank, and the trace's art).  Refused unless the run's AOT Build against the trace equalled it.  Leaves
# manifests/records.json changed, for the commit "re-pin ROW" that the row's Build of record then runs at.
#   [TRACE_ART=art:... (a trace preserved by trace_art.sh on its own)] [REF (cursor/glm-v2-21f3)] integrations/vllm/manifests/glm_v2_tree/jobs/repin.sh ROW RUN [NODE (vy-glm53-cpu-2)]
set -euo pipefail
ROW=${1:?ROW} RUN=${2:?RUN} NODE=${3:-vy-glm53-cpu-2}
cd "$(git rev-parse --show-toplevel)"
read -r -a R <<<"${RESEARCH:-uv run --frozen --no-sync research}"
read -r -a V <<<"${VERITY_VLLM:-uv run --frozen --no-sync verity-vllm}"
"${R[@]}" fetch "$RUN" --all --on "$NODE" >/dev/null
O=$HOME/.research/runs/$RUN/out
need="build_summary.json comparison.json aot2_vs_trace.json tree.txt"
[ -n "${TRACE_ART:-}" ] || need="$need trace_art.txt"
for f in $need; do
  [ -s "$O/$f" ] || { echo "$RUN has no out/$f: its rehearsal did not get to the re-pin"; exit 3; }
done
python3 - "$O" "$ROW" <<'EOF'
import json, sys
o, row = sys.argv[1], sys.argv[2]
c, a = json.load(open(f"{o}/comparison.json")), json.load(open(f"{o}/aot2_vs_trace.json"))
if c.get("row") != row or c.get("equal") is not True:
    sys.exit(f"comparison.json is of {c.get('row')}, equal={c.get('equal')}: the predictor does not give the trace")
if a.get("equal") is not True:
    sys.exit(f"the AOT Build against the trace did not equal it: {a.get('differences')}")
EOF
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
git diff --quiet -- integrations/vllm/manifests/records.json || { echo "manifests/records.json has uncommitted changes"; exit 2; }
trap 'git checkout -q -- integrations/vllm/manifests/records.json; echo "re-pin of $ROW failed: manifests/records.json restored"' ERR
mkdir -p "$T/$ROW" && cp "$O/build_summary.json" "$T/$ROW/"
art=${TRACE_ART:-$(cat "$O/trace_art.txt")}
src=$(cat "$O/tree.txt")
(cd integrations/vllm &&
  "${V[@]}" neutrality "$ROW" --tier heavy --compare --update --out "$T" --run-id "$RUN" --source-sha "$src" &&
  "${V[@]}" records switch "$ROW" --comparison "$O/comparison.json" --run "$RUN" --trace "$art")
# `neutrality --update` names the Build it re-pinned from by its path here (the trace's art is that Build) and its tree by sha
# alone, and keeps the old record's id, which none of the new pins are from: the row names no record until its v2 publish
uv run --frozen --no-sync python - "$ROW" "$art" "${REF:-cursor/glm-v2-21f3}@$src" <<'EOF'
import sys
from verity_vllm.pipeline import config_record as CR
doc = CR.load_index(CR.INDEX)
e = doc["rows"][sys.argv[1]]
e["pinned_from"].update(build=sys.argv[2], tree=sys.argv[3])
e["record"] = None
CR.write_index(doc, CR.INDEX)
EOF
git diff --stat -- integrations/vllm/manifests/records.json
