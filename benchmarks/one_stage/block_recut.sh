#!/usr/bin/env bash
# The proof-block census cut again from a stored census: its ranks' sized families (`sized-rank<r>.json`, each rank's
# `m1_families.py --estimate`, fetched from the census artifact ART) cut by this tree's `block_census.py`, which writes
# census.json, table.md and card.json under $OUT. The families are not sized again, so a change to the cut alone
# (`block_census.cut`) needs no Commit Build or row directory. The card takes the model and the Build of the planner's card
# (`benchmarks/sampling_plan/glm53_blocks.json`) and, as its census, census.json put in the store from $OUT; `compare.json`
# holds the Definitions the new card adds, drops or changes against the planner's card, and both cards' totals per request.
# It needs the store:
#   research run --on auto --store-read --cwd source -- bash benchmarks/one_stage/block_recut.sh ART
set -uo pipefail
ART=${1:?the census artifact holding sized-rank<r>.json}
REPO=$(pwd); O=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$O"
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
git rev-parse HEAD 'HEAD^{tree}' > "$O/head.txt" 2>/dev/null
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }
PY=$REPO/.venv/bin/python
R=("$REPO/.venv/bin/research")
"${R[@]}" data fetch "$ART" --to "$D" --path '*sized-rank*.json' > "$O/fetch.log" 2>&1 || { tail -5 "$O/fetch.log"; exit 4; }
SIZED=$(find "$D" -name 'sized-rank*.json' | sort -V)
[ -n "$SIZED" ] || { echo "no sized-rank<r>.json in $ART"; exit 4; }
sha256sum $SIZED | sed "s|$D/||" > "$O/sized.sha256"
export PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/proofs/flock/prover/python:$REPO/adapters/vllm
start=$(date +%s)
$PY benchmarks/one_stage/block_census.py --sized $SIZED --out "$O" 2>&1 | tee "$O/block_census.log"
rc=${PIPESTATUS[0]}; echo "block_census.py: rc $rc in $(( $(date +%s) - start )) s" | tee -a "$O/block_census.log"
[ "$rc" -eq 0 ] || exit "$rc"
CENSUS=$("${R[@]}" run-preserve "$O/census.json" 2>"$O/preserve.log" | grep -oE '[0-9a-f]{64}' | head -1)
[ -n "$CENSUS" ] || { cat "$O/preserve.log"; exit 6; }
CENSUS=art:$CENSUS; echo "census $CENSUS"
$PY - "$O/card.json" benchmarks/sampling_plan/glm53_blocks.json "$CENSUS" "$O/compare.json" <<'PYEOF'
import json, sys
new, old = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
json.dump({"model": old["model"], "source": sys.argv[3], "build": old["build"]} | new, open(sys.argv[1], "w"), indent=1)
open(sys.argv[1], "a").write("\n")
a, b = old["definitions"], new["definitions"]
keep = ("blocks", "out_bits", "ands", "macs")
def totals(ds):
    t = dict.fromkeys(keep, 0)
    for d in ds.values():
        n = d["prefill"] + d["decode"]
        for k, v in (("blocks", n), ("out_bits", n * d["out_bits"]), ("ands", n * d["ands"]), ("macs", n * d["macs"])):
            t[k] += v
    return t
cmp = {"census": sys.argv[3], "types": [len(a), len(b)], "added": sorted(set(b) - set(a)), "dropped": sorted(set(a) - set(b)),
       "changed": {s: {k: [a[s][k], b[s][k]] for k in a[s] if a[s][k] != b[s][k]} for s in sorted(set(a) & set(b)) if a[s] != b[s]},
       "totals": {"before": totals(a), "after": totals(b)}}
json.dump(cmp, open(sys.argv[4], "w"), indent=1)
print(f"CARD types {len(a)} -> {len(b)}: {len(cmp['added'])} added, {len(cmp['dropped'])} dropped, {len(cmp['changed'])} changed; "
      + "; ".join(f"{k} {cmp['totals']['before'][k]:,} -> {cmp['totals']['after'][k]:,}" for k in keep))
PYEOF
