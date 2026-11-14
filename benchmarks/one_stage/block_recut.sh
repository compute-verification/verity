#!/usr/bin/env bash
# The proof-block census made again at this tree from a stored census's families (`census-rank<r>.json`, each rank's
# `m1_families.py --steps all`, fetched from the census artifact ART): each family's unit templates and template
# recomputed from its spec at this tree (`serving_rows.unit_templates`, `serving_rows.template`), each rank sized
# (`m1_families.py --estimate`) and cut (`block_census.py`), which writes census.json, table.md and card.json under $OUT.
# The Programs, rows and steps are the census's, so a change to the cut or to a family's units needs no Commit Build or row
# directory. The card takes the model and the Build of the planner's card (`benchmarks/sampling_plan/glm53_blocks.json`)
# and, as its census, census.json put in the store from $OUT; `compare.json` holds the Definitions the new card adds, drops
# or changes against the planner's card, and both cards' totals per request. It needs the store:
#   research run --on auto --store-read --cwd source -- bash benchmarks/one_stage/block_recut.sh ART
set -uo pipefail
ART=${1:?the census artifact holding census-rank<r>.json}
REPO=$(pwd); O=${OUT:-$RESEARCH_RUN_DIR/out}; JOBS=${BC_JOBS:-4}; mkdir -p "$O"
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
git rev-parse HEAD 'HEAD^{tree}' > "$O/head.txt" 2>/dev/null
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }
PY=$REPO/.venv/bin/python
R=("$REPO/.venv/bin/research")
"${R[@]}" data fetch "$ART" --to "$D" --path '*census-rank*.json' > "$O/fetch.log" 2>&1 || { tail -5 "$O/fetch.log"; exit 4; }
FAMS=$(find "$D" -name 'census-rank*.json' | sort -V)
[ -n "$FAMS" ] || { echo "no census-rank<r>.json in $ART"; exit 4; }
sha256sum $FAMS | sed "s|$D/||" > "$O/families.sha256"
export PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/proofs/flock/prover/python:$REPO/adapters/vllm
start=$(date +%s)
$PY - $FAMS <<'PYEOF' 2>&1 | tee "$O/retemplate.log"
import json, sys
from verity_vllm.pipeline import serving_rows as SRW
memo, moved = {}, 0
for p in sys.argv[1:]:
    doc = json.load(open(p))
    for req in doc["requests"].values():
        for spec, f in req["families"].items():
            if spec not in memo:
                try:
                    t = SRW.template(spec)
                    tmpl = {"ok": True, "layout": t.layout, "word": t.word}
                except (LookupError, ValueError) as e:
                    tmpl = {"ok": False, "why": str(e)[:300]}
                memo[spec] = ([list(u) for u in SRW.unit_templates(spec)], tmpl)
            units, tmpl = memo[spec]
            moved += f.get("unit_templates") != units or f.get("template") != tmpl
            f["unit_templates"], f["template"] = units, tmpl
    json.dump(doc, open(p, "w"), indent=1, default=str)
print(f"retemplated {len(memo)} families over {len(sys.argv) - 1} ranks; {moved} family entries changed")
PYEOF
[ "${PIPESTATUS[0]}" -eq 0 ] || exit 6
pids=()
for f in $FAMS; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  r=$(basename "$f" .json); r=${r#census-}
  $PY benchmarks/one_stage/m1_families.py --estimate "$f" --out "$D/sized-$r.json" > "$O/sized-$r.log" 2>&1 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
[ $rc -eq 0 ] || { for f in "$O"/sized-*.log; do echo "== $f"; tail -5 "$f"; done; exit 7; }
$PY benchmarks/one_stage/block_census.py --sized $(find "$D" -name 'sized-rank*.json' | sort -V) --out "$O" 2>&1 | tee "$O/block_census.log"
rc=${PIPESTATUS[0]}; echo "block_recut: rc $rc in $(( $(date +%s) - start )) s" | tee -a "$O/block_census.log"
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
