#!/bin/bash
# The published tables rebuilt with each search record's tokens a step restated (restate.py), beside the published
# layer rebuilt as it is. A research run's workload on the scorer's node; reads K, writes none of it, publishes nothing.
#
#   TE=COMMIT [LAYERS=TREE,...] [WORKERS=N] [TRACE_TIMEOUT=S] [NULL=DIR,...] bash correct.sh
#
# The run's inputs/ holds: replot.bundle (a bundle holding COMMIT, the checkout whose te_core traced the published
# layer), replot-code.tar (the checkout's benchmarks/covert_witness that builds), replot-inputs.tar (replot.py node's
# --inputs), layers.tar (the published layer, a trace40.py node run's out/, as ./new), meta.json (the last refresh's)
# and this directory's restate.py, restate.json, replot.py and trace40.py.
#
# 1. restate.py plan, over the runs of the published search table's rows (OUT/runs.txt): the runs restated, from the
#    factor of the trace the published tables read of each (K's trace caches LAYERS, in the order the tables read
#    them), and the runs kept as they are with why (OUT/plan.json);
# 2. every restated record traced by COMMIT's te_core (trace40.py pack and node, OUT/traced), each checked to be the
#    plan's record by its sha256;
# 3. the corrected layer: the published layer with those traces over it (restate.py layer, OUT/layer-corr.json). A run
#    whose restated record did not trace keeps its record as it is and the published trace of it;
# 4. two builds, each a replot.py node on a scratch state of its own (one te_core tree's layers share a cache name, so a
#    state holds one of them): `fixed`, the published layer alone on K's records, which is the published tables again,
#    and `corr`, the corrected layer on a state holding the restated record of each run the layer took (OUT/build-NAME).
#    With NULL (trace40.py node out/ directories of COMMIT's traces of the records as they are), a third, `null`: the
#    published layer with those traces of the same runs over it, on K's records.
#
# What the correction moves is `corr` against `fixed`, never against replot.py's control: the control is the state as
# the last refresh left it, and the published layer already moves rows against it. `corr` against `null` is the
# restating alone, with the same te_core on both sides.
set -uo pipefail
K=${TE_SCORER_K:-/workspace/research/lanes/training/te-scorer}
IN=$RESEARCH_RUN_DIR/inputs
OUT=${OUT:-$RESEARCH_RUN_DIR/out}
TE=${TE:?the commit whose te_core traced the published layer}
LAYERS=${LAYERS:-280a017fcfa5,34220f8d3c76,538dc468a120}
WORKERS=${WORKERS:-20}
TRACE_TIMEOUT=${TRACE_TIMEOUT:-3000}
NULL=${NULL:-}
S=$(mktemp -d)
SAMPLER=
trap '[ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null; rm -rf "$S"' EXIT
mkdir -p "$OUT" "$S/layers"
echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) $(hostname) cores $(nproc) scratch $S"
free -g | head -2
# this job's own memory: the resident sizes of this shell's session, every 10 s (the cgroup is the node's, shared)
( peak=0; while sleep 10; do
    now=$(ps -o rss= -s "$(ps -o sid= -p $$ | tr -d ' ')" | awk '{s += $1} END {print int(s / 1048576)}')
    [ "$now" -gt "$peak" ] && peak=$now && echo "$(date -u +%H:%M:%SZ) $peak" > "$OUT/memory-peak-gb.txt"
  done ) &
SAMPLER=$!
(cd "$IN" && sha256sum -- *) > "$OUT/inputs.sha256.txt"

git init -q --bare "$S/repo.git"
echo "$K/repo.git/objects" > "$S/repo.git/objects/info/alternates"
git -C "$S/repo.git" fetch -q "$IN/replot.bundle" 'refs/heads/*:refs/heads/*' || exit 1
PIN=$(git -C "$K/repo.git" rev-parse origin/cursor/covert-witness-4207) || exit 1
for w in te fixed corr null; do
  git -C "$S/repo.git" worktree add -q --detach --no-checkout "$S/wt/$w" "$TE" || exit 1
done
TREE=$(git -C "$S/wt/te" rev-parse HEAD:benchmarks/covert_witness/te_core) || exit 1
echo "te: commit $(git -C "$S/wt/te" rev-parse HEAD) te_core tree $TREE $(git -C "$S/wt/te" show HEAD:benchmarks/covert_witness/te_core/__init__.py | grep '^VERSION'); pin $PIN" | tee "$OUT/checkouts.txt"
echo "K's trace caches: $(ls "$K/state/trace40" | tr '\n' ' '); read for the factor: $LAYERS" | tee -a "$OUT/checkouts.txt"
PY=$K/venv/bin/python
export TE_LEDGER=$K/ledger TE_REPO=$S/repo.git TE_PIN_REV=$PIN
TE_V1_PINS=$("$PY" -c 'import ast, json, sys; p = json.load(open(sys.argv[1]))["pins"]; print(",".join(p if isinstance(p, list) else ast.literal_eval(p)))' "$IN/meta.json") || exit 1
export TE_V1_PINS
TE_SCORER=$K/state "$PY" - "$IN" "$OUT/scope.json" <<'PY' || exit 1
import json, sys
sys.path.insert(0, sys.argv[1])
import trace40
s = trace40.scope()
json.dump(s, open(sys.argv[2], "w"))
print("scope:", len(s), "runs")
PY
if ! command -v research >/dev/null; then
  mkdir -p "$S/bin"
  printf '#!/bin/sh\nexec python3 -m research "$@"\n' > "$S/bin/research"
  chmod +x "$S/bin/research"
  export PATH="$S/bin:$PATH"
fi
tar -xf "$IN/layers.tar" -C "$S/layers" ./new || exit 1

echo "== $(date -u +%H:%M:%SZ) the plan, and a state holding every restated record"
mkdir -p "$S/pub"
tar -xf "$IN/replot-inputs.tar" -C "$S/pub" ./last/algorithm-policy.json || exit 1
"$PY" - "$S/pub/last/algorithm-policy.json" "$OUT/runs.txt" <<'PY' || exit 1
# the runs of the published search table's rows
import json, re, sys
q = json.load(open(sys.argv[1]))
i = q["columns"].index("record")
runs = sorted({m.group(0) for r in q["rows"] for m in [re.match(r"r\d{8}-\d{6}-[0-9a-f]{4}", r[i])] if m})
open(sys.argv[2], "w").write("".join(r + "\n" for r in runs))
print(f"the published search table: {len(q['rows'])} rows of {len(runs)} runs")
PY
"$PY" "$IN/restate.py" plan --state "$K/state" --scope "$OUT/scope.json" --layers "$LAYERS" --runs "$OUT/runs.txt" \
    --out "$OUT/plan.json" --workers "$WORKERS" || exit 1
"$PY" "$IN/restate.py" overlay --state "$K/state" --plan "$OUT/plan.json" --overlay "$S/state-all" \
    --out "$OUT/overlay-all.json" || exit 1

echo "== $(date -u +%H:%M:%SZ) packing and tracing the restated records with $WORKERS workers"
only=$("$PY" -c 'import json, sys; print(",".join(sorted(json.load(open(sys.argv[1]))["restated"])))' "$OUT/plan.json")
mkdir -p "$S/tmp"
TE_SCORER=$S/state-all TMPDIR="$S/tmp" "$PY" "$IN/trace40.py" pack --te "$S/wt/te" --all --only "$only" \
    --out "$S/pack.tar" > "$OUT/pack.json" 2> "$OUT/pack.err" || { echo "pack failed"; exit 1; }
"$PY" - "$S/pack.tar" "$OUT/plan.json" "$OUT/packed.json" <<'PY' || exit 1
# every packed record is the plan's restated one
import gzip, hashlib, json, sys, tarfile
plan = json.load(open(sys.argv[2]))["restated"]
bad = []
with tarfile.open(sys.argv[1]) as t:
    packed = sorted(r["run_id"] for r in json.load(t.extractfile("./runs.json")))
    for run in packed:
        got = hashlib.sha256(gzip.decompress(t.extractfile(f"./runs/{run}/run.json.gz").read())).hexdigest()
        if got != plan[run]["after_sha256"]:
            bad.append(run)
json.dump({"restated": len(plan), "packed": len(packed), "not_packed": sorted(set(plan) - set(packed)),
           "not_the_restated_record": bad}, open(sys.argv[3], "w"), indent=1)
print(f"packed {len(packed)} of {len(plan)} restated; not packed: {sorted(set(plan) - set(packed))}; not the restated record: {bad}")
sys.exit(1 if bad or not packed else 0)
PY
TE_TRACE_TIMEOUT=$TRACE_TIMEOUT TMPDIR="$S/tmp" "$K/trace-venv/bin/python" "$IN/trace40.py" node --pack "$S/pack.tar" \
    --out "$OUT/traced" --workers "$WORKERS" > "$OUT/traced.txt" 2>&1
echo "trace40 node rc $? $(date -u +%H:%M:%SZ)" >> "$OUT/traced.txt"
rm -rf "$S/pack.tar" "$S/tmp"
echo "== $(date -u +%H:%M:%SZ) traced: $(tail -n 2 "$OUT/traced.txt" | tr '\n' ' ' | cut -c1-300); peak so far $(cat "$OUT/memory-peak-gb.txt" 2>/dev/null) GB"

"$PY" "$IN/restate.py" layer --base "$S/layers/new" --over "$OUT/traced" --dest "$S/layers/corr" \
    --out "$OUT/layer-corr.json" || exit 1
"$PY" - "$OUT/layer-corr.json" "$OUT/plan.json" "$S/only-corr.txt" <<'PY' || exit 1
import json, sys
layer, plan = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))["restated"]
took = sorted(set(layer["from_over"]) & set(plan))
open(sys.argv[3], "w").write("".join(r + "\n" for r in took))
print(f"the corrected layer holds {len(took)} runs at their restated record; restated in the plan and not traced (they "
      f"keep the record as it is and its published trace): {sorted(set(plan) - set(took))}")
PY
: > "$S/only-fixed.txt"
: > "$S/only-null.txt"
cp -r "$S/layers/new" "$S/layers/fixed"
BUILDS="fixed corr"
if [ -n "$NULL" ]; then
  over=()
  for d in $(echo "$NULL" | tr , ' '); do [ -f "$d/traced.json" ] && over+=(--over "$d"); done
  "$PY" "$IN/restate.py" layer --base "$S/layers/new" "${over[@]}" --only "$S/only-corr.txt" --dest "$S/layers/null" \
      --out "$OUT/layer-null.json" && BUILDS="$BUILDS null"
fi

build() {  # NAME: the tables with the layer NAME, on a state holding the restated records of only-NAME.txt
  local name=$1 rc=0
  mkdir -p "$S/K-$name" "$S/code-$name" "$S/in-$name"
  for p in "$K"/*; do [ "$(basename "$p")" = state ] || ln -s "$p" "$S/K-$name/$(basename "$p")"; done
  "$PY" "$IN/restate.py" overlay --state "$K/state" --plan "$OUT/plan.json" --overlay "$S/K-$name/state" \
      --only "$S/only-$name.txt" --reuse "$S/state-all" --drop-cache "${TREE:0:12}" --out "$OUT/state-$name.json" || return 1
  tar -xf "$IN/replot-code.tar" -C "$S/code-$name"
  cp "$IN/replot.py" "$S/code-$name/benchmarks/covert_witness/scorer/replot.py"
  tar -xf "$IN/replot-inputs.tar" -C "$S/in-$name"
  ( cd "$S/code-$name/benchmarks/covert_witness" && "$PY" scorer/replot.py node --state "$S/K-$name" --inputs "$S/in-$name" \
        --scratch "$S/work-$name" --out "$OUT/build-$name" --layer "$name=$S/wt/$name=$S/layers/$name" --publish "$name" ) \
      > "$OUT/build-$name.txt" 2>&1 || rc=$?
  rm -rf "$S/work-$name" "$S/code-$name" "$S/in-$name"
  echo "== $(date -u +%H:%M:%SZ) build $name: rc $rc; peak so far $(cat "$OUT/memory-peak-gb.txt" 2>/dev/null) GB" | tee -a "$OUT/build-$name.txt"
  [ "$rc" = 0 ] && date -u +%H:%M:%SZ > "$OUT/build-$name.done"
  return "$rc"
}
PIDS=()
for n in $BUILDS; do build "$n" & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait "$p"; done
du -sh "$OUT" 2>/dev/null
echo "== $(date -u +%H:%M:%SZ) done; this job's peak: $(cat "$OUT/memory-peak-gb.txt" 2>/dev/null) GB"
[ -f "$OUT/build-fixed.done" ] && [ -f "$OUT/build-corr.done" ]
