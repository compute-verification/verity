#!/bin/bash
# A measure for one run, on a scratch branch only: circuit-check's test suite as a candidate's check runs it (suites.py, full
# tier, the opt-in walks out), with ccmem_peak.py's rows from every xdist worker.  From the tree's root, in a clone of the run's own:
#
#   MODE=records bash scratch/cir227_peak/job.sh    the bindings' records made first, in a cache of the run's own, so they count
#   MODE=clean   bash scratch/cir227_peak/job.sh    a clean cache of the run's own: no record counts, the first worker builds apart
#   MODE=keys    bash scratch/cir227_peak/job.sh    only: which variables key a record in a session suites.py starts
#   MODE=order   bash scratch/cir227_peak/job.sh    that, and the registered roots' order in a session against circuit-check's own
#
# A record counts only under the key it was stored with, and the key holds every VERITY_ and CIRCUIT_CHECK_ variable of the
# process (`cache.key`).  suites.py inside a `research run` sets two for its suites (`store_custody`), as check.py's `env()`
# does for every step of a recorded check, its circuit-check among them.  So the records are made with the variables a
# session's worker has (ccmem_plan.py writes them from the smoke's session), as a check's own circuit-check would have them.
#
# PASSES: suites.py's --mem-gb for each pass of the suite.  100 is pytest's share of a 200 GB train slot (check.py's
# `tests_gb`), which gives the suite alone 12 workers at worker_gb = 8; 50 is what that share leaves the parallel suites beside
# 11 one-process lanes (100 - 11 * 4.5 = 50.5, suites.py's `wide_gb`), which gives it 6.
set -u
MODE="${MODE:?records or clean}"
PASSES="${PASSES:-50 100}"
RECORDS_GB="${RECORDS_GB:-90}"
SMOKE_GB="${SMOKE_GB:-100}"
ROOT=$PWD
S=tools/verity/circuit_check
D="$ROOT/scratch/cir227_peak"
stamp() { date -u +%H:%M:%SZ; }

echo "== $MODE: commit $(git rev-parse HEAD 2>/dev/null) tree $(git rev-parse 'HEAD^{tree}' 2>/dev/null) host $(hostname) cpus $(nproc) $(stamp)"
git log -2 --format='   %H %s' 2>/dev/null
echo "   the slot's memory as this job sees it: ${CHECK_SLOT_MEM_GB:-unset} GB; the opt-in walks' variable is $([ -n "${VERITY_CIRCUIT_CHECK_SLOW:-}" ] && echo set || echo unset)"
echo "   keyed names in the environment: $(compgen -e | grep -E '^(VERITY_|CIRCUIT_CHECK_)' | tr '\n' ' ')"
uv sync -q --all-packages --extra torch-cpu < /dev/null || { echo "uv sync failed"; exit 9; }
PY="$ROOT/.venv/bin/python"
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cp "$D/ccmem_peak.py" "$D/ccmem_plan.py" "$D/ccmem_first.py" "$SITE/" || exit 9
echo "   synced; the plugins are at ${SITE#"$ROOT"/}/ccmem_{peak,plan,first}.py $(stamp)"

SCR=$(mktemp -d)
trap 'rm -rf "$SCR"' EXIT
# a home of this run's own: the node's, but for circuit-check's records and suites.py's kept passes
REAL="$HOME"
H="$SCR/home"
C="$H/.cache/verity-check/circuit-check"
mkdir -p "$C" "$H/.cache/verity"
link() {  # link FROM TO SKIP...: each entry of FROM into TO, but those named
  local from=$1 to=$2 p b; shift 2
  for p in "$from"/* "$from"/.[!.]*; do
    [ -e "$p" ] || continue
    b=$(basename "$p")
    case " $* " in *" $b "*) ;; *) ln -s "$p" "$to/$b" ;; esac
  done
}
link "$REAL" "$H" .cache
link "$REAL/.cache" "$H/.cache" verity-check verity
link "$REAL/.cache/verity-check" "$H/.cache/verity-check" circuit-check
link "$REAL/.cache/verity" "$H/.cache/verity" tests
OWN="$REAL/.cache/verity-check/circuit-check"
echo "   the node's own records, which this run neither reads nor writes: $([ -d "$OWN" ] && echo "$(find "$OWN" -mindepth 1 -maxdepth 1 -type d | wc -l) version directories" || echo none)"

why() {  # why LABEL PYTHONPATH: which records count, looked up from the suite's directory with a session's keyed variables
  echo "== records looked up from the suite's directory, $1 $(stamp)"
  ( cd "$ROOT/$S" && HOME="$H" PYTHONPATH="$2" timeout 1500 "$PY" "$D/with_env.py" "$K" "$PY" -P "$D/ccmem_why.py" "$C" < /dev/null 2>&1 | grep -a "^WHY\|Error" | cut -c1-1800 )
}

suite() {  # suite DIR MEM_GB JOBS [PYTEST ARGS...]: one suites.py run of the suite with the plugin, its rows and logs in DIR
  local dir=$1 gb=$2 jobs=$3; shift 3   # WRAP: a command to run it under (a bound on its time)
  mkdir -p "$dir/rows" "$dir/logs"
  HOME="$H" VERITY_TEST_CACHE="$SCR/kept-$(basename "$dir")" CCMEM_OUT="$dir/rows" \
    ${WRAP:-} "$PY" tools/verity/check/suites.py --synced "$S" --fresh --timeouts --jobs "$jobs" --mem-gb "$gb" \
    --logs "$dir/logs" --json "$dir/suites.json" -- -p ccmem_peak "$@" < /dev/null
}

rows() { cat "$1"/rows/gw*.jsonl 2>/dev/null | grep -a "\"event\": \"$2\"" | cut -c1-700; }

# the measure is in place: one cheap test that asks for no catalog, through suites.py as every pass runs
CHEAP=test_a_circuit_check_in_an_xdist_worker_keeps_to_the_worker_s_share
K="$SCR/keyed/gw0.json"
echo "== the measure, on one test $(stamp)"
CCMEM_KEYED="$SCR/keyed" suite "$OUT/smoke" "$SMOKE_GB" 2 -p ccmem_plan -k "$CHEAP"
echo "   rc=$? rows: $(cat "$OUT"/smoke/rows/gw*.jsonl 2>/dev/null | wc -l)"
"$PY" - "$SCR/keyed" <<'EOF' || { echo "== no worker wrote what keys a record in its session, or two workers differ: stopping"; exit 5; }
import json, pathlib, sys
got = {p.name: json.loads(p.read_text()) for p in sorted(pathlib.Path(sys.argv[1]).glob("gw*.json"))}
for name, env in got.items():
    print(f"   what keys a record in the session ({name}), names only: {sorted(env) or 'nothing'}; each set to 1: "
          f"{all(v == '1' for v in env.values())}")
sys.exit(0 if got and len({json.dumps(v, sort_keys=True) for v in got.values()}) == 1 else 1)
EOF
rows "$OUT/smoke" start | head -n 2
rows "$OUT/smoke" test | head -n 2
"$PY" - "$OUT/smoke/suites.json" <<'EOF' || { echo "== the suite does not pass on one test with the plugin: stopping"; exit 5; }
import json, sys
got = json.load(open(sys.argv[1]))
for s in got:
    print("   suites.json:", {k: s.get(k) for k in ("status", "workers", "counts", "outside", "child_outside", "unguarded")})
sys.exit(0 if got and all(s.get("status") == "passed" for s in got) else 1)
EOF
[ "$(rows "$OUT/smoke" test | wc -l)" -ge 1 ] || { echo "== the plugin wrote no test row: stopping"; exit 5; }
[ "$MODE" = keys ] && { echo "done (keys only) $(stamp)"; exit 0; }

status() {  # status DIR: what suites.py said of the one suite it ran there
  "$PY" -c 'import json, sys
for s in json.load(open(sys.argv[1])):
    print("   suites.json:", {k: s.get(k) for k in ("status", "seconds", "workers", "counts", "outside", "child_outside", "unguarded")})' "$1/suites.json" 2>/dev/null \
    || echo "   suites.json: not read"
}

if [ "$MODE" != clean ]; then
  # the registered roots' order (which the records of <registered>:0..7 are cut from): circuit-check's own process, a session's
  # workers, and those of a session that loads the registries before pytest imports a test module (ccmem_first.py)
  echo "== the registered roots' order $(stamp)"
  ORD="$OUT/order"
  mkdir -p "$ORD"
  ( cd "$ROOT/$S" && PYTHONPATH="$ROOT/tools/verity/check/guard/child" "$PY" -P -c 'import json, sys
from circuit_check import cache, cli, targets, units
targets.load_registries()
json.dump([getattr(r, "id", repr(r)[:80]) for r in targets._registered()], open(sys.argv[1], "w"))' "$ORD/cli.json" < /dev/null ) \
    || echo "   the order of circuit-check's own process: not written"
  CCMEM_ORDER="$ORD/session" suite "$OUT/order-session" "$SMOKE_GB" 2 -p ccmem_plan -k "$CHEAP" > "$OUT/order-session.log" 2>&1
  status "$OUT/order-session"
  CCMEM_ORDER="$ORD/first" suite "$OUT/order-first" "$SMOKE_GB" 2 -p ccmem_first -p ccmem_plan -k "$CHEAP" > "$OUT/order-first.log" 2>&1
  status "$OUT/order-first"
  "$PY" "$D/ccmem_order.py" "$ORD" | cut -c1-1500
fi
[ "$MODE" = order ] && { echo "done (order only) $(stamp)"; exit 0; }
PRE=""

if [ "$MODE" = records ]; then
  t0=$SECONDS
  HOME="$H" "$PY" "$D/with_env.py" "$K" "$PY" -m circuit_check --all --records-only -q --cache "$C" --jobs "$(nproc)" \
    --mem-gb "$RECORDS_GB" --out "$OUT/circuit-records.json" < /dev/null > "$OUT/records.log" 2>&1
  echo "== records, made with the session's keyed variables: circuit-check --all --records-only --jobs $(nproc) --mem-gb $RECORDS_GB rc=$? wall=$((SECONDS - t0))s, $(du -sh "$C" | cut -f1), $(find "$C" -type f | wc -l) files $(stamp)"
  tail -n 4 "$OUT/records.log" | cut -c1-400
  why "with suites.py's path" "$ROOT/tools/verity/check/guard/child"
  # conftest's own `_from_records`, called in each worker of a session suites.py starts, with why the first record misses
  counted() { [ "$(cat "$1"/rows/plan.gw*.jsonl 2>/dev/null | grep -c '"how": "from records')" -eq 2 ]; }
  echo "== conftest's _from_records in a session, as suites.py starts it $(stamp)"
  CCMEM_PLAN=1 WRAP="timeout -k 120 900" suite "$OUT/plan" 100 2 -p ccmem_plan -k "$CHEAP" > "$OUT/plan.log" 2>&1
  echo "   rc=$?"
  status "$OUT/plan"
  cat "$OUT"/plan/rows/plan.gw*.jsonl 2>/dev/null | cut -c1-2200
  if ! counted "$OUT/plan"; then
    echo "== the same in a session that loads the registries first (-p ccmem_first) $(stamp)"
    CCMEM_PLAN=1 WRAP="timeout -k 120 900" suite "$OUT/plan-first" 100 2 -p ccmem_first -p ccmem_plan -k "$CHEAP" > "$OUT/plan-first.log" 2>&1
    echo "   rc=$?"
    status "$OUT/plan-first"
    cat "$OUT"/plan-first/rows/plan.gw*.jsonl 2>/dev/null | cut -c1-2200
    counted "$OUT/plan-first" || { echo "== _from_records gave the catalog from the records in neither session: stopping"; exit 4; }
    PRE="-p ccmem_first"
  fi
  echo "== every session from here runs with: ${PRE:-nothing added}"
  # and the session's own: one cheap catalog test; a session they do not count in builds apart, for long
  echo "== the records in a session, on one catalog test $(stamp)"
  WRAP="timeout -k 120 900" suite "$OUT/first" 100 2 $PRE -k test_circuit_check_keeps_every_task
  echo "   rc=$? (124: stopped at 900 s)"
  rows "$OUT/first" catalog
  rows "$OUT/first" built_apart
  rows "$OUT/first" catalog | grep -q '"how": "from records' || { echo "== the session's catalog did not come from the records: stopping"; exit 4; }
fi

rc_all=0
for gb in $PASSES; do
  if [ "$MODE" = clean ]; then rm -rf "$C"; mkdir -p "$C"; fi
  before=$(find "$C" -type f | wc -l)
  L="$OUT/pass-$gb"
  t0=$(date -u +%s)
  echo "== pass: suites.py --fresh --timeouts --jobs $(nproc) --mem-gb $gb ${PRE:+-- $PRE }starts $(stamp); files in the records cache: $before"
  suite "$L" "$gb" "$(nproc)" $PRE
  rc=$?
  [ $rc -eq 0 ] || rc_all=$rc
  t1=$(date -u +%s)
  echo "== pass --mem-gb $gb rc=$rc wall=$((t1 - t0))s $(stamp); files in the records cache after: $(find "$C" -type f | wc -l)"
  tail -n 4 "$L/logs/tools_verity_circuit_check.log" 2>/dev/null | cut -c1-300
  echo "   catalog rows: $(rows "$L" catalog | grep -o '"how": "[^"]*"' | sort | uniq -c | tr '\n' ';')"
  rows "$L" built_apart | grep '"test": null' | cut -c1-500
done
echo "done rc=$rc_all $(stamp)"
exit $rc_all
