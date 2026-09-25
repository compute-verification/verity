#!/usr/bin/env bash
# verify_lane.sh — PROTECTED.  The sealed verifier: judges a lane's candidate commit in a FRESH checkout and emits the
# machine-generated configuration card.  The lane never produces its own card.
#
#   bash verity_vllm/ops/verify_lane.sh <lane-ref> <run-dir> [--base <core-ref>] [--holdouts <dir>] [--laptop]
#                                            [--pod <ssh-wrapper> --pod-run-dir </vol/cp/...> [--pod-parallel 8]]
#
#   <lane-ref>   the lane's candidate commit / branch (must be reachable from this repo's object store)
#   <run-dir>    the lane's run directory produced by run_config.py's GPU phase on its pod and copied back
#                (log.jsonl.gz, snapshots/, census.json, noninterference/, versions.json, tokens.json, ...).  The CPU phase
#                (resolve, fold x2, root policy, replay tiers, census reconcile, golden, gates, card) is RE-RUN here from the
#                fresh checkout; whatever the lane computed for those stages is ignored.
#   --base       the core commit the lane branched from (default: lane/capture)
#   --holdouts   directory of holdout run dirs (each with gates.json from the orchestrator's holdout runs); assembled into
#                <run-dir>/holdouts.json for gate G7.  Absent => G7 skipped => verdict INCOMPLETE.
#   --laptop     pass --laptop to run_config.py (D69 memory/time discipline; skip stages that need more than 1.5 GB)
#   --pod        run step 4 (the CPU phase) on a pod instead of the laptop: `git archive <lane-ref>` is shipped to
#                /workspace/cp/verify/<name> over the given ssh wrapper (out/capture/pods/cp-<x>.ssh), the driver runs there with
#                /workspace/venv-py312 over --pod-run-dir (the run directory ON THE POD, e.g. /vol/cp/<lane>/runs/<name>), and the
#                small outputs (card, gates, stages, replay/root-policy/golden/fold reports, CARD.md) are copied into <run-dir> here.
#                Steps 1-3 (protected paths, tests, quarantine lint, holdouts) still run here from the core checkout.  Use this for
#                every >= 1B model: tier b of a 32-row prefill is hours of CPU there and infeasible on the laptop (D69).
#
# Steps: 1. `git worktree add` a fresh checkout of <lane-ref> under /private/tmp/verify/<name>;  2. protected.py: the diff
# base...lane must touch no protected path (-> provenance.json; a violation still produces a card, verdict REJECTED);
# 3. pytest of the lane's tests in the fresh checkout (results -> tests.json);  4. run_config.py --phase cpu in the fresh
# checkout over <run-dir>;  5. holdouts.json;  6. card.  Output: <run-dir>/card.json + card.md, provenance.json.
set -euo pipefail
LANE_REF=${1:?lane ref}; RUN_DIR=${2:?run dir}; shift 2
BASE=lane/capture; HOLDOUTS=""; LAPTOP=""; POD=""; POD_RUN_DIR=""; POD_PARALLEL=8; REPLAY_BUDGET=""
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE=$2; shift 2;;
    --holdouts) HOLDOUTS=$2; shift 2;;
    --laptop) LAPTOP=--laptop; shift;;
    --pod) POD=$2; shift 2;;
    --pod-run-dir) POD_RUN_DIR=$2; shift 2;;
    --pod-parallel) POD_PARALLEL=$2; shift 2;;
    --replay-budget-leaves) REPLAY_BUDGET="--replay-budget-leaves $2"; shift 2;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
REPO=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$RUN_DIR"; RUN_DIR=$(cd "$RUN_DIR" && pwd)
[ -n "$POD" ] && [ -z "$POD_RUN_DIR" ] && { echo "--pod needs --pod-run-dir" >&2; exit 2; }
NAME=$(echo "$LANE_REF" | tr '/' '_')-$(date -u +%Y%m%dT%H%M%SZ)
WT=/private/tmp/verify/$NAME
PY=${VERIFY_PYTHON:-$REPO/.venv/bin/python}
log() { echo "[verify_lane] $(date -u +%FT%TZ) $*" | tee -a "$RUN_DIR/verify_lane.log"; }

mkdir -p /private/tmp/verify "$RUN_DIR"
log "lane=$LANE_REF base=$BASE run_dir=$RUN_DIR worktree=$WT"
# D86: the fresh checkout is SPARSE (same exclusions as the lane worktrees: results/bundles/eod/contract/night/scale/curve/pdfs) --
# a full checkout of this tree is 3.5 GB and the laptop cannot afford one per verification; the CPU phase needs none of the excluded roots.
git -C "$REPO" worktree add --detach --no-checkout "$WT" "$LANE_REF" >/dev/null 2>&1 || { log "worktree add failed (is $LANE_REF fetched?)"; exit 2; }
trap 'git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true' EXIT
git -C "$WT" config --worktree core.sparseCheckout true 2>/dev/null || git -C "$WT" config core.sparseCheckout true
SPARSE=$(git -C "$WT" rev-parse --git-path info/sparse-checkout); mkdir -p "$(dirname "$SPARSE")"
printf '%s\n' '/*' '!/fixtures/results' '!/vllm-poc/bundles' '!/out/eod' '!/data/contract' '!/out/night' '!/out/scale' '!/out/curve' '!/docs/**/*.pdf' > "$SPARSE"
git -C "$WT" checkout --detach "$LANE_REF" >/dev/null 2>&1 || git -C "$WT" read-tree -mu HEAD || { log "sparse checkout failed"; exit 2; }
# the fresh checkout uses the repo venv (torch-free half) but its own sources
cd "$WT"; ln -s "$REPO/.venv" .venv 2>/dev/null || true

# 1. protected paths (from the CORE checkout's protected.py — never the lane's copy)
( cd "$REPO" && "$PY" -m verity_vllm.properties.protected --base "$BASE" --head "$LANE_REF" --repo "$REPO" --head-checkout "$WT" --out "$RUN_DIR/provenance.json" ) 2>&1 | tee -a "$RUN_DIR/verify_lane.log" || log "protected paths touched (card will say REJECTED)"

# 1b. the verifier's fresh sampling seed (D82: the orchestrator, not the lane author, selects the tier-a / chain sample);
#     recorded in provenance.json so the card can quote it
SEED=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
"$PY" - "$RUN_DIR/provenance.json" "$SEED" <<'EOF2' || true
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception:
    d = {}
d["sampling_seed"] = int(sys.argv[2])
d["sampling_seed_source"] = "verify_lane.sh /dev/urandom"
json.dump(d, open(p, "w"), indent=1)
EOF2
log "sampling seed: $SEED"

# 2. the lane's tests in the fresh checkout (laptop limits: pytest -x, 10 min cap)
( cd "$WT" && timeout 600 "$PY" -m pytest tests/observe verity_vllm/program/registry/quarantine -q -p no:cacheprovider --junitxml="$RUN_DIR/tests.xml" 2>&1 | tail -3 | tee -a "$RUN_DIR/verify_lane.log" ) || log "pytest failed or timed out"
"$PY" - "$RUN_DIR/tests.xml" "$RUN_DIR/tests.json" <<'EOF' || true
import json, sys, xml.etree.ElementTree as ET
try:
    r = ET.parse(sys.argv[1]).getroot()
    s = r if r.tag == "testsuite" else r.find("testsuite")
    a = s.attrib
    doc = {"tests": int(a.get("tests", 0)), "failures": int(a.get("failures", 0)), "errors": int(a.get("errors", 0)), "skipped": int(a.get("skipped", 0))}
    doc["status"] = "pass" if doc["tests"] and not doc["failures"] and not doc["errors"] else "fail"
except Exception as e:
    doc = {"status": "fail", "reason": f"{type(e).__name__}: {e}"}
json.dump(doc, open(sys.argv[2], "w"), indent=1)
print("[verify_lane] tests:", doc)
EOF
# 2a. G4 admission: `quarantine.json` (written by `difftest check`, copied into RUN_DIR by the lane) gets its
#     `negative_matcher_tests` verdict from the lane's pytest run above; it requires a `test_gen_<lane>.py` with
#     recognizer tests (admission rule 3) and a passing suite.  Without a quarantine.json nothing is written (D83).
if [ -f "$RUN_DIR/quarantine.json" ]; then
"$PY" - "$RUN_DIR/quarantine.json" "$RUN_DIR/tests.json" "$WT" <<'EOF' || true
import glob, json, sys
q = json.load(open(sys.argv[1])); t = json.load(open(sys.argv[2]))
has_gen_tests = bool(glob.glob(f"{sys.argv[3]}/tests/observe/test_gen_*.py"))
q["negative_matcher_tests"] = "pass" if (t.get("status") == "pass" and has_gen_tests) else "fail"
q["negative_matcher_tests_detail"] = {"pytest": t.get("status"), "test_gen_files": has_gen_tests}
json.dump(q, open(sys.argv[1], "w"), indent=1)
print("[verify_lane] quarantine negative_matcher_tests:", q["negative_matcher_tests"])
EOF
fi

# 2b. quarantine purity/identity lint (from the CORE checkout's lint over the LANE's quarantine tree)
( cd "$REPO" && "$PY" -m verity_vllm.properties.quarantine_lint --root "$WT/verity_vllm/program/registry/quarantine" --out "$RUN_DIR/quarantine_lint.json" ) 2>&1 | tee -a "$RUN_DIR/verify_lane.log" || log "quarantine lint failed"

# 3. holdouts (orchestrator-run) -> holdouts.json
if [ -n "$HOLDOUTS" ]; then
  ( cd "$WT" && "$PY" -m verity_vllm.properties.holdout --assemble "$HOLDOUTS" --out "$RUN_DIR/holdouts.json" ) 2>&1 | tee -a "$RUN_DIR/verify_lane.log" || true
fi

# 4. the CPU phase of the sealed driver in the fresh checkout: resolve, fold x2, root policy, replay tiers, census reconcile,
#    golden corpus, gates, card (the lane's copies of these outputs are overwritten)
if [ -z "$POD" ]; then
  ( cd "$WT" && "$PY" -m verity_vllm.pipeline.match --run-dir "$RUN_DIR" --phase cpu --seed "$SEED" --cards-dir "$RUN_DIR/cards" $LAPTOP $REPLAY_BUDGET ) 2>&1 | tee -a "$RUN_DIR/verify_lane.log" || log "run_config cpu phase returned non-zero"
else
  # the fresh checkout goes to the pod as `git archive <lane-ref>` (exactly the candidate commit: no dirty state, no lane run dirs)
  PWT=/workspace/cp/verify/$NAME
  log "pod: shipping git archive of $LANE_REF -> $PWT; run dir on the pod: $POD_RUN_DIR"
  # D86: ship the same sparse set (pathspecs = the tree's top-level entries minus the excluded roots), not the 3.5 GB full tree
  SHIP_PATHS=$(git -C "$REPO" ls-tree --name-only "$LANE_REF" | grep -v -x -E 'out|vllm-poc|docs'; git -C "$REPO" ls-tree --name-only "$LANE_REF" out/ vllm-poc/ | grep -v -x -E 'out/(eod|contract|night|scale|curve)|vllm-poc/(results|bundles)')
  git -C "$REPO" archive --format=tar "$LANE_REF" $SHIP_PATHS | gzip -1 | "$POD" "mkdir -p $PWT && tar -xzf - -C $PWT" || { log "pod ship failed"; exit 2; }
  # provenance.json and tests.json travel with the run dir so the card's sealed rows are filled from the SAME verifier invocation
  tar -czf - -C "$RUN_DIR" provenance.json tests.json quarantine_lint.json $( [ -f "$RUN_DIR/holdouts.json" ] && echo holdouts.json ) $( [ -f "$RUN_DIR/quarantine.json" ] && echo quarantine.json ) 2>/dev/null | "$POD" "mkdir -p $POD_RUN_DIR && tar -xzf - -C $POD_RUN_DIR" || true
  "$POD" "cd $PWT && PYTHONPATH=. HF_HOME=/workspace/hf /workspace/venv-py312/bin/python -m verity_vllm.pipeline.match --run-dir $POD_RUN_DIR --phase cpu --seed $SEED --python-cpu /workspace/venv-py312/bin/python --parallel $POD_PARALLEL --cards-dir $POD_RUN_DIR/cards $REPLAY_BUDGET" 2>&1 | tee -a "$RUN_DIR/verify_lane.log" || log "run_config cpu phase (pod) returned non-zero"
  # small outputs back (never the snapshots)
  "$POD" "cd $POD_RUN_DIR && tar -czf - --ignore-failed-read card.json CARD.md gates.json stages.json run.json provenance.json tests.json quarantine_lint.json holdouts.json fold_summary.json fold_summary_2.json root_policy.json golden.json wiring.json replay_chain.json replay_a.json replay_b_step*.json compare.json resolution.json resolution.md census_reconcile.json census_reconcile.md nonint/report.json nonint/report.md logs cards 2>/dev/null" | tar -xzf - -C "$RUN_DIR" || log "copy-back incomplete"
  "$POD" "rm -rf $PWT" || true
fi

log "done: $RUN_DIR/CARD.md"
[ -f "$RUN_DIR/CARD.md" ] && sed -n 1,60p "$RUN_DIR/CARD.md"
