#!/usr/bin/env bash
# stoch_negatives.sh — the two named negatives of a `Sampling=stochastic` workload.
#
#   N1 wrong-seed      the RUN is served with a different per-request seed than the MANIFEST OF RECORD declares
#   N2 chain-mutation  the RUN is served with a different sampler chain (top_p 0.9) than the Program was BUILT for (top_p 0.95)
#
# Each negative re-serves the row's workload from a MUTATED copy of the manifest (GPU: m1_capture with snapshots of steps 0,1),
# folds it, and then runs the row's OWN Match checks against the UNMUTATED Program / manifest of record:
#   cb_a.compare  (derived request Program vs the fold)      -- N2 must FAIL here: the fold's Const32[top_p] differs from the Program's
#   stoch_recompute --workload <manifest of record>          -- N1 and N2 must FAIL here, NAMED (header seed/top_p != manifest; recomputed token != served)
# A negative PASSES (as a negative) iff the Match verdict is FAIL with a named reason; anything else is recorded as NEGATIVE-NOT-CAUGHT.
# Nothing is retried; every artefact stays under $OUT/<N1|N2>/.
#
# usage: stoch_negatives.sh <row-id> <role> <out-dir>        (env: SWEEP_DIR, PY, PY312, GPU_UTIL as row_pod.sh)
set -uo pipefail
ROW=${1:?row-id}; ROLE=${2:?role}; OUT=${3:?out-dir}
# the tree that runs is the one this script lives in, never an environment default
TREE_SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [ -n "${VERITY_TREE:-}" ] && [ "$(cd "$VERITY_TREE" 2>/dev/null && pwd -P)" != "$TREE_SELF" ]; then
  echo "PRECHECK_FAIL_SOURCE_IDENTITY: stoch_negatives.sh at $TREE_SELF but \$VERITY_TREE=$VERITY_TREE names another tree" >&2; exit 3
fi
export VERITY_TREE=$TREE_SELF
cd "$VERITY_TREE" || exit 1
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1 TOKENIZERS_PARALLELISM=false
PY=${PY:-/workspace/venv-cu129/bin/python}; PY312=${PY312:-/workspace/venv-py312/bin/python}
GPU_UTIL=${GPU_UTIL:-0.3}
D=${SWEEP_DIR:-/workspace/cp/sweep}/$ROW                  # the row's Build (build_request = the Program of record)
WL=workloads/$ROW.json                       # the manifest of record
"${PY312%/*}/verity-vllm" workload-file "$WL" > /dev/null || exit 2
[ -f "$WL" ] || { echo "no manifest $WL"; exit 2; }
[ -d "$D/build_request" ] || { echo "no build_request under $D (run the row's Build first)"; exit 2; }
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) [stoch-neg $ROW] $*" | tee -a "$OUT/negatives.log"; }
B=$($PY312 -c "import json;print(len(json.load(open('$WL'))['requests']))")
MML=$($PY312 -c "import json;w=json.load(open('$WL'));print(max(r['prompt_len']+r['max_tokens'] for r in w['requests']))")
PROFILE=${FOLD_PROFILE:-$($PY312 -c "from verity_vllm.observe.fold.profiles import profile_module_for_case as f; print(f('$ROLE'))")}   # as run_config does

run_one() {   # $1 = N1|N2, $2 = python mutation snippet applied to the manifest dict `w`
  N=$OUT/$1; mkdir -p "$N/match/capture"
  $PY312 - "$WL" "$N/workload.json" "$2" <<'EOF'
import json,sys
w=json.load(open(sys.argv[1])); exec(sys.argv[3]); json.dump(w, open(sys.argv[2],"w"), indent=1)
EOF
  log "$1: mutated manifest $N/workload.json ($2)"
  "${PY%/*}/verity-vllm" m1-capture --workload "$N/workload.json" --case "$ROLE" --max-num-seqs "$B" \
      --checkpoints manifests/checkpoints.json --engine-arg=max_num_batched_tokens=2048 --engine-arg=max_model_len="$MML" \
      --engine-arg=gpu_memory_utilization="$GPU_UTIL" --out "$N/match/capture" --snapshot-steps 0,1 --verify-weights > "$N/capture.log" 2>&1
  rc1=$?; log "$1: capture rc=$rc1"
  "${PY312%/*}/verity-vllm" match --worker fold --log "$N/match/capture/log.jsonl.gz" --profile "$PROFILE" \
      --out "$N/match/fold_summary.json" --persist-dir "$N/match" > "$N/fold.log" 2>&1
  rc2=$?; log "$1: fold rc=$rc2"
  "${PY312%/*}/verity-vllm" program-compare --a "$D/build_request" --b "$N/match" --out "$N/match_compare.json" > "$N/compare.log" 2>&1
  rc3=$?; log "$1: compare rc=$rc3"
  "${PY312%/*}/verity-vllm" stoch-recompute --run "$N/match" --workload "$WL" --out "$N/match_stoch.json" > "$N/recompute.log" 2>&1
  rc4=$?; log "$1: recompute rc=$rc4"
  $PY312 - "$N" "$1" "$rc1" "$rc2" "$rc3" "$rc4" <<'EOF' | tee -a "$OUT/negatives.log"
import json, os, sys
n, name = sys.argv[1], sys.argv[2]; rcs = [int(x) for x in sys.argv[3:7]]
cmp = json.load(open(f"{n}/match_compare.json")) if os.path.exists(f"{n}/match_compare.json") else {}
c = cmp.get("compare") or {}
canon = (c.get("canonical_sha256") or {}).get("equal")
tot = c.get("totals") or {}
st = json.load(open(f"{n}/match_stoch.json")) if os.path.exists(f"{n}/match_stoch.json") else {}
sm = st.get("summary") or {}
reasons = []
if canon is False:
    hd = cmp.get("histogram_diff_T_normalised") or {}
    reasons.append(f"cb_a.compare: canonical_equal=False (steps spec-equal {tot.get('steps_spec_equal')}, operands_different {tot.get('operands_different')}; "
                   f"first spec differences {json.dumps(dict(list(hd.items())[:2]))[:300]})")
if sm.get("verdict") == "FAIL":
    reasons.append(f"stoch_recompute: FAIL rows={sm.get('rows')} named={json.dumps(st.get('reasons') or [])[:500]}")
caught = bool(reasons)
verdict = {"negative": name, "expected": "Match FAIL (named)", "caught": caught, "rcs": {"capture": rcs[0], "fold": rcs[1], "compare": rcs[2], "recompute": rcs[3]},
           "compare_canonical_equal": canon, "stoch_verdict": sm.get("verdict"), "reasons": reasons}
json.dump(verdict, open(f"{n}/verdict.json", "w"), indent=1)
print(f"{name}: {'NEGATIVE-CAUGHT' if caught else 'NEGATIVE-NOT-CAUGHT'} canonical_equal={canon} stoch={sm.get('verdict')}")
for r in reasons: print("   ", r[:600])
EOF
}

# N1: every request served under a DIFFERENT seed than the manifest of record declares (deterministic mutation, not a retry)
run_one N1 "
for r in w['requests']:
    if r.get('seed') is not None: r['seed'] = (int(r['seed']) ^ 0x5bd1e995) & 0xFFFFFFFF
for c in w.get('children', []) or []:
    if c.get('seed') is not None: c['seed'] = (int(c['seed']) ^ 0x5bd1e995) & 0xFFFFFFFF
if isinstance(w.get('seeds'), list): w['seeds'] = [(int(s) ^ 0x5bd1e995) & 0xFFFFFFFF for s in w['seeds']]
"
# N2: the sampler CHAIN is mutated (top_p 0.95 -> 0.9): the Program was built for the declared chain; the run serves another
run_one N2 "
v = w['sampling']['variant']; w['variants'][v]['top_p'] = 0.9
for r in w['requests']:
    if r.get('top_p') is not None: r['top_p'] = 0.9
"
log "done: $($PY312 -c "import json; print({n: json.load(open('$OUT/'+n+'/verdict.json'))['caught'] for n in ('N1', 'N2')})")"
