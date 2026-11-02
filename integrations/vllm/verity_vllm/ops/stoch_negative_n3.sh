#!/usr/bin/env bash
# stoch_negative_n3.sh — N3, the CHRONOLOGY negative of a B > 1 `Sampling=stochastic` workload.
#
#   The request Program's per-event split count S_r[s] = splits_for(n_live(arrive_r + s)) is computed from the DECLARED chronology
#   (arrivals + caps of the whole population).  N3 declares a WRONG chronology — one batch-mate's arrival moved — builds the first
#   request's Program from that manifest (CPU: derive_step on meta), and compares it against the workload's REAL fold with its own
#   Match tool (cb_a.batch_decomp, the other shapes' Programs from its Build).  The Match must FAIL: the fold's launches carry the
#   S vLLM chose for the realized batch, the mutated Program declares another S at the affected events (spec-id inequality, named).
#   A negative PASSES (as a negative) iff batch_decomp reports the first request canonical-unequal on GumbelTopPTokenSelect spec ids.
#   Nothing is served again and nothing is retried; every artefact stays under $OUT/N3/.
#
# usage: stoch_negative_n3.sh <row-id> <role> <hf-repo> <revision> <out-dir>
#        env: SWEEP_DIR, PY, PY312 as row_pod.sh; N3_MUTATION = python over the manifest dict `w` (default: move the
#        arrival of the request that arrives last-but-one ... see below) — the mutation must change some batch-mate's live count
#        across a split boundary (4/5 = 32/16 at 142 SMs) or the negative is vacuous (reported as NEGATIVE-VACUOUS, not as caught).
set -uo pipefail
ROW=${1:?row-id}; ROLE=${2:?role}; REPO=${3:?hf repo}; REV=${4:?revision}; OUT=${5:?out-dir}
# the tree that runs is the one this script lives in, never an environment default
TREE_SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [ -n "${VERITY_TREE:-}" ] && [ "$(cd "$VERITY_TREE" 2>/dev/null && pwd -P)" != "$TREE_SELF" ]; then
  echo "PRECHECK_FAIL_SOURCE_IDENTITY: stoch_negative_n3.sh at $TREE_SELF but \$VERITY_TREE=$VERITY_TREE names another tree" >&2; exit 3
fi
export VERITY_TREE=$TREE_SELF
cd "$VERITY_TREE" || exit 1
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=${PY:-/workspace/venv-cu129/bin/python}; PY312=${PY312:-/workspace/venv-py312/bin/python}
D=${SWEEP_DIR:-/workspace/cp/sweep}/$ROW
WL=workloads/$ROW.json
"${PY312%/*}/verity-vllm" workload-file "$WL" > /dev/null || exit 2
[ -f "$WL" ] || { echo "no manifest $WL"; exit 2; }
[ -f "$D/match/program.json" ] || { echo "no fold under $D/match (run the row's Match first)"; exit 2; }
N=$OUT/N3; mkdir -p "$N"
log() { echo "$(date -u +%FT%TZ) [stoch-neg-N3 $ROW] $*" | tee -a "$OUT/negatives.log"; }
MML=$($PY312 -c "import json;w=json.load(open('$WL'));print(max(r['prompt_len']+r['max_tokens'] for r in w['requests']))")
# default mutation (empty N3_MUTATION): the first non-first request whose arrival moved 9 engine steps earlier changes r0's declared S
# somewhere (crosses a live-count boundary of SplitsFor_v1 at the Builds' num_SMs: 4/5 at 142, 5/6 at 188); an explicit N3_MUTATION
# is python over the manifest dict `w`
MUT=${N3_MUTATION:-}
# the mutated manifest + the declared schedules before/after (the negative is vacuous if r0's schedule does not change), at the
# num_SMs the Builds under $D declare (sampling.geometry.num_sms)
$PY312 - "$WL" "$N/workload.json" "$MUT" "$N/schedules.json" "$D" <<'EOF' || { log "N3: no schedule (see above)"; exit 2; }
import glob, json, sys
from verity_vllm.pipeline.build import declared_live_counts
from verity.ml.definitions.registry.topp_split import splits_for
nsms = {int(g["num_sms"]) for p in glob.glob(f"{sys.argv[5]}/build_request*/result.json")
        for g in [((json.load(open(p)).get("sampling") or {}).get("geometry") or {})] if g.get("num_sms") is not None}
if len(nsms) != 1:
    sys.exit(f"the Builds ({sys.argv[5]}/build_request*/result.json) declare num_SMs {sorted(nsms)}, not one value: "
             "the declared split schedule has no constant")
NSM = nsms.pop()
w = json.load(open(sys.argv[1])); w0 = json.loads(json.dumps(w))
def sched(wl, i, nsm=NSM):
    n = declared_live_counts(wl["requests"]); r = wl["requests"][i]; a = int(r.get("arrive_step", 0) or 0)
    return [splits_for(n[a + s], nsm) for s in range(int(r["max_tokens"]))]
mut = sys.argv[3]
if mut.strip():
    exec(mut)
else:
    for k in range(1, len(w["requests"])):
        cand = json.loads(json.dumps(w0)); a = int(cand["requests"][k].get("arrive_step", 0) or 0)
        cand["requests"][k]["arrive_step"] = max(0, a - 9)
        if sched(cand, 0) != sched(w0, 0):
            w = cand; mut = f"w['requests'][{k}]['arrive_step'] = {max(0, a - 9)}  # (auto: was {a})"; break
json.dump(w, open(sys.argv[2], "w"), indent=1)
sys.argv[3] = mut
s0, s1 = sched(w0, 0), sched(w, 0)
diff = [s for s in range(len(s0)) if s0[s] != s1[s]]
json.dump({"r0_shape": {"LP": int(w0["requests"][0]["prompt_len"]), "T": int(w0["requests"][0]["max_tokens"]) - 1}, "num_sms": NSM,
           "declared_schedule_of_record": s0, "declared_schedule_mutated": s1, "events_with_different_S": diff,
           "mutation": sys.argv[3], "requests_mutated": [(a["request_id"][:8], a.get("arrive_step"), b.get("arrive_step"), a["max_tokens"], b["max_tokens"])
                                                         for a, b in zip(w0["requests"], w["requests"]) if a != b]}, open(sys.argv[4], "w"), indent=1)
print(len(diff))
EOF
NDIFF=$(tail -n 1 "$N/schedules.json" >/dev/null; $PY312 -c "import json;print(len(json.load(open('$N/schedules.json'))['events_with_different_S']))")
read -r LP0 T0 <<< "$($PY312 -c "import json;s=json.load(open('$N/schedules.json'))['r0_shape'];print(s['LP'], s['T'])")"
log "N3: mutated manifest $N/workload.json; r0 (LP=$LP0, T=$T0) declared S differs at $NDIFF events ($(head -c 300 "$N/schedules.json" | tr '\n' ' '))"
if [ "$NDIFF" = "0" ]; then
  echo "{\"negative\": \"N3\", \"caught\": false, \"vacuous\": true, \"why\": \"the mutation changes no S of r0's schedule\"}" > "$N/verdict.json"
  log "N3: NEGATIVE-VACUOUS (mutation changes no S of r0) -> $N/verdict.json"; exit 3
fi
# r0's Program under the WRONG declared chronology (meta-device derivation; no serving)
"${PY%/*}/verity-vllm" build --model "$REPO" --revision "$REV" --wrapper request --LP "$LP0" --T "$T0" \
    --max-model-len "$MML" --workload "$N/workload.json" --out "$N/build_request_mut" > "$N/derive.log" 2>&1
rc1=$?; log "N3: derive (mutated chronology) rc=$rc1 $(tail -n 1 "$N/derive.log" | cut -c1-200)"
# the row's Match tool against the row's REAL fold: r0's shape -> the mutated Program; every other shape -> the row's Build
DERIVED=(--derived "$N/build_request_mut")
for dd in "$D"/build_request_LP*_T*; do
  [ -d "$dd" ] || continue
  [[ "$(basename "$dd")" == "build_request_LP${LP0}_T${T0}" ]] && continue
  DERIVED+=(--derived "$dd")
done
"${PY312%/*}/verity-vllm" batch-decomp --oracle "$D/match" "${DERIVED[@]}" --out "$N/match_decomp.json" > "$N/decomp.log" 2>&1
rc2=$?; log "N3: batch_decomp rc=$rc2"
$PY312 - "$N" "$rc1" "$rc2" <<'EOF' | tee -a "$OUT/negatives.log"
import json, os, sys
n = sys.argv[1]; rcs = [int(x) for x in sys.argv[2:4]]
sch = json.load(open(f"{n}/schedules.json"))
d = json.load(open(f"{n}/match_decomp.json")) if os.path.exists(f"{n}/match_decomp.json") else {}
LP, T = sch["r0_shape"]["LP"], sch["r0_shape"]["T"]
r0 = next((r for r in d.get("requests", []) if r.get("LP") == LP and r.get("T") == T), None)
reasons = []
if r0 is not None and r0.get("canonical_equal") is False:
    fd = r0.get("first_divergences") or []
    hd = [x for x in fd if "GumbelTopPTokenSelect" in json.dumps(x)]
    reasons.append(f"batch_decomp: r0 (LP={LP}, T={T}) canonical_equal=False, {r0.get('n_divergent_steps')} divergent steps "
                   f"(declared S differs at events {sch['events_with_different_S'][:6]}..); first divergences {json.dumps(fd[:2])[:400]}")
    if not hd and fd:
        reasons.append("(the first divergences do not name GumbelTopPTokenSelect: inspect match_decomp.json)")
caught = bool(reasons) and d.get("verdict") != "PASS"
verdict = {"negative": "N3", "expected": "Match FAIL (r0 canonical-unequal on the sampler spec ids at the mutated-chronology events)", "caught": caught,
           "rcs": {"derive": rcs[0], "batch_decomp": rcs[1]}, "decomp_verdict": d.get("verdict"), "r0": None if r0 is None else
           {k: r0.get(k) for k in ("LP", "T", "steps", "canonical_equal", "n_divergent_steps")}, "events_with_different_S": sch["events_with_different_S"],
           "others_equal": [(r.get("LP"), r.get("T"), r.get("canonical_equal")) for r in d.get("requests", []) if r is not r0], "reasons": reasons}
json.dump(verdict, open(f"{n}/verdict.json", "w"), indent=1)
print(f"N3: {'NEGATIVE-CAUGHT' if caught else 'NEGATIVE-NOT-CAUGHT'} decomp_verdict={d.get('verdict')} r0={verdict['r0']}")
for r in reasons: print("   ", r[:700])
EOF
