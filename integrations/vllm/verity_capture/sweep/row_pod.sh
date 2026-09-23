#!/usr/bin/env bash
# sweep/row_pod.sh — pod-side runner of ONE sweep row (SCHEMA §6): Build (Bet A step + request) -> Match (eager capture + fold ->
# cb_a.compare, control tokens) -> Commit + overhead (commit_delta --pairs 3, native_collect_v2b, FA2 matReq tap, 64 openings).
# Early stop at the first failing stage.  Evidence under /workspace/cp/sweep/<row-id>/; a one-line verdict per stage in stages.txt.
# Glossary (Draft 3 §12): a sweep 'row' here is one workload of the qualification table (not a Program Call); timeline.jsonl is the
# execution evidence; manifest.json is the required-value manifest; the checks' verdicts are in commit_summary.json.
#
#   bash verity_capture/sweep/row_pod.sh <row-id> <ROLE> <hf-repo> <revision> [stages=build,match,commit]
#
# [R19, Verity] The tree is the Verity checkout the research runner shipped (`research run --source`): this script's tree is
# integrations/vllm; the core (packages/verity/src) and the runner's telemetry (tools/research/src) sit beside it and go on
# PYTHONPATH.  One Python 3.12 environment (pod_bootstrap.sh: /workspace/venv312 -- torch cu129, vLLM d9105ea80) serves the GPU and
# the CPU halves (PY and PY312 default to it; the two-venv split of the 3.11 pods is gone).  When RESEARCH_RUN_DIR is set the row
# writes $RESEARCH_RUN_DIR/result.json on exit (verity_capture/sweep/research_result.py) so `research inspect` reports the stage
# verdicts separately from the exit code.
#
# Env: PY (venv312 python), PY312 (defaults to PY), HIDDEN_SO, GPU_UTIL (gpu_memory_utilization, default 0.5), PAIRS (3), LP/T (request wrapper shape;
# default = the manifest's longest prompt / largest cap MINUS ONE = decode steps, REL-01/R-16), MATCH_SNAP_STEPS (0,1), MATCH_SNAP_MAX_BYTES
# (unset = run_config derives the per-tensor snapshot cap, D87; set = explicit cap in bytes, recorded in match/run.json snapshot_max_bytes),
# BUILD_TIMEOUT / MATCH_TIMEOUT / COMPARE_TIMEOUT / COMMIT_TIMEOUT (s; Match ones default to B-scaled, REL-06), REQUIRED_CLASSES (REL-02: comma list passed to commit_delta --required-classes),
# EXTRA_ENGINE_ARGS ("k=v k=v" appended to Commit).  [r18-int-3, admit] $D/admission.json = the ADVISORY host-memory plan written
# before Build (verity_capture/telemetry/admission.py plan; per-stage reserved_bytes, named terms, verdict, short_by) -- never a stop.
#
# PROTOCOL SUMMARY (one row = precheck -> Build -> Match -> Commit; stop at the first FAIL; one verdict line per stage in stages.txt)
#   Purpose: run ONE workload row (workloads/<row>.json) from the tree THIS script lives in and leave its evidence in
#     $D = ${SWEEP_DIR:-/workspace/cp/sweep}/<row>  (the launcher puts the release/lane tag in SWEEP_DIR).
#   Inputs: <row-id> <ROLE> <hf-repo> <revision> [stages]; workloads/<row>.json; manifests/checkpoints.json; the env knobs above.
#   Outputs: $D/stages.txt row.log timeline.jsonl source_identity*.json target_family*.json admission.json release.json tree.json;
#     build_step/ build_request*/ build_workload/ build_summary.json; match/ match_decomp.json|match_compare.json global_match.json
#     match_summary.json; manifest.json commit/ commit_summary.json weights_of_record.log; $RESEARCH_RUN_DIR/result.json on EXIT.
#   Invariants: the source-identity and target-family prechecks run before any engine and fail by name PRECHECK_FAIL_* (exit 3);
#     the Program digest handed to the Commit is read from build_summary.json, never from the required-value manifest; a Match verdict over a record
#     this attempt did not produce is FAIL (freshness rule) unless MATCH_PHASE=cpu resumes it by name; a Commit never starts over a
#     Match record missing its account files (PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE, exit 14); `required_manifest verify` FAIL = Commit FAIL.
#   Exit: 0 every requested stage PASS; 10 Build FAIL; 11 Match FAIL; 12 Commit FAIL; 13 B>=2 global manifest not built; 3 precheck FAIL;
#     2 bad argv / MATCH_PHASE; 128+signal when cancelled (TERM/INT/HUP is forwarded to the running stage's process group, see run_to).
#   Protocol spec: Notion -> "vLLM integration — protocol spec (Build · Match · Commit)".
set -uo pipefail
ROW=${1:?row-id}; ROLE=${2:?role}; REPO=${3:?hf repo}; REV=${4:?revision}; STAGES=${5:-build,match,commit}
# [R18 infra, F-r17-tp-06 / BRIEF Phase 0] the tree that runs is the tree THIS SCRIPT lives in -- never the environment, never a default
# path.  $VERITOR_REPO is exported FROM here for every child; a caller that exported another tree is the mixed-tree case and is refused
# by name (a row of record never runs tree A's row_pod.sh over tree B's code).
TREE_SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [ -n "${VERITOR_REPO:-}" ] && [ "$(cd "$VERITOR_REPO" 2>/dev/null && pwd -P)" != "$TREE_SELF" ]; then
  echo "PRECHECK_FAIL_SOURCE_IDENTITY: row_pod.sh at $TREE_SELF but \$VERITOR_REPO=$VERITOR_REPO names another tree" >&2; exit 3
fi
export VERITOR_REPO=$TREE_SELF
cd "$VERITOR_REPO" || exit 3
VERITY_CORE=$(cd "$TREE_SELF/../../packages/verity/src" 2>/dev/null && pwd -P) || { echo "PRECHECK_FAIL_SOURCE_IDENTITY: no packages/verity/src beside $TREE_SELF (not a shipped Verity tree)" >&2; exit 3; }
VERITY_RESEARCH=$(cd "$TREE_SELF/../../tools/research/src" 2>/dev/null && pwd -P) || { echo "PRECHECK_FAIL_SOURCE_IDENTITY: no tools/research/src beside $TREE_SELF" >&2; exit 3; }
export VERITY_CORE VERITY_RESEARCH
export PYTHONPATH=.:$VERITY_CORE:$VERITY_RESEARCH HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
# [r19-int-12, F-r19-int-13] The Match of record is MATCH_IMPL=fast inside MATCH_PIPELINE=shared (veritor policy r17-rules-1 [checker],
# ruling M-1332: "the Match of record for every configuration class rev has signed"; the record checker's route is the asynchronous
# audit from the archive on a CPU host).  In R17/R18 the dead launcher run_row.sh carried that default (`have MATCH_IMPL || ENVS+=
# ("MATCH_IMPL=fast")`) and it did not move with row_pod.sh into the Verity tree: every R19 launch without the env ran the record checker
# -- #73 qwen3-4b B8 GM-01 single-core 14400 s -> "match FAIL rc=0/0/124" (its FP8 twin #74 under cheap's explicit env: 12 min), #60
# 2 h 47 min, canary #3's 2 h 32 min orphan.  The runner owns the policy now; an explicit MATCH_IMPL=record still selects the audit checker.
# (MATCH_SNAP_MAX_BYTES stays as it was: unset = run_config's derived cap, D87; R18's explicit 32 MiB is the launcher's choice.)
export MATCH_IMPL=${MATCH_IMPL:-fast} MATCH_PIPELINE=${MATCH_PIPELINE:-shared}
PY=${PY:-/workspace/venv312/bin/python}; PY312=${PY312:-$PY}
[ -x "$PY" ] || { echo "no python at $PY (run pod_bootstrap.sh first)" >&2; exit 3; }
export PATH=$(dirname "$PY"):/usr/local/cuda/bin:$PATH
NCB_SET=${NATIVE_COLLECT_BUILD:-}   # [R17 rel] the caller's explicit choice wins over the release's per-tree build dir (below)
export NATIVE_COLLECT_BUILD=${NATIVE_COLLECT_BUILD:-/workspace/cp/nc_build}
# hidden_gpu_tree: the worker resolves the default torch-extensions cache that pod_hidden_gpu.sh (X-00) built; only export
# HIDDEN_GPU_BUILD if the caller set it (and ran pod_hidden_gpu.sh with the same value) -- R13 rel: pod_ready checks that cache
[ -n "${HIDDEN_GPU_BUILD:-}" ] && export HIDDEN_GPU_BUILD
HIDDEN_SO_SET=${HIDDEN_SO:-}                                                        # explicit override wins over the target's FA version
HIDDEN_SO=${HIDDEN_SO:-/workspace/cp/fa2/build/matReq/verity_fa2_matReq.so}
GPU_UTIL=${GPU_UTIL:-0.5}; PAIRS=${PAIRS:-3}
D=${SWEEP_DIR:-/workspace/cp/sweep}/$ROW; mkdir -p "$D" "$NATIVE_COLLECT_BUILD"      # SWEEP_DIR: canary.sh keeps its evidence per release tag
WL=workloads/$ROW.json
[ -f "$WL" ] || { echo "no manifest $WL"; exit 2; }
log() { echo "$(date -u +%FT%TZ) [row $ROW] $*" | tee -a "$D/row.log"; }
# [R19] the research runner's result envelope, written once on exit from the stage verdicts (no-op without RESEARCH_RUN_DIR)
_research_result() { [ -n "${RESEARCH_RUN_DIR:-}" ] && "$PY" -m verity_capture.sweep.research_result "$D" "$ROW" "$ROLE" "$STAGES" >> "$D/row.log" 2>&1 || true; }
trap _research_result EXIT
# has_flag <--flag> <cmd...>: feature-detect a CLI flag from the command's --help.  The WHOLE help text is captured first; the former
# `... --help 2>/dev/null | grep -q -- "--flag"` pipeline is unsound under `set -o pipefail`: grep -q exits at the first match (the usage line),
# Python's remaining stdout write hits EPIPE and exits 120, and the pipeline reads as "flag absent" -- a race that became deterministic once
# commit_delta's help text passed 16 KiB (r16-int-2 candidate a2ee36ec2: 17,007 bytes; r16-int-1: 16,065): the qwen canary Commit armed neither
# --match-dir nor --program-dir nor the weights pin and FAILED closed 'C2 of record not established ... not run (no --program-dir)'.  A tree
# whose command does not import (or has no such flag) still reads as absent -> the callers arm nothing, exactly as before (fail-closed).
has_flag() { local _flag=$1; shift; local _h; _h=$("$@" --help 2>/dev/null) || true; [[ "$_h" == *"$_flag"* ]]; }
verdict() { echo "$1 $2 $(date -u +%FT%TZ) $3" | tee -a "$D/stages.txt"; }
# [R16 sys, BRIEF §5/§10] automatic stage telemetry: append-only $D/timeline.jsonl (schema verity-sweep/timeline/v1, verity_capture/sweep/timeline.py).
# This shell marks every stage / sub-step with `date +%s.%N`; the python stages (commit_delta, m1_capture, run_config) append their own
# events through VERITY_TIMELINE (engine build + reuse/restart reason, warm-up, arms, validation components).  Measured, never estimated;
# `python3 -m verity_capture.sweep.timeline report $D/timeline.jsonl` folds it into the BRIEF §5 table (startup / preparation / workload /
# validation / closure) and counts engine starts vs reuses.  Values that parse as numbers / true / false / null are written unquoted.
export VERITY_TIMELINE=${VERITY_TIMELINE:-$D/timeline.jsonl}
tl() {  # <stage> <start|end|mark> [k=v ...]   (+ $TL_EXTRA k=v words, e.g. par=true on concurrent spans -- SYS-06)
  local st=$1 ev=$2; shift 2; local kv="" x k v
  for x in "$@" ${TL_EXTRA:-}; do k=${x%%=*}; v=${x#*=}
    if [[ "$v" =~ ^-?[0-9]+(\.[0-9]+)?$ ]] || [ "$v" = true ] || [ "$v" = false ] || [ "$v" = null ]; then kv="$kv,\"$k\":$v"; else v=${v//\\/\\\\}; kv="$kv,\"$k\":\"${v//\"/\\\"}\""; fi
  done
  printf '{"t":%s,"utc":"%s","schema":"verity-sweep/timeline/v1","proc":"row_pod","pid":%d,"stage":"%s","ev":"%s","hot":false%s}\n' \
    "$(date +%s.%N)" "$(date -u +%FT%TZ)" $$ "$st" "$ev" "$kv" >> "$VERITY_TIMELINE" 2>/dev/null || true
}
# [REL-09] the row cites the release it actually ran on: copy this pod's /workspace/cp/RELEASE.json (tag, commit, artifact shas, env) into the
# evidence at row start and put a `release` line in stages.txt -- the importer must take the tag from HERE, per pod, not from RELEASES.md
# (08:43Z: the dispatcher cited r13-rel5 for every pod while cp-sw2 was still running r13-rel3).  A tree shipped without pod_release has none.
# [REL-14, R15 rev-pm M-0118] the release line describes the TREE WHOSE CODE RUNS (this cwd = VERITOR_REPO), never the pod's global
# RELEASE.json by itself: a lane tree (/workspace/cp/lane-<id>) on a pod whose release tree is r15-int-N must not cite r15-int-N.  The pod's
# RELEASE.json applies only when it names THIS tree (its `tree` == realpath of cwd) and its commit + dirty-patch digest equal this tree's
# EXPORT.json; otherwise the row is UNRELEASED with the tree's own commit/branch/dirty digest and path (the finalizer takes release_tag from here).
TREE_HERE=$(realpath . 2>/dev/null || pwd)
# [R17 rel, exec M-0477 / M-0709] the line comes from verity_capture/sweep/release_json.py: the TREE-LOCAL record (<tree>/RELEASE.json,
# written by the ship that installed this tree) first, the pod-wide /workspace/cp/RELEASE.json (the pod's LAST ship) second -- so a second
# tree shipped beside this one never erases this tree's citation, and a record that names another tree never applies here.  The record
# actually used is copied into the evidence (release.json), not the pod-wide file blindly.
TAB=$'\t'
REL_OUT=$(python3 verity_capture/sweep/release_json.py line "$TREE_HERE" 2>/dev/null) || REL_OUT="UNRELEASED|commit=unknown tree=$TREE_HERE (release_json.py failed; not installed here by pod_release.sh)${TAB}-"
REL_LINE=${REL_OUT%%"$TAB"*}; REL_SRC=${REL_OUT#*"$TAB"}
if [ "${REL_LINE%%|*}" != UNRELEASED ] && [ -f "$REL_SRC" ]; then cp "$REL_SRC" "$D/release.json"; else rm -f "$D/release.json"; fi
# [R17 rel] a released tree loads its collector from the dir the release built it in (RELEASE.json build_dirs.native_collect: per tree for an
# own tree, so two trees on one pod never JIT-build in one directory) unless the caller set NATIVE_COLLECT_BUILD explicitly
if [ -z "$NCB_SET" ] && [ -f "$D/release.json" ]; then
  NCB_REL=$(python3 -c "import json,sys;print((json.load(open(sys.argv[1])).get('build_dirs') or {}).get('native_collect') or '')" "$D/release.json" 2>/dev/null)
  if [ -n "$NCB_REL" ] && [ "$NCB_REL" != "$NATIVE_COLLECT_BUILD" ]; then export NATIVE_COLLECT_BUILD=$NCB_REL; mkdir -p "$NATIVE_COLLECT_BUILD"; log "collector build dir from the release: $NATIVE_COLLECT_BUILD"; fi
fi
python3 -c "import json,sys;json.dump({'schema':'verity-sweep/executing-tree/v1','tree':sys.argv[1],'export':json.load(open('EXPORT.json')) if __import__('os').path.exists('EXPORT.json') else None,'release_line':sys.argv[2]},open(sys.argv[3],'w'),indent=1)" "$TREE_HERE" "$REL_LINE" "$D/tree.json" 2>/dev/null
verdict release "${REL_LINE%%|*}" "${REL_LINE#*|}" >/dev/null
# [R18 infra, F-r17-tp-06 / BRIEF Phase 0] PRECHECK before any stage: under the SAME interpreters and PYTHONPATH the stages use, every
# important package (verity.ir, verity_capture, vllm-poc, hidden_gpu, telemetry) must resolve from THIS tree; the record
# ($D/source_identity.json: resolved_path / repo_root / git_sha per package + sys.path[0]) is the attempt's evidence.  A mismatch is
# `precheck FAIL PRECHECK_FAIL_SOURCE_IDENTITY` in stages.txt and the row stops here -- no engine, no inference.
tl precheck start
SID_SHA=$(python3 -c "import json;print(json.load(open('EXPORT.json')).get('commit') or '')" 2>/dev/null || true)
SID_RC=0
for _py in "$PY" $([ "$PY312" != "$PY" ] && echo "$PY312"); do
  [ -x "$_py" ] || continue
  if ! "$_py" -m verity_capture.sweep.source_identity --tree "$TREE_HERE" ${SID_SHA:+--sha "$SID_SHA"} --stage precheck --out "$D/source_identity.$(basename "$(dirname "$(dirname "$_py")")").json" >> "$D/row.log" 2>&1; then SID_RC=3; fi
done
if [ "$SID_RC" != 0 ]; then
  tl precheck end rc=3 fail_class=PRECHECK_FAIL_SOURCE_IDENTITY
  verdict precheck FAIL "PRECHECK_FAIL_SOURCE_IDENTITY (see $D/source_identity.*.json and row.log)"; exit 3
fi
cp "$D/source_identity.$(basename "$(dirname "$(dirname "$PY")")").json" "$D/source_identity.json" 2>/dev/null || cp "$(ls "$D"/source_identity.*.json | head -1)" "$D/source_identity.json"
# the attempt manifest (increment A job.json, written by run_row.sh's `tele init` before this script) carries the record too
if [ -f "$D/job.json" ]; then
  python3 - "$D/job.json" "$D/source_identity.json" <<'EOF' 2>/dev/null || true
import json, os, sys
p, s = sys.argv[1], sys.argv[2]
d = json.load(open(p)); d.setdefault("computation", {})["source_identity"] = json.load(open(s)); d["computation"]["source_identity_reason"] = None
tmp = p + ".tmp"; open(tmp, "w").write(json.dumps(d, indent=1) + "\n"); os.replace(tmp, p)
EOF
fi
tl precheck end rc=0
verdict precheck PASS "source identity == $TREE_HERE${SID_SHA:+ @ $SID_SHA} (verity verity_vllm verity_capture verity_vllm_adapter hidden_gpu research)" >/dev/null
IFS=', ' read -r B I O LP_DEF T_DEF SNAP_DEF CK <<< "$($PY - "$WL" <<'EOF'
import json, sys
w = json.load(open(sys.argv[1])); s = w["sweep"]
ck = json.load(open("manifests/checkpoints.json"))
# REL-01 (R13 rel; reviewer R-16): request-wrapper --T = number of DECODE steps = max_output - 1 (the prefill step samples
# token 1; max_tokens=O is generated in O engine steps).  --T O built a program with O+1 steps and failed Match at the step count.
lp = max(s["input_lengths"]); t = max(s["output_caps"]) - 1
# [R15 stoch, DECISION M-0529 (A)] the request Program of a STOCHASTIC row carries the per-event split count S of the request its shape
# identifies in the declared chronology, so build_request must be the shape of a DECLARED request: the first request (r0: prompt_len,
# cap - 1), not the row envelope (max input, max cap - 1), which no request of a mixed workload has.  X-09 derives the other shapes
# in Match as before.  Greedy rows keep the envelope (their Program has no batch-dependent constant).  [rel: no apostrophes in this
# heredoc -- bash 3.2 (laptop release gate) parses quotes inside a heredoc nested in \$( ); pod bash 5 does not care]
sp = s.get("sampling_params") or {}
snap = "0,1"     # MATCH_SNAP_STEPS default: greedy rows snapshot the two X-09 steps
if float(sp.get("temperature", 0) or 0) > 0 and w.get("requests"):
    r0 = w["requests"][0]
    lp = int(r0.get("prompt_len") or len(r0.get("prompt_token_ids") or [])); t = int(r0["max_tokens"]) - 1
    # [R15 stoch M-0713, rev-pm M-0541 T1] a stochastic row snapshots EVERY engine step: stoch_recompute is PASS only when every sampled
    # row of the record is value-checked (seed / T / p / pos consumed by the kernels, masked row, token); k of n is INCOMPLETE = Match FAIL.
    snap = "all"
import os
local = None
for c in ck["checkpoints"]:
    if c.get("repo") == w["model"] and str(c.get("revision","")).startswith(w["tokenizer_revision"][:10]):
        local = c.get("local_path"); break
if not local or not os.path.exists(local):
    base = f"/workspace/hf/hub/models--{w['model'].replace('/','--')}/snapshots"
    for d in sorted(os.listdir(base)) if os.path.exists(base) else []:
        if d.startswith(w["tokenizer_revision"][:10]): local = f"{base}/{d}"
# [rel r16-int-2 canary] the read above splits on IFS=", " -- a comma INSIDE a field splits it: the greedy default "0,1" became SNAP_DEF=0
# and CK="1 /workspace/hf/..." (no such dir -> weights of record NOT derived -> Commit FAIL weights unpinned).  Emit ":" and map it back.
print(s["batch"], s["max_input"], s["max_output"], lp, t, snap.replace(",", ":"), local)
EOF
)"
SNAP_DEF=${SNAP_DEF//:/,}
LP=${LP:-$LP_DEF}; T=${T:-$T_DEF}; MML=$((I + O))
# [R13 hopper] the row's declared TargetProfile (manifest `target` block, e.g. {"compute_capability":[9,0],"num_sms":132,"flash_attn_version":2}):
# Build gets it as --target; the GPU stages (m1_capture / census / noninterference / commit_delta) read the same block from the manifest.
TGT=$($PY -c 'import json,sys; t=json.load(open(sys.argv[1])).get("target"); print(json.dumps(t, sort_keys=True) if t else "")' "$WL")
TGT_ARGS=(); [ -n "$TGT" ] && TGT_ARGS=(--target "$TGT")
# [R13 hopper] the attention implementation the row executes = TargetProfile.effective_flash_attn_version (declared knob, else the pinned
# vLLM's default for this device's capability: cc 9.x -> FA3).  It selects the M1 tap .so (FA2: fa2_tap_src / pod_fa2_tap.sh; FA3:
# fa3_tap_src / pod_fa3_tap.sh) unless HIDDEN_SO is given; commit_delta re-derives the same fact from the built engine and refuses a
# tap of the other implementation (F-hw-08 fail-closed), and requires the matching hidden class (fa2_/fa3_hidden_m1_stream).
FA_VER=$($PY - "$WL" <<'EOF' 2>/dev/null || echo ""
import json, sys
from verity_vllm.frontend.target_profile import TargetProfile
wl = json.load(open(sys.argv[1]))
base = TargetProfile.from_device()
d = base.to_json(); d.update({k: v for k, v in (wl.get("target") or {}).items() if v is not None})
print(TargetProfile.from_json(d).effective_flash_attn_version())
EOF
)
if [ -z "${HIDDEN_SO_SET:-}" ] && [ "${FA_VER:-}" = "3" ]; then HIDDEN_SO=/workspace/cp/fa2/build/fa3_matReq/verity_fa3_matReq.so; fi
CODE9=$(python3 -c "import json;print(json.load(open('EXPORT.json'))['commit'][:9])" 2>/dev/null)
# [R18 infra, r18-int-2; x09 M-0009 / coord M-0026] TARGET-FAMILY PREFLIGHT before Build and before any engine: the row id's device class
# (`__l40s__` = cc 8.9), the manifest's declared target (if any) and the pod GPU the stages run on must be ONE family -- #60's Build of
# record was a Hopper Program built on the H100 host for an L40S row.  Fails by name PRECHECK_FAIL_TARGET_FAMILY (rc 3, no Build);
# $TARGET_FAMILY_WAIVER=<reason> (the release canary, whose __l40s__ base rows also run on the Hopper canary host) records
# `precheck-target WAIVED <reason>` instead -- never silent, never on a row of record.  Record: $D/target_family.json.  Checked again
# after Build against build_summary.json step.target (verity_capture/sweep/target_family.py).
tl precheck-target start
TF_LINE=$(python3 -m verity_capture.sweep.target_family --row "$ROW" --workload "$WL" --stages "$STAGES" --out "$D/target_family.json" 2>&1); TF_RC=$?
echo "$TF_LINE" >> "$D/row.log"
if [ $TF_RC -ne 0 ]; then
  tl precheck-target end rc=3 fail_class=PRECHECK_FAIL_TARGET_FAMILY
  verdict precheck-target FAIL "$TF_LINE"; exit 3
fi
tl precheck-target end rc=0
case "$TF_LINE" in TARGET-FAMILY-WAIVED*) verdict precheck-target WAIVED "$TF_LINE" >/dev/null ;; *) verdict precheck-target PASS "$TF_LINE" >/dev/null ;; esac
# [R18 admit, F-r18-admit-1 / S-r18-3 C2; coord M-0061 (2), r18-int-3] ADVISORY host-memory admission plan before Build: the empirical
# planner (verity_capture/telemetry/admission.py: base serving + simultaneously-live named Verity terms + two named constant margins,
# no multiplier) predicts each stage's peak from the workload file and this cgroup's memory.max and writes $D/admission.json (per stage
# reserved_bytes, the named terms, verdict ADMIT / REFUSE / LOWER_BOUND, short_by) beside the attempt; one line in row.log; a WARN line
# when a stage WOULD be refused.  ADVISORY ONLY: this never stops the row (no row of record is stopped by a day-old model -- coord M-0061;
# enforcement is post-campaign, after the record half C2 on the R18 attempts).  `plan` exits 0 on every path, including its own failure.
tl admission start
ADM_LINE=$(python3 -m verity_capture.telemetry.admission plan --row "$ROW" --workload "$WL" --stages "$STAGES" --pairs "$PAIRS" \
             --snapshot-steps "${MATCH_SNAP_STEPS:-$SNAP_DEF}" ${MATCH_SNAP_MAX_BYTES:+--snapshot-max-bytes "$MATCH_SNAP_MAX_BYTES"} \
             --out "$D/admission.json" 2>&1) || ADM_LINE="ADMISSION advisory unavailable rc=$? $ADM_LINE"
log "$ADM_LINE"
case "$ADM_LINE" in *"WARN would-refuse"*) log "WARN admission advisory: a stage would be REFUSED by the planner (see $D/admission.json); advisory in r18-int-3 -- the row continues" ;; esac
tl admission end rc=0 advisory=true
log "start B=$B I=$I O=$O LP=$LP T=$T ckpt=$CK stages=$STAGES target=${TGT:-device} hidden_so=$HIDDEN_SO code=$CODE9"
tl row start stages="$STAGES" B="$B" I="$I" O="$O" LP="$LP" T="$T" code="$CODE9" release="${REL_LINE%%|*}" pod="$(hostname 2>/dev/null)"

# [r19-int-19, F-r19-int-6] a cancelled row must take its stage with it.  The research runner (tele run) starts this shell as a session
# leader and cancels with killpg(pgid, SIGTERM).  GNU `timeout` puts itself and the stage in a NEW process group (so that its own expiry
# can kill the stage's forked workers -- fold workers, commit_delta's legs); that group never sees the runner's killpg, so the shell died
# and the stage lived on with ppid 1 under its own 14400 s clock: #23's record GM-01 ran 54 min / 46 GB past its cancelled run, #11's the
# same class (R19).  `--foreground` would fix the cancel path and break the expiry path (timeout then signals only its direct child).
# Fix: keep timeout's group and FORWARD: the stage runs in the background under `wait` (bash runs a trap only after a foreground child
# ends, but `wait` returns at once on a trapped signal); the trap signals timeout's pid, and timeout forwards to the stage and to its
# whole group (coreutils cleanup(): send_sig(child), send_sig(0, sig) when not --foreground).  The shell then waits for the stage to be
# gone, records the cancel on the timeline / stage log, and exits 128+signal -- the EXIT trap still writes the research result envelope.
# --- row_pod.signal-forwarding begin (extracted by verity_capture/tests/test_row_pod_cancel_forwarding.py)
RUN_TO_PID=""; ROW_SIGNALLED=""
_on_signal() {  # <TERM|INT|HUP>
  ROW_SIGNALLED=$1
  if [ -n "$RUN_TO_PID" ] && kill -0 "$RUN_TO_PID" 2>/dev/null; then
    log "SIG$1 received: forwarding to the running stage (timeout pid $RUN_TO_PID -> its process group)"
    kill -s "$1" "$RUN_TO_PID" 2>/dev/null || true
  fi
}
trap '_on_signal TERM' TERM; trap '_on_signal INT' INT; trap '_on_signal HUP' HUP
_signum() { case "$1" in TERM) echo 15;; INT) echo 2;; HUP) echo 1;; *) echo 15;; esac; }

run_to() {  # <timeout-s> <logfile> <cmd...>
  local to=$1 lf=$2; shift 2
  echo "\$ $*" > "$lf"
  timeout "$to" "$@" >> "$lf" 2>&1 &
  RUN_TO_PID=$!
  local rc
  while :; do
    wait "$RUN_TO_PID"; rc=$?
    # a trapped signal returns from `wait` (rc > 128) while the stage is still alive: the trap forwarded it -- keep waiting for the stage
    kill -0 "$RUN_TO_PID" 2>/dev/null || break
  done
  RUN_TO_PID=""
  [ $rc -eq 124 ] && echo "TIMEOUT after ${to}s" >> "$lf"
  if [ -n "$ROW_SIGNALLED" ]; then
    echo "CANCELLED by SIG$ROW_SIGNALLED (stage rc=$rc)" >> "$lf"
    log "row cancelled by SIG$ROW_SIGNALLED; stage ended rc=$rc -> stop"
    tl row cancelled signal="$ROW_SIGNALLED" rc="$rc"
    tl row end ok=false stopped_at=cancel
    exit $((128 + $(_signum "$ROW_SIGNALLED")))
  fi
  return $rc
}
# --- row_pod.signal-forwarding end
# [R16 sys] run_to with a timeline span named after the sub-step: `run_tl <stage> <timeout-s> <logfile> <cmd...>` (rc on the end event)
run_tl() { local st=$1; shift; tl "$st" start; run_to "$@"; local rc=$?; tl "$st" end rc=$rc; return $rc; }

# [R16 sys SYS-02e] one engine per pod: a live hot Commit worker (HOT_ENGINE=1, idle up to HOT_IDLE s) holds its engine's GPU memory
# (22.5 GiB of the L40S for the 1B row at util 0.5); a Build / Match / cold Commit engine started beside it fails vLLM's startup check
# ("Free memory ... less than desired GPU memory utilization" -- seen live on cp-sys, Match control 06:27Z).  So every stage that starts
# its own engine first asks the pod's hot workers to stop: an IDLE worker exits (its memory is back within seconds), a worker BUSY with
# another row's Commit is waited for -- never killed -- and the wait is on the timeline.  Torch-free (hot_commit stop = worker.json + signals).
gpu_exclusive() {  # <why>
  local root=${HOT_ROOT:-/workspace/cp/hot} t0 out
  ls "$root"/*/worker.json >/dev/null 2>&1 || return 0
  t0=$(date +%s.%N)
  out=$($PY -m verity_capture.sweep.hot_commit stop --root "$root" 2>&1 | tail -3 | tr '\n' ' ' | cut -c1-300)
  local s; s=$(python3 -c "print(round($(date +%s.%N) - $t0, 3))")
  case "$out" in *'"stopped": []'*|"") ;; *) log "hot worker(s) stopped before $1 (one engine per pod, ${s}s): $out";; esac
  tl hot.gpu_exclusive mark why="$1" s="$s" out="$out"
}

# ---------------------------------------------------------------- Build (Bet A)
if [[ ",$STAGES," == *,build,* ]]; then
  # [REL-08] the request-wrapper derive is O(LP x T) instances (and grows with the model): 900 s fit 1B at 256/31 but Mistral-7B at
  # 1024/127 took 1195 s (fa2 07:50Z) -> default scales with the envelope, 900 s x max(1, LP/256) x max(1, T/31), capped at 4 h; BUILD_TIMEOUT overrides.
  ENV_SCALE=$(( (LP < 256 ? 256 : LP) / 256 * ((T < 31 ? 31 : T) / 31) )); ENV_SCALE=$(( ENV_SCALE < 1 ? 1 : ENV_SCALE ))
  TO=${BUILD_TIMEOUT:-$(( 900 * ENV_SCALE > 14400 ? 14400 : 900 * ENV_SCALE ))}
  # [X-05] the manifest's sampling policy is a structural input of the construction (a stochastic row is refused, not built greedy);
  # [R13 hopper] the manifest's declared target (--target) is another
  CMD_S=($PY -m verity_capture.experimental.cb_a.derive_step --model "$REPO" --revision "$REV" --wrapper step --M "$B" --pos0 8 --max-model-len "$MML" --workload "$WL" --out "$D/build_step" "${TGT_ARGS[@]}")
  # [R15 hopper HOP-03, M-0225] B >= 2: every request Program is built under its LAUNCH CONTEXT (per request step the varlen launch's
  # max_seqlen_q over the co-scheduled prompts, verity_capture.sweep.launch_context from the workload's arrivals/lengths/caps): FA3's tile
  # rule reads the launch's value, not the request's own rows (an 8-token prompt batched with a 1024-token one runs the 192 x 128 tile).
  # B = 1 passes nothing (own rows = the launch: digests/identities unchanged).  A shape whose requests are co-scheduled differently gets
  # 'conflict' -> derive_step refuses under FA3 (fail closed), records it inert under FA2.
  # [HOP-03b, rev-pm M-0326 T3] a manifest `launch_context` cannot read (rc != 0) FAILS the Build of that shape -- it is not "no context"
  # (which would build the H100 B >= 2 request Program as a batch of one).  The shape's derive is skipped, so its result.json is absent and
  # build_summary marks it FAIL; the module's stderr is kept beside the row log.
  launch_args() { local lc rc; [ "$B" -gt 1 ] || return 0
    lc=$($PY -m verity_capture.sweep.launch_context "$WL" "$1" "$2" 2>"$D/launch_context_LP$1_T$2.err"); rc=$?
    if [ $rc -ne 0 ]; then log "Build FAIL shape LP$1 T$2: launch_context rc=$rc -- $(tr '\n' ' ' < "$D/launch_context_LP$1_T$2.err" | tail -c 300)"; return $rc; fi
    rm -f "$D/launch_context_LP$1_T$2.err"; [ -n "$lc" ] && printf -- '--launch-max-seqlen-q\n%s\n' "$lc"; return 0; }
  LCTX_R=(); lc_ok=0
  if lcr=$(launch_args "$LP" "$T"); then [ -n "$lcr" ] && mapfile -t LCTX_R <<< "$lcr"; else lc_ok=1; fi
  CMD_R=($PY -m verity_capture.experimental.cb_a.derive_step --model "$REPO" --revision "$REV" --wrapper request --LP "$LP" --T "$T" --max-model-len "$MML" --workload "$WL" --out "$D/build_request" "${TGT_ARGS[@]}" "${LCTX_R[@]}")
  t0=$(date +%s); tl stage.build start
  gpu_exclusive build
  # [R16 sys SYS-06] BUILD_JOBS: the derives are independent CPU-only processes (derive_step runs the model on the meta device -- GPU
  # 0 MiB during a derive, one thread at ~100 % CPU, RSS 1-18 GB per shape as measured on cp-r1/r2/r3 08:52Z) and row_pod ran them one
  # after another: a B8 1024/128 row spends 1000-3800 s in 8 serial derives on a 128-256 vCPU host (cp-r1 smollm2 B8 1024/128: 3758 s
  # and counting; cp-r3 qwen B8 1024/128: 2859 s).  BUILD_JOBS=N runs up to N derives at once (step + main request + the extra shapes);
  # every derive is the same process with the same arguments and its own --out, so the Program digests cannot change -- only the
  # schedule does (measured: llama B8 256/32 Build 8-wide, all 10 Program digests + workload_digest == the serial row of record).
  # Default 1 = the serial order below, unchanged.  BUILD_JOBS=auto = clamp(headroom / 20 GiB, 1, 8) where headroom = cgroup memory.max -
  # memory.current (F-dA-15: a container's bound is the cgroup; a derive that dies of OOM is rc 137 -> Build FAIL by rc, never a wrong
  # Program).  Each background job writes its rc beside its log; Build FAILs if any rc != 0.  The decision is on the timeline
  # (`build.jobs`) and in the row log.
  BUILD_JOBS_REQ=${BUILD_JOBS:-1}; BUILD_JOBS=$BUILD_JOBS_REQ
  if [ "$BUILD_JOBS" = auto ]; then
    BUILD_JOBS=$(python3 - <<'EOF'
def rd(p):
    try:
        v = open(p).read().strip(); return int(v) if v.isdigit() else None
    except OSError:
        return None
mx = rd("/sys/fs/cgroup/memory.max") or rd("/sys/fs/cgroup/memory/memory.limit_in_bytes")
cur = rd("/sys/fs/cgroup/memory.current") or rd("/sys/fs/cgroup/memory/memory.usage_in_bytes")
if mx is None or mx >= (1 << 60) or cur is None:
    kv = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in open("/proc/meminfo") if l.split()[1].isdigit()}
    head = kv.get("MemAvailable", 0)
else:
    head = mx - cur
print(max(1, min(8, head // (20 << 30))))
EOF
    )
  fi
  [[ "$BUILD_JOBS" =~ ^[0-9]+$ ]] || BUILD_JOBS=1
  tl build.jobs mark jobs="$BUILD_JOBS" requested="$BUILD_JOBS_REQ"
  # [R13 X-09 M-BATCH-DECOMP] a B>1 mixed batch is B request Programs, one per distinct (prompt length, cap-1) of the manifest -- not only
  # the longest/largest one: derive the other shapes here (Build declares the whole batch; Match's decomposition still derives a shape it
  # meets that is not on disk, e.g. an early EOS makes T_r < cap_r-1). Failures of the extra shapes fail Build like the main one.
  # [R16 stoch, M-0713 (2)] the shape EXCLUDED here is the one build_request was derived for -- ($LP, $T): the envelope for greedy rows,
  # r0's declared shape for stochastic rows (M-0529 (A)) -- never the envelope by name: with r0 = (8, 114) the envelope exclusion built
  # build_request_LP8_T114 beside build_request and the GP-01 compose refused 'two derived dirs for shape (8, 114)' (B8 stoch, 05:13Z).
  extra_shapes() { [ "$B" -gt 1 ] || return 0; $PY - "$WL" "$LP" "$T" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))["sweep"]
main = (int(sys.argv[2]), int(sys.argv[3]))
pairs = sorted({(int(lp), int(c) - 1) for lp, c in zip(s["input_lengths"], s["output_caps"])})
print(" ".join(f"{lp},{t}" for lp, t in pairs if (lp, t) != main))
EOF
  }
  # [R19 F-r19-int-3] a shape is on disk only if its derive SUCCEEDED: derive_step writes result.json {"ok": false, ...} on a crash too
  # (#73 h100b 06:22Z: 8 shapes crashed on a short --revision, the 06:39Z re-run skipped all 8 on `-f result.json`, the step + main
  # request re-derived for 35 min, and build_summary read the 8 stale failures -> Build FAIL).  A failed shape is re-derived.
  derived_ok() { [ -f "$1/result.json" ] && $PY -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("ok") is True else 1)' "$1/result.json" 2>/dev/null; }
  if [ "$BUILD_JOBS" -le 1 ]; then
    run_tl build.derive_step "$TO" "$D/build_step.log" "${CMD_S[@]}"; rc1=$?
    if [ $lc_ok -eq 0 ]; then run_tl build.derive_request "$TO" "$D/build_request.log" "${CMD_R[@]}"; rc2=$?; else rc2=1; fi
    for lpt in $(extra_shapes); do
      IFS=, read -r lp t <<< "$lpt"
      derived_ok "$D/build_request_LP${lp}_T${t}" && continue
      # [R15 fp8] the batch's other shapes are built for the SAME declared target as the main one (rc3 dropped --target here: the FP8 row's
      # extra shapes derived for vLLM's default kernel selection -> DeepGEMM custom op -> Build FAIL while the main shape passed)
      LCTX_X=()
      if lcx=$(launch_args "$lp" "$t"); then [ -n "$lcx" ] && mapfile -t LCTX_X <<< "$lcx"; else rc2=1; continue; fi
      CMD_X=($PY -m verity_capture.experimental.cb_a.derive_step --model "$REPO" --revision "$REV" --wrapper request --LP "$lp" --T "$t" --max-model-len "$MML" --workload "$WL" --out "$D/build_request_LP${lp}_T${t}" "${TGT_ARGS[@]}" "${LCTX_X[@]}")
      run_tl "build.derive_request_LP${lp}_T${t}" "$TO" "$D/build_request_LP${lp}_T${t}.log" "${CMD_X[@]}"; rcx=$?
      [ $rcx -eq 0 ] || rc2=$rcx
    done
  else
    # SYS-06 parallel schedule: same commands, same logs, same timeline span names; at most BUILD_JOBS derives alive at once.
    # par=true on every concurrent span: the report's nesting rule (a span strictly containing another is a wrapper, not a leaf) must not
    # eat a long derive that happens to overlap a short one -- concurrent spans are never containers (timeline.report).
    build_job() { local st=$1 lf=$2; shift 2; TL_EXTRA="par=true" run_tl "$st" "$TO" "$lf" "$@"; echo $? > "$lf.rc"; }
    launch() {  # <stage> <logfile> <cmd...>
      while [ "$(jobs -rp | wc -l)" -ge "$BUILD_JOBS" ]; do wait -n 2>/dev/null || sleep 1; done
      rm -f "$2.rc"; build_job "$@" & }
    rc_of() { cat "$1.rc" 2>/dev/null || echo 1; }
    RCF=()
    launch build.derive_step "$D/build_step.log" "${CMD_S[@]}"
    [ $lc_ok -eq 0 ] && launch build.derive_request "$D/build_request.log" "${CMD_R[@]}"
    rc2x=0
    for lpt in $(extra_shapes); do
      IFS=, read -r lp t <<< "$lpt"
      derived_ok "$D/build_request_LP${lp}_T${t}" && continue
      LCTX_X=()
      if lcx=$(launch_args "$lp" "$t"); then [ -n "$lcx" ] && mapfile -t LCTX_X <<< "$lcx"; else rc2x=1; continue; fi
      CMD_X=($PY -m verity_capture.experimental.cb_a.derive_step --model "$REPO" --revision "$REV" --wrapper request --LP "$lp" --T "$t" --max-model-len "$MML" --workload "$WL" --out "$D/build_request_LP${lp}_T${t}" "${TGT_ARGS[@]}" "${LCTX_X[@]}")
      launch "build.derive_request_LP${lp}_T${t}" "$D/build_request_LP${lp}_T${t}.log" "${CMD_X[@]}"; RCF+=("$D/build_request_LP${lp}_T${t}.log")
    done
    wait
    rc1=$(rc_of "$D/build_step.log")
    if [ $lc_ok -eq 0 ]; then rc2=$(rc_of "$D/build_request.log"); else rc2=1; fi
    for lf in "${RCF[@]}"; do rcx=$(rc_of "$lf"); [ "$rcx" -eq 0 ] || rc2=$rcx; done
    [ $rc2x -eq 0 ] || rc2=$rc2x
    log "Build derives ran $BUILD_JOBS-wide (SYS-06): step rc=$rc1 request rc=$rc2 shapes=$(( ${#RCF[@]} + 1 )) wall=$(( $(date +%s) - t0 ))s"
  fi
  wall=$(( $(date +%s) - t0 ))
  tl build.summary start
  ok=$($PY - "$D" <<'EOF'
import json, sys, os
d = sys.argv[1]; out = {}
for w in ("step", "request"):
    p = f"{d}/build_{w}/result.json"
    r = json.load(open(p)) if os.path.exists(p) else {"ok": False, "message": "no result.json"}
    art = f"{d}/build_{w}/artifact.json"
    a = json.load(open(art)) if os.path.exists(art) else {}
    out[w] = {"ok": bool(r.get("ok")), "unsupported": r.get("unsupported"), "exception": r.get("exception"), "message": (r.get("message") or "")[:300],
              "program_digest": (r.get("program") or {}).get("digest") or a.get("program_digest"), "identity": a.get("identity"),
              "target": (a.get("target_profile") or {}), "applicability": a.get("applicability")}
import glob
for p in sorted(glob.glob(f"{d}/build_request_LP*_T*/result.json")):  # [X-09] the batch's other request shapes
    r = json.load(open(p)); w = os.path.basename(os.path.dirname(p))
    out[w] = {"ok": bool(r.get("ok")), "unsupported": r.get("unsupported"), "message": (r.get("message") or "")[:300], "program_digest": (r.get("program") or {}).get("digest")}
json.dump(out, open(f"{d}/build_summary.json", "w"), indent=1, default=str)
print("PASS" if all(v["ok"] for v in out.values()) else "FAIL")
EOF
)
  # [R14 GP-01] Build under r14-v1 = ONE rooted workload Program of the complete declared manifest: the request Programs above are its
  # components (weights once, one call node per request, declared arrivals + caps, global address map + chronology). Any B >= 1.
  # Its digest is the row's `program_digest` for the identity key (TABLE-RULES); the request digests stay recorded beside it.
  WL_DIGEST=""
  tl build.summary end ok="$ok"
  if [ "$ok" = PASS ]; then
    DER=(--derived "$D/build_request"); for dd in "$D"/build_request_LP*_T*; do [ -d "$dd" ] && DER+=(--derived "$dd"); done
    # [rev-pm F-pm-24] the pinned checkpoint's eos_token_id (config.json / generation_config.json) is the Program's declared STOP RULE;
    # GM-01 G6 verifies every request that stopped before its cap ended on one of these ids (fails closed without the declaration)
    run_tl build.global_program "$TO" "$D/build_workload.log" $PY312 -m verity_capture.experimental.cb_a.global_program --workload "$WL" "${DER[@]}" --out "$D/build_workload" --checkpoint-dir "$CK"; rcw=$?
    WL_DIGEST=$(python3 -c "import json;print(json.load(open('$D/build_workload/workload_program.json')).get('workload_digest') or '')" 2>/dev/null)
    [ $rcw -eq 0 ] && [ -n "$WL_DIGEST" ] || { ok=FAIL; log "workload Program compose failed rc=$rcw: $(python3 -c "import json;w=json.load(open('$D/build_workload/workload_program.json'));print(w.get('reason'),w.get('message') or '')" 2>/dev/null) (see build_workload.log)"; }
    python3 - "$D" "$WL_DIGEST" "$rcw" <<'EOF'
import json, sys
d, wd, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = json.load(open(f"{d}/build_summary.json"))
import os
W = json.load(open(f"{d}/build_workload/workload_program.json")) if os.path.exists(f"{d}/build_workload/workload_program.json") else {}
# [COORD M-0063 (1)] the workload Program's declared applicability (execution profile + engine conditions it models); a refusal is a
# named Build FAIL (e.g. a bi-compiled row over eager components), not a compose error
s["workload"] = {"ok": rc == 0 and bool(wd), "workload_digest": wd or None, "mechanism": "GP-01 cb_a.global_program", "artifact": "build_workload/workload_program.json",
                 "reason": W.get("reason"), "message": W.get("message"), "applicability": W.get("applicability"),
                 "stop_rule": W.get("stop_rule")}
json.dump(s, open(f"{d}/build_summary.json", "w"), indent=1, default=str)
EOF
  fi
  # [R18 infra, r18-int-2] the Build that came out must be the row's family too (build_summary.json step.target.compute_capability):
  # a PASS Build for another target is a named FAIL (PRECHECK_FAIL_TARGET_FAMILY), or WAIVED by name under the canary's waiver.
  if [ "$ok" = PASS ]; then
    TFB_LINE=$(python3 -m verity_capture.sweep.target_family --row "$ROW" --workload "$WL" --stages "$STAGES" --no-device --build-summary "$D/build_summary.json" --out "$D/target_family.build.json" 2>&1); TFB_RC=$?
    echo "$TFB_LINE" >> "$D/row.log"
    if [ $TFB_RC -ne 0 ]; then ok=FAIL; verdict build-target FAIL "$TFB_LINE"; else case "$TFB_LINE" in TARGET-FAMILY-WAIVED*) verdict build-target WAIVED "$TFB_LINE" >/dev/null ;; esac; fi
  fi
  verdict build "$ok" "rc=$rc1/$rc2 wall=${wall}s digest=$(python3 -c "import json;print((json.load(open('$D/build_summary.json'))['step'].get('program_digest') or '')[:16])") workload_digest=${WL_DIGEST:0:16}"
  tl stage.build end ok="$ok" wall_s="$wall"
  [ "$ok" = PASS ] || { log "BUILD FAIL -> stop"; tl row end ok=false stopped_at=build; exit 10; }
fi

# ---------------------------------------------------------------- Match (eager capture + fold -> comparator; control tokens)
if [[ ",$STAGES," == *,match,* ]]; then
  # [REL-06] Match-tool cost scales with the batch (fold: events x steps; comparator: instances): the R12 constants (1800 s fold, 600 s
  # comparator) were tuned on B=1 256/32 and produced INFRA-TIMEOUTs, not verdicts, at B=4 (F-report-03/04) and b16/1024 (F-tp-05/06).
  # Defaults now scale with B (floor at the old values x1 / x4), capped at 4 h; MATCH_TIMEOUT / COMPARE_TIMEOUT override.
  SCALE_B=$(( B < 1 ? 1 : B )); cap4h() { [ "$1" -gt 14400 ] && echo 14400 || echo "$1"; }
  # [REL-06b, fp8 M-0230] the fold also scales with the token envelope (instances ~ (I+O) x layers x steps): a 1024/128 B1 fold under
  # MATCH_SNAP_STEPS=all streamed 2.24 M instances and was killed by the 1800 s floor ~1 min short (INFRA-TIMEOUT, not a verdict).  The floor
  # is multiplied by ceil((I+O)/288) (the 256/32 envelope the constants were tuned on): 1024/128 -> x4 = 7200 s.  Only ever longer; cap 4 h.
  SCALE_T=$(( (I + O + 287) / 288 )); [ "$SCALE_T" -lt 1 ] && SCALE_T=1
  TO=${MATCH_TIMEOUT:-$(cap4h $(( 1800 * SCALE_B * SCALE_T )))}
  TO_C=${COMPARE_TIMEOUT:-$(cap4h $(( 600 * (SCALE_B < 4 ? 4 : SCALE_B) * SCALE_T )))}
  SKIP="census,census_reconcile,nonint,nonint_compare,compare,root_policy,golden,wiring,replay_a,replay_chain,fold_2"
  # [R17 mfast F2, COORD R17-34 (2) / M-0496, rev M-0500, S-r17-33] the fold's instances.jsonl of record is written in the PROGRESSIONS form
  # (verity-gen/instances/v2: node-space operand references as compact progressions built from the fold's own Refs, no runs() expansion at
  # write; 1.63 GB -> 534 MB on a llama B8 1024/128 row, lossless, every reader accepts both forms).  VERITY_INSTANCES_FORM=runs restores v1.
  export VERITY_INSTANCES_FORM=${VERITY_INSTANCES_FORM:-progressions}
  log "fold: instances.jsonl stored form = $VERITY_INSTANCES_FORM (F2)"
  # [r19-int-13, F-r19-int-14] MATCH_PHASE=cpu resumes the CPU half (resolve -> fold -> gates) over a capture already in $D/match/capture
  # (run_config's own --phase cpu, D87: realized shape from the existing capture, spec kept): #69 olmoe B128's fold died on ENOSPC at its
  # program.json write 2 h 39 min in, with the 74 GB / 188k-file capture complete on disk.  Default all = capture + CPU, unchanged.
  export MATCH_PHASE=${MATCH_PHASE:-all}; case "$MATCH_PHASE" in all|cpu) ;; *) log "MATCH_PHASE=$MATCH_PHASE is not all|cpu"; exit 2;; esac   # exported: the summary's record-freshness rule reads it (r19-int-18)
  [ "$MATCH_PHASE" = cpu ] && { [ -d "$D/match/capture" ] || { verdict match FAIL "MATCH_PHASE=cpu without $D/match/capture"; exit 12; }; log "match: MATCH_PHASE=cpu over the existing capture (no GPU stage)"; }
  CMD_M=($PY312 -m verity_capture.bench.run_config --case "$ROLE" --workload "$WL" --out "$D/match" --checkpoint-dir "$CK" --phase "$MATCH_PHASE"
         --python-gpu "$PY" --python-cpu "$PY312" --max-num-seqs "$B" --snapshot-steps "${MATCH_SNAP_STEPS:-$SNAP_DEF}"
         ${MATCH_SNAP_MAX_BYTES:+--snapshot-max-bytes "$MATCH_SNAP_MAX_BYTES"} --skip "$SKIP"
         --engine-arg "gpu_memory_utilization=$GPU_UTIL" --cards-dir "$D/match/cards")
  # [R18 infra, r18-int-2; cheap M-0080/M-0082, F-r18-cheap-2] MATCH_SNAP_MAX_BYTES set = an explicit per-tensor snapshot cap (run_config
  # records it in match/run.json snapshot_max_bytes with derivation null); unset = byte-identical argv to int-1 (run_config derives, D87).
  t0=$(date +%s); tl stage.match start
  gpu_exclusive match
  run_tl match.run_config "$TO" "$D/match.log" "${CMD_M[@]}"; rcm=$?
  tl match.compare start
  if [ -f "$D/match/program.json" ] && [ -f "$D/match/instances.jsonl" ]; then
    if [ "$B" -gt 1 ]; then
      # [R13 X-09 M-BATCH-DECOMP] at B>1 the fold is the BATCH (one instance per row of every co-scheduled request) while build_request is
      # ONE request's Program: cb_a.compare can never be canonical-equal (F-report-02). The verdict is the batch decomposition: every fold
      # instance attributed to its request (StepStart rows / logits rows), each request's projection compared to the derived Program of
      # ITS OWN shape (LP_r = prompt length, T_r = served decode steps). Shapes not derived yet (shorter prompts / caps of the mixed
      # workload, early EOS) are reported by exit 4 -> derived here (one request-wrapper derivation per distinct (LP,T)) and rerun once.
      DERIVED=(--derived "$D/build_request"); for dd in "$D"/build_request_LP*_T*; do [ -d "$dd" ] && DERIVED+=(--derived "$dd"); done
      if [ "${MATCH_PIPELINE:-}" = "shared" ] && [ -f "$D/build_workload/workload_program.json" ]; then
        # [COORD R17-28, mprof P1] ONE X-09 per row.  The standalone batch_decomp below and GM-01's x09 block are the SAME decompose() over
        # the same fold and the same derived components (fold decode 108 s + 8 component decodes 115 s + Projections / compare_steps /
        # dag_hashes: 365 s at llama B8 1024/128, M-0231) -- run twice, in two processes.  Under MATCH_PIPELINE=shared GM-01 runs first and
        # writes the X-09 document it computed to match_decomp.json (--decomp-out, batch_decomp's CLI schema; exit 4 = missing derived
        # shapes, the same derive-and-rerun protocol), and the batch_decomp process is not started.  Unset = the flow of record.
        tl match.compare end rc=null shared=match.global_match
        GM_SHARED=1
        # the same --derived list batch_decomp got (build_request + every build_request_LP*_T*, + the shapes derived here after an exit 4):
        # X-09 inside GM-01 may use them beside the global Program's components, so match_decomp.json is the document of record key for key
        for pass in 1 2; do
          CMD_G=($PY312 -m verity_capture.experimental.cb_a.global_match --gprog "$D/build_workload/workload_program.json" --workload "$WL" --oracle "$D/match"
                 --out "$D/global_match.json" --within-step "${WITHIN_STEP:-dag}" --decomp-out "$D/match_decomp.json" "${DERIVED[@]}")
          run_tl match.global_match "$TO_C" "$D/global_match.log" "${CMD_G[@]}"; GM_RC=$?
          [ $GM_RC -eq 4 ] && [ $pass -eq 1 ] || break
          for lpt in $($PY312 -c "import json;print(' '.join(f'{lp},{t}' for lp,t in json.load(open('$D/match_decomp.json'))['missing_derived']))"); do
            IFS=, read -r lp t <<< "$lpt"
            log "X-09 (inside GM-01): deriving request Program LP=$lp T=$t"
            CMD_R=($PY -m verity_capture.experimental.cb_a.derive_step --model "$REPO" --revision "$REV" --wrapper request --LP "$lp" --T "$t" --max-model-len "$MML" --workload "$WL" --out "$D/build_request_LP${lp}_T${t}" "${TGT_ARGS[@]}")
            run_to "${BUILD_TIMEOUT:-900}" "$D/build_request_LP${lp}_T${t}.log" "${CMD_R[@]}"
            DERIVED+=(--derived "$D/build_request_LP${lp}_T${t}")
          done
        done
        # the compare stage's rc, as batch_decomp's CLI would have returned it (0 PASS / 4 MISSING_DERIVED / 3 otherwise), read from the document
        rcc=$($PY312 -c "import json,os;p='$D/match_decomp.json';v=json.load(open(p)).get('verdict') if os.path.exists(p) else None;print(0 if v=='PASS' else 4 if v=='MISSING_DERIVED' else 3)" 2>/dev/null || echo 3)
        echo "\$ (shared, R17-28 P1) X-09 document written by cb_a.global_match --decomp-out; see global_match.log" > "$D/match_compare.log"
      else
      for pass in 1 2; do
        CMD_C=($PY312 -m verity_capture.experimental.cb_a.batch_decomp --oracle "$D/match" "${DERIVED[@]}" --out "$D/match_decomp.json" --within-step "${WITHIN_STEP:-dag}")
        run_to "$TO_C" "$D/match_compare.log" "${CMD_C[@]}"; rcc=$?
        [ $rcc -eq 4 ] && [ $pass -eq 1 ] || break
        for lpt in $($PY312 -c "import json;print(' '.join(f'{lp},{t}' for lp,t in json.load(open('$D/match_decomp.json'))['missing_derived']))"); do
          IFS=, read -r lp t <<< "$lpt"
          log "X-09: deriving request Program LP=$lp T=$t"
          CMD_R=($PY -m verity_capture.experimental.cb_a.derive_step --model "$REPO" --revision "$REV" --wrapper request --LP "$lp" --T "$t" --max-model-len "$MML" --workload "$WL" --out "$D/build_request_LP${lp}_T${t}" "${TGT_ARGS[@]}")   # [R15 fp8] same declared target
          run_to "${BUILD_TIMEOUT:-900}" "$D/build_request_LP${lp}_T${t}.log" "${CMD_R[@]}"
          DERIVED+=(--derived "$D/build_request_LP${lp}_T${t}")
        done
      done
      fi
    else
      # [R13 MOE-05: early EOS at B=1] the Program to compare against is the one of the request's SERVED shape (LP, T_r = fold steps - 1),
      # exactly X-09's rule at B>1: when greedy decoding hits EOS before cap-1 (OLMoE 1024/128 stopped after 20 tokens, F-moe-09) the
      # cap-1 Program has more steps than the execution and cb_a.compare can never be canonical-equal although every served step is.
      # Derive the served shape here if it is not on disk (one request-wrapper derivation per distinct (LP,T)); criteria unchanged.
      REQ_DIR="$D/build_request"
      T_R=$($PY312 -c "import json;print(int(json.load(open('$D/match/fold_summary.json')).get('n_steps') or 0)-1)" 2>/dev/null || echo "$T")
      if [ "$T_R" -gt 0 ] && [ "$T_R" -ne "$T" ]; then
        REQ_DIR="$D/build_request_LP${LP}_T${T_R}"
        # [COORD 12:33Z] the record labels such a row `eos-truncated` (actual generated length shown, not the cap): a stages.txt line the importer reads
        verdict eos-truncated "T_r=$T_R" "cap_T=$T LP=$LP (greedy hit EOS before the cap; Match against the served shape, MOE-05)" >/dev/null
        if [ ! -f "$REQ_DIR/result.json" ]; then
          log "early EOS: served T_r=$T_R != cap-1=$T -> deriving request Program LP=$LP T=$T_R"
          ENV_SCALE=$(( (LP < 256 ? 256 : LP) / 256 * ((T_R < 31 ? 31 : T_R) / 31) )); ENV_SCALE=$(( ENV_SCALE < 1 ? 1 : ENV_SCALE ))
          # (same --target as the Build derivation, HOP-01: the served-shape Program is the same construction at a shorter T)
          CMD_R=($PY -m verity_capture.experimental.cb_a.derive_step --model "$REPO" --revision "$REV" --wrapper request --LP "$LP" --T "$T_R" --max-model-len "$MML" --workload "$WL" --out "$REQ_DIR" "${TGT_ARGS[@]}")
          run_to "${BUILD_TIMEOUT:-$(( 900 * ENV_SCALE > 14400 ? 14400 : 900 * ENV_SCALE ))}" "$REQ_DIR.log" "${CMD_R[@]}"
        fi
      fi
      echo "{\"LP\": $LP, \"T\": $T_R, \"cap_T\": $T, \"derived\": \"$REQ_DIR\"}" > "$D/match_shape.json"
      CMD_C=($PY312 -m verity_capture.experimental.cb_a.compare --a "$REQ_DIR" --b "$D/match" --out "$D/match_compare.json")
      run_to "$TO_C" "$D/match_compare.log" "${CMD_C[@]}"; rcc=$?
    fi
    # [R15 GM-01 on the integrated path] TABLE-RULES Match = the applicable GLOBAL Program corresponds to the execution (all requests and
    # steps, global I/O associations, chronology): cb_a.global_match/v1 over the SAME fold against THIS row's Build workload Program
    # (GP-01, request ids r<k>; the record's UUIDs are mapped through the manifest). X-09 / cb_a.compare above are evidence inside it.
    # Any B (B=1: the one-request workload Program, COORD R14 17:35Z item 3). Within-step criterion of record = dag (COORD ruling M-0177
    # on DECISION M-0021, from r15-int-2; approved by Daniel 21:33Z M-0281 -- no longer provisional); the sequence verdict is recorded beside it in
    # global_match.json (alternate_verdict) for traceability. WITHIN_STEP=sequence re-runs the r14-v1 criterion as the verdict of record.
    if [ -n "${GM_SHARED:-}" ]; then
      :   # [R17-28 P1] GM-01 already ran above (one X-09 per row); GM_RC and match_decomp.json are its
    else
    GM_RC=""
    tl match.compare end rc="${rcc:-}"
    if [ -f "$D/build_workload/workload_program.json" ]; then
      CMD_G=($PY312 -m verity_capture.experimental.cb_a.global_match --gprog "$D/build_workload/workload_program.json" --workload "$WL" --oracle "$D/match"
             --out "$D/global_match.json" --within-step "${WITHIN_STEP:-dag}")
      run_tl match.global_match "$TO_C" "$D/global_match.log" "${CMD_G[@]}"; GM_RC=$?
    else
      echo "no build_workload/workload_program.json: the global Match needs this row's Build workload Program (GP-01)" > "$D/global_match.log"; GM_RC=98
    fi
    fi
  else
    rcc=99; GM_RC=99; echo "no program.json/instances.jsonl in $D/match (fold did not run or failed)" > "$D/match_compare.log"
    tl match.compare end rc=99
  fi
  # [R15 stoch] a stochastic row's Match (manifest variant with temperature > 0) also checks the VALUE of the sampler chain: every
  # snapshotted sampled row is recomputed with the Program's own sampler Definition (GumbelTopPTokenSelect_v1{V,S}) under the DECLARED
  # seed / temperature / top_p / position and must equal the served token, bit for bit; the kernel-consumed masked fp32 row and the
  # small buffers the kernels read (seed, temperature, top_p, pos) must equal the declared ones.  A wrong manifest seed (manifest vs run)
  # or a mutated sampler chain fails HERE by name (match_stoch.json `reasons`).  Fail closed: nothing checked (no snapshot) -> FAIL.
  rcs=-1
  if [ -f "$D/match/capture/log.jsonl.gz" ] && $PY312 - "$WL" <<'EOF'
import json, sys
w = json.load(open(sys.argv[1])); v = str((w.get("sampling") or {}).get("variant", "greedy"))
p = (w.get("variants") or {}).get(v) or {"temperature": 0.0}
sys.exit(0 if float(p.get("temperature", 0.0)) > 0.0 else 1)
EOF
  then
    run_tl match.stoch_recompute "$TO_C" "$D/match_stoch.log" $PY312 -m verity_capture.bench.stoch_recompute --run "$D/match" --workload "$WL" --build "$D" --out "$D/match_stoch.json"; rcs=$?
    log "stoch value check rc=$rcs: $(tail -n 1 "$D/match_stoch.log" 2>/dev/null | cut -c1-300)"
  fi
  wall=$(( $(date +%s) - t0 ))
  tl match.summary start
  ok=$($PY312 - "$D" "$rcm" "$rcc" "${GM_RC:-97}" "$t0" "$rcs" <<'EOF'
import json, sys, os, time
d, rcm, rcc, rcg, t0, rcs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
cmp = json.load(open(f"{d}/match_compare.json")) if os.path.exists(f"{d}/match_compare.json") else {}
c = cmp.get("compare") or {}
tot = c.get("totals") or {}
steps_a, steps_b = (c.get("a") or {}).get("steps"), (c.get("b") or {}).get("steps")
canon = (c.get("canonical_sha256") or {}).get("equal")
# PASS = canonical instance sequences equal (spec sequence + operand wiring per step), same step count, no differing operands
verdict = None
decomp = json.load(open(f"{d}/match_decomp.json")) if os.path.exists(f"{d}/match_decomp.json") else None
if decomp is not None:
    # [X-09] B>1: the batch decomposition is the verdict -- every request canonical-equal to its own (LP_r, T_r) Program, every instance
    # attributed, no cross-request reads, no missing derivation
    reqs = decomp.get("requests") or []
    canon = all(r.get("canonical_equal") is True for r in reqs) and bool(reqs)
    tot = {"requests": len(reqs), "operands_equal": sum(int((r.get("totals") or {}).get("operands_equal", 0) or 0) for r in reqs),
           "operands_different": sum(int((r.get("totals") or {}).get("operands_different", 0) or 0) for r in reqs),
           "unattributed": (decomp.get("attribution") or {}).get("unattributed"), "xreq": decomp.get("xreq_total"),
           "shapes": sorted({f"LP{r.get('LP')}_T{r.get('T')}" for r in reqs}),
           # [M-0177] the operand counts are X-09's POSITIONAL (sequence) comparison; under the dag criterion a request step that is a
           # dependency-safe permutation of its component's counts its displaced operands here while its DAG multiset is equal
           "within_step_criterion": decomp.get("within_step_criterion"),
           "operands_note": "positional comparison counts (sequence); verdict per within_step_criterion, see match_decomp.json per request"}
    steps_a = steps_b = sum(int(r.get("steps") or 0) for r in reqs)
    verdict = "PASS" if decomp.get("verdict") == "PASS" else f"FAIL batch_decomp {decomp.get('verdict')}: {'; '.join(decomp.get('reasons') or [])[:300]}"
elif cmp:
    verdict = "PASS" if (canon is True and steps_a == steps_b and int(tot.get("operands_different", 0) or 0) == 0) else \
              f"FAIL canonical_equal={canon} steps={steps_a}/{steps_b} operands_different={tot.get('operands_different')} hist_diff={len(cmp.get('histogram_diff_T_normalised') or {})}"
stages = {}
sp = f"{d}/match/stages.json"
if os.path.exists(sp):
    for s in json.load(open(sp)).get("stages", []):
        stages[s.get("name") or s.get("stage")] = {"rc": s.get("returncode", s.get("rc")), "status": s.get("status"), "reason": (s.get("reason") or "")[:200]}
# [R15 global 23:40Z, F-global-rec] the record this verdict is about must have been PRODUCED BY THIS ATTEMPT: when the row dir is reused
# (`match` re-run after a Build) and the GPU capture fails (cp-dev1 B16 256/32 rerun: vLLM refused to start on 12 GiB free), run_config
# keeps the previous attempt's capture/control record, the fold + GM-01 run on that stale record and this stage would report PASS with
# rcm=1. The record's log must be newer than the stage start and the capture/control stages must have rc 0 in stages.json.
rec_log = f"{d}/match/capture/log.jsonl.gz"
rec_mtime = os.path.getmtime(rec_log) if os.path.exists(rec_log) else None
# [r19-int-18, F-r19-int-18] MATCH_PHASE=cpu is an EXPLICIT resume over the preserved capture (F-r19-int-14): the record is older than
# this stage by construction, so the mtime rule cannot apply -- the resumed record is accepted when it is complete (capture + control
# tokens.json beside the log) and labelled `resumed` with its mtime; #57's cpu resume reported "record predates this match stage" on a
# record that was 37 minutes old and whole.  Without the flag the rule of record stands: the log must be newer than the stage start.
resumed = os.environ.get("MATCH_PHASE") == "cpu"
rec_complete = rec_mtime is not None and all(os.path.exists(f"{d}/match/{p}/tokens.json") for p in ("capture", "control"))
rec_fresh = rec_mtime is not None and (rec_mtime >= t0 - 5 or (resumed and rec_complete))
# the capture/control stage records must say rc 0; when the stage record was set aside after an interrupted save (F-r19-int-17) a resume
# has none for the GPU stages -- accepted on the completion markers above, labelled stage_record_absent
rec_stage_absent = resumed and not any(k in stages for k in ("capture", "control"))
rec_stage_ok = all((stages.get(k) or {}).get("rc") == 0 for k in ("capture", "control")) or (rec_stage_absent and rec_complete)
record = {"fresh": bool(rec_fresh and rec_stage_ok), "log_mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(rec_mtime)) if rec_mtime else None,
          "stage_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
          "resumed": resumed, "resumed_record_complete": rec_complete if resumed else None, "stage_record_absent": rec_stage_absent,
          "capture_rc": (stages.get("capture") or {}).get("rc"), "control_rc": (stages.get("control") or {}).get("rc"),
          "reason": None if (rec_fresh and rec_stage_ok) else
                    ("record predates this match stage (stale record from a previous attempt)" if not rec_fresh else
                     f"capture/control stage failed: capture rc={(stages.get('capture') or {}).get('rc')} control rc={(stages.get('control') or {}).get('rc')}")}
tok_eq = None
try:
    ct = json.load(open(f"{d}/match/control/tokens.json")); it = json.load(open(f"{d}/match/capture/tokens.json"))
    # m1_capture tokens.json = {requests: [{request_id, prompt_token_ids, generated_token_ids, ...}], per_step_sampled, timing, observer, ...}:
    # compare the TOKENS per request (prompt + generated ids), not the whole document (timing / observer / profile_id always differ).
    # (R13 fa2: the whole-document compare reported tokens_equal=False on a run whose 32 generated ids were identical.)
    def toks(doc):
        if isinstance(doc, dict) and "requests" in doc:
            return [(r.get("request_id"), r.get("prompt_token_ids"), r.get("generated_token_ids")) for r in doc["requests"]]
        return doc.get("tokens", doc) if isinstance(doc, dict) else doc
    a = toks(ct); b = toks(it)
    tok_eq = (a == b) and len(a) > 0
except Exception as e:  # noqa
    tok_eq = None
fold_ok = os.path.exists(f"{d}/match/program.json") and os.path.exists(f"{d}/match/instances.jsonl")
# [R15 GM-01] the global check is the Match criterion; the per-request comparison above is evidence inside it
G = json.load(open(f"{d}/global_match.json")) if os.path.exists(f"{d}/global_match.json") else {}
gm_verdict = G.get("verdict")
gm_ok = gm_verdict == "PASS" and rcg == 0
if not G:
    gm_verdict = {97: "NOT RUN", 98: "NO WORKLOAD PROGRAM", 99: "NO FOLD", 124: "TIMEOUT"}.get(rcg, f"rc={rcg}")
gm = {"comparator": G.get("comparator"), "rules_version": G.get("rules_version"), "rc": rcg, "verdict": gm_verdict, "failed_checks": G.get("failed_checks"),
      "checks": {c.get("id"): c.get("status") for c in (G.get("checks") or [])}, "within_step_criterion": G.get("within_step_criterion"),
      "workload_digest": (G.get("global_program") or {}).get("workload_digest"), "achieved_concurrency": G.get("achieved_concurrency"),
      "generated_lengths": {e.get("request_id"): e.get("generated_length") for e in (G.get("requests") or [])},
      "eos_truncated_requests": G.get("eos_truncated_requests"),
      "eos_verified": {e["request_id"]: e.get("eos_verified") for e in (G.get("requests") or []) if "eos_verified" in e} or None,
      # [R17-1] the two lengths kept distinct per request + the Program's declared LAG vs the record's attestation (GM-01 res.lag)
      "served_output_lens": {e.get("request_id"): e.get("served_output_len") for e in (G.get("requests") or []) if "served_output_len" in e} or None,
      "executed_prefix_lens": {e.get("request_id"): e.get("executed_prefix_len") for e in (G.get("requests") or []) if "executed_prefix_len" in e} or None,
      "lag": G.get("lag"), "alerts": G.get("alerts"),
      # [rev-pm M-0622 TIP (b)] the declared stop rule (set + source) beside every verdict with an early stop: outside the digest
      "stop_rule": (next((c.get("stop_rule") for c in (G.get("checks") or []) if c.get("id") == "G6"), None)
                    if G.get("eos_truncated_requests") else None),
      "seconds": G.get("seconds"), "artifact": "global_match.json"}
# [R15 stoch] the sampler value check (stochastic rows only; rcs == -1 -> a greedy row, not applicable)
stoch = None
if rcs != -1:
    sr = json.load(open(f"{d}/match_stoch.json")) if os.path.exists(f"{d}/match_stoch.json") else {}
    ss = sr.get("summary") or {}
    stoch = {"rc": rcs, "verdict": ss.get("verdict", "NOT-RUN"), "rows": ss.get("rows"), "device_facts_all_match": ss.get("device_facts_all_match"),
             "masked_rows_all_equal": ss.get("masked_rows_all_equal"), "reasons": (sr.get("reasons") or [])[:6], "variant": sr.get("variant")}
    if verdict == "PASS" and (rcs != 0 or stoch["verdict"] != "PASS"):
        verdict = f"FAIL stoch value check {stoch['verdict']}: {'; '.join(stoch['reasons'])[:400]}"
ok = (verdict == "PASS") and fold_ok and (tok_eq is not False) and gm_ok and record["fresh"] and (rcs in (-1, 0))   # [R15 stoch] the sampler value check must pass on a stochastic row
if not record["fresh"] and verdict == "PASS":
    verdict = f"FAIL record: {record['reason']}"
shape = json.load(open(f"{d}/match_shape.json")) if os.path.exists(f"{d}/match_shape.json") else None  # [MOE-05] served (LP, T_r) at B=1
json.dump({"run_config_rc": rcm, "compare_rc": rcc, "compare_verdict": verdict, "canonical_equal": canon, "steps": [steps_a, steps_b], "totals": tot,
           "request_shape": shape, "record": record, "stoch_value_check": stoch,   # [R15 stoch]
           "histogram_diff_T_normalised": cmp.get("histogram_diff_T_normalised"), "fold_ok": fold_ok,
           "comparator": (G.get("comparator") or "cb_a.global_match/v1") + (" (X-09 batch_decomp inside)" if decomp is not None else " (cb_a.compare inside)"),
           "per_request_comparator": "cb_a.batch_decomp (X-09)" if decomp is not None else "cb_a.compare",
           "global_match": gm,
           "control_vs_instrumented_tokens_equal": tok_eq, "stages": stages, "pass": ok}, open(f"{d}/match_summary.json", "w"), indent=1, default=str)
print("PASS" if ok else "FAIL")
EOF
)
  tl match.summary end ok="$ok"
  verdict match "$ok" "rc=$rcm/$rcc/${GM_RC:-97} wall=${wall}s $(python3 -c "import json;s=json.load(open('$D/match_summary.json'));g=s.get('global_match') or {};print('verdict',s['compare_verdict'],'global',g.get('verdict'),g.get('failed_checks') or '',g.get('within_step_criterion') or '','concurrency',(g.get('achieved_concurrency') or {}).get('observed_max'),'tokens_equal',s['control_vs_instrumented_tokens_equal'],'fold',s['fold_ok'])")"
  tl stage.match end ok="$ok" wall_s="$wall"
  [ "$ok" = PASS ] || { log "MATCH FAIL -> stop"; tl row end ok=false stopped_at=match; exit 11; }
fi

# ---------------------------------------------------------------- Commit + overhead (one run)
if [[ ",$STAGES," == *,commit,* ]]; then
  # [r19-int-10, F-r19-int-10] 2400 s was the R12 constant (B=1 256/32).  #60 mistral B8 1024/128 on l40s2b (r20260922-073827-2712): pairs 0
  # and 1 complete and green (sampled replay 15216/15216, linkage 516/516, weights pin, 68/68 openings, tokens equal; RSS 155 GB -- admit's
  # F-r18-admit-1 fix held on memory) and pair 2 inside its replay when `timeout 2400` fired -> "commit FAIL rc=124", an INFRA-TIMEOUT
  # with no verdict.  The wall scales with the pairs x per-pair replay (population VUs ~ B x tokens): the floor is multiplied by
  # ceil(B/4) x ceil((I+O)/288) like Match's (REL-06/06b), capped at 4 h.  B1 256/32 keeps 2400 s; B8 1024/128 -> 14400 s.  COMMIT_TIMEOUT overrides.
  K_SCALE_B=$(( (B + 3) / 4 )); [ "$K_SCALE_B" -lt 1 ] && K_SCALE_B=1
  K_SCALE_T=$(( (I + O + 287) / 288 )); [ "$K_SCALE_T" -lt 1 ] && K_SCALE_T=1
  TO=${COMMIT_TIMEOUT:-$(( 2400 * K_SCALE_B * K_SCALE_T > 14400 ? 14400 : 2400 * K_SCALE_B * K_SCALE_T ))}
  log "commit timeout ${TO}s (B=$B I=$I O=$O; x$K_SCALE_B x$K_SCALE_T over the 2400 s floor)"
  tl stage.commit start
  [ -f "$HIDDEN_SO" ] || { verdict commit FAIL "hidden .so missing: $HIDDEN_SO"; tl stage.commit end ok=false; tl row end ok=false stopped_at=commit; exit 12; }
  tl commit.prepare start
  CMD_K=($PY -m verity_capture.bench.commit_delta --case "$ROLE" --workload "$WL" --committer native_collect_v2b --fast-copy --gpu-tree
         --hidden-so "$HIDDEN_SO" --openings 64 --pairs "$PAIRS" --warmup 2 --max-num-seqs "$B" --engine-arg "gpu_memory_utilization=$GPU_UTIL"
         --label "$ROW" --out "$D/commit")
  # [REL-11] the P0 collector is the default (COORD 10:22Z: "row_pod.sh's default Commit still fails C-COLLECTOR-FOOTPRINT -- the b4 1024/128 pass
  # used P0-01..03's flags hand-run"): windowed device staging (ring of WINDOW_SLOTS x WINDOW_MB MiB, absorb/release per window), raw copies
  # RETAINed on host (arena block recycled), and the openings-after-release acceptance (P0-02: every raw device copy released, then the 64
  # challenge positions opened from the host copy and verified against the ORIGINAL run's roots -- a miss is a Commit FAIL).  Layout
  # chunk-leaf-v1 -> roots unchanged (p0 test_e2e_windowed_roots; release canaries pin them).  Opt out per row: WINDOW_MB=0 (per-step blocks),
  # RETAIN=device (R9..R12 footprint), OPEN_AFTER_RELEASE=0.
  WINDOW_MB=${WINDOW_MB:-256}; WINDOW_SLOTS=${WINDOW_SLOTS:-8}; RETAIN=${RETAIN:-host}
  [ "$WINDOW_MB" -gt 0 ] && CMD_K+=(--window-mb "$WINDOW_MB" --window-slots "$WINDOW_SLOTS")
  CMD_K+=(--retain "$RETAIN")
  [ "${OPEN_AFTER_RELEASE:-1}" = "0" ] || CMD_K+=(--open-after-release)
  # [R18 tap, F-r17-23] the FA2 M1 tap's device residency is CAPPED (default 2048 MiB): a multi-sequence prefill launch whose padded tap
  # buffer exceeds the cap is split into sequence groups tapped one after another into ONE bounded buffer (names/bytes/roots unchanged;
  # B=1 and swapped-decode launches untouched).  Cited on the record (commit.log line 1, versions collector.fa2_tap_bounded,
  # commit/fa2_tap_bound.json + fa2_tap_windows.jsonl).  TAP_CAP_MB=0 restores the unbounded buffers (B64 llama 1024: 24.53 GiB, r17_0738 OOM).
  TAP_CAP_MB=${TAP_CAP_MB:-2048}
  CMD_K+=(--tap-cap-mb "$TAP_CAP_MB")
  # X-02 lever: EXTRA_ENGINE_ARGS="max_num_batched_tokens=1024" caps the prefill step (chunked prefill) so the staging block stays prompt-sized
  for ea in ${EXTRA_ENGINE_ARGS:-}; do CMD_K+=(--engine-arg "$ea"); done
  # P0 windowed collector flags (`--window-mb 256 --window-slots 8 --retain host --open-after-release`) until rel makes them the default
  # (COORD 10:22Z (2)); the full command line is the first line of commit.log, so the row cites the flags it actually ran with.
  for ca in ${EXTRA_COMMIT_ARGS:-}; do CMD_K+=("$ca"); done
  # [R16 sys M-0043 / coord M-0090 (5)] REPLAY_CACHE (default 1): C2 sampled replay evaluated once per distinct (run root, manifest digest,
  # seed) within the Commit stage -- pairs with the same run root reference pair 0's record (`reused_from` on the copy, `shared_by` on the
  # original, verdict.replay_cache), distinct roots are each replayed, openings stay per pair.  Validated on cp-sys 05:02Z: cold with/without
  # the cache and hot -> identical roots, verdict, replay artifacts (modulo timing); Commit stage 232 s -> 106 s on llama32-1b B1 (3 pairs).
  # REPLAY_CACHE=0 opts out (every pair replays).
  [ "${REPLAY_CACHE:-1}" = "1" ] && CMD_K+=(--replay-cache)
  [ -n "${REQUIRED_CLASSES:-}" ] && CMD_K+=(--required-classes "$REQUIRED_CLASSES")     # REL-02 override (documented reason required)
  # [R14 MAN-01] the expected required-value population comes from THIS row's Build (the request Program) x the canonical VU query, not from
  # the collector: build `manifest.json` from build_request and hand it to commit_delta (--required-manifest); the REL-02 counts+classes
  # source stays only as the labelled fallback when the row has no request Program (verdict.json `required_classes_source` says which).
  REQUIRED_MANIFEST=${REQUIRED_MANIFEST:-}
  # [R17 F-r17b-42 (exec M-0846 / reqval M-0849 / rev M-0850 / revb M-0865), resolution (a); R17-26 / R17-27] B=1 PROGRAM OF RECORD AT
  # COMMIT = the CAP-shape build_request, exactly as at B>=2: "population stays fixed (cap-length manifest, same digest); the R17-1 rule
  # classifies it" -- the positions >= executed_prefix_len of an EOS-stopped request are NOT-EXECUTED BY RULE (executed_prefix_of_record
  # -> binding / sampled_replay's not-executed class, N_excluded named), never cut out of the population by choosing a shorter Program.
  # The R16 M-0402 rule this replaces took the SERVED-shape Program (build_request_LP${LP}_T${T_R}, derived at Match) as the Program of
  # record whenever match_shape.json said T != cap_T: its reason (a cap-shape population asked the collector to bind steps that never ran;
  # cp-sys earlystop int-3c 8,203 unbound) is GONE since commit_delta takes the replay / linkage horizons and the schedule's n_exec from
  # executed_prefix_of_record -- and what it left was a population that DEPENDS ON WHERE THE RUN STOPPED (one B=1 row, two Program digests:
  # Build's declared 0da657b7... vs Commit's served 4d25b5f8... on the sync twin), the class R17-26 exists to remove; on >= int-5 R17-27's
  # manifest_cut_before_classification refused every B=1 early-stop Commit by name.  The Match-side comparison against the served shape
  # (MOE-05 / X-09: the fold has only executed steps; match_shape.json, stages.txt `eos-truncated`) is UNCHANGED; the Commit logs it beside
  # the cap-shape Program so the row says both.  B>=2 unchanged (GLOBAL manifest over the declared shapes, build-global --programs-root).
  PD_REQ="$D/build_request"
  if [ "${B:-1}" -le 1 ] && [ -f "$D/match_shape.json" ]; then
    SERVED_SHAPE=$(python3 -c '
import json, sys
s = json.load(open(sys.argv[1]))
try:
    lp, t, cap = int(s["LP"]), int(s["T"]), int(s["cap_T"])
except (KeyError, TypeError, ValueError):
    raise SystemExit
if t > 0 and t != cap:
    print("LP=%d T_r=%d cap_T=%d" % (lp, t, cap))
' "$D/match_shape.json" 2>/dev/null || true)
    if [ -n "$SERVED_SHAPE" ]; then
      log "B=1 EOS-stopped request ($SERVED_SHAPE, match_shape.json): Program of record at Commit = the CAP-shape build_request (R17-26/27, F-r17b-42 (a)); the positions past the executed prefix are classified not-executed by rule, not cut from the population (the R16 M-0402 served-shape rule is retired here; the Match-side served-shape comparison stands)"
    fi
  fi
  if [ -z "$REQUIRED_MANIFEST" ] && [ "${B:-1}" -gt 1 ] && ls -d "$D"/build_request_LP*_T* >/dev/null 2>&1; then
    # [R15 manbind B8] B>=2: the manifest of record is the GLOBAL manifest over the declared requests (one request Program per declared
    # shape, build_request_LP<prompt_len>_T<max_tokens-1>; request ids r<index>, the workload UUID beside); program_digest = GP-01's
    # workload Program digest when the Build produced it (build_workload/workload_program.json), else the labelled components fold.
    # The one-request manifest of build_request is NOT a population at B>=2 (rc3 F-global: coverage fails on it) -> no fallback: fail closed.
    WD=$($PY -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('workload_digest') or d.get('program_digest') or d.get('digest') or '')" "$D/build_workload/workload_program.json" 2>/dev/null || true)   # [rel] GP-01 writes workload_digest
    tl commit.required_manifest start mode=build-global
    if $PY -m verity_capture.commit.required_manifest build-global --fixture "$WL" --programs-root "$D" ${WD:+--workload-digest "$WD"} --out "$D/manifest.json" > "$D/manifest.log" 2>&1; then
      tl commit.required_manifest end rc=0
      REQUIRED_MANIFEST="$D/manifest.json"; log "required-value manifest (GLOBAL, B=$B${WD:+, workload digest ${WD:0:16}}): $(tail -n 1 "$D/manifest.log")"
    else
      tl commit.required_manifest end rc=1
      log "required-value manifest build-global FAILED (see manifest.log)"; verdict commit FAIL "B>=2 global required manifest not built (manifest.log)"; tl stage.commit end ok=false; tl row end ok=false stopped_at=commit; exit 13
    fi
  fi
  if [ -z "$REQUIRED_MANIFEST" ] && [ -f "$PD_REQ/instances.json.gz" ]; then
    tl commit.required_manifest start mode=build
    if $PY -m verity_capture.commit.required_manifest build --program "$PD_REQ" --workload "$WL" --out "$D/manifest.json" > "$D/manifest.log" 2>&1; then
      tl commit.required_manifest end rc=0
      REQUIRED_MANIFEST="$D/manifest.json"; log "required-value manifest: $(tail -n 1 "$D/manifest.log")"
    else
      tl commit.required_manifest end rc=1
      log "required-value manifest build FAILED (see manifest.log): falling back to REL-02 counts+classes (labelled in verdict.json)"
    fi
  fi
  [ -n "$REQUIRED_MANIFEST" ] && CMD_K+=(--required-manifest "$REQUIRED_MANIFEST")
  # [R15 COORD 5 / rev-cb F-cb-06] the Build's Program digest of record, read from the BUILD (build_summary.json), never from the manifest:
  # B=1 the request Program's digest; B>=2 GP-01's workload digest, else the fold of the per-shape request digests (required_manifest's rule).
  # commit_delta refuses a manifest whose program_digest differs (wrong-Program, fail closed) and keys the binding map by it.
  if [ -z "${PROGRAM_DIGEST:-}" ]; then
    PROGRAM_DIGEST=$($PY -c '
import hashlib, json, os, sys
b = json.load(open(sys.argv[1])); B = int(sys.argv[2]); wd = ""
if B <= 1:
    # [R17 F-r17b-42 (a)] the DECLARED request Program (cap shape) is the one digest of record; the R16 M-0402 served-shape branch is gone
    print((b.get("request") or {}).get("program_digest") or ""); raise SystemExit
if os.path.exists(sys.argv[3]):
    d = json.load(open(sys.argv[3])); wd = d.get("workload_digest") or d.get("program_digest") or d.get("digest") or ""   # [rel] GP-01 key
comps = sorted(v["program_digest"] for k, v in b.items() if k.startswith("build_request_LP") and isinstance(v, dict) and v.get("program_digest"))
print(wd or (hashlib.sha256(("workload-components/" + ",".join(comps)).encode()).hexdigest() if comps else ""))
' "$D/build_summary.json" "${B:-1}" "$D/build_workload/workload_program.json" "$PD_REQ" 2>/dev/null || true)
  fi
  [ -n "$PROGRAM_DIGEST" ] && CMD_K+=(--program-digest "$PROGRAM_DIGEST")
  # [R13 hopper runner] COMMIT_ARGS: extra commit_delta flags verbatim (P0 established set: "--retain host --window-mb 256 --window-slots 8
  # --open-after-release"; layout stays chunk-leaf-v1 by default) -- the harness knobs the completed B>1 rows were measured with
  for ca in ${COMMIT_ARGS:-}; do CMD_K+=("$ca"); done
  # [R15 EXEC-04 / COORD M-0111 (2), manbind M-0242] LIVE C2 = oracle-compare: commit_delta opens every committed identity's bytes and
  # compares them with the Match record's value for the same identity from the SAME attempt ($D/match: capture/snapshots.json + values/,
  # instances.jsonl, the Program-evaluated fold -- a separate process and acquisition path from the collector), and at B>=2 checks the
  # per-(step, request) attribution (bound input ids / sampled id == THAT request's Match tokens, F-cb-23); manbind owns the check inside
  # commit_verdict (flag `--match-dir`, lane/r15-manbind 757521b3a), row_pod hands it the Match dir.  Feature-detected on the flag name; a
  # tree without it runs unchanged (B=1: C2 stays root-compare-vs-independent-run per attempt.py; B>=2 on such a tree: manbind's verdict
  # names 'live C2 oracle-compare not run' once released).  MATCH_SNAP_STEPS=all at Match makes the compare COMPLETE (every step's values).
  # [r19-int-21, F-r19-int-19] the Match ACCOUNT this Commit links its executed prefix to (R17-27 a/b: a_r per request, the lag attestation)
  # is match/capture/tokens.json + match/control/tokens.json beside global_match.json -- INSIDE the capture directory that the disk-freeing
  # move to /vault (F-r19-int-14) takes away whole.  #39's Commit (l40s4, r20260922-135818-be81) started 8 min after its capture was moved,
  # ran 47 min and failed BY NAME ("a_r is not linked", linked=False, lag_attested=False) on the missing file.  A Commit over a Match record
  # refuses to START without the account files, as an INFRA class -- the files are small: restore them from the archive, or run match.
  # --- row_pod.commit-record-precheck begin (extracted by verity_capture/tests/test_row_pod_commit_record_precheck.py)
  if [ -d "$D/match" ] && [ -f "$D/match/instances.jsonl" ]; then
    REC_MISSING=""
    for f in match/capture/tokens.json match/control/tokens.json global_match.json; do [ -f "$D/$f" ] || REC_MISSING="$REC_MISSING $f"; done
    if [ -n "$REC_MISSING" ]; then
      log "PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE: the Match record beside this attempt lacks$REC_MISSING (capture moved / removed? F-r19-int-14) -> Commit not started"
      tl stage.commit end rc=3 fail_class=PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE missing="${REC_MISSING# }"
      verdict commit FAIL "PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE: Match record lacks$REC_MISSING -- restore the file(s) (small) or run match; the Commit was NOT run (infra, not a verdict)"
      tl row end ok=false stopped_at=commit; exit 14
    fi
  fi
  # --- row_pod.commit-record-precheck end
  if [ -d "$D/match" ] && [ -f "$D/match/instances.jsonl" ] && has_flag --match-dir $PY -m verity_capture.bench.commit_delta; then
    CMD_K+=(--match-dir "$D/match"); log "C2 live oracle-compare armed: --match-dir $D/match (Match record of this attempt; snapshot steps ${MATCH_SNAP_STEPS:-$SNAP_DEF})"
  elif [ "$B" -gt 1 ]; then
    log "WARNING B=$B: commit_delta on this tree has no --match-dir (or no $D/match/instances.jsonl): live C2 oracle-compare NOT armed -- request attribution of decode-step identities unverified (rev-cb F-cb-23, M-0216)"
  fi
  # [R15 DECISIONS-2 §4 C2 / M-0356 (1), manbind M-0394] C2 OF RECORD = sampled exact VU replay + ordered boundary linkage from the COMMITTED
  # bytes (registered relation evaluators on the Program's dataflow; needs no Match record).  commit_delta gets the Build's Program dir(s):
  # B=1 the request Program (`--program-dir $D/build_request`), B>=2 the per-request Programs under the row dir (`--programs-root $D`,
  # build-global's resolution: *LP<n>_T<t> else build_request*/result.json declaring (LP, T)).  Feature-detected on the flag name; a tree
  # without it runs unchanged.  REPLAY_ARGS: extra flags verbatim (--replay-per-stratum N --replay-seed S --replay-max M).
  if [ -f "$PD_REQ/instances.json.gz" ] && has_flag --program-dir $PY -m verity_capture.bench.commit_delta; then
    if [ "$B" -gt 1 ]; then CMD_K+=(--programs-root "$D"); else CMD_K+=(--program-dir "$PD_REQ"); fi
    for ra in ${REPLAY_ARGS:-}; do CMD_K+=("$ra"); done
    log "C2 of record armed: sampled exact VU replay + boundary linkage ($([ "$B" -gt 1 ] && echo "--programs-root $D" || echo "--program-dir $PD_REQ") ${REPLAY_ARGS:-})"
    # [R15 manbind MAN-09; rev-cb M-0610, coord M-0614 (1)] WEIGHTS OF RECORD: the served weights are authenticated against the PINNED CHECKPOINT
    # SHARDS (manifests/checkpoints.json sha256), not against the committer's own registration.  The record is derived CPU-side from the
    # Program of record x the checkpoint dir ($CK, the same snapshot Build/Match load) and written beside the Program once
    # (build_request/weights_of_record.json, reusable by Match); commit_delta compares every registered parameter with it by name.
    if has_flag --weights-of-record $PY -m verity_capture.bench.commit_delta; then
      # [R15 rel int-6, hopper BLOCKER M-0723 F-hopper-WOR] at B>1 the record must be derived from a Program dir that IS a component of the
      # workload Program: rev-cb W2 binds the record to the of-record set {--program-digest (workload digest)} + manifest component digests,
      # and the row default wrapper shape (build_request = longest prompt x largest cap - 1) is a component only when some request has
      # both maxima -- a mixed B>=2 row has none, so a record derived from build_request FAILs by name (measured cp-h100b Llama H100 B8:
      # 98 compared / 0 mismatched / program 9699b09e != of record).  The weights struct is model-level: hopper verified the record from a
      # component dir has identical fields, digests and root_of_record; only program_digest changes.  Pick the FIRST manifest component
      # (manifest.json components[r].LP/T -> build_request_LP<LP>_T<T>, or build_request when its own result.json declares that shape);
      # the record lives beside THAT Program so a default-shape record left by int-5 is never reused.  B1 unchanged.  No apostrophes in
      # the heredoc (bash 3.2 laptop gate).
      PD_WOR="$PD_REQ"
      if [ "$B" -gt 1 ] && [ -f "$D/manifest.json" ]; then
        PD_WOR=$(python3 - "$D" <<'EOF'
import json, os, sys
d = sys.argv[1]
m = json.load(open(os.path.join(d, "manifest.json")))
comps = m.get("components") or {}
for rid in sorted(comps, key=lambda r: (len(r), r)):
    c = comps[rid] or {}
    lp, t = c.get("LP"), c.get("T")
    if lp is None or t is None:
        continue
    for cand in (os.path.join(d, "build_request_LP%d_T%d" % (int(lp), int(t))), os.path.join(d, "build_request")):
        rj = os.path.join(cand, "result.json")
        if os.path.exists(rj):
            try:
                r = json.load(open(rj))
            except Exception:  # noqa: BLE001
                continue
            if r.get("ok") and int(r.get("LP", -1)) == int(lp) and int(r.get("T", -1)) == int(t):
                print(cand)
                sys.exit(0)
print(os.path.join(d, "build_request"))
EOF
)
        [ "$PD_WOR" != "$D/build_request" ] && log "weights of record: B=$B -> derived from the component Program $(basename "$PD_WOR") (rev-cb W2 of-record set; M-0723)"
      fi
      WOR="$PD_WOR/weights_of_record.json"
      # [R16 pin, wave-1 Commit-only reruns] a record already beside the picked Program is reused ONLY when it is THIS Program's
      # (program_digest == the component's result.json digest), of the current schema (MAN-09c stamps engine_assumptions; an int-6 record
      # leaves the 65 engine constants unpinned by name) and complete.  Otherwise it is moved aside (weights_of_record.stale-<utc>.json,
      # never deleted) and re-derived: a stale record FAILs W2 by name at the checker (M-0727) and a pre-MAN-09c record is a weaker pin
      # than the tree can derive -- re-derivation is the remedy, not an operator rm.  No apostrophes in the heredoc (bash 3.2 laptop gate).
      if [ -f "$WOR" ] && [ -n "${CK:-}" ] && [ -d "$CK" ]; then
        # [R16 pin, schema v2] a record of an older schema is re-derived too: v1 roots are keyed by the deriving host's fp32 inv_freq
        # (two roots for one model across cp-r1 / cp-a100, exec M-0178); the tree's schema is asked of the module, never assumed
        WOR_SCHEMA=$($PY -c "from verity_capture.commit.weights_of_record import SCHEMA; print(SCHEMA)" 2>/dev/null || true)
        WOR_STALE=$(python3 - "$WOR" "$PD_WOR/result.json" "${WOR_SCHEMA:-}" <<'EOF'
import json, os, sys
try:
    rec = json.load(open(sys.argv[1]))
except Exception as e:  # noqa: BLE001
    print("unreadable record (%s)" % str(e)[:80]); sys.exit(0)
res = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else {}
cur = sys.argv[3] if len(sys.argv) > 3 else ""
# the component Program digest as a real result.json carries it (artifact.program_digest / program.digest; a top-level key on the
# synthetic ones), else the instances.json.gz digest the derivation itself falls back to -- an empty value here would skip the check
rd = str(res.get("program_digest") or (res.get("artifact") or {}).get("program_digest") or (res.get("program") or {}).get("digest") or "")
if not rd:
    ig = os.path.join(os.path.dirname(sys.argv[2]), "instances.json.gz")
    if os.path.exists(ig):
        import gzip
        try:
            rd = str(json.load(gzip.open(ig, "rt")).get("digest") or "")
        except Exception:  # noqa: BLE001
            rd = ""
pd = str(rec.get("program_digest") or "")
if rd and pd != rd:
    print("built for Program %s, this component is %s (stale)" % (pd[:16] or "?", rd[:16]))
elif not isinstance(rec.get("engine_assumptions"), dict):
    print("predates MAN-09c (no engine_assumptions; %d engine constant(s) unpinned by name)" % len(rec.get("unpinned") or []))
elif cur and str(rec.get("schema") or "") != cur:
    print("schema %s, this tree derives %s (root formula differs: the analytic entry is keyed by its inputs, not the host)" % (rec.get("schema") or "?", cur))
elif rec.get("complete") is not True:
    print("incomplete (%d unresolved field(s))" % len(rec.get("unresolved") or []))
EOF
)
        if [ -n "$WOR_STALE" ]; then
          WOR_PREV="$PD_WOR/weights_of_record.stale-$(date -u +%Y%m%dT%H%M%SZ).json"
          mv "$WOR" "$WOR_PREV"
          log "weights of record: existing record beside $(basename "$PD_WOR") not reused -- $WOR_STALE; moved aside ($(basename "$WOR_PREV")), re-deriving"
        fi
      fi
      if [ ! -f "$WOR" ] && [ -n "${CK:-}" ] && [ -d "$CK" ]; then
        WOR_SUP=""    # a scalar, not an array: set -u + an empty array breaks bash 3.2; the path carries no spaces (the row dir is ours)
        if [ -n "${WOR_PREV:-}" ] && has_flag --supersedes $PY -m verity_capture.commit.weights_of_record; then WOR_SUP="--supersedes=$WOR_PREV"; fi
        tl commit.weights_of_record start
        $PY -m verity_capture.commit.weights_of_record --program-dir "$PD_WOR" --checkpoint-dir "$CK" --out "$WOR" ${WOR_SUP:+"$WOR_SUP"} > "$D/weights_of_record.log" 2>&1 \
          && { tl commit.weights_of_record end rc=0; log "weights of record: $(tail -n 1 "$D/weights_of_record.log")"; } \
          || { tl commit.weights_of_record end rc=1; log "WARNING weights of record not derived (see weights_of_record.log): $(tail -n 1 "$D/weights_of_record.log")"; }
      elif [ ! -f "$WOR" ]; then
        # [R16 pin] name the cause in row.log: the checker will FAIL this Commit 'weights unpinned (MAN-09)' (fail-closed), and the reason is
        # here, not in the engine -- the workload's (model, tokenizer_revision) resolved to no local checkpoint dir (manifests/checkpoints.json
        # local_path, else /workspace/hf/hub/models--<repo>/snapshots/<revision>): pin the snapshot on this pod, do not hand-write a record
        log "WARNING weights of record NOT derived: no local checkpoint dir for the workload's (model, tokenizer_revision) (CK='${CK:-}'; checkpoints.json local_path / HF hub snapshot absent on this pod) -- the Commit will FAIL 'weights unpinned (MAN-09)' by name"
        tl commit.weights_of_record mark reused=false derived=false
      else
        tl commit.weights_of_record mark reused="$([ -f "$WOR" ] && echo true || echo false)"
      fi
      if [ -f "$WOR" ]; then CMD_K+=(--weights-of-record "$WOR"); elif [ -n "${CK:-}" ]; then CMD_K+=(--checkpoint-dir "$CK"); fi
    fi
  else
    log "WARNING: C2 sampled replay NOT armed (no $PD_REQ/instances.json.gz or commit_delta without --program-dir): C2 of record not run (DECISIONS-2 §4)"
  fi
  tl commit.prepare end
  t0=$(date +%s)
  if [ "${HOT_ENGINE:-0}" = "1" ]; then
    # [R16 sys, BRIEF §5] hot-engine reuse (default OFF): the same commit_delta ARGS run on the pod's hot worker for this engine key
    # (verity_capture/sweep/hot_commit.py: key = code tree + native .so + env + engine args; state reset + pristine check between jobs;
    # HOT-UNSAFE / key mismatch / dead worker -> automatic cold fallback with the reason in commit.log + timeline).  Same commit.log content,
    # same evidence files; versions.json `hot` and the timeline `engine.reuse` / `engine.build` events say which happened.
    log "HOT_ENGINE=1: Commit on the hot worker (root ${HOT_ROOT:-/workspace/cp/hot}, idle ${HOT_IDLE:-900}s)"
    run_tl commit.commit_delta "$TO" "$D/commit.log" $PY -m verity_capture.sweep.hot_commit submit --root "${HOT_ROOT:-/workspace/cp/hot}" --idle "${HOT_IDLE:-900}" -- "${CMD_K[@]:3}"; rck=$?
  else
    gpu_exclusive commit-cold
    run_tl commit.commit_delta "$TO" "$D/commit.log" "${CMD_K[@]}"; rck=$?
  fi
  wall=$(( $(date +%s) - t0 ))
  # [R15 exec] the summary (same fields as before) also records the execution configuration the overhead is labelled with -- the run's
  # own collector record (variant/retention/window), the commit_delta command line, --engine-arg settings -- so the finalizer never has
  # to guess "whole-block/device" for a windowed/host run (verity_capture/sweep/commit_summary.py; re-runnable offline on the evidence)
  tl closure.commit_summary start
  ok=$($PY -m verity_capture.sweep.commit_summary "$D" "$rck")
  tl closure.commit_summary end ok="$ok"
  # [R15 EXEC-05, manbind MAN-05 M-0340 (3) / rev-cb F-cb-11] MANIFEST OF RECORD: after the Commit, rebuild the required-value manifest
  # from THIS attempt's Program x workload with `required_manifest verify` (CPU) and record it at $D/commit/manifest_verify.json -- the
  # finalizer grades Commit ◐ without it ("pinned by digest recompute only").  B=1: --program build_request --workload WL; B>=2:
  # --fixture WL --programs-root $D --workload-digest <GP-01's FULL workload digest from build_workload/workload_program.json> (the
  # 16-hex prefix in stages.txt is NOT accepted: the rebuilt rows embed the digest, so a prefix fails the rows compare).  A verify FAIL
  # means the Commit did not use the manifest of record -> the Commit verdict is FAIL (fail-closed), never a footnote.  Feature-detected:
  # a tree without the verify sub-command runs unchanged.
  if [ "$ok" = PASS ] && $PY -m verity_capture.commit.required_manifest verify --help >/dev/null 2>&1; then
    MV=("$PY" -m verity_capture.commit.required_manifest verify --manifest "$D/manifest.json" --out "$D/commit/manifest_verify.json")
    if [ "$B" -gt 1 ]; then
      WDF=$($PY -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('workload_digest') or d.get('program_digest') or d.get('digest') or '')" "$D/build_workload/workload_program.json" 2>/dev/null || true)
      MV+=(--fixture "$WL" --programs-root "$D"); [ -n "$WDF" ] && MV+=(--workload-digest "$WDF")
    else
      MV+=(--program "$PD_REQ" --workload "$WL")
    fi
    tl validate.manifest_verify start
    if "${MV[@]}" > "$D/manifest_verify.log" 2>&1; then
      tl validate.manifest_verify end rc=0
      log "manifest of record VERIFIED (required_manifest verify: rebuilt from this attempt's Program x workload == used; commit/manifest_verify.json)"
    else
      tl validate.manifest_verify end rc=1
      ok=FAIL; log "manifest of record VERIFY FAILED: $(tail -1 "$D/manifest_verify.log" | cut -c1-200)"
    fi
  fi
  verdict commit "$ok" "rc=$rck wall=${wall}s $(python3 -c "import json;s=json.load(open('$D/commit_summary.json'));print('runs',s['n_runs'],'delta',s['delta_wall_mean'],'openings',s['openings'],'tokens',s['tokens_equal'])" 2>/dev/null) manifest_verify=$([ -f "$D/commit/manifest_verify.json" ] && python3 -c "import json;print(json.load(open('$D/commit/manifest_verify.json')).get('ok'))" 2>/dev/null || echo not-run)"
  tl stage.commit end ok="$ok" wall_s="$wall"
  [ "$ok" = PASS ] || { log "COMMIT FAIL -> stop"; tl row end ok=false stopped_at=commit; exit 12; }
fi
tl row end ok=true
# [R16 sys] the measured BRIEF §5 table of this attempt beside the evidence (timeline_report.md/json); never fails the row
python3 -m verity_capture.sweep.timeline report "$VERITY_TIMELINE" --json "$D/timeline_report.json" --md "$D/timeline_report.md" --title "$ROW" >/dev/null 2>&1 || true
log "DONE"
