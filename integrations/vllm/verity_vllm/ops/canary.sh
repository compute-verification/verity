#!/usr/bin/env bash
# R13 rel: the release canary suite (BRIEF §5A), run on the verification worker (cp-rev) for EVERY release candidate before it is tagged.
#
#   cd /workspace/cp/veritor && bash verity_vllm/ops/canary.sh <candidate-tag> [positives=smollm2,llama[,llama_b2,llama_b2_bm,phi3,qwen,olmoe,stoch]] [negatives=arena,omitted,lateread[,drop1dup0,nopd,nosr,flipweight,lateflush,wl_swapped]]
#   -> evidence under /workspace/cp/canary/<tag>/, summary canary.json; last line  CANARY-GREEN <tag>  or  CANARY-RED <tag> <first failing check>
#
# Positives (the two base canaries, through the SAME row procedure the sweep uses -- verity_vllm/ops/row_pod.sh, PAIRS=1):
#   SmolLM2-135M B1 256/32 and Llama-3.2-1B B1 256/32 on this GPU: Build (Bet A step + request, --T = O-1 per REL-01) -> Match (eager
#   capture + fold -> cb_a.compare canonical-equal, same step count, tokens equal) -> fail-closed Commit (X-03 + REL-02, 64 openings, 1 pair)
#   AND the run root must equal the known-good root pinned for this compute capability in verity_vllm/ops/known_roots.json
#   (R12: cb129578b7846e1f… / cfd86e06b12b61c1…, reproduced bit-for-bit across pods).  A different root with an otherwise green Commit is RED:
#   what the collector binds changed (or the values did) and that needs a decision, not a tag.
# Negatives (SmolLM2, must FAIL the way stated):
#   arena     --arena-mb 1 --arena-unchecked            -> exit 3, reason expected_tensors != committed_tensors / arena block too small (X-03)
#   omitted   no --hidden-so (FA2 producer never attached) -> exit 3, reason "required class 'fa2_hidden_m1_stream' not declared" (REL-02 / R-14)
#             [R15 hopper] on a Hopper pod (cc 9.x, FlashAttention-3 engine) the class is 'fa3_hidden_m1_stream' and the tap fa3_matReq
#   lateread  --late-read qkv_proj (R11 regression)      -> the run root MUST differ from the known-good root (openings still verify: binding
#             is not correctness -- the harness must never take a verifying root as evidence the right buffer was read)
# Env: everything row_pod.sh takes (PY, PY312, HIDDEN_SO, GPU_UTIL, timeouts); CANARY_DIR (/workspace/cp/canary).
set -uo pipefail
TAG=${1:?candidate tag, e.g. r13-rel0}; POS=${2:-smollm2,llama}; NEG=${3:-arena,omitted,lateread}
# [R17 rel, COORD R17-19 (4)] the canary runs from the tree it is given (VERITOR_REPO, default the pod's live tree): a second canary GPU
# (cp-big GPU 1) takes the SAME candidate installed as an OWN tree (pod_release.sh REMOTE_TREE=/workspace/cp/<tag>) without replacing that
# pod's live tree; the collector build dir is the one the release built for THIS tree (RELEASE.json build_dirs.native_collect, per tree for
# an own tree -- never another tree's nc_build), unless the caller set NATIVE_COLLECT_BUILD explicitly.  CUDA_VISIBLE_DEVICES passes through.
# [R18 infra, F-r17-tp-06] the tree that runs is the tree THIS SCRIPT lives in (the own-tree install's canary.sh is invoked from that
# tree): $VERITOR_REPO is set FROM here, never read as the default; a caller exporting another tree is refused by name.
TREE_SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [ -n "${VERITOR_REPO:-}" ] && [ "$(cd "$VERITOR_REPO" 2>/dev/null && pwd -P)" != "$TREE_SELF" ]; then
  echo "PRECHECK_FAIL_SOURCE_IDENTITY: canary.sh at $TREE_SELF but \$VERITOR_REPO=$VERITOR_REPO names another tree" >&2; exit 3
fi
export VERITOR_REPO=$TREE_SELF
cd "$VERITOR_REPO" || exit 1
# [R18 infra, r18-int-2] the canary's base rows are `__l40s__` ids and the canary also runs on the Hopper host: the target-family
# preflight (row_pod.sh, PRECHECK_FAIL_TARGET_FAMILY) is WAIVED BY NAME here -- `precheck-target WAIVED canary <tag>` in stages.txt,
# never on a row of record.  On an L40S the same rows PASS the check unwaived (the waiver only names a would-be failure).
export TARGET_FAMILY_WAIVER=${TARGET_FAMILY_WAIVER:-"canary $TAG"}
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1 TOKENIZERS_PARALLELISM=false CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
PY=${PY:-/workspace/venv-cu129/bin/python}; export PATH=$(dirname "$PY"):/usr/local/cuda/bin:$PATH
if [ -z "${NATIVE_COLLECT_BUILD:-}" ]; then
  NCB_REL=$(python3 - "$PWD" <<'EOF' 2>/dev/null
import json, os, sys
tree = sys.argv[1]
for p in (os.path.join(tree, "RELEASE.json"), "/workspace/cp/RELEASE.json"):   # tree-local first (the ship that installed THIS tree), pod-wide second
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if d.get("tree") in (None, tree):
        print(((d.get("build_dirs") or {}).get("native_collect")) or ""); break
EOF
)
  export NATIVE_COLLECT_BUILD=${NCB_REL:-/workspace/cp/nc_build}
fi
export NATIVE_COLLECT_BUILD
# [R15 hopper] the tap = the attention implementation this device executes (TargetProfile.effective_flash_attn_version(), as row_pod.sh):
# cc 9.x runs FlashAttention-3 -> the FA3 M1 tap (HOP-01) and the required hidden class is fa3_hidden_m1_stream; the omitted negative's
# expected reason names that class.  HIDDEN_SO overrides (a foreign tap is refused by commit_delta's engine check, not silently accepted).
FA_VER=$($PY -c "from verity_vllm.program.frontend.target_profile import TargetProfile; print(TargetProfile.from_device().effective_flash_attn_version())" 2>/dev/null || echo 2)
HIDDEN_SO=${HIDDEN_SO:-$([ "$FA_VER" = 3 ] && echo /workspace/cp/fa2/build/fa3_matReq/verity_fa3_matReq.so || echo /workspace/cp/fa2/build/matReq/verity_fa2_matReq.so)}; export HIDDEN_SO
HIDDEN_CLASS=$([ "$FA_VER" = 3 ] && echo fa3_hidden_m1_stream || echo fa2_hidden_m1_stream)
GPU_UTIL=${GPU_UTIL:-0.5}; export GPU_UTIL
CD=${CANARY_DIR:-/workspace/cp/canary}/$TAG; mkdir -p "$CD"; export SWEEP_DIR=$CD
CC=$($PY -c "import torch;p=torch.cuda.get_device_properties(0);print(f'{p.major}.{p.minor}')")
COMMIT=$(python3 -c "import json;print(json.load(open('EXPORT.json'))['commit'][:9])" 2>/dev/null || echo unknown)
SMOL=smollm2-135m__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager
LLAMA=llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager
LLAMA_B2=llama32-1b__bf16__l40s__tp1__b2__i256__o32__mixed__greedy__bi-eager   # [X-09] B>1: Match by batch decomposition (positives=...,llama_b2)
PHI3=phi3-mini__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager         # [FA2-01] head dim 96 through the portable tap (positives=...,phi3; needs PHI3_MINI seeded)
QWEN=qwen25-15b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager        # [X-04] Qwen2 qkv bias (BroadcastBiasAdd) + hd-128 tap; root of record from the cp-sw1 sweep row (positives=...,qwen; needs B1 seeded)
# [R16 rel] the MoE family (OLMoE-1B-7B: fused-MoE block source, moe_block_stream class, MAN-06f plane-fed strata not-evaluable BY NAME, FLUSH-02
# gather flush): root of record = moe's post-FLUSH-02 B1 root (r15_flush02 / r15_int4b, known_roots.json).  R15 finding (rel M-0813): the MoE
# and the Qwen2.5 BiasAdd defect classes were invisible to every canary; both are positives from r16-int-1 on (positives=...,qwen,olmoe; needs OLMOE seeded).
OLMOE=olmoe-1b-7b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager
log() { echo "$(date -u +%FT%TZ) [canary $TAG] $*" | tee -a "$CD/canary.log"; }
RESULTS=()   # "name|PASS/FAIL|detail"
FIRST_FAIL=""
rec() { RESULTS+=("$1|$2|$3"); log "$1 $2 -- $3"; [ "$2" = PASS ] || [ -n "$FIRST_FAIL" ] || FIRST_FAIL="$1: $3"; }
known_root() { python3 -c "import json;print((json.load(open('verity_vllm/ops/known_roots.json'))['roots'].get('$1') or {}).get('$CC',''))"; }
root_of() { python3 -c "import json,sys;v=json.load(open('$1'));print((v.get('run_roots') or [''])[0])" 2>/dev/null; }
reason_of() { python3 -c "import json;print(json.load(open('$1')).get('first_fail_reason') or '')" 2>/dev/null; }
# [R14 BIND-01/MAN-01] the row's identity-coverage line: commit_delta prints "[delta] BINDING (pair k): ... coverage OK|FAIL bound n/N missing m ..."
# per pair (the binding record itself lives in the pair rows); verdict.json carries the required manifest's digest.  Empty on pre-R14 trees.
coverage_of() {   # <row dir>
  local cov man
  cov=$(grep -o 'coverage \(OK\|FAIL\) bound [0-9]*/[0-9]* missing [0-9]*' "$1/commit.log" 2>/dev/null | sort | uniq -c | awk '{printf "%s x%d; ", substr($0, index($0,$2)), $1}')
  man=$(python3 -c "import json;m=json.load(open('$1/commit/verdict.json')).get('required_manifest') or {};print('manifest', (m.get('manifest_digest') or '')[:16] or 'none')" 2>/dev/null)
  [ -n "$cov" ] && echo "${cov% }; $man"
}

log "start commit=$COMMIT cc=$CC flash_attn=$FA_VER hidden_so=$HIDDEN_SO hidden_class=$HIDDEN_CLASS positives=$POS negatives=$NEG"

# ------------------------------------------------------------------ positives: the row procedure itself, then the root pin
positive() {   # <name> <row-id> <ROLE> <repo> <revision>
  local name=$1 row=$2 role=$3 repo=$4 rev=$5 t0=$(date +%s)
  local exp; exp=$(known_root "$row")
  PAIRS=1 bash verity_vllm/ops/row_pod.sh "$row" "$role" "$repo" "$rev" build,match,commit > "$CD/$name.row.log" 2>&1; local rc=$?
  local stages; stages=$(tr '\n' ';' < "$CD/$row/stages.txt" 2>/dev/null | cut -c1-400)
  if [ $rc -ne 0 ]; then rec "$name" FAIL "row_pod rc=$rc stages: $stages"; return; fi
  local root; root=$(root_of "$CD/$row/commit/verdict.json")
  if [ -z "$root" ]; then rec "$name" FAIL "no run root in commit/verdict.json; stages: $stages"; return; fi
  if [ -n "$exp" ] && [ "$root" != "$exp" ]; then rec "$name" FAIL "Build/Match/Commit green but run root $root != known-good $exp (cc $CC): what is bound changed -- decision needed"; return; fi
  # [R14 BIND-01/MAN-01] identity coverage against the row's required-value manifest must be complete (the verdict is fail-closed on it already; shown for the record)
  local cov; cov=$(coverage_of "$CD/$row")
  if [[ "$cov" == *"coverage FAIL"* ]]; then rec "$name" FAIL "Commit verdict green but binding $cov; $stages"; return; fi
  rec "$name" PASS "Build+Match+Commit green in $(( $(date +%s) - t0 ))s; root ${root:0:16} $([ -n "$exp" ] && echo '== known-good' || echo '(no reference for this cc)'); ${cov:-no binding record}; $stages"
}
[[ ",$POS," == *,smollm2,* ]] && positive smollm2 "$SMOL" B0 HuggingFaceTB/SmolLM2-135M 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
[[ ",$POS," == *,llama,* ]]   && positive llama "$LLAMA" LLAMA32_1B unsloth/Llama-3.2-1B 9535bd9b1d1dea6acafbdc4813b728796aeb28da
# [R13 X-09] the first B>1 Match of the campaign (M-BATCH-DECOMP): Build derives both request shapes, Match = cb_a.batch_decomp, Commit at B=2;
# root pinned from dense's cp-dB run.  Default positives stay the two B=1 rows; releases touching Build/Match/row_pod run with llama_b2 too.
[[ ",$POS," == *,llama_b2,* ]] && positive llama_b2 "$LLAMA_B2" LLAMA32_1B unsloth/Llama-3.2-1B 9535bd9b1d1dea6acafbdc4813b728796aeb28da
# [R16 rel, int-4] the STOCHASTIC B1 panel row as a positive (coord M-0295/M-0326: no release canary had a stochastic positive; the derived-S
# units make S a request-owned Program INPUT `splits` and REQ-03 makes it a required family, so this line is the live check that a stochastic
# row on the tree Builds (derived S), Matches (GM-01 G6 reads the bound operand) and Commits with the population COMPLETE (sampler_splits acquired
# from the launched kernel's constexpr, never recomputed).  No known-good root is pinned for the derived-S population yet (the root changes with
# what is bound): PASS reads "(no reference for this cc)" until a reviewer pins it.  Same model/revision as the llama lines.
STOCH=llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__stoch-t0.8-p0.95__bi-eager
[[ ",$POS," == *,stoch,* ]] && positive stoch "$STOCH" LLAMA32_1B unsloth/Llama-3.2-1B 9535bd9b1d1dea6acafbdc4813b728796aeb28da
# [R15 rel, GP-01/GM-01 (global M-0095)] the B=2 row through Build+Match ONLY: Build must compose the workload Program (GP-01, build_workload/
# workload_program.json, request ids r<k>) and Match must PASS the GLOBAL comparator (cb_a.global_match/v1 G1..G8, --within-step ${WITHIN_STEP:-dag}, COORD M-0177)
# over the same fold.  No Commit: at rc3 the B>=2 required manifest is the one-request manifest, so B>=2 Commit fails closed on identity
# coverage (global M-0097) until manbind's B>=2 manifest lands -- then positives=...,llama_b2 (full row + root pin) replaces this one.
positive_bm() {   # <name> <row-id> <ROLE> <repo> <revision>   Build+Match positive; PASS = row_pod rc 0 + match_summary.pass + global_match PASS
  local name=$1 row=$2 role=$3 repo=$4 rev=$5 t0=$(date +%s)
  PAIRS=1 bash verity_vllm/ops/row_pod.sh "$row" "$role" "$repo" "$rev" build,match > "$CD/$name.row.log" 2>&1; local rc=$?
  local stages; stages=$(tr '\n' ';' < "$CD/$row/stages.txt" 2>/dev/null | cut -c1-400)
  if [ $rc -ne 0 ]; then rec "$name" FAIL "row_pod rc=$rc stages: $stages"; return; fi
  local gm; gm=$(python3 - "$CD/$row" <<'EOF'
import json, os, sys
d = sys.argv[1]
s = json.load(open(f"{d}/match_summary.json")) if os.path.exists(f"{d}/match_summary.json") else {}
g = s.get("global_match") or {}
b = json.load(open(f"{d}/build_summary.json")) if os.path.exists(f"{d}/build_summary.json") else {}
wd = ((b.get("workload") or {}).get("workload_digest") or "")[:16]
ok = bool(s.get("pass")) and g.get("verdict") == "PASS" and g.get("rc") == 0 and bool(wd)
ck = g.get("checks") or {}
print(("PASS" if ok else "FAIL") + f"|global {g.get('verdict')} rc={g.get('rc')} comparator={g.get('comparator')} within_step={g.get('within_step_criterion')} "
      f"checks={','.join(f'{k}:{v}' for k, v in sorted(ck.items()))} concurrency={(g.get('achieved_concurrency') or {}).get('observed_max')} "
      f"workload_digest={wd or 'none'} per_request={s.get('per_request_comparator')} tokens_equal={s.get('control_vs_instrumented_tokens_equal')}")
EOF
)
  local ok=${gm%%|*} detail=${gm#*|}
  if [ "$ok" != PASS ]; then rec "$name" FAIL "Build+Match rc 0 but the global Match is not PASS: $detail; $stages"; return; fi
  rec "$name" PASS "Build+Match green in $(( $(date +%s) - t0 ))s (no Commit: B>=2 manifest pending, M-0097); $detail; $stages"
}
[[ ",$POS," == *,llama_b2_bm,* ]] && positive_bm llama_b2_bm "$LLAMA_B2" LLAMA32_1B unsloth/Llama-3.2-1B 9535bd9b1d1dea6acafbdc4813b728796aeb28da
# [R13 FA2-01] head dim 96 (Phi-3-mini): the tap's hd-96 instantiation + X-04-class QKV handling; root pinned from fa2's cp-dC run.
[[ ",$POS," == *,phi3,* ]]     && positive phi3 "$PHI3" PHI3_MINI microsoft/Phi-3-mini-4k-instruct f39ac1d28e925b323eae81227eaba4464caced4e
# [R13 X-04] the Qwen family (qkv bias add, head dim 128): the third completed dense model family; root of record = the sweep row on cp-sw1 (r13-rel3).
[[ ",$POS," == *,qwen,* ]]     && positive qwen "$QWEN" B1 Qwen/Qwen2.5-1.5B 8faed761d45a263340a0528343f099c05c9a4323
# [R16 rel] OLMoE B1 through the row procedure (MoE block source + MAN-06f by-name strata + weights pin on 3 shards); root pinned from moe's B1 row.
# [R16 rel, r16-int-2; rev F-r16-07 M-0068, pin 3ef2e7219, COORD M-0090 (6)] the OLMoE Commit verdict OF RECORD is a NAMED FAIL: the plane-fed
# (MAN-06f) strata are value-checked by neither form until the row->plane mapping (moe M-0085 option (A)) lands, and the checker must say so.
# The positive therefore requires Build+Match green, the run root == known-good, coverage OK, AND commit/verdict.json's first_fail_reason to be
# the ruled text.  A Commit PASS here is the F-r16-02 overclaim and FAILS the canary; so does any other reason.  When the mapping lands, this
# line goes back to `positive` (a PASS becomes the outcome of record again) -- never a canary that accepts both.
positive_named() {   # <name> <row-id> <ROLE> <repo> <revision> <required first_fail_reason, grep -E pattern>
  local name=$1 row=$2 role=$3 repo=$4 rev=$5 want=$6 t0=$(date +%s)
  local exp; exp=$(known_root "$row")
  PAIRS=1 bash verity_vllm/ops/row_pod.sh "$row" "$role" "$repo" "$rev" build,match,commit > "$CD/$name.row.log" 2>&1; local rc=$?
  local stages; stages=$(tr '\n' ';' < "$CD/$row/stages.txt" 2>/dev/null | cut -c1-400)
  [ -f "$CD/$row/commit/verdict.json" ] || { rec "$name" FAIL "no commit/verdict.json (row_pod rc=$rc: Build or Match did not reach Commit); stages: $stages"; return; }
  local root reason; root=$(root_of "$CD/$row/commit/verdict.json"); reason=$(reason_of "$CD/$row/commit/verdict.json")
  if [ -z "$root" ]; then rec "$name" FAIL "no run root in commit/verdict.json; stages: $stages"; return; fi
  if [ -n "$exp" ] && [ "$root" != "$exp" ]; then rec "$name" FAIL "run root $root != known-good $exp (cc $CC): what is bound changed -- decision needed"; return; fi
  local cov; cov=$(coverage_of "$CD/$row")
  if [[ "$cov" == *"coverage FAIL"* ]]; then rec "$name" FAIL "binding $cov; $stages"; return; fi
  if ! echo "$reason" | grep -qE "$want"; then rec "$name" FAIL "expected the ruled Commit verdict /$want/, got rc=$rc reason '${reason:-none: Commit PASS = the F-r16-02 overclaim (rev M-0068, COORD M-0090 (6))}'; $stages"; return; fi
  rec "$name" PASS "Build+Match green, Commit FAILS BY NAME as ruled (COORD M-0090 (6)) in $(( $(date +%s) - t0 ))s: '${reason:0:140}'; root ${root:0:16} $([ -n "$exp" ] && echo '== known-good' || echo '(no reference for this cc)'); ${cov:-no binding record}; $stages"
}
# The ruled reason has two stages (COORD M-0130 (b)/(c), rev F-r16-15): until pin's F-r16-MOE-PIN registers the fused-expert stacks
# (vLLM's routed_experts.w13_weight / w2_weight = concat(gate|up) / stack(down) over experts), the Commit fails closed at the WEIGHTS PIN with
# exactly the 32 fused-expert fields (16 layers x w13|w2) unresolved -- that is the verdict of record on r16-int-1/-2.  When F-r16-MOE-PIN lands,
# OLMOE_EXPECT becomes the plane-fed C2 reason (F-r16-07) until the row->plane mapping lands, then the line goes back to `positive`.
# r16-int-3 carries F-r16-MOE-PIN (pin b0f142740: 212/212 pinned on the int-1 OLMoE Program): the weights pin now PASSES and the verdict of
# record reaches the plane-fed rule -- commit_verdict._partial_replay_named_gap names the moe/ plane strata value-checked by NEITHER form.
# The stage-1 (weights-pin) reason would now be a REGRESSION of the pin and reads RED here, as does a bare PASS (the F-r16-02 overclaim).
# r16-int-3 carries the row->plane mapping too (moe MAN-06f v3 c1d42a908: the fused-MoE planes are VUs of the sampled replay) and FLUSH-03
# (3826b41be: the up plane is flushed before the block's w2 launch overwrites its alias) -- OLMoE B1 Commit PASS x3 with every plane
# value-checked on moe's lane tree (M-0249/M-0258).  The line is back to `positive`: Build+Match+Commit PASS.  A named FAIL of any stage
# reads RED again.  OLMOE_EXPECT (set in the environment) switches the line back to the named-verdict form when a ruling needs it.
if [[ ",$POS," == *,olmoe,* ]]; then
  if [ -n "${OLMOE_EXPECT:-}" ]; then positive_named olmoe "$OLMOE" OLMOE allenai/OLMoE-1B-7B-0924 6d84c48581ece794365f2b8e9cfb043c68ade9c5 "$OLMOE_EXPECT"
  else positive olmoe "$OLMOE" OLMOE allenai/OLMoE-1B-7B-0924 6d84c48581ece794365f2b8e9cfb043c68ade9c5; fi
fi

# ------------------------------------------------------------------ negatives on SmolLM2 (fast; the same commit_delta the rows use)
WL=workloads/$SMOL.json
BASE=($PY -m verity_vllm.pipeline.commit --case B0 --workload "$WL" --committer native_collect_v2b --fast-copy --gpu-tree --openings 64
      --pairs 1 --warmup 1 --max-num-seqs 1 --engine-arg max_model_len=288 --engine-arg "gpu_memory_utilization=$GPU_UTIL")
negative() {   # <name> <expect: exit3|rootdiff> <reason-regex> <args...>
  local name=$1 expect=$2 rx=$3; shift 3
  local d=$CD/neg_$name; rm -rf "$d"
  timeout 1500 "${BASE[@]}" --label "canary_${TAG}_neg_$name" --out "$d" "$@" > "$d.log" 2>&1; local rc=$?
  local reason root; reason=$(reason_of "$d/verdict.json"); root=$(root_of "$d/verdict.json")
  case $expect in
    exit3)
      if [ $rc -eq 3 ] && [[ "$reason" =~ $rx ]]; then rec "neg_$name" PASS "exit 3 as required; reason: ${reason:0:160}"
      else rec "neg_$name" FAIL "expected exit 3 with reason ~ /$rx/, got rc=$rc reason='${reason:0:160}' $(grep -m1 -E 'Error|Traceback' "$d.log" | cut -c1-120)"; fi ;;
    exit1)   # refused before the engine starts: rc 1 and the named refusal in the log (no verdict.json is written)
      if [ $rc -eq 1 ] && grep -qE "$rx" "$d.log"; then rec "neg_$name" PASS "refused as required (rc 1): $(grep -m1 -E "$rx" "$d.log" | cut -c1-160)"
      else rec "neg_$name" FAIL "expected rc 1 with /$rx/ in the log, got rc=$rc: $(grep -m1 -E 'REFUSED|Error|Traceback|COMMIT' "$d.log" | cut -c1-160)"; fi ;;
    rootdiff)
      local exp; exp=$(known_root "$SMOL")
      if [ -z "$root" ]; then rec "neg_$name" FAIL "no run root (rc=$rc): $(grep -m1 -E 'Error|Traceback|COMMIT FAIL' "$d.log" | cut -c1-160)"
      elif [ -n "$exp" ] && [ "$root" = "$exp" ]; then rec "neg_$name" FAIL "late read produced the KNOWN-GOOD root $root: the regression no longer detects a wrong-buffer read"
      else rec "neg_$name" PASS "root ${root:0:16} != known-good ${exp:0:16} (rc=$rc; openings still verified: binding != correctness)"; fi ;;
  esac
}
[[ ",$NEG," == *,arena,* ]]    && negative arena exit3 "expected_tensors .* != committed_tensors|arena block too small" --hidden-so "$HIDDEN_SO" --arena-mb 1 --arena-unchecked
[[ ",$NEG," == *,omitted,* ]]  && negative omitted exit3 "required class '$HIDDEN_CLASS' not declared|$HIDDEN_CLASS.*never fired"
[[ ",$NEG," == *,lateread,* ]] && negative lateread rootdiff "" --hidden-so "$HIDDEN_SO" --late-read qkv_proj
# [R14 BIND-01, F-rev-01] drop `/1` + bind `/0` twice on every layer>0 input_layernorm: counts + classes unchanged, so only the identity-coverage
# check against the row's required-value manifest (MAN-01, built by the smollm2 positive's Commit) can reject it -- exit 3 "identity coverage".
# [R15 manbind C1, COORD M-0205] --required-manifest is refused without --program-digest (the Build's digest of record, never the manifest's):
# the negative passes the smollm2 positive's Build digest (build_summary.json request.program_digest, as row_pod does) so the fault reaches
# the identity-coverage check; the r15-int-3 canary #1 (0f534a802) hit the C1 refusal here (rc 1) instead -- a harness gap, not a weakening.
if [[ ",$NEG," == *,drop1dup0,* ]]; then
  MAN="$CD/$SMOL/manifest.json"
  PD=$($PY -c 'import json,sys; print((json.load(open(sys.argv[1])).get("request") or {}).get("program_digest") or "")' "$CD/$SMOL/build_summary.json" 2>/dev/null || true)
  if [ ! -f "$MAN" ]; then rec neg_drop1dup0 FAIL "no required-value manifest at $MAN (run the smollm2 positive first; row_pod commit builds it)"
  elif [ -z "$PD" ]; then rec neg_drop1dup0 FAIL "no Build program digest in $CD/$SMOL/build_summary.json (request.program_digest): cannot arm --required-manifest"
  else export VERITY_FAULT=drop1_dup0; negative drop1dup0 exit3 "identity coverage" --hidden-so "$HIDDEN_SO" --required-manifest "$MAN" --program-digest "$PD"; unset VERITY_FAULT; fi
fi
# [R15 manbind C1] the refusal itself is a canary negative: a required manifest WITHOUT the Build's digest must be refused before the engine
# starts (rc 1, named), never silently accepted (rev-cb M-0334 F-cb-05/06).
if [[ ",$NEG," == *,nopd,* ]]; then
  MAN="$CD/$SMOL/manifest.json"
  if [ ! -f "$MAN" ]; then rec neg_nopd FAIL "no required-value manifest at $MAN (run the smollm2 positive first)"
  else negative nopd exit1 "REFUSED: --required-manifest requires --program-digest" --hidden-so "$HIDDEN_SO" --required-manifest "$MAN"; fi
fi
# [R15 manbind MAN-06 O1, COORD M-0522 (1)] sampled-replay ABSENCE: a row-procedure Commit (binding block armed by --required-manifest) that is
# given NO Program dir (--program-dir / --programs-root) must FAIL named "C2 of record not established" (exit 3) -- never a PASS on C1 + C3
# alone (DECISIONS-2 §4; rev-cb M-0411 O1).  The smollm2 positive's own Commit (row_pod) runs WITH the Program dir: that is the replay positive.
if [[ ",$NEG," == *,nosr,* ]]; then
  MAN="$CD/$SMOL/manifest.json"
  PD=$($PY -c 'import json,sys; print((json.load(open(sys.argv[1])).get("request") or {}).get("program_digest") or "")' "$CD/$SMOL/build_summary.json" 2>/dev/null || true)
  if [ ! -f "$MAN" ]; then rec neg_nosr FAIL "no required-value manifest at $MAN (run the smollm2 positive first)"
  elif [ -z "$PD" ]; then rec neg_nosr FAIL "no Build program digest in $CD/$SMOL/build_summary.json (request.program_digest)"
  else negative nosr exit3 "C2 of record not established" --hidden-so "$HIDDEN_SO" --required-manifest "$MAN" --program-digest "$PD"; fi
fi
# [R16 pin, BRIEF 4A "substituted weight input"; MAN-09 / rev-cb M-0610] WEIGHTS PIN negative on the real engine: the smollm2 positive's
# Commit derived weights_of_record.json (row_pod MAN-09 hunk: the pinned checkpoint shards composed by the stacking rule, for the Build's
# Program).  The negative serves the SAME checkpoint with ONE element of one live parameter changed before register_weights
# (commit_delta --flip-weight): the pin must FAIL (exit 3) naming that served field -- first_fail_reason "weights_pin: 1 served weight(s)
# differ from the checkpoint of record: model.layers.0.self_attn.qkv_proj.weight" (the value_correspondence composite names the failing
# component, as manbind M-0684 measured on int-5; "weights of record: ..." when only the verdict's own pin rule fires).  The record is checked
# against the positive's Build digest (W2), as row_pod does.
if [[ ",$NEG," == *,flipweight,* ]]; then
  WOR="$CD/$SMOL/build_request/weights_of_record.json"     # B1: row_pod derives the record beside the Build Program (PD_WOR)
  PD=$($PY -c 'import json,sys; print((json.load(open(sys.argv[1])).get("request") or {}).get("program_digest") or "")' "$CD/$SMOL/build_summary.json" 2>/dev/null || true)
  if [ ! -f "$WOR" ]; then rec neg_flipweight FAIL "no weights_of_record.json at $WOR (run the smollm2 positive first; its row_pod Commit derives the pin beside build_request, MAN-09)"
  elif [ -z "$PD" ]; then rec neg_flipweight FAIL "no Build program digest in $CD/$SMOL/build_summary.json (request.program_digest)"
  else negative flipweight exit3 "(weights_pin|weights of record): .*served weight\(s\) differ from the checkpoint of record: .*model\.layers\.0\.self_attn\.qkv_proj\.weight" \
         --hidden-so "$HIDDEN_SO" --weights-of-record "$WOR" --program-digest "$PD" --flip-weight model.layers.0.self_attn.qkv_proj.weight:7; fi
fi
# [R15 moe FLUSH-02, ALERT M-0479 / COORD M-0522 (1)] the un-fixed late read as a canary negative: the panel's QK-norm / fused-MoE models are
# not on this pod, so the check is moe's byte-level test file over the BUILT C++ collector (the committer's own hook bodies with an in-place
# RoPE / fused-add stand-in): the positive must equal the produced bytes and each VERITY_FAULT=late_flush_qk / late_flush_norm negative must
# equal the OVERWRITTEN bytes (= the r15-int-3 OLMoE record).  A skip (no torch / no CUDA / no collector build) is a FAIL, not a pass.
if [[ ",$NEG," == *,lateflush,* ]]; then
  d=$CD/neg_lateflush; rm -rf "$d"; mkdir -p "$d"
  NATIVE_COLLECT_BUILD=${NATIVE_COLLECT_BUILD:-/workspace/cp/nc_build} timeout 900 $PY -m pytest -q -p no:cacheprovider \
      tests/commit/test_native_collect_flush.py > "$d/pytest.log" 2>&1; rc=$?
  summ=$(grep -E "^[0-9]+ (passed|failed|skipped|error)|^[0-9]+ passed" "$d/pytest.log" | tail -1); summ=${summ:-$(tail -1 "$d/pytest.log")}
  npass=$(echo "$summ" | sed -nE 's/.*\b([0-9]+) passed.*/\1/p'); npass=${npass:-0}
  if [ $rc -eq 0 ] && [ "$npass" -ge 4 ] && [[ "$summ" != *skipped* ]] && [[ "$summ" != *failed* ]]; then
    rec neg_lateflush PASS "collector flush tests: $summ (positive == produced bytes; late_flush_qk / late_flush_norm == overwritten bytes)"
  else rec neg_lateflush FAIL "collector flush tests rc=$rc: ${summ:0:160} $(grep -m1 -E 'Error|FAILED|SKIPPED' "$d/pytest.log" | cut -c1-120)"; fi
fi
# [R15 rel, GM-01 (global M-0095)] swapped-request workload: the two B=2 requests' prompts/caps exchanged under the SAME UUIDs, composed into a
# workload Program from the llama_b2_bm positive's derived components (the shape set is unchanged, so GP-01 composes), checked against the
# positive's real execution -> global_match must FAIL with G6 (output association) among the failed checks (global's fixture: G1,G3,G4,G6,G7).
# A PASS here would mean the global comparator no longer ties declared requests to what was executed.
if [[ ",$NEG," == *,wl_swapped,* ]]; then
  PY312=${PY312:-/workspace/venv-py312/bin/python}
  RB=$CD/$LLAMA_B2; NW=$CD/neg_wl_swapped; rm -rf "$NW"; mkdir -p "$NW"
  if [ ! -f "$RB/build_request/result.json" ] || [ ! -f "$RB/match/program.json" ]; then rec neg_wl_swapped FAIL "no llama_b2_bm Build/Match under $RB (run positives=...,llama_b2_bm first)"
  else
    python3 - workloads/$LLAMA_B2.json "$NW/wl_swapped.json" <<'EOF'
import json, sys
wl = json.load(open(sys.argv[1])); a, b = wl["requests"][0], wl["requests"][1]
for k in ("prompt_len", "max_tokens", "prompt_token_ids", "prompt_text", "origin"):   # identities (request_id, index, arrival) stay; contents exchange
    a[k], b[k] = b.get(k), a.get(k)
for k in ("prompt_lengths", "max_tokens"):
    if isinstance(wl.get(k), list) and len(wl[k]) == 2: wl[k] = [wl[k][1], wl[k][0]]
wl["note"] = "CANARY NEGATIVE wl_swapped: " + str(wl.get("note", ""))
json.dump(wl, open(sys.argv[2], "w"), indent=1)
EOF
    DER=(--derived "$RB/build_request"); for dd in "$RB"/build_request_LP*_T*; do [ -d "$dd" ] && DER+=(--derived "$dd"); done
    timeout 1500 $PY312 -m verity_vllm.pipeline.global_program --workload "$NW/wl_swapped.json" "${DER[@]}" --out "$NW/gp_swapped" > "$NW/gp_swapped.log" 2>&1; rcg=$?
    timeout 1500 $PY312 -m verity_vllm.check.match.global_match --gprog "$NW/gp_swapped/workload_program.json" --workload "$NW/wl_swapped.json" --oracle "$RB/match" \
        --out "$NW/gm_swapped.json" --within-step "${WITHIN_STEP:-dag}" > "$NW/gm_swapped.log" 2>&1; rcm=$?
    verd=$(python3 -c "import json;g=json.load(open('$NW/gm_swapped.json'));print(g.get('verdict'), ','.join(g.get('failed_checks') or []))" 2>/dev/null)
    if [ $rcg -ne 0 ]; then rec neg_wl_swapped FAIL "GP-01 compose of the swapped manifest failed rc=$rcg (the negative needs a composed Program): $(grep -m1 -E 'Error|Traceback' "$NW/gp_swapped.log" | cut -c1-120)"
    elif [[ "$verd" == FAIL* ]] && [[ ",${verd#FAIL },," == *,G6,* ]]; then rec neg_wl_swapped PASS "global_match FAIL as required (rc=$rcm): failed checks ${verd#FAIL }"
    else rec neg_wl_swapped FAIL "expected global_match FAIL with G6, got '${verd:-no verdict}' rc=$rcm $(grep -m1 -E 'Error|Traceback' "$NW/gm_swapped.log" | cut -c1-120)"; fi
  fi
fi

# ------------------------------------------------------------------ summary
python3 - "$CD/canary.json" "$TAG" "$COMMIT" "$CC" "$HIDDEN_SO" "${RESULTS[@]}" <<'EOF'
import json, sys, time
out, tag, commit, cc, so, *res = sys.argv[1:]
checks = [dict(zip(("check", "result", "detail"), r.split("|", 2))) for r in res]
json.dump({"schema": "verity-sweep/canary/v1", "tag": tag, "commit": commit, "compute_capability": cc, "hidden_so": so, "green": all(c["result"] == "PASS" for c in checks),
           "checks": checks, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(out, "w"), indent=1)
EOF
if [ -z "$FIRST_FAIL" ]; then log "CANARY-GREEN $TAG"; else log "CANARY-RED $TAG $FIRST_FAIL"; exit 1; fi
