#!/usr/bin/env bash
# The release canary suite, run on the verification worker for every release candidate before it is tagged.
#
#   cd <tree>/integrations/vllm && bash verity_vllm/ops/canary.sh <candidate-tag> [positives=smollm2,llama[,llama_b2,llama_b2_bm,phi3,qwen,olmoe,stoch]] [negatives=arena,omitted,lateread[,drop1dup0,nopd,nosr,flipweight,lateflush,wl_swapped]]
#   -> evidence under /workspace/cp/canary/<tag>/, summary canary.json; last line  CANARY-GREEN <tag>  or  CANARY-RED <tag> <first failing check>
#
# Positives (the two base canaries, through the same per-workload procedure the sweep uses: `verity-vllm row run`, PAIRS=1):
#   SmolLM2-135M B1 256/32 and Llama-3.2-1B B1 256/32 on this GPU: Build (the step and request Programs, --T = O-1) -> Match (eager
#   capture + fold -> cb_a.compare canonical-equal, same step count, tokens equal) -> fail-closed Commit (64 openings, 1 pair),
#   and the run root must equal the known-good root pinned for this compute capability in verity_vllm/ops/known_roots.json
#   (cb129578b7846e1f… / cfd86e06b12b61c1…, reproduced bit for bit across pods).  A different root with an otherwise green Commit is RED:
#   what the collector binds changed (or the values did) and that needs a decision, not a tag.
# Negatives (SmolLM2, must FAIL the way stated):
#   arena     --arena-mb 1 --arena-unchecked            -> exit 3, reason expected_tensors != committed_tensors / arena block too small
#   omitted   no --hidden-so (FA2 producer never attached) -> exit 3, reason "required class 'fa2_hidden_m1_stream' not declared"
#             on a Hopper pod (cc 9.x, FlashAttention-3 engine) the class is 'fa3_hidden_m1_stream' and the tap fa3_matReq
#   lateread  --late-read qkv_proj (a wrong-buffer read)  -> the run root MUST differ from the known-good root (openings still verify: binding
#             is not correctness -- the harness must never take a verifying root as evidence the right buffer was read)
# Env: everything `verity-vllm row` takes (PY, PY312, HIDDEN_SO, GPU_UTIL, timeouts); CANARY_DIR (/workspace/cp/canary).
set -uo pipefail
TAG=${1:?candidate tag, e.g. r13-rel0}; POS=${2:-smollm2,llama}; NEG=${3:-arena,omitted,lateread}
# The tree that runs is the tree this script lives in: $VERITY_TREE is set from here, never read as the default, and a caller exporting
# another tree is refused by name.  A second canary GPU takes the same candidate installed as its own tree (pod_release.sh
# REMOTE_TREE=/workspace/cp/<tag>) without replacing the pod's live tree.  The collector build dir is the one the release built for this
# tree (RELEASE.json build_dirs.native_collect, never another tree's nc_build), unless the caller sets NATIVE_COLLECT_BUILD.
# CUDA_VISIBLE_DEVICES passes through.
TREE_SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [ -n "${VERITY_TREE:-}" ] && [ "$(cd "$VERITY_TREE" 2>/dev/null && pwd -P)" != "$TREE_SELF" ]; then
  echo "PRECHECK_FAIL_SOURCE_IDENTITY: canary.sh at $TREE_SELF but \$VERITY_TREE=$VERITY_TREE names another tree" >&2; exit 3
fi
export VERITY_TREE=$TREE_SELF
cd "$VERITY_TREE" || exit 1
# The canary's base workloads are `__l40s__` ids and the canary also runs on the Hopper host, so the target-family preflight
# (`verity-vllm row`, PRECHECK_FAIL_TARGET_FAMILY) is waived by name here (`precheck-target WAIVED canary <tag>` in stages.txt), never on a
# workload of record.  On an L40S the same workloads pass the check unwaived: the waiver only names a would-be failure.
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
# The tap is the attention implementation this device executes (TargetProfile.effective_flash_attn_version(), as row_pod.sh):
# cc 9.x runs FlashAttention-3 -> the FA3 M1 tap and the required hidden class is fa3_hidden_m1_stream; the omitted negative's
# expected reason names that class.  HIDDEN_SO overrides (a foreign tap is refused by commit_delta's engine check, not silently accepted).
FA_VER=$($PY -c "from verity_vllm.program.frontend.target_profile import TargetProfile; print(TargetProfile.from_device().effective_flash_attn_version())" 2>/dev/null || echo 2)
G=$([ "${GUARDED_MAX_TAP:-1}" = 1 ] && echo G)   # the guarded-max build unless GUARDED_MAX_TAP=0 (config.CommitConfig), as a workload selects it
HIDDEN_SO=${HIDDEN_SO:-$([ "$FA_VER" = 3 ] && echo /workspace/cp/fa2/build/fa3_matReq$G/verity_fa3_matReq$G.so || echo /workspace/cp/fa2/build/matReq$G/verity_fa2_matReq$G.so)}; export HIDDEN_SO
HIDDEN_CLASS=$([ "$FA_VER" = 3 ] && echo fa3_hidden_m1_stream || echo fa2_hidden_m1_stream)
GPU_UTIL=${GPU_UTIL:-0.5}; export GPU_UTIL
CD=${CANARY_DIR:-/workspace/cp/canary}/$TAG; mkdir -p "$CD"; export SWEEP_DIR=$CD
CC=$($PY -c "import torch;p=torch.cuda.get_device_properties(0);print(f'{p.major}.{p.minor}')")
COMMIT=$(python3 -c "import json;print(json.load(open('EXPORT.json'))['commit'][:9])" 2>/dev/null || echo unknown)
SMOL=smollm2-135m__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager
LLAMA=llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager
LLAMA_B2=llama32-1b__bf16__l40s__tp1__b2__i256__o32__mixed__greedy__bi-eager   # B>1: Match by batch decomposition (positives=...,llama_b2)
PHI3=phi3-mini__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager         # head dim 96 through the portable tap (positives=...,phi3; needs PHI3_MINI seeded)
QWEN=qwen25-15b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager        # Qwen2 qkv bias (BroadcastBiasAdd) + hd-128 tap; root of record from the sweep (positives=...,qwen; needs B1 seeded)
# The MoE family (OLMoE-1B-7B: the fused-MoE block source, the moe_block_stream class, the gather-staging flush), root of record the B1 root
# in known_roots.json.  The qwen and olmoe positives cover the qkv-bias and MoE defect classes (positives=...,qwen,olmoe; needs OLMOE seeded).
OLMOE=olmoe-1b-7b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager
log() { echo "$(date -u +%FT%TZ) [canary $TAG] $*" | tee -a "$CD/canary.log"; }
RESULTS=()   # "name|PASS/FAIL|detail"
FIRST_FAIL=""
rec() { RESULTS+=("$1|$2|$3"); log "$1 $2 -- $3"; [ "$2" = PASS ] || [ -n "$FIRST_FAIL" ] || FIRST_FAIL="$1: $3"; }
known_root() { python3 -c "import json;print((json.load(open('verity_vllm/ops/known_roots.json'))['roots'].get('$1') or {}).get('$CC',''))"; }
root_of() { python3 -c "import json,sys;v=json.load(open('$1'));print((v.get('run_roots') or [''])[0])" 2>/dev/null; }
reason_of() { python3 -c "import json;print(json.load(open('$1')).get('first_fail_reason') or '')" 2>/dev/null; }
# The workload's identity-coverage line: commit_delta prints "[delta] BINDING (pair k): ... coverage OK|FAIL bound n/N missing m ..."
# per pair (the binding record itself lives in the pair rows); verdict.json carries the required manifest's digest.
coverage_of() {   # <workload dir>
  local cov man
  cov=$(grep -o 'coverage \(OK\|FAIL\) bound [0-9]*/[0-9]* missing [0-9]*' "$1/commit.log" 2>/dev/null | sort | uniq -c | awk '{printf "%s x%d; ", substr($0, index($0,$2)), $1}')
  man=$(python3 -c "import json;m=json.load(open('$1/commit/verdict.json')).get('required_manifest') or {};print('manifest', (m.get('manifest_digest') or '')[:16] or 'none')" 2>/dev/null)
  [ -n "$cov" ] && echo "${cov% }; $man"
}

log "start commit=$COMMIT cc=$CC flash_attn=$FA_VER hidden_so=$HIDDEN_SO hidden_class=$HIDDEN_CLASS positives=$POS negatives=$NEG"

# ------------------------------------------------------------------ positives: the workload procedure itself, then the root pin
positive() {   # <name> <row-id> <ROLE> <repo> <revision>
  local name=$1 row=$2 role=$3 repo=$4 rev=$5 t0=$(date +%s)
  local exp; exp=$(known_root "$row")
  PAIRS=1 "${PY%/*}/verity-vllm" row run "$row" "$role" "$repo" "$rev" --stages build,match,commit > "$CD/$name.row.log" 2>&1; local rc=$?
  local stages; stages=$(tr '\n' ';' < "$CD/$row/stages.txt" 2>/dev/null | cut -c1-400)
  if [ $rc -ne 0 ]; then rec "$name" FAIL "row_pod rc=$rc stages: $stages"; return; fi
  local root; root=$(root_of "$CD/$row/commit/verdict.json")
  if [ -z "$root" ]; then rec "$name" FAIL "no run root in commit/verdict.json; stages: $stages"; return; fi
  if [ -n "$exp" ] && [ "$root" != "$exp" ]; then rec "$name" FAIL "Build/Match/Commit green but run root $root != known-good $exp (cc $CC): what is bound changed -- decision needed"; return; fi
  # identity coverage against the workload's required-value manifest must be complete (the verdict is fail-closed on it already; shown for the record)
  local cov; cov=$(coverage_of "$CD/$row")
  if [[ "$cov" == *"coverage FAIL"* ]]; then rec "$name" FAIL "Commit verdict green but binding $cov; $stages"; return; fi
  rec "$name" PASS "Build+Match+Commit green in $(( $(date +%s) - t0 ))s; root ${root:0:16} $([ -n "$exp" ] && echo '== known-good' || echo '(no reference for this cc)'); ${cov:-no binding record}; $stages"
}
[[ ",$POS," == *,smollm2,* ]] && positive smollm2 "$SMOL" B0 HuggingFaceTB/SmolLM2-135M 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
[[ ",$POS," == *,llama,* ]]   && positive llama "$LLAMA" LLAMA32_1B unsloth/Llama-3.2-1B 9535bd9b1d1dea6acafbdc4813b728796aeb28da
# B>1: Build derives both request shapes, Match = cb_a.batch_decomp, Commit at B=2, root pinned in known_roots.json.  The default
# positives are the two B=1 workloads; a release touching Build, Match or row_pod runs with llama_b2 too.
[[ ",$POS," == *,llama_b2,* ]] && positive llama_b2 "$LLAMA_B2" LLAMA32_1B unsloth/Llama-3.2-1B 9535bd9b1d1dea6acafbdc4813b728796aeb28da
# The stochastic B1 panel workload as a positive: S is a request-owned Program input `splits` and a required family, so this line is the
# live check that a stochastic workload on the tree Builds (derived S), Matches (the global comparator's G6 reads the bound operand) and
# Commits with the population complete (sampler_splits acquired from the launched kernel's constexpr, never recomputed).  No known-good root
# is pinned for the derived-S population yet (the root changes with what is bound): PASS reads "(no reference for this cc)" until a
# reviewer pins it.  Same model and revision as the llama lines.
STOCH=llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__stoch-t0.8-p0.95__bi-eager
[[ ",$POS," == *,stoch,* ]] && positive stoch "$STOCH" LLAMA32_1B unsloth/Llama-3.2-1B 9535bd9b1d1dea6acafbdc4813b728796aeb28da
# The B=2 workload through Build and Match only: Build must compose the workload Program (build_workload/workload_program.json, request
# ids r<k>) and Match must PASS the global comparator (cb_a.global_match/v1 G1..G8, --within-step ${WITHIN_STEP:-dag}) over the same fold.
# llama_b2 runs the same workload with its Commit and root pin; the wl_swapped negative reads this positive's Build and Match.
positive_bm() {   # <name> <row-id> <ROLE> <repo> <revision>   Build+Match positive; PASS = row_pod rc 0 + match_summary.pass + global_match PASS
  local name=$1 row=$2 role=$3 repo=$4 rev=$5 t0=$(date +%s)
  PAIRS=1 "${PY%/*}/verity-vllm" row run "$row" "$role" "$repo" "$rev" --stages build,match > "$CD/$name.row.log" 2>&1; local rc=$?
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
  rec "$name" PASS "Build+Match green in $(( $(date +%s) - t0 ))s (no Commit: llama_b2 runs it); $detail; $stages"
}
[[ ",$POS," == *,llama_b2_bm,* ]] && positive_bm llama_b2_bm "$LLAMA_B2" LLAMA32_1B unsloth/Llama-3.2-1B 9535bd9b1d1dea6acafbdc4813b728796aeb28da
# Head dim 96 (Phi-3-mini): the tap's hd-96 instantiation and its QKV handling; root pinned in known_roots.json.
[[ ",$POS," == *,phi3,* ]]     && positive phi3 "$PHI3" PHI3_MINI microsoft/Phi-3-mini-4k-instruct f39ac1d28e925b323eae81227eaba4464caced4e
# The Qwen family (qkv bias add, head dim 128); root of record from the sweep.
[[ ",$POS," == *,qwen,* ]]     && positive qwen "$QWEN" B1 Qwen/Qwen2.5-1.5B 8faed761d45a263340a0528343f099c05c9a4323
# OLMoE B1 through the per-workload procedure (MoE block source, weights pin on 3 shards); root pinned from the MoE B1 run.
# positive_named: a positive whose Commit verdict of record is a named FAIL.  It requires Build+Match green, the run root == known-good,
# coverage OK, and commit/verdict.json's first_fail_reason to match the expected pattern; a Commit PASS fails the canary, as does any
# other reason.  Never a canary that accepts both.
positive_named() {   # <name> <row-id> <ROLE> <repo> <revision> <required first_fail_reason, grep -E pattern>
  local name=$1 row=$2 role=$3 repo=$4 rev=$5 want=$6 t0=$(date +%s)
  local exp; exp=$(known_root "$row")
  PAIRS=1 "${PY%/*}/verity-vllm" row run "$row" "$role" "$repo" "$rev" --stages build,match,commit > "$CD/$name.row.log" 2>&1; local rc=$?
  local stages; stages=$(tr '\n' ';' < "$CD/$row/stages.txt" 2>/dev/null | cut -c1-400)
  [ -f "$CD/$row/commit/verdict.json" ] || { rec "$name" FAIL "no commit/verdict.json (row_pod rc=$rc: Build or Match did not reach Commit); stages: $stages"; return; }
  local root reason; root=$(root_of "$CD/$row/commit/verdict.json"); reason=$(reason_of "$CD/$row/commit/verdict.json")
  if [ -z "$root" ]; then rec "$name" FAIL "no run root in commit/verdict.json; stages: $stages"; return; fi
  if [ -n "$exp" ] && [ "$root" != "$exp" ]; then rec "$name" FAIL "run root $root != known-good $exp (cc $CC): what is bound changed -- decision needed"; return; fi
  local cov; cov=$(coverage_of "$CD/$row")
  if [[ "$cov" == *"coverage FAIL"* ]]; then rec "$name" FAIL "binding $cov; $stages"; return; fi
  if ! echo "$reason" | grep -qE "$want"; then rec "$name" FAIL "expected the named Commit verdict /$want/, got rc=$rc reason '${reason:-none: a Commit PASS where a named FAIL is expected}'; $stages"; return; fi
  rec "$name" PASS "Build+Match green, Commit FAILS BY NAME as expected in $(( $(date +%s) - t0 ))s: '${reason:0:140}'; root ${root:0:16} $([ -n "$exp" ] && echo '== known-good' || echo '(no reference for this cc)'); ${cov:-no binding record}; $stages"
}
# OLMoE runs as a `positive`: the weights pin registers the fused-expert stacks (vLLM's routed_experts.w13_weight / w2_weight =
# concat(gate|up) / stack(down) over experts), the fused-MoE planes are VUs of the sampled replay, and the up plane is flushed before the
# block's w2 launch overwrites its alias, so Build+Match+Commit PASS with every plane value-checked.  A named FAIL of any stage reads RED.
# OLMOE_EXPECT, set in the environment, switches the line to the named-verdict form with that expected reason.
if [[ ",$POS," == *,olmoe,* ]]; then
  if [ -n "${OLMOE_EXPECT:-}" ]; then positive_named olmoe "$OLMOE" OLMOE allenai/OLMoE-1B-7B-0924 6d84c48581ece794365f2b8e9cfb043c68ade9c5 "$OLMOE_EXPECT"
  else positive olmoe "$OLMOE" OLMOE allenai/OLMoE-1B-7B-0924 6d84c48581ece794365f2b8e9cfb043c68ade9c5; fi
fi

# ------------------------------------------------------------------ negatives on SmolLM2 (fast; the same commit_delta the workloads use)
WL=workloads/$SMOL.json
BASE=("${PY%/*}/verity-vllm" commit --case B0 --workload "$WL" --committer native_collect_v2b --fast-copy --gpu-tree --openings 64
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
# drop1dup0: drop `/1` and bind `/0` twice on every layer>0 input_layernorm.  Counts and classes are unchanged, so only the
# identity-coverage check against the workload's required-value manifest (built by the smollm2 positive's Commit) can reject it: exit 3
# "identity coverage".  --required-manifest is refused without --program-digest (the Build's digest of record, never the manifest's), so
# the negative passes the smollm2 positive's Build digest (build_summary.json request.program_digest, as row_pod does) and the fault reaches
# the identity-coverage check.
if [[ ",$NEG," == *,drop1dup0,* ]]; then
  MAN="$CD/$SMOL/manifest.json"
  PD=$($PY -c 'import json,sys; print((json.load(open(sys.argv[1])).get("request") or {}).get("program_digest") or "")' "$CD/$SMOL/build_summary.json" 2>/dev/null || true)
  if [ ! -f "$MAN" ]; then rec neg_drop1dup0 FAIL "no required-value manifest at $MAN (run the smollm2 positive first; row_pod commit builds it)"
  elif [ -z "$PD" ]; then rec neg_drop1dup0 FAIL "no Build program digest in $CD/$SMOL/build_summary.json (request.program_digest): cannot arm --required-manifest"
  else export VERITY_FAULT=drop1_dup0; negative drop1dup0 exit3 "identity coverage" --hidden-so "$HIDDEN_SO" --required-manifest "$MAN" --program-digest "$PD"; unset VERITY_FAULT; fi
fi
# The refusal itself is a canary negative: a required manifest without the Build's digest must be refused before the engine starts
# (rc 1, named), never silently accepted.
if [[ ",$NEG," == *,nopd,* ]]; then
  MAN="$CD/$SMOL/manifest.json"
  if [ ! -f "$MAN" ]; then rec neg_nopd FAIL "no required-value manifest at $MAN (run the smollm2 positive first)"
  else negative nopd exit1 "REFUSED: --required-manifest requires --program-digest" --hidden-so "$HIDDEN_SO" --required-manifest "$MAN"; fi
fi
# Sampled-replay absence: a Commit of the per-workload procedure (binding block armed by --required-manifest) that is given no Program
# dir (--program-dir / --programs-root) must FAIL named "C2 of record not established" (exit 3), never PASS on C1 and C3 alone.  The
# smollm2 positive's own Commit (row_pod) runs with the Program dir: that is the replay positive.
if [[ ",$NEG," == *,nosr,* ]]; then
  MAN="$CD/$SMOL/manifest.json"
  PD=$($PY -c 'import json,sys; print((json.load(open(sys.argv[1])).get("request") or {}).get("program_digest") or "")' "$CD/$SMOL/build_summary.json" 2>/dev/null || true)
  if [ ! -f "$MAN" ]; then rec neg_nosr FAIL "no required-value manifest at $MAN (run the smollm2 positive first)"
  elif [ -z "$PD" ]; then rec neg_nosr FAIL "no Build program digest in $CD/$SMOL/build_summary.json (request.program_digest)"
  else negative nosr exit3 "C2 of record not established" --hidden-so "$HIDDEN_SO" --required-manifest "$MAN" --program-digest "$PD"; fi
fi
# Weights-pin negative (a substituted weight input) on the real engine: the smollm2 positive's Commit derived weights_of_record.json
# (row_pod: the pinned checkpoint shards composed by the stacking rule, for the Build's Program).  The negative serves the same checkpoint
# with one element of one live parameter changed before register_weights (commit_delta --flip-weight): the pin must FAIL (exit 3) naming
# that served field, first_fail_reason "weights_pin: 1 served weight(s) differ from the checkpoint of record:
# model.layers.0.self_attn.qkv_proj.weight" (the value_correspondence composite names the failing component; "weights of record: ..." when
# only the verdict's own pin rule fires).  The record is checked against the positive's Build digest, as row_pod does.
if [[ ",$NEG," == *,flipweight,* ]]; then
  WOR="$CD/$SMOL/build_request/weights_of_record.json"     # B1: row_pod derives the record beside the Build Program (PD_WOR)
  PD=$($PY -c 'import json,sys; print((json.load(open(sys.argv[1])).get("request") or {}).get("program_digest") or "")' "$CD/$SMOL/build_summary.json" 2>/dev/null || true)
  if [ ! -f "$WOR" ]; then rec neg_flipweight FAIL "no weights_of_record.json at $WOR (run the smollm2 positive first; its row_pod Commit derives the pin beside build_request)"
  elif [ -z "$PD" ]; then rec neg_flipweight FAIL "no Build program digest in $CD/$SMOL/build_summary.json (request.program_digest)"
  else negative flipweight exit3 "(weights_pin|weights of record): .*served weight\(s\) differ from the checkpoint of record: .*model\.layers\.0\.self_attn\.qkv_proj\.weight" \
         --hidden-so "$HIDDEN_SO" --weights-of-record "$WOR" --program-digest "$PD" --flip-weight model.layers.0.self_attn.qkv_proj.weight:7; fi
fi
# The late read without its flush as a canary negative: the panel's QK-norm / fused-MoE models are not on this pod, so the check is the
# byte-level test file over the built C++ collector (the committer's own hook bodies with an in-place RoPE / fused-add stand-in): the
# positive must equal the produced bytes and each VERITY_FAULT=late_flush_qk / late_flush_norm negative must equal the overwritten bytes.
# A skip (no torch / no CUDA / no collector build) is a FAIL, not a pass.
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
# Swapped-request workload: the two B=2 requests' prompts/caps exchanged under the same UUIDs, composed into a workload Program from the
# llama_b2_bm positive's derived components (the shape set is unchanged, so it composes), checked against the positive's real execution
# -> global_match must FAIL with G6 (output association) among the failed checks (the comparator's fixture fails G1,G3,G4,G6,G7).
# A PASS here would mean the global comparator no longer ties declared requests to what was executed.
if [[ ",$NEG," == *,wl_swapped,* ]]; then
  PY312=${PY312:-/workspace/venv-py312/bin/python}
  RB=$CD/$LLAMA_B2; NW=$CD/neg_wl_swapped; rm -rf "$NW"; mkdir -p "$NW"
  if [ ! -f "$RB/build_request/result.json" ] || [ ! -f "$RB/match/program.json" ]; then rec neg_wl_swapped FAIL "no llama_b2_bm Build/Match under $RB (run positives=...,llama_b2_bm first)"
  else
    "${PY312%/*}/verity-vllm" workload-file "workloads/$LLAMA_B2.json" > /dev/null
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
    timeout 1500 "${PY312%/*}/verity-vllm" global-program --workload "$NW/wl_swapped.json" "${DER[@]}" --out "$NW/gp_swapped" > "$NW/gp_swapped.log" 2>&1; rcg=$?
    timeout 1500 "${PY312%/*}/verity-vllm" global-match --gprog "$NW/gp_swapped/workload_program.json" --workload "$NW/wl_swapped.json" --oracle "$RB/match" \
        --out "$NW/gm_swapped.json" --within-step "${WITHIN_STEP:-dag}" > "$NW/gm_swapped.log" 2>&1; rcm=$?
    verd=$(python3 -c "import json;g=json.load(open('$NW/gm_swapped.json'));print(g.get('verdict'), ','.join(g.get('failed_checks') or []))" 2>/dev/null)
    if [ $rcg -ne 0 ]; then rec neg_wl_swapped FAIL "composing the swapped manifest's workload Program failed rc=$rcg (the negative needs a composed Program): $(grep -m1 -E 'Error|Traceback' "$NW/gp_swapped.log" | cut -c1-120)"
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
