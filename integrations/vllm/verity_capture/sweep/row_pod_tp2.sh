#!/usr/bin/env bash
# sweep/row_pod_tp2.sh -- pod-side runner of ONE tensor-parallel (TP2) sweep row, the TP2 analogue of row_pod.sh (which is the TP1 harness:
# run_config Match, commit_delta Commit, no --tp/--rank).  Build (derive_step --tp 2 --rank r per rank: step Program + one request Program per
# declared (prompt_len, cap-1) shape, GP-01 compose per rank) -> Match (tp2_capture observer+collective recorder, control arm, tp2_match --arm
# check = collective-level structure/provenance/tokens; then, for a case with a TP2 RANK PROFILE, the D90 per-rank fold Match tp2_fold_match)
# -> Commit (required_manifest build-global over both ranks, tp2_commit native_collect_v2b --value-check).  Early stop at the first failing
# stage.  Evidence under $SWEEP_DIR/<row-id>/ (build/rank<r>/, match/, manifest.json, commit/); one verdict line per stage in stages.txt.
#
#   WORLD=2 NCCL_P2P_DISABLE=1 bash verity_capture/sweep/row_pod_tp2.sh <row-id> <ROLE> <hf-repo> <revision> [stages=build,match,commit]
#
# PROVENANCE [R19 lane tp2]: the stage sequence is veritor lane/r17-tp's evidence-dir orchestrators, verbatim in flags --
#   out/gen/sweep/evidence/tp/olmoe-1b-7b__...__tp2__b8.../r17int6/run_olmoe_b8.sh   (#70: MML 1152, max_num_batched_tokens 3763, GPU_UTIL 0.6,
#     B8, derive per rank --tp 2 --rank r, tp2_capture --record-collectives + --no-observer control, tp2_match --arm check, required_manifest
#     build-global --programs-root per rank --workload-digest per rank, tp2_commit --value-check --replay-workers)
#   out/gen/sweep/evidence/tp/qwen3-30b-a3b__...__tp2__b2.../r17int9/scripts/run_qwen3_int6.sh (#75: B2, shapes LP1024_T127 + LP9_T1,
#     max_num_batched_tokens 2048, Commit GPU util 0.80; the D90 fold Match tp2_fold_match --profile <rank profile> per lane qwen3's
#     R19 recipe, board 06:02Z)
# and the R19 board posts of lanes moe (06:14Z, F-r19-moe-1) and qwen3 (06:02Z).  Differences from r17, stated: (1) the request Programs are
# derived BEFORE the capture from the manifest's declared shapes (r17int6 derived them after the capture from the served lengths; on #70 the
# served lengths equalled the caps) -- a request that stops early is detected after the capture (served T_r = generated-1 != cap-1), its
# served-shape Program derived then and GP-01 recomposed, `eos-truncated` in stages.txt (row_pod.sh's MOE-05 rule); (2) no --launch-max-seqlen-q
# (FA2 target: inert; r17 passed none, so the r17 digests reproduce); (3) row_pod.sh's conventions: source-identity + target-family precheck,
# stages.txt / row.log / timeline.jsonl, research_result.py on EXIT, SWEEP_DIR per release tag.  Env: PY, GPU_UTIL (0.6), COMMIT_GPU_UTIL (=GPU_UTIL),
# PAIRS (3), REPLAY_WORKERS (32), BUILD_JOBS (max concurrent derives, default 2 x shapes), BUILD_TIMEOUT / MATCH_TIMEOUT / FOLD_TIMEOUT /
# COMMIT_TIMEOUT (s), FOLD_PROFILE (rank profile module; default from the case table below, empty = collective-level Match only, named so),
# HIDDEN_SO, EXTRA_COMMIT_ARGS (verbatim tp2_commit flags).
set -uo pipefail
ROW=${1:?row-id}; ROLE=${2:?role}; REPO=${3:?hf repo}; REV=${4:?revision}; STAGES=${5:-build,match,commit}
TREE_SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [ -n "${VERITOR_REPO:-}" ] && [ "$(cd "$VERITOR_REPO" 2>/dev/null && pwd -P)" != "$TREE_SELF" ]; then
  echo "PRECHECK_FAIL_SOURCE_IDENTITY: row_pod_tp2.sh at $TREE_SELF but \$VERITOR_REPO=$VERITOR_REPO names another tree" >&2; exit 3
fi
export VERITOR_REPO=$TREE_SELF
cd "$VERITOR_REPO" || exit 3
VERITY_CORE=$(cd "$TREE_SELF/../../packages/verity/src" 2>/dev/null && pwd -P) || { echo "PRECHECK_FAIL_SOURCE_IDENTITY: no packages/verity/src beside $TREE_SELF" >&2; exit 3; }
VERITY_RESEARCH=$(cd "$TREE_SELF/../../tools/research/src" 2>/dev/null && pwd -P) || { echo "PRECHECK_FAIL_SOURCE_IDENTITY: no tools/research/src beside $TREE_SELF" >&2; exit 3; }
export VERITY_CORE VERITY_RESEARCH
export PYTHONPATH=.:$VERITY_CORE:$VERITY_RESEARCH HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}     # the manifests' notes_tp: P2P hangs on the SYS topology
WORLD=${WORLD:-2}; [ "$WORLD" = 2 ] || { echo "row_pod_tp2.sh: WORLD=$WORLD unsupported (tp2_match / tp2_commit are world-2 tools)" >&2; exit 2; }
PY=${PY:-/workspace/venv312/bin/python}
[ -x "$PY" ] || { echo "no python at $PY (run pod_bootstrap.sh first)" >&2; exit 3; }
export PATH=$(dirname "$PY"):/usr/local/cuda/bin:$PATH
export NATIVE_COLLECT_BUILD=${NATIVE_COLLECT_BUILD:-/workspace/cp/nc_build}
HIDDEN_SO=${HIDDEN_SO:-/workspace/cp/fa2/build/matReq/verity_fa2_matReq.so}
GPU_UTIL=${GPU_UTIL:-0.6}; COMMIT_GPU_UTIL=${COMMIT_GPU_UTIL:-$GPU_UTIL}; PAIRS=${PAIRS:-3}; REPLAY_WORKERS=${REPLAY_WORKERS:-32}
D=${SWEEP_DIR:-/workspace/cp/sweep}/$ROW; B=$D/build; M=$D/match; mkdir -p "$D" "$B" "$M" "$NATIVE_COLLECT_BUILD"
WL=workloads/$ROW.json
[ -f "$WL" ] || { echo "no manifest $WL"; exit 2; }
log() { echo "$(date -u +%FT%TZ) [row $ROW] $*" | tee -a "$D/row.log"; }
# [R19] the research runner's result envelope, written once on exit from the stage verdicts (no-op without RESEARCH_RUN_DIR)
_research_result() { [ -n "${RESEARCH_RUN_DIR:-}" ] && "$PY" -m verity_capture.sweep.research_result "$D" "$ROW" "$ROLE" "$STAGES" >> "$D/row.log" 2>&1 || true; }
trap _research_result EXIT
verdict() { echo "$1 $2 $(date -u +%FT%TZ) $3" | tee -a "$D/stages.txt"; }
export VERITY_TIMELINE=${VERITY_TIMELINE:-$D/timeline.jsonl}
tl() { local st=$1 ev=$2; shift 2; local kv="" x k v
  for x in "$@"; do k=${x%%=*}; v=${x#*=}
    if [[ "$v" =~ ^-?[0-9]+(\.[0-9]+)?$ ]] || [ "$v" = true ] || [ "$v" = false ] || [ "$v" = null ]; then kv="$kv,\"$k\":$v"; else v=${v//\\/\\\\}; kv="$kv,\"$k\":\"${v//\"/\\\"}\""; fi; done
  printf '{"t":%s,"utc":"%s","schema":"verity-sweep/timeline/v1","proc":"row_pod_tp2","pid":%d,"stage":"%s","ev":"%s","hot":false%s}\n' \
    "$(date +%s.%N)" "$(date -u +%FT%TZ)" $$ "$st" "$ev" "$kv" >> "$VERITY_TIMELINE" 2>/dev/null || true; }
run_to() { local to=$1 lf=$2; shift 2; echo "\$ $*" > "$lf"; timeout "$to" "$@" >> "$lf" 2>&1; local rc=$?; [ $rc -eq 124 ] && echo "TIMEOUT after ${to}s" >> "$lf"; return $rc; }
run_tl() { local st=$1; shift; tl "$st" start; run_to "$@"; local rc=$?; tl "$st" end rc=$rc; return $rc; }
jq_() { "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {'d': d, 'json': json}))" "$1" "$2" 2>/dev/null; }
derived_ok() { [ -f "$1/result.json" ] && [ "$(jq_ "$1/result.json" "bool(d.get('ok'))")" = True ]; }

# ---------------------------------------------------------------- release line + PRECHECK (row_pod.sh's, verbatim in spirit)
TREE_HERE=$(realpath . 2>/dev/null || pwd); TAB=$'\t'
REL_OUT=$(python3 verity_capture/sweep/release_json.py line "$TREE_HERE" 2>/dev/null) || REL_OUT="UNRELEASED|commit=unknown tree=$TREE_HERE${TAB}-"
REL_LINE=${REL_OUT%%"$TAB"*}; verdict release "${REL_LINE%%|*}" "${REL_LINE#*|} tp=$WORLD" >/dev/null
tl precheck start
SID_SHA=$(python3 -c "import json;print(json.load(open('EXPORT.json')).get('commit') or '')" 2>/dev/null || true)
if ! "$PY" -m verity_capture.sweep.source_identity --tree "$TREE_HERE" ${SID_SHA:+--sha "$SID_SHA"} --stage precheck --out "$D/source_identity.json" >> "$D/row.log" 2>&1; then
  tl precheck end rc=3 fail_class=PRECHECK_FAIL_SOURCE_IDENTITY
  verdict precheck FAIL "PRECHECK_FAIL_SOURCE_IDENTITY (see $D/source_identity.json and row.log)"; exit 3
fi
tl precheck end rc=0
verdict precheck PASS "source identity == $TREE_HERE${SID_SHA:+ @ $SID_SHA} (verity verity_vllm verity_capture verity_vllm_adapter hidden_gpu research)" >/dev/null
TF_LINE=$(python3 -m verity_capture.sweep.target_family --row "$ROW" --workload "$WL" --stages "$STAGES" --out "$D/target_family.json" 2>&1); TF_RC=$?
echo "$TF_LINE" >> "$D/row.log"
[ $TF_RC -eq 0 ] || { verdict precheck-target FAIL "$TF_LINE"; exit 3; }
verdict precheck-target PASS "$TF_LINE" >/dev/null
# the row's shape from the manifest: B, I, O, the declared request shapes (LP, cap-1) and the pinned checkpoint dir
IFS=' ' read -r BATCH I O CK <<< "$($PY - "$WL" <<'EOF'
import json, os, sys
w = json.load(open(sys.argv[1])); s = w["sweep"]
ck = json.load(open("manifests/checkpoints.json")); local = None
for c in ck["checkpoints"]:
    if c.get("repo") == w["model"] and str(c.get("revision", "")).startswith(w["tokenizer_revision"][:10]):
        local = c.get("local_path"); break
if not local or not os.path.exists(local):
    base = f"/workspace/hf/hub/models--{w['model'].replace('/', '--')}/snapshots"
    for d in sorted(os.listdir(base)) if os.path.exists(base) else []:
        if d.startswith(w["tokenizer_revision"][:10]): local = f"{base}/{d}"
print(s["batch"], s["max_input"], s["max_output"], local or "-")
EOF
)"
SHAPES=$($PY -c "import json,sys;s=json.load(open(sys.argv[1]))['sweep'];print(' '.join(f'{lp},{c-1}' for lp,c in sorted({(int(a),int(b)) for a,b in zip(s['input_lengths'],s['output_caps'])})))" "$WL")
MML=$($PY -c "import json,sys;w=json.load(open(sys.argv[1]));print((w.get('engine_args_required') or {}).get('max_model_len') or int(sys.argv[2])+int(sys.argv[3]))" "$WL" "$I" "$O")
MNBT=$($PY -c "import json,sys;w=json.load(open(sys.argv[1]));print((w.get('engine_args_required') or {}).get('max_num_batched_tokens') or int(sys.argv[2]))" "$WL" "$MML")
TGT=$($PY -c 'import json,sys; t=json.load(open(sys.argv[1])).get("target"); print(json.dumps(t, sort_keys=True) if t else "")' "$WL"); TGT_ARGS=(); [ -n "$TGT" ] && TGT_ARGS=(--target "$TGT")
# the TP2 RANK profile of the fold Match, per case (a lane adds its module + this line; empty = collective-level Match only, named in the verdict)
case "${FOLD_PROFILE-unset}" in unset) case "$ROLE" in QWEN3_30B_A3B) FOLD_PROFILE=gen_ov_moe_qwen3 ;; *) FOLD_PROFILE="" ;; esac ;; esac
ENGINE=(--engine-arg "max_model_len=$MML" --engine-arg "max_num_batched_tokens=$MNBT" --engine-arg disable_custom_all_reduce=true)
CODE9=$(python3 -c "import json;print(json.load(open('EXPORT.json'))['commit'][:9])" 2>/dev/null)
log "start tp=$WORLD B=$BATCH I=$I O=$O MML=$MML mnbt=$MNBT shapes=[$SHAPES] ckpt=$CK stages=$STAGES fold_profile=${FOLD_PROFILE:-none} gpu_util=$GPU_UTIL hidden_so=$HIDDEN_SO code=$CODE9"
tl row start stages="$STAGES" B="$BATCH" tp="$WORLD" code="$CODE9" pod="$(hostname 2>/dev/null)"
[ -d "$CK" ] || { verdict build FAIL "no local checkpoint dir for $REPO @ $REV (checkpoints.json local_path / HF snapshot absent; HF_HUB_OFFLINE=1)"; exit 10; }

derive() {  # <rank> <out-dir> <wrapper args...>   (one CPU derive; the .rc beside the log carries the exit code for the parallel schedule)
  local r=$1 out=$2; shift 2
  run_to "$TO_B" "$out.log" "$PY" -m verity_capture.experimental.cb_a.derive_step --model "$REPO" --revision "$REV" --max-model-len "$MML" --workload "$WL" \
         --tp "$WORLD" --rank "$r" --out "$out" "${TGT_ARGS[@]}" "$@"; echo $? > "$out.rc"; }
launch() { while [ "$(jobs -rp | wc -l)" -ge "$BUILD_JOBS" ]; do wait -n 2>/dev/null || sleep 1; done; derive "$@" & }
compose() {  # <rank>: GP-01 workload Program of the rank over its request Programs
  local r=$1 DER=() dd
  for dd in "$B/rank$r"/build_request_LP*_T*; do [ -f "$dd/result.json" ] && DER+=(--derived "$dd"); done
  run_tl "build.global_program_r$r" "$TO_B" "$B/rank$r/build_workload.log" "$PY" -m verity_capture.experimental.cb_a.global_program --workload "$WL" "${DER[@]}" --out "$B/rank$r/build_workload" --checkpoint-dir "$CK"; }
wd_of() { jq_ "$B/rank$1/build_workload/workload_program.json" "d.get('workload_digest') or ''"; }
build_summary() {  # -> PASS|FAIL; writes $D/build_summary.json (per rank: step + request Programs + GP-01)
  "$PY" - "$B" "$WORLD" <<'EOF'
import glob, json, os, sys
b, world = sys.argv[1], int(sys.argv[2]); out = {"tp": {"world": world}, "ranks": {}}; ok = True
for r in range(world):
    rd = f"{b}/rank{r}"; rk = {}
    for p in [f"{rd}/build_step"] + sorted(glob.glob(f"{rd}/build_request_LP*_T*")):
        res = json.load(open(f"{p}/result.json")) if os.path.exists(f"{p}/result.json") else {"ok": False, "message": "no result.json"}
        rk[os.path.basename(p)] = {"ok": bool(res.get("ok")), "unsupported": res.get("unsupported"), "message": (res.get("message") or "")[:300],
                                   "program_digest": (res.get("program") or {}).get("digest")}
        ok &= bool(res.get("ok"))
    wp = f"{rd}/build_workload/workload_program.json"; W = json.load(open(wp)) if os.path.exists(wp) else {}
    rk["workload"] = {"ok": bool(W.get("workload_digest")), "workload_digest": W.get("workload_digest"), "reason": W.get("reason"), "message": W.get("message"), "mechanism": "GP-01 cb_a.global_program"}
    ok &= bool(W.get("workload_digest"))
    out["ranks"][f"rank{r}"] = rk
out["step"] = {"program_digest": (out["ranks"]["rank0"].get("build_step") or {}).get("program_digest")}    # research_result reads step.program_digest
json.dump(out, open(f"{os.path.dirname(b)}/build_summary.json", "w"), indent=1, default=str); print("PASS" if ok else "FAIL")
EOF
}

# ---------------------------------------------------------------- Build (CPU): per rank step + declared request shapes, then GP-01
if [[ ",$STAGES," == *,build,* ]]; then
  NSH=$(echo "$SHAPES" | wc -w); BUILD_JOBS=${BUILD_JOBS:-$(( WORLD * (NSH + 1) ))}
  TO_B=${BUILD_TIMEOUT:-7200}     # r17: OLMoE B8 request derives 50-840 s each 16-wide; Qwen3-30B LP1024_T127 ~75 min
  t0=$(date +%s); tl stage.build start jobs="$BUILD_JOBS"
  for r in $(seq 0 $((WORLD - 1))); do
    mkdir -p "$B/rank$r"
    derived_ok "$B/rank$r/build_step" || launch "$r" "$B/rank$r/build_step" --wrapper step --M "$BATCH" --pos0 8
    for sh in $SHAPES; do IFS=, read -r lp t <<< "$sh"; out=$B/rank$r/build_request_LP${lp}_T${t}
      derived_ok "$out" || { rm -rf "$out"; launch "$r" "$out" --wrapper request --LP "$lp" --T "$t"; }
    done
  done
  wait
  for r in $(seq 0 $((WORLD - 1))); do for f in "$B/rank$r"/*.rc; do [ -f "$f" ] && log "derive $(basename "${f%.rc}") rank$r rc=$(cat "$f") $(tail -c 200 "${f%.rc}.log" | tr '\n' ' ')"; done; done
  for r in $(seq 0 $((WORLD - 1))); do derived_ok "$B/rank$r/build_step" && compose "$r" & done; wait
  ok=$(build_summary); wall=$(( $(date +%s) - t0 ))
  verdict build "$ok" "rc=$( [ "$ok" = PASS ] && echo 0 || echo 1) wall=${wall}s digest=$(jq_ "$D/build_summary.json" "(d['step']['program_digest'] or '')[:16]") digest_r1=$(jq_ "$D/build_summary.json" "((d['ranks']['rank1'].get('build_step') or {}).get('program_digest') or '')[:16]") workload_digest_r0=$(wd_of 0 | cut -c1-16) workload_digest_r1=$(wd_of 1 | cut -c1-16) shapes=$NSH"
  tl stage.build end ok="$ok" wall_s="$wall"
  [ "$ok" = PASS ] || { log "BUILD FAIL -> stop ($(grep -hE 'Error|refused|Unsupported' "$B"/rank*/*.log 2>/dev/null | sort | uniq -c | sort -rn | head -3 | cut -c1-300 | tr '\n' ' '))"; tl row end ok=false stopped_at=build; exit 10; }
fi

# ---------------------------------------------------------------- Match: capture (observer + recorder) + control -> tp2_match check -> fold Match
if [[ ",$STAGES," == *,match,* ]]; then
  TO_M=${MATCH_TIMEOUT:-3600}; TO_F=${FOLD_TIMEOUT:-7200}; TO_B=${BUILD_TIMEOUT:-7200}
  for r in 0 1; do derived_ok "$B/rank$r/build_step" || { verdict match FAIL "no step Program for rank $r ($B/rank$r/build_step): Build first"; exit 11; }; done
  CAP=("$PY" -m verity_capture.bench.tp2_capture --workload "$WL" --case "$ROLE" --tp "$WORLD" --max-num-seqs "$BATCH" --engine-arg "gpu_memory_utilization=$GPU_UTIL" "${ENGINE[@]}")
  t0=$(date +%s); tl stage.match start
  run_tl match.capture "$TO_M" "$M/capture.log" "${CAP[@]}" --record-collectives --out "$M/capture"; rc_cap=$?
  log "match capture rc=$rc_cap $(grep -E '^\[tp2\] rank [01]:' "$M/capture.log" | tr '\n' ' ' | cut -c1-300)"
  run_tl match.control "$TO_M" "$M/control.log" "${CAP[@]}" --no-observer --out "$M/control"; rc_ctl=$?
  run_tl match.check "$TO_M" "$M/check.log" "$PY" -m verity_capture.bench.tp2_match --arm check --out "$M" --build-rank0 "$B/rank0/build_step" --build-rank1 "$B/rank1/build_step"; rc_chk=$?
  CHK=$(jq_ "$M/tp2_match.json" "' '.join(str(x) for x in ['pass',d['pass'],'structure',d['structure']['pass'],'n_coll_r0',d['structure']['per_rank']['0']['n_collectives'],'sig_len',d['structure']['per_rank']['0']['build_signature_len'],'steps',d['structure']['per_rank']['0']['n_steps'],'provenance',d['provenance']['pass'],d['provenance']['collectives_checked'],'mismatches',d['provenance']['n_mismatches'],'tokens',d['tokens'].get('pass')])")
  log "match check rc=$rc_chk: ${CHK:-no tp2_match.json}"
  # served shapes: a request that stopped before its cap (T_r = generated-1 != cap-1) is matched against its served-shape Program (MOE-05 / X-09)
  NEED=$(jq_ "$M/capture/tokens.json" "' '.join(f\"{len(q['prompt_token_ids'])},{len(q['generated_token_ids'])-1}\" for q in d['requests'])")
  MISSING=""; for sh in $NEED; do IFS=, read -r lp t <<< "$sh"; for r in 0 1; do derived_ok "$B/rank$r/build_request_LP${lp}_T${t}" || MISSING="$MISSING $sh"; done; done
  if [ -n "$MISSING" ]; then
    verdict eos-truncated "served=[$NEED]" "declared=[$SHAPES]: request(s) stopped before the cap; deriving the served-shape Program(s) [$MISSING] per rank and recomposing GP-01 (MOE-05)" >/dev/null
    BUILD_JOBS=${BUILD_JOBS:-16}
    for sh in $(echo "$MISSING" | tr ' ' '\n' | sort -u); do IFS=, read -r lp t <<< "$sh"; for r in 0 1; do out=$B/rank$r/build_request_LP${lp}_T${t}; derived_ok "$out" || { rm -rf "$out"; launch "$r" "$out" --wrapper request --LP "$lp" --T "$t"; }; done; done; wait
    for r in 0 1; do compose "$r" & done; wait
    build_summary >/dev/null; log "served-shape derivation done: workload digests r0=$(wd_of 0 | cut -c1-16) r1=$(wd_of 1 | cut -c1-16)"
  fi
  # the D90 per-rank fold Match (only with a TP2 rank profile; its absence is NAMED in the verdict, never a silent skip)
  FOLD="collective-level only (no TP2 rank profile for $ROLE: fold Match not run)"; rc_fold=-1
  if [ -n "$FOLD_PROFILE" ]; then
    FARGS=(); for r in 0 1; do first=1
      for dd in $(ls -d "$B/rank$r"/build_request_LP*_T* 2>/dev/null | sort); do [ -f "$dd/result.json" ] || continue
        if [ $first = 1 ]; then FARGS+=("--build-rank$r" "$dd"); first=0; else FARGS+=("--derived-rank$r" "$dd"); fi; done
      [ -f "$B/rank$r/build_workload/workload_program.json" ] && FARGS+=("--gprog-rank$r" "$B/rank$r/build_workload/workload_program.json"); done
    run_tl match.fold_match "$TO_F" "$M/fold_match.log" "$PY" -m verity_capture.bench.tp2_fold_match --capture "$M/capture" --profile "$FOLD_PROFILE" "${FARGS[@]}" \
           --workload "$WL" --tp2-match "$M/tp2_match.json" --within-step dag --rank-workers 0 --python "$PY" --timeout "$TO_F" --out "$M/fold_match"; rc_fold=$?
    FOLD="fold_match rc=$rc_fold $(jq_ "$M/fold_match/tp2_fold_match.json" "' '.join(['pass',str(d['pass']),'legs',json.dumps(d['legs']),'problems',json.dumps(d['problems'][:3])[:400]])")"
    log "match fold ($FOLD_PROFILE): $FOLD"
  fi
  wall=$(( $(date +%s) - t0 ))
  ok=FAIL
  if [ $rc_cap -eq 0 ] && [ $rc_ctl -eq 0 ] && [ $rc_chk -eq 0 ] && [ "$(jq_ "$M/tp2_match.json" "d['pass']")" = True ]; then
    if [ -z "$FOLD_PROFILE" ]; then ok=PASS
    elif [ $rc_fold -eq 0 ] && [ "$(jq_ "$M/fold_match/tp2_fold_match.json" "d['pass']")" = True ]; then ok=PASS; fi
  fi
  "$PY" - "$M" "$rc_cap" "$rc_ctl" "$rc_chk" "$rc_fold" "$ok" "$FOLD_PROFILE" <<'EOF'
import json, os, sys
m, rc_cap, rc_ctl, rc_chk, rc_fold, ok, prof = sys.argv[1], *map(int, sys.argv[2:6]), sys.argv[6], sys.argv[7]
tm = json.load(open(f"{m}/tp2_match.json")) if os.path.exists(f"{m}/tp2_match.json") else {}
fm = json.load(open(f"{m}/fold_match/tp2_fold_match.json")) if os.path.exists(f"{m}/fold_match/tp2_fold_match.json") else None
json.dump({"pass": ok == "PASS", "level": "fold (D90 per-rank fold Match + collective-level)" if prof else "collective-level (tp2_match check; no TP2 rank profile)",
           "capture_rc": rc_cap, "control_rc": rc_ctl, "check_rc": rc_chk, "fold_rc": None if rc_fold < 0 else rc_fold, "fold_profile": prof or None,
           "tp2_match": {k: tm.get(k) for k in ("pass", "structure", "provenance", "tokens") if k in tm}, "tp2_fold_match": None if fm is None else {k: fm.get(k) for k in ("pass", "legs", "problems", "seconds")}},
          open(f"{os.path.dirname(m)}/match_summary.json", "w"), indent=1, default=str)
EOF
  verdict match "$ok" "rc=$rc_cap/$rc_ctl/$rc_chk/$rc_fold wall=${wall}s tp2_match[${CHK:-none}] $FOLD"
  tl stage.match end ok="$ok" wall_s="$wall"
  [ "$ok" = PASS ] || { log "MATCH FAIL -> stop"; tl row end ok=false stopped_at=match; exit 11; }
fi

# ---------------------------------------------------------------- Commit: global required manifest over both ranks, then tp2_commit
if [[ ",$STAGES," == *,commit,* ]]; then
  TO_K=${COMMIT_TIMEOUT:-7200}
  t0=$(date +%s); tl stage.commit start
  [ -f "$HIDDEN_SO" ] || { verdict commit FAIL "hidden .so missing: $HIDDEN_SO"; exit 12; }
  wd0=$(wd_of 0); wd1=$(wd_of 1)
  [ -n "$wd0" ] && [ -n "$wd1" ] || { verdict commit FAIL "GP-01 workload digests absent (r0='${wd0:0:16}' r1='${wd1:0:16}'): Build first"; exit 12; }
  run_tl commit.required_manifest "$TO_K" "$D/manifest.log" "$PY" -m verity_capture.commit.required_manifest build-global --fixture "$WL" \
         --programs-root "$B/rank0" --rank 0 --programs-root "$B/rank1" --rank 1 --ranks "$WORLD" --workload-digest "$wd0" --workload-digest "$wd1" --out "$D/manifest.json"; rc_man=$?
  MAN=$(jq_ "$D/manifest.json" "' '.join(str(x) for x in ['complete',d.get('complete'),'identities',(d.get('populations') or {}).get('identities'),'tp_peer_binding_n_unbound',(d.get('tp_peer_binding') or {}).get('n_unbound'),'unmodelled',json.dumps(d.get('unmodelled'))[:300],'digest',(d.get('manifest_digest') or '')[:16]])")
  log "manifest rc=$rc_man: ${MAN:-no manifest.json}"
  if [ $rc_man -ne 0 ] || [ "$(jq_ "$D/manifest.json" "d.get('complete')")" != True ]; then
    verdict commit FAIL "required manifest not complete (rc=$rc_man): ${MAN:-no manifest.json} (manifest.log)"; tl stage.commit end ok=false; tl row end ok=false stopped_at=commit; exit 13
  fi
  CMD_K=("$PY" -m verity_capture.bench.tp2_commit --workload "$WL" --case "$ROLE" --tp "$WORLD" --committer native_collect_v2b --hidden-so "$HIDDEN_SO" --openings 64
         --max-num-seqs "$BATCH" --engine-arg "gpu_memory_utilization=$COMMIT_GPU_UTIL" "${ENGINE[@]}" --build-rank0 "$B/rank0" --build-rank1 "$B/rank1"
         --required-manifest "$D/manifest.json" --match-dir "$M" --checkpoint-dir "$CK" --checkpoint-model "$REPO"
         --pairs "$PAIRS" --warmup 2 --value-check --replay-workers "$REPLAY_WORKERS" --label "$ROW" --out "$D/commit")
  [ -n "$FOLD_PROFILE" ] && [ -f "$M/fold_match/tp2_fold_match.json" ] && CMD_K+=(--require-fold-match)     # D90 R1: a fold Match on disk must bind
  for ca in ${EXTRA_COMMIT_ARGS:-}; do CMD_K+=("$ca"); done
  run_tl commit.tp2_commit "$TO_K" "$D/commit.log" "${CMD_K[@]}"; rc_k=$?
  grep -E '^\[tp2commit\] +(PASS|FAIL|sampled replay|match oracle|weights pin|cross-rank collectives|fold-Match|C2 population|coverage)' "$D/commit.log" | tail -16 | cut -c1-400 >> "$D/row.log"
  wall=$(( $(date +%s) - t0 ))
  SUMM=$(jq_ "$D/commit/summary.json" "' '.join(str(x) for x in ['pass',d.get('pass'),'pairs',d.get('pairs'),'delta',d.get('delta_wall_mean'),'roots',json.dumps(d.get('tp_run_roots'))[:80],'tokens',d.get('tokens_equal'),'replay',(d.get('sampled_replay') or {}).get('result'),'linkage',(d.get('boundary_linkage') or {}).get('result'),'xrank',(d.get('cross_rank_collectives') or {}).get('result'),'fold_binding',(d.get('fold_match_binding') or {}).get('result'),'weights_pin',(d.get('weights_pin') or {}).get('result')])")
  ok=FAIL; [ $rc_k -eq 0 ] && [ "$(jq_ "$D/commit/summary.json" "d.get('pass')")" = True ] && ok=PASS
  verdict commit "$ok" "rc=$rc_k wall=${wall}s ${SUMM:-no commit/summary.json} $(grep -E '^\[tp2commit\] (PASS|FAIL)' "$D/commit.log" | tail -1 | cut -c1-200)"
  tl stage.commit end ok="$ok" wall_s="$wall"
  [ "$ok" = PASS ] || { log "COMMIT FAIL -> stop"; tl row end ok=false stopped_at=commit; exit 12; }
fi
tl row end ok=true
python3 -m verity_capture.sweep.timeline report "$VERITY_TIMELINE" --json "$D/timeline_report.json" --md "$D/timeline_report.md" --title "$ROW" >/dev/null 2>&1 || true
log "DONE"
