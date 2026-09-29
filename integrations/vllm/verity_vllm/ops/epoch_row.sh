#!/bin/bash
# epoch_row.sh: one re-baseline row inside its `research run` job on a pod, with every stop the row's outcome depends on in the job.
#
#   research run --on <pod> --project verity --campaign <campaign> --custody-r2 --custody-ttl <T> --timeout <S> \
#     --source <clean worktree at EPOCH_SHA> --cwd source/integrations/vllm \
#     --env ROW=<row id> --env ROWNUM=<n> --env CLASS=<GREEN|FAIL> --env EPOCH_SHA=<sha> [--env K=V ...] \
#     -- bash verity_vllm/ops/epoch_row.sh
#
# One line per step in evidence/progress.txt; "STOP <why>" ends the row, and every stop, deadline or signal ends the same way (`finish`):
# the row's small files, the store (epoch_store.sh: Build + records as fixture/v1 trees, `research data put --preserve`, then
# `research data preserved` on what it stored), the digests, "END".  Nothing the outcome depends on runs off the pod.
#   0 guard      the shipped tree is EPOCH_SHA; no epoch default is switched off; the custody key outlives the run's timeout by >= 1 h
#   1 bootstrap  B0 + the row's checkpoint role (epoch_failfast.sh: pod_bootstrap.sh under the 15-minute fail-fast)
#   2 Build, Match (`verity-vllm row run --stages build`, `--stages match`); a GREEN row whose Match FAILs stops (deferred)
#   3 word check epoch_word_check.py, from the Build's own manifest step
#   4 Commit     `verity-vllm row run --stages commit` (manifest-verify and the verdict record inside)
#   5 store      as in `finish`; then `rebaseline run` (candidate = this sweep; the frozen reference from the store)
#
# Deadlines (epoch seconds are compared, UTC stamps accepted): the job ends its last stage STORE_RESERVE_S (default 2400) before the run's
# timeout (status.json's start + timeout_s, or control/timeout_s after `tele set-timeout`), or at JOB_DEADLINE_UTC if that is earlier;
# DEADLINE_BUILD / DEADLINE_MATCH / DEADLINE_COMMIT end one stage earlier still.  A stage still running then is stopped (TERM, then KILL
# after 120 s) and the row stops "STOP <stage>: deadline" (exit 124).  STOP_AFTER=build|match stops the row after that stage (exit 23).
# Env: PY (/workspace/venv312/bin/python), SWEEP_DIR (/workspace/epoch/sweep), RESEARCH_STORE (/workspace/epoch/store), RESEARCH_CLI
# (`python3 -m research`), EPOCH_BOOTSTRAP (`bash verity_vllm/ops/epoch_failfast.sh`), ALLOWED_QWORD_MAX_GATES (a GO's one allowed
# VERITY_QWORD_MAX_GATES value for the row).  The sweep is outside the run dir: custody uploads every run-dir file, so evidence/ holds only
# small files.
set -u
export EV=$RESEARCH_RUN_DIR/evidence; mkdir -p "$EV"
P=$EV/progress.txt
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$P"; }
: "${ROW:?}" "${ROWNUM:?}" "${CLASS:?}" "${EPOCH_SHA:?}" "${RESEARCH_RUN_ID:?}"
OPS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
T=$(cd "$OPS/../../../.." && pwd -P)
export PY=${PY:-/workspace/venv312/bin/python} HF_HOME=${HF_HOME:-/workspace/hf}
export PYTHONPATH=$T/integrations/vllm:$T/packages/verity/src:$T/tools/research/src:$T/protocols/sampled_proofs
export SWEEP_DIR=${SWEEP_DIR:-/workspace/epoch/sweep} RESEARCH_STORE=${RESEARCH_STORE:-/workspace/epoch/store} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export CUSTODY_DIR=${CUSTODY_DIR:-$(dirname "$(dirname "$RESEARCH_RUN_DIR")")/requests/$RESEARCH_RUN_ID/custody}
D=$SWEEP_DIR/$ROW
mkdir -p "$SWEEP_DIR"
cd "$OPS/../.." || exit 2

epoch_of() { [ -n "${1:-}" ] && date -u -d "$1" +%s 2>/dev/null; }
RUN_END=$(python3 - "$RESEARCH_RUN_DIR" <<'EOF'
import calendar, json, sys, time
from pathlib import Path
d = Path(sys.argv[1])
try:
    run = [t for t in json.loads((d / "status.json").read_text())["transitions"] if t.get("state") == "running"][-1]
    start = calendar.timegm(time.strptime(run["started_utc"][:19], "%Y-%m-%dT%H:%M:%S"))
    timeout = run.get("timeout_s")
    try:
        timeout = json.loads((d / "control" / "timeout_s").read_text()).get("timeout_s", timeout)
    except (OSError, ValueError):
        pass
    print(int(start + float(timeout)) if timeout else "")
except (OSError, ValueError, KeyError, IndexError):
    print("")
EOF
)
JOB_END=$(epoch_of "${JOB_DEADLINE_UTC:-}")
if [ -n "$RUN_END" ]; then
  cap=$(( RUN_END - ${STORE_RESERVE_S:-2400} ))
  [ -z "$JOB_END" ] || [ "$JOB_END" -gt "$cap" ] && JOB_END=$cap
fi
stage_end() {  # stage_end STAGE: the earlier of the job's end and DEADLINE_<STAGE>, epoch seconds ("" = none)
  local s; s=$(epoch_of "$(eval echo "\${DEADLINE_${1^^}:-}")")
  if [ -n "$s" ] && { [ -z "$JOB_END" ] || [ "$s" -lt "$JOB_END" ]; }; then echo "$s"; else echo "$JOB_END"; fi
}

CHILD=
run_stage() {  # run_stage STAGE LOG CMD...: CMD in the background under the stage's deadline; rc 124 when the deadline stopped it
  local stage=$1 log=$2 end left; shift 2
  end=$(stage_end "$stage")
  if [ -n "$end" ]; then
    left=$(( end - $(date +%s) ))
    [ "$left" -gt 0 ] || return 124
    timeout --kill-after=120 "$left" "$@" > "$log" 2>&1 &
  else
    "$@" > "$log" 2>&1 &
  fi
  CHILD=$!; wait "$CHILD"; local rc=$?; CHILD=
  return $rc
}

small() {   # the row's small files into evidence/ (idempotent; called after each step)
  mkdir -p "$EV/row"
  for f in stages.txt row.log timeline.jsonl build_summary.json verdict.json target_family.json admission.json manifest.log \
           match_summary.json global_match.json commit/manifest_verify.json commit/verdict.json commit/summary.json commit/runs.jsonl; do
    [ -f "$D/$f" ] && [ "$(stat -c %s "$D/$f")" -lt 50000000 ] && { mkdir -p "$EV/row/$(dirname "$f")"; cp "$D/$f" "$EV/row/$f"; }
  done
  [ -f "$D/manifest.json" ] && python3 - "$D/manifest.json" "$EV/row/manifest_header.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
json.dump({k: m.get(k) for k in ("schema", "manifest_digest", "program_digest", "complete", "populations", "query")}, open(sys.argv[2], "w"), indent=1)
EOF
}
src=0
store() {  # once per run, whenever a Build exists: deferred rows keep their Build too
  [ -f "$D/build_summary.json" ] || [ -d "$D/build" ] || return 0
  [ -f "$EV/stored" ] && return 0
  bash "$OPS/epoch_store.sh" "$D" > "$EV/store.log" 2>&1
  src=$?; touch "$EV/stored"
  say "store rc=$src $(grep -E '^(STORED|PRESERVED-CHECK)' "$EV/store.log" | cut -d' ' -f2-4 | tr '\n' ' ')"
  "$PY" "$OPS/epoch_digests.py" "$D" --out "$EV/digests.json" --markdown "$EV/digests.md" > "$EV/digests.log" 2>&1
  say "digests rc=$? $(tail -n 1 "$EV/digests.md" 2>/dev/null | cut -c1-240)"
}
finish() { small; store; say "END"; }
stop() { say "STOP $2"; finish; exit "$1"; }   # stop RC WHY
on_signal() {  # a TERM/INT/HUP the job receives (the run's own timeout): stop the stage, then the same end as every stop
  trap '' TERM INT HUP
  [ -n "$CHILD" ] && { kill -TERM "$CHILD" 2>/dev/null; wait "$CHILD" 2>/dev/null; }
  stop $(( 128 + $2 )) "signal $1: the job was told to end"
}
trap 'on_signal TERM 15' TERM; trap 'on_signal INT 2' INT; trap 'on_signal HUP 1' HUP

# ---- 0 guard ----------------------------------------------------------------------------------------------------------------------------
[ "${RESEARCH_SOURCE_SHA:-}" = "$EPOCH_SHA" ] || stop 3 "guard: shipped tree ${RESEARCH_SOURCE_SHA:-?} is not the epoch sha $EPOCH_SHA"
for v in NORM_TAP GUARDED_MAX_TAP ROUTER_TAP VOCAB_TAP ROW_TAPS; do
  [ "${!v:-1}" = "1" ] || stop 3 "guard: $v=${!v} switches an epoch default off"
done
case "${VERITY_WORD_CHECK:-16/32}" in 16/32) ;; *) stop 3 "guard: VERITY_WORD_CHECK=$VERITY_WORD_CHECK is not the record's 16/32";; esac
[ -z "${VERITY_QUERY_ID:-}" ] || stop 3 "guard: VERITY_QUERY_ID is set ($VERITY_QUERY_ID); the row runs the query of record"
[ -z "${TARGET_FAMILY_WAIVER:-}" ] || stop 3 "guard: a row of record carries no target-family waiver"
expires=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('expires_utc') or '')" "$CUSTODY_DIR/cred.json" 2>/dev/null)
[ -n "$expires" ] || stop 3 "guard: no custody key at $CUSTODY_DIR (the store could not run)"
[ -z "$RUN_END" ] || [ "$(epoch_of "$expires")" -ge $(( RUN_END + 3600 )) ] \
  || stop 3 "guard: the custody key expires $expires, less than 1 h after the run's timeout ($(date -u -d "@$RUN_END" +%FT%TZ)): raise --custody-ttl"
env | grep -E '^(PAIRS|VU_EXPORT|GPU_UTIL|COMMIT_GPU_UTIL|BUILD_TIMEOUT|VERITY_QWORD_MAX_GATES|NCCL_P2P_DISABLE|ROW|ROWNUM|CLASS|EPOCH_SHA|STOP_AFTER|JOB_DEADLINE_UTC|DEADLINE_[A-Z]+)=' \
  | sort > "$EV/row_env.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$EV/gpus.txt" 2>&1
export GPU_NAME=$(head -n 1 "$EV/gpus.txt" | cut -d, -f1)
{ grep MemTotal /proc/meminfo; cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null; } > "$EV/host_mem.txt"
say "guard PASS tree=$EPOCH_SHA row=#$ROWNUM class=$CLASS job_end=${JOB_END:+$(date -u -d "@$JOB_END" +%FT%TZ)} key_expires=$expires $(tr '\n' ' ' < "$EV/row_env.txt")"

read -r ROLE REPO REV WORLD < <(python3 - "$ROW" <<'EOF'
import json, sys
row = sys.argv[1]
wl = json.load(open(f"workloads/{row}.json"))
role = wl["case"]
alias = {"B0": "HuggingFaceTB/SmolLM2-135M", "B1": "Qwen/Qwen2.5-1.5B"}
cps = json.load(open("manifests/checkpoints.json"))["checkpoints"]
cp = next(c for c in cps if c.get("role") == role or (role in alias and c["repo"] == alias[role] and c.get("local_path")))
tp = int((wl.get("sweep") or {}).get("tp") or wl.get("tp") or (2 if "__tp2__" in row else 1))
print(role, cp["repo"], cp["revision"], tp)
EOF
)
[ -n "${REV:-}" ] || stop 2 "resolve: no checkpoint for $ROW"
say "resolved role=$ROLE repo=$REPO rev=${REV:0:12} world=$WORLD"

# ---- 1 bootstrap ------------------------------------------------------------------------------------------------------------------------
CASES=B0; [ "$ROLE" = B0 ] || CASES=B0,$ROLE
run_stage bootstrap "$EV/bootstrap.run.log" ${EPOCH_BOOTSTRAP:-bash $OPS/epoch_failfast.sh} "$CASES" "$EV/bootstrap"
rc=$?
[ "$rc" = 124 ] && stop 124 "bootstrap: the job's deadline reached"
[ "$rc" = 40 ] && { finish; exit 40; }
say "bootstrap cases=$CASES rc=$rc"
[ "$rc" = 0 ] || { tail -n 30 "$EV/bootstrap.log" > "$EV/bootstrap.tail" 2>/dev/null; stop 4 "bootstrap rc=$rc (bootstrap.tail)"; }
export HF_HUB_OFFLINE=1
stage() {  # stage NAME: `verity-vllm row run --stages NAME` under its deadline; the verdict line, or the deadline's STOP
  run_stage "$1" "$EV/$1.log" "$PY" -m verity_vllm.pipeline.cli row run "$ROW" "$ROLE" "$REPO" "$REV" --stages "$1"
  rc=$?; small
  local end; end=$(stage_end "$1")
  if { [ "$rc" = 124 ] || [ "$rc" = 137 ]; } && [ -n "$end" ] && [ "$(date +%s)" -ge $(( end - 5 )) ]; then
    stop 124 "$1: deadline $(date -u -d "@$end" +%FT%TZ) reached, the stage was stopped"
  fi
  say "$1 rc=$rc $(grep "^$1 " "$D/stages.txt" 2>/dev/null | tail -n 1 | cut -c1-240)"
}
after() { [ "${STOP_AFTER:-}" = "$1" ] && stop 23 "after $1 (STOP_AFTER=$1)"; return 0; }

# ---- 2 Build, Match ---------------------------------------------------------------------------------------------------------------------
stage build
[ "$rc" = 0 ] || stop "$rc" "build rc=$rc"
fams=$(python3 -c "import json;print(','.join((json.load(open('$D/manifest.json')).get('query') or {}).get('required_families') or []))" 2>/dev/null)
echo "$fams" > "$EV/required_families.txt"
[ -n "$fams" ] || stop 21 "the manifest states no query.required_families"
say "required_families: $fams"
after build
stage match
case "$rc" in
  0) ;;
  11) [ "$CLASS" = FAIL ] || stop 11 "GREEN row, Match FAIL: deferred, not committed"
      say "Match FAIL on a FAIL-class row: the Commit runs (the class's record carries one)";;
  *) stop "$rc" "match rc=$rc";;
esac
after match

# ---- 3 word check -----------------------------------------------------------------------------------------------------------------------
"$PY" "$OPS/epoch_word_check.py" "$ROW" "$D" --world "$WORLD" --out "$EV/strict_word.json" > "$EV/strict_word.log" 2>&1
rc=$?; say "strict word check rc=$rc $(tail -n 1 "$EV/strict_word.log" | cut -c1-240)"
[ "$rc" = 0 ] || stop 20 "strict word check FAILED: not committed (strict_word.json)"

# ---- 4 Commit ---------------------------------------------------------------------------------------------------------------------------
stage commit
out=$("$PY" -c "import json;print(json.load(open('$D/verdict.json')).get('outcome'))" 2>/dev/null)
mv=$("$PY" -c "import json;print(json.load(open('$D/commit/manifest_verify.json')).get('ok'))" 2>/dev/null)
say "commit verdict=${out:-none} manifest_verify=${mv:-none}"

# ---- 5 store, then the regression record ------------------------------------------------------------------------------------------------
small; store
VERITY_REGRESSION_CANDIDATE=$SWEEP_DIR run_stage record "$EV/rebaseline_run.log" \
  "$PY" -m tests.regression.rebaseline run --record "$EV/record" --tier T0,T1,T2 -- -k "r$ROWNUM" -ra
say "rebaseline run rc=$? $(grep -E 'passed|failed|error' "$EV/rebaseline_run.log" | tail -n 1 | cut -c1-200)"
finish
[ "$src" = 0 ] || exit 30
exit 0
