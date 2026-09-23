#!/usr/bin/env bash
# run_row_v2.sh -- the v2 wrapper that runs each sweep stage as ONE `research run --tool vllm.<stage>` Attempt (Phase 5a).  row_pod.sh (or
# tp_stage.sh, for a `__tp<N>__` row with N >= 2) is untouched: it is called for one stage at a time; this script only maps flags to its env knobs, checks the cited inputs, and on exit
# declares the stage's typed outputs (`outputs.json`, verity_capture/sweep/research_outputs.py) for the runner's publish step.
#
#   run_row_v2.sh stage <build|match|commit> <row-id> <ROLE> <hf-repo> <revision> [--flag value ...]
#       The WORKLOAD of one Attempt.  Run it under the harness with the REPO ROOT as cwd (the Tool's source root; this script cds to
#       integrations/vllm, row_pod.sh's tree, by itself):
#         research run --tool vllm.build --campaign <c> --project verity -- \
#             bash integrations/vllm/verity_capture/sweep/run_row_v2.sh stage build <row> <ROLE> <repo> <rev> [--retain host --build-jobs 4 ...]
#         research run --tool vllm.match  ... --input build=art:<the Build artifact> -- bash ... stage match  <row> ...
#         research run --tool vllm.commit ... --input build=art:... --input match=art:... -- bash ... stage commit <row> ...
#       Flags (the Tool's params; research_tools.py says which are key): --retain host|device|gpu  --sweep-dir DIR  --build-jobs N|auto
#       --match-impl fast|record  --match-phase all|gpu|cpu  --gpu-util F  --pairs N  --snapshot-steps 0,1|all|row-default
#       --snapshot-max-bytes N  --within-step dag|sequence  --window-mb N  --window-slots N  --tap-cap-mb N  --open-after-release 0|1
#       --replay-cache 0|1  --hot-engine 0|1  --required-classes LIST  --extra-engine-args "k=v ..."  --replay-args "..."
#       --commit-args "..."  --lp N  --t N  --query ID (recorded; default Q_module_body_v1)  --py PATH
#       Inputs are read from the run's job.json (`store.inputs`, what `--input name=art:` recorded); a Match/Commit whose $D holds a Build
#       with a different program_digest than the cited Build artifact stops by name (PRECHECK_FAIL_BUILD_INPUT_MISMATCH, exit 3).
#       On exit (any rc): outputs.json + $RESEARCH_RUN_DIR/outputs/<name>/ are written, and the run id is registered with the pod guard's
#       durability list ($GUARD_PRESERVE_FILE, default /root/dm/guard.preserve) so the pod will not terminate before it is PRESERVED.
#
#   run_row_v2.sh row <row-id> <ROLE> <hf-repo> <revision> [--stages build,match,commit] [--campaign C] [--project P] [stage flags...]
#       Pod-side orchestration: one `research run --tool vllm.<stage>` per stage, in order, each citing the earlier stages' artifacts
#       (`--input build=art:...`, `--input match=art:...`) read from the published attempt; stops at the first stage whose run exits
#       non-zero or publishes no account artifact.  Prints `<stage> <run-id> <artifact-id>` lines; `row.json` in $SWEEP_DIR/<row>/ keeps them.
#
#   run_row_v2.sh shed [--force]
#       The sweep's teardown: `research data preserved` over every attempt this pod registered; the pod is terminated (pod_guard.sh
#       guard_terminate_self, which applies the same gate) only when everything is PRESERVED.  --force = GUARD_FORCE_TERMINATE=1.
#
# Env: RESEARCH_RUN_DIR / RESEARCH_RUN_ID (set by the runner), RESEARCH_BIN (the research CLI; default `research` on PATH, else
# `python3 -m research` with the shipped tool snapshot on PYTHONPATH), RESEARCH_STORE / RESEARCH_STORE_CONFIG (the store + remote the
# publish uses; absolutised here before any nested run), GUARD_PRESERVE_FILE, ROW_POD (a stand-in for row_pod.sh: dry runs / tests only),
# plus everything row_pod.sh reads (PY, SWEEP_DIR, ...).
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VLLM_ROOT=$(cd "$HERE/../.." && pwd)                     # integrations/vllm: row_pod.sh's tree (cwd it requires)
REPO_ROOT=$(cd "$VLLM_ROOT/../.." && pwd)
GUARD_PRESERVE_FILE=${GUARD_PRESERVE_FILE:-/root/dm/guard.preserve}

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }

research_cmd() {
  if [ -n "${RESEARCH_BIN:-}" ]; then "$RESEARCH_BIN" "$@"
  elif command -v research >/dev/null 2>&1; then research "$@"
  else
    local py=${RESEARCH_PY:-python3}
    if ! "$py" -c "import research" >/dev/null 2>&1 && [ -d "$REPO_ROOT/tools/research/src" ]; then
      PYTHONPATH="$REPO_ROOT/tools/research/src${PYTHONPATH:+:$PYTHONPATH}" "$py" -m research "$@"
    else
      "$py" -m research "$@"
    fi
  fi
}

abs_store_env() {
  # a relative RESEARCH_STORE_CONFIG / RESEARCH_STORE (e.g. tools/research/store.pod.toml given to `research run --env`) resolves against
  # the cwd of the process that reads it; nested runs and row_pod.sh change cwd, so pin them here
  local v
  for v in RESEARCH_STORE_CONFIG RESEARCH_STORE RESEARCH_RUNS; do
    if [ -n "${!v:-}" ] && [ "${!v:0:1}" != "/" ] && [ "${!v:0:1}" != "~" ]; then export "$v=$(cd "$(dirname "${!v}")" 2>/dev/null && pwd)/$(basename "${!v}")"; fi
  done
}

# ---- flag -> row_pod.sh env knob ---------------------------------------------------------------------------------------------------------------
STAGE_FLAGS=()          # the stage flags as given (re-passed to nested runs in `row` mode)
apply_flag() {
  local k=$1 v=$2
  case "$k" in
    retain) export RETAIN=$v;;                 sweep-dir) export SWEEP_DIR=$v;;               build-jobs) export BUILD_JOBS=$v;;
    match-impl) export MATCH_IMPL=$v;;         match-phase) export MATCH_PHASE=$v;;           gpu-util) export GPU_UTIL=$v;;
    pairs) export PAIRS=$v;;                   snapshot-max-bytes) export MATCH_SNAP_MAX_BYTES=$v;;
    snapshot-steps) [ "$v" = row-default ] || export MATCH_SNAP_STEPS=$v;;
    within-step) export WITHIN_STEP=$v;;       window-mb) export WINDOW_MB=$v;;               window-slots) export WINDOW_SLOTS=$v;;
    tap-cap-mb) export TAP_CAP_MB=$v;;         open-after-release) export OPEN_AFTER_RELEASE=$v;;  replay-cache) export REPLAY_CACHE=$v;;
    hot-engine) export HOT_ENGINE=$v;;         required-classes) export REQUIRED_CLASSES=$v;;  extra-engine-args) export EXTRA_ENGINE_ARGS=$v;;
    replay-args) export REPLAY_ARGS=$v;;       commit-args) export EXTRA_COMMIT_ARGS=$v;;     lp) export LP=$v;;   t) export T=$v;;
    query) export VERITY_QUERY_ID=$v;;         py) export PY=$v;;
    *) echo "run_row_v2.sh: unknown flag --$k" >&2; exit 2;;
  esac
}
parse_stage_flags() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --*=*) local kv=${1#--}; apply_flag "${kv%%=*}" "${kv#*=}"; STAGE_FLAGS+=("$1"); shift;;
      --*) [ $# -ge 2 ] || { echo "run_row_v2.sh: --${1#--} needs a value" >&2; exit 2; }; apply_flag "${1#--}" "$2"; STAGE_FLAGS+=("$1" "$2"); shift 2;;
      *) echo "run_row_v2.sh: unexpected argument $1" >&2; exit 2;;
    esac
  done
}

row_batch() { local r=$1 b; b=$(echo "$r" | sed -n 's/.*__b\([0-9]*\)__.*/\1/p'); echo "${b:-1}"; }
row_world() { local r=$1 w; w=$(echo "$r" | sed -n 's/.*__tp\([0-9]*\)__.*/\1/p'); echo "${w:-1}"; }

# ---- stage ------------------------------------------------------------------------------------------------------------------------------------
job_input() {  # job_input <name> -> art id recorded by `research run --input <name>=art:...` (job.json store.inputs), or ""
  [ -n "${RESEARCH_RUN_DIR:-}" ] && [ -f "$RESEARCH_RUN_DIR/job.json" ] || return 0
  python3 -c 'import json,sys
d=json.load(open(sys.argv[1])); print(((d.get("store") or {}).get("inputs") or {}).get(sys.argv[2]) or "")' "$RESEARCH_RUN_DIR/job.json" "$1" 2>/dev/null
}

artifact_meta() {  # artifact_meta <art> <key> -> meta[key] of the artifact's manifest (local store, or the remote), or ""
  research_cmd data show "$1" --json 2>/dev/null | python3 -c 'import json,sys
d=json.load(sys.stdin); m=(d.get("manifest") or d).get("meta") or {}; v=m.get(sys.argv[1]); print("" if v is None else v)' "$2" 2>/dev/null
}

cmd_stage() {
  [ $# -ge 5 ] || usage
  local stage=$1 row=$2 role=$3 repo=$4 rev=$5; shift 5
  case "$stage" in build|match|commit) ;; *) echo "run_row_v2.sh: stage must be build|match|commit, got $stage" >&2; exit 2;; esac
  parse_stage_flags "$@"
  abs_store_env
  cd "$VLLM_ROOT" || exit 2
  export SWEEP_DIR=${SWEEP_DIR:-/workspace/cp/sweep}
  local d="$SWEEP_DIR/$row"; mkdir -p "$d"
  local build_in match_in
  build_in=$(job_input build); match_in=$(job_input match)
  # on exit: the typed outputs for the publish step + the durability registration; the stage's own rc stands
  _v2_exit() {
    local rc=$?
    if [ -n "${RESEARCH_RUN_DIR:-}" ]; then
      python3 -m verity_capture.sweep.research_outputs "$d" "$row" "$stage" ${build_in:+--input build=$build_in} ${match_in:+--input match=$match_in} >> "$d/row.log" 2>&1 || true
      if [ -n "${RESEARCH_RUN_ID:-}" ]; then
        mkdir -p "$(dirname "$GUARD_PRESERVE_FILE")" 2>/dev/null && { grep -qxF "$RESEARCH_RUN_ID" "$GUARD_PRESERVE_FILE" 2>/dev/null || echo "$RESEARCH_RUN_ID" >> "$GUARD_PRESERVE_FILE"; } 2>/dev/null || true
      fi
    fi
    exit "$rc"
  }
  trap _v2_exit EXIT
  # input provenance: the Build this stage consumes IS the cited artifact (program_digest of record from build_summary.json, never the manifest)
  if [ "$stage" != build ] && [ -n "$build_in" ]; then
    local here cited
    here=$(python3 -m verity_capture.sweep.research_outputs digest "$d" "$row" 2>/dev/null || true)
    if [ -z "$here" ]; then
      echo "precheck FAIL $(date -u +%FT%TZ) PRECHECK_FAIL_BUILD_INPUT_ABSENT (cited $build_in but $d has no Build; run the build stage on this pod first)" | tee -a "$d/stages.txt"; exit 3
    fi
    cited=$(artifact_meta "$build_in" program_digest)
    if [ -z "$cited" ]; then
      echo "precheck FAIL $(date -u +%FT%TZ) PRECHECK_FAIL_BUILD_INPUT_UNREADABLE (cited $build_in: manifest not readable here, or no program_digest in meta)" | tee -a "$d/stages.txt"; exit 3
    fi
    if [ "$cited" != "$here" ]; then
      echo "precheck FAIL $(date -u +%FT%TZ) PRECHECK_FAIL_BUILD_INPUT_MISMATCH (cited $build_in program_digest=${cited:0:16} != this dir's Build ${here:0:16})" | tee -a "$d/stages.txt"; exit 3
    fi
    echo "$(date -u +%FT%TZ) [row $row] v2 $stage: consumes Build $build_in (program_digest ${here:0:16} verified)${match_in:+, Match $match_in}" | tee -a "$d/row.log"
  fi
  local rc runner=verity_capture/sweep/row_pod.sh
  [ "$(row_world "$row")" -gt 1 ] && runner=verity_capture/sweep/tp_stage.sh      # a `__tp<N>__` row: the WORLD=N runner (it builds its own manifest)
  bash "${ROW_POD:-$runner}" "$row" "$role" "$repo" "$rev" "$stage"; rc=$?     # ROW_POD: a stand-in stage for dry runs / tests
  # the Build artifact carries the required-value manifest: build it here (CPU; the same commands row_pod.sh's Commit uses) when the Build passed
  if [ "$stage" = build ] && [ "$rc" -eq 0 ] && [ ! -f "$d/manifest.json" ]; then
    local py=${PY:-/workspace/venv312/bin/python} b wl="workloads/$row.json" wd
    b=$(row_batch "$row")
    if [ "$b" -gt 1 ] && ls -d "$d"/build_request_LP*_T* >/dev/null 2>&1; then
      wd=$("$py" -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d.get("workload_digest") or d.get("program_digest") or d.get("digest") or "")' "$d/build_workload/workload_program.json" 2>/dev/null || true)
      "$py" -m verity_capture.commit.required_manifest build-global --fixture "$wl" --programs-root "$d" ${wd:+--workload-digest "$wd"} --out "$d/manifest.json" > "$d/manifest.log" 2>&1 \
        || echo "$(date -u +%FT%TZ) [row $row] v2 build: required-value manifest build-global FAILED (manifest.log); the Build artifact has none" | tee -a "$d/row.log"
    elif [ -f "$d/build_request/instances.json.gz" ]; then
      "$py" -m verity_capture.commit.required_manifest build --program "$d/build_request" --workload "$wl" --out "$d/manifest.json" > "$d/manifest.log" 2>&1 \
        || echo "$(date -u +%FT%TZ) [row $row] v2 build: required-value manifest build FAILED (manifest.log); the Build artifact has none" | tee -a "$d/row.log"
    fi
  fi
  exit "$rc"
}

# ---- row (pod-side orchestration) -------------------------------------------------------------------------------------------------------------
attempt_output() {  # attempt_output <run-id> <name> -> outputs.<name> of the published attempt, or ""
  research_cmd data show "$1" --json 2>/dev/null | python3 -c 'import json,sys
d=json.load(sys.stdin)
if isinstance(d, list): d = d[0] if d else {}
print(((d.get("outputs") or {}).get(sys.argv[1])) or "")' "$2" 2>/dev/null
}

cmd_row() {
  [ $# -ge 4 ] || usage
  local row=$1 role=$2 repo=$3 rev=$4; shift 4
  local stages="build,match,commit" campaign=${RESEARCH_CAMPAIGN:-} project=${RESEARCH_PROJECT:-verity} rest=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --stages) stages=$2; shift 2;; --campaign) campaign=$2; shift 2;; --project) project=$2; shift 2;;
      *) rest+=("$1"); shift;;
    esac
  done
  parse_stage_flags "${rest[@]}"          # validates; also exports SWEEP_DIR when given
  abs_store_env
  cd "$VLLM_ROOT" || exit 2
  export SWEEP_DIR=${SWEEP_DIR:-/workspace/cp/sweep}
  local d="$SWEEP_DIR/$row"; mkdir -p "$d"
  local build_art="" match_art="" stage run_id rc art name
  local ids=()
  for stage in ${stages//,/ }; do
    run_id="r$(date -u +%Y%m%d-%H%M%S)-$(od -An -N2 -tx1 /dev/urandom | tr -d ' \n')"
    local args=(run --id "$run_id" --tool "vllm.$stage" --project "$project" --cwd "$REPO_ROOT")     # the Tool's source root is the repo; this script cds to integrations/vllm itself
    [ -n "$campaign" ] && args+=(--campaign "$campaign")
    [ "$stage" != build ] && [ -n "$build_art" ] && args+=(--input "build=$build_art")
    [ "$stage" = commit ] && [ -n "$match_art" ] && args+=(--input "match=$match_art")
    echo "$(date -u +%FT%TZ) [row $row] v2 row: research ${args[*]} -- run_row_v2.sh stage $stage ..." | tee -a "$d/row.log"
    research_cmd "${args[@]}" -- bash integrations/vllm/verity_capture/sweep/run_row_v2.sh stage "$stage" "$row" "$role" "$repo" "$rev" "${STAGE_FLAGS[@]}"; rc=$?
    case "$stage" in build) name=build;; match) name=match;; commit) name=verdict;; esac
    art=$(attempt_output "$run_id" "$name")
    ids+=("$stage $run_id ${art:--}")
    echo "$stage $run_id ${art:--} rc=$rc"
    python3 - "$d/row.json" "$stage" "$run_id" "$art" "$rc" <<'PY' 2>/dev/null || true
import json, os, sys
p, stage, rid, art, rc = sys.argv[1:6]
doc = json.load(open(p)) if os.path.exists(p) else {"schema": "verity-sweep/row-v2/v1", "stages": {}}
doc["stages"][stage] = {"run_id": rid, "artifact": art or None, "rc": int(rc)}
json.dump(doc, open(p, "w"), indent=1)
PY
    if [ "$rc" -ne 0 ] || [ -z "$art" ]; then
      echo "$(date -u +%FT%TZ) [row $row] v2 row: stopping at $stage (rc=$rc, artifact=${art:-none})" | tee -a "$d/row.log"; exit "$(( rc == 0 ? 20 : rc ))"
    fi
    case "$stage" in build) build_art=$art;; match) match_art=$art;; esac
  done
  exit 0
}

# ---- shed (teardown gated on durability) -------------------------------------------------------------------------------------------------------
cmd_shed() {
  local force=0
  [ "${1:-}" = --force ] && force=1
  abs_store_env
  local guard
  for guard in /root/dm/pod_guard.sh "$REPO_ROOT/tools/research/src/research/pods/sh/pod_guard.sh"; do [ -f "$guard" ] && break; done
  if [ ! -f "$guard" ]; then echo "run_row_v2.sh shed: no pod_guard.sh found; not terminating" >&2; exit 2; fi
  # shellcheck disable=SC1090
  source "$guard"
  guard_research() { research_cmd "$@"; }                 # the guard's CLI lookup, replaced by this script's (RESEARCH_BIN / snapshot aware)
  if [ "$force" = 1 ]; then export GUARD_FORCE_TERMINATE=1; fi
  if ! guard_preserved_ok; then
    echo "run_row_v2.sh shed: outputs NOT preserved (see /root/dm/guard.log, guard.preserved.log); attempting research data push" >&2
    guard_preserve_repair || { [ "$force" = 1 ] || { echo "run_row_v2.sh shed: REFUSING to terminate (--force overrides)" >&2; exit 1; }; }
  fi
  guard_terminate_self "run_row_v2.sh shed: stage outputs preserved"
}

case "${1:-}" in
  stage) shift; cmd_stage "$@";;
  row) shift; cmd_row "$@";;
  shed) shift; cmd_shed "$@";;
  *) usage;;
esac
