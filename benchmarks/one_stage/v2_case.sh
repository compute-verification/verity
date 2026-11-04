#!/usr/bin/env bash
# One-stage v2's audit on a real session (v2_case.py), on a CPU machine, on hidden_zk's builds
# (benchmarks/pouw/hidden_zk/job.sh, sourced: the proving build of flock-circuit and a venv) and the sampled-proofs Lean
# (sampled-proofs-verifier for the challenger's coins, sampled-proofs-one-stage-zk to decide the case). From a clone:
#   research run --on HOST --source . --as LANE --project verity --cwd clone --cpus 16 --mem-gb 32 \
#     -- bash benchmarks/one_stage/v2_case.sh
# Kept in $OUT: case/ (case.json, the case as the difftests' `serve` reads it, and verdict.json, its answer), the session's
# serve and prove logs and the builds' logs. The job exits 1 unless the case is accepted.
# env: V2_CACHE (the builds' cache directory, kept across runs), V2_MEM_GB (the job's --mem-gb, default 32)
set -uo pipefail
export HZ_CONFIGS=v2-case HZ_MEM_GB=${V2_MEM_GB:-32}
[ -n "${V2_CACHE:-}" ] && export HZ_CACHE=$V2_CACHE
source benchmarks/pouw/hidden_zk/job.sh
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/benchmarks/numerical/python
command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
SP=$SRC/verity/core/service/sampled_proofs
( cd "$SP" && lake build sampled-proofs-verifier sampled-proofs-one-stage-zk > "$OUT/build-sampled-proofs.log" 2>&1
  echo $? > "$W/sp.rc" ) &
setup
wait
[ "$(cat "$W/sp.rc" 2>/dev/null)" = 0 ] || { log "sampled-proofs build FAILED: $(tail -5 "$OUT/build-sampled-proofs.log")"; exit 1; }

"$PY" "$SRC/verity/ml/flock/tests/plain_hidden.py" --out "$W/stage" > "$OUT/stage.json" 2> "$OUT/stage.err" \
  || { log "stage FAILED: $(tail -2 "$OUT/stage.err")"; exit 1; }
# a run's clone has the shipped tree as its origin, which holds main only as its own origin/main (a5_tp.sh's fetch)
UNSHALLOW=(); [ "$(git -C "$SRC" rev-parse --is-shallow-repository)" = true ] && UNSHALLOW=(--unshallow)
{ git -C "$SRC" fetch -q "${UNSHALLOW[@]}" origin +refs/heads/main:refs/remotes/origin/main \
    || git -C "$SRC" fetch -q "${UNSHALLOW[@]}" origin +refs/remotes/origin/main:refs/remotes/origin/main; } > "$OUT/main.err" 2>&1
git -C "$SRC" rev-list --first-parent refs/remotes/origin/main > "$W/main.txt" 2>> "$OUT/main.err" && [ -s "$W/main.txt" ] \
  || { log "no origin/main first-parent history in the clone: $(tail -2 "$OUT/main.err")"; exit 1; }
log "staged; main's first-parent line has $(wc -l < "$W/main.txt") commits"

"$PY" "$SRC/benchmarks/one_stage/v2_case.py" --stage "$W/stage" --bin "$BIN" \
  --zk-check "$SP/.lake/build/bin/sampled-proofs-one-stage-zk" --main "$W/main.txt" --out "$OUT" \
  > "$OUT/v2_case.out" 2> "$OUT/v2_case.err"
rc=$?
log "v2-case: exit $rc: $(tail -c 600 "$OUT/v2_case.out") $(tail -3 "$OUT/v2_case.err")"
exit $rc
