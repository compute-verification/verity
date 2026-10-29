#!/usr/bin/env bash
# A served model's proof blocks under Flock --zk on CPUs, each proved by Rust `prove --zk` and verified by the Lean verifier of record
# (`flock-verify`). Run from the source tree's root; results go to
# $RESEARCH_RUN_DIR/out, scratch to a `mktemp -d` directory the job removes, builds and stagings to $SU_CACHE (kept, keyed by their
# inputs) or the scratch directory. The builds, the run and the selftest controls are hidden_zk's (its job.sh, sourced):
#   1. builds flock-circuit twice from Flock b684b12 and verity/ml/flock's patches (the proving build; the selftest build, with
#      seed-injection), the Lean flock-verify from verity/core/service/flock, and a venv with numpy and blake3
#   2. per UNIT in SU_UNITS (stage.py's TYPE[:S=V,...]): stage N inputs of its Q_word proof block from synthetic values, one loopback
#      --zk run sampling all N (subset:N) with the proving build, the selftest build's recorded controls drawn the same way,
#      and the Lean verifier on all of them:
#        honest-own   the honest run                                                                         ACCEPT
#        st-<case>    the selftest build's records under verity/flock-circuit+seed-injection: honest ACCEPT;
#                     output_row_committed_false (a forged output word, committed and proved), output_claim_false and
#                     unit_internal_bit_flipped REJECT
#   3. summarize.py: out/<tag>/summary.json and out/summary.json
# env: SU_UNITS="fp8-coord:K=6144"  SU_N=4  SU_MEM_GB (the job's --mem-gb)  SU_RUNS=1  SU_WARM=0  SU_CASES (selftest cases)
#      SU_CACHE  FLOCK_PROVE / FLOCK_SELFTEST / FLOCK_VERIFY / HZ_PYTHON (prebuilt)  SU_STAGE_ONLY=1 (stage, nothing else)
set -uo pipefail
SRC=$(pwd)
export HZ_CONFIGS=${SU_UNITS:?} HZ_N=${SU_N:-4} HZ_MEM_GB=${SU_MEM_GB:?} HZ_RUNS=${SU_RUNS:-1} HZ_WARM=${SU_WARM:-0}
export HZ_CASES=${SU_CASES:-honest output_row_committed_false output_claim_false unit_internal_bit_flipped}
[ -n "${SU_CACHE:-}" ] && export HZ_CACHE=$SU_CACHE
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src${PYTHONPATH:+:$PYTHONPATH}
source benchmarks/pouw/hidden_zk/job.sh
SU=$SRC/benchmarks/served_units

stage_one() {   # stage_one SPEC D O: D holds the proof block's staged files, from the stage cache or staged now
  local spec=$1 D=$2 O=$3 key S rc=0
  key=$(tree_key benchmarks/served_units/stage.py verity/ml/flock/python verity integrations/vllm/verity_vllm/program \
    tools/verity/circuit_check/src)
  S=${HZ_CACHE:-$W/cache}/stage/$(basename "$D")-n$N-$(basename "$(dirname "$(dirname "$PY")")")-$key
  mkdir -p "$(dirname "$S")"
  (
    flock 9
    if [ -e "$S/.done" ]; then
      echo '{"hit": true, "dir": "'"$S"'"}' > "$O/stage-cache.json"
    else
      rm -rf "$S" && mkdir -p "$S"
      $TIME "$O/stage.time" bash -c "$(vlimit $STAGE_GB) exec $PY $SU/stage.py $spec $N $S" > "$O/stage.out" 2> "$O/stage.err" || exit 1
      cp "$O/stage.time" "$S/stage.time" && touch "$S/.done"
      echo '{"hit": false, "dir": "'"$S"'"}' > "$O/stage-cache.json"
    fi
  ) 9> "$S.lock" || rc=1
  [ $rc = 0 ] || { rm -rf "$S"; return 1; }
  cp "$S/stage.time" "$O/stage.time"
  for f in "$S"/*; do [ "$(basename "$f")" = stage.time ] || ln -s "$f" "$D/"; done
}

run_units() {
  local spec tag D O
  for spec in $CONFIGS; do
    tag=$(echo "$spec" | tr ':,=' '---' | tr 'A-Z' 'a-z')
    D=$W/$tag O=$OUT/$tag
    mkdir -p "$D" "$O"
    log "$tag: staging $spec"
    if ! stage_one "$spec" "$D" "$O"; then
      log "$tag: staging FAILED: $(tail -1 "$O/stage.err")"; rm -rf "$D"; "$PY" "$SU/summarize.py" "$OUT" > /dev/null; continue
    fi
    cp "$D/stage.json" "$D/pin.txt" "$O/"
    log "$tag: staged: $("$PY" -c "import json; s = json.load(open('$O/stage.json')); c = json.load(open('$O/stage-cache.json')); print(s['definition'], s.get('class') or '', s['n'], 'instances,', s['ands'], 'ANDs,', s['unit_nnz'], 'nonzeros, k_log', s['k_log'], s['circuit_bytes'], 'B, sha512', s['circuit_sha512'][:16], 'peak', s['maxrss_gb'], 'GB,', s['seconds']['total'], 's, twin', (s['twin'] or {}).get('agrees'), ', cache', 'hit' if c['hit'] else 'miss')")"
    if [ "${SU_STAGE_ONLY:-0}" = 1 ]; then "$PY" "$SU/summarize.py" "$OUT" > /dev/null; rm -rf "$D"; continue; fi
    log "$tag: --zk session ($WARM warm + $RUNS timed)"
    session "$D" "$O" || log "$tag: --zk session FAILED: $(tail -2 "$O/prove.err")"
    controls "$D" "$O" tile        # hidden_zk's "tile" form: a proof block without public input ports
    lean_jobs "$D" "$O" tile
    "$PY" "$SU/summarize.py" "$OUT" > /dev/null
    rm -rf "$D"
  done
  "$PY" "$SU/summarize.py" "$OUT" | tee -a "$OUT/job.log"
  log "served-units: done"
}

setup
run_units
