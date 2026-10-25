#!/usr/bin/env bash
# The multi-unit statement forms (`verity_flock.multi_unit`) of ScaledMmFp8Block_v2{K,N,128} on CPUs, measured side by side.
# Run from the source tree's root in a proofs slot (`research run --pool proofs --cwd source`; the slot's memory is
# CHECK_SLOT_MEM_GB); results go to $RESEARCH_RUN_DIR/out, scratch to a `mktemp -d` directory the job removes. The builds are
# hidden_zk's (its job.sh, sourced: flock-circuit, the Lean flock-verify, a venv with numpy), plus flock-live's test binary.
# Per shape K:N:n in MU_SHAPES (n instances, one x row each):
#   c       one statement: N * K/128 k-step units and N scale-stage slots per VU, the stage returning the outputs
#   b       two statements joined by a row: b-writer's units return temp and s, b-reader's units are the scale stage
#   a       two tables with glue: c's unit and stage circuits as tables, every stage port a glue relation (no rows, no --zk)
# c and b: one loopback session without --zk and one with it (prove.py, all n drawn); the Lean verifier of record on each --zk
# session; for shapes in MU_SELFTEST, c's selftest (every case on a shape in MU_SELFTEST_ALL, else MU_CASES).
# env: MU_SHAPES="1024:4:4 6144:16:2"  MU_SELFTEST="1024:4:4 6144:16:2"  MU_SELFTEST_ALL="1024:4:4"  MU_CACHE (builds)
set -uo pipefail
SRC=$(pwd)
MU=$SRC/benchmarks/multi_unit
SHAPES=${MU_SHAPES:-1024:4:4 6144:16:2}
SELFTEST=${MU_SELFTEST:-1024:4:4 6144:16:2}
SELFTEST_ALL=${MU_SELFTEST_ALL:-1024:4:4}
MU_CASES=${MU_CASES:-private_cut_word_tampered unit_internal_bit_flipped output_row_committed_false tail_swapped}
export HZ_CONFIGS=multi-unit HZ_N=1 HZ_MEM_GB=${CHECK_SLOT_MEM_GB:-${MU_MEM_GB:?}} HZ_RUNS=1 HZ_WARM=0
[ -n "${MU_CACHE:-}" ] && export HZ_CACHE=$MU_CACHE
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src${PYTHONPATH:+:$PYTHONPATH}
source benchmarks/pouw/hidden_zk/job.sh

build_tests() {   # build_tests DEST: DEST/flock-live-tests, flock-live's lib tests (sha512, glue)
  export PATH="$HOME/.cargo/bin:$PATH"
  local D=$1 F=$W/flock-tests exe
  rm -rf "$F" "$D.tmp"
  git clone -q https://github.com/succinctlabs/flock "$F" && cd "$F" && git checkout -q b684b12 || return 1
  for p in link gpu-link; do git apply "$SRC/verity/ml/flock/flock-$p-b684b12.patch" || return 1; done
  cp -r "$SRC/verity/ml/flock/live" crates/flock-live
  grep -q '"crates/flock-live"' Cargo.toml || sed -i 's|    "crates/flock-transcript",|    "crates/flock-transcript",\n    "crates/flock-live",|' Cargo.toml
  rm -f crates/flock-merkle/src/hm96.rs
  for p in sha512 glue zk vtime; do git apply "$SRC/verity/ml/flock/flock-$p-b684b12.patch" || return 1; done
  mkdir -p "$D.tmp"
  exe=$(CARGO_TARGET_DIR=$W/target-tests cargo test -q --release -p flock-live --features sha512,glue --lib --no-run \
    --message-format=json -j "$TH" | "$PY" -c "import json, sys; print([j['executable'] for j in map(json.loads, sys.stdin) \
if j.get('reason') == 'compiler-artifact' and j.get('executable') and j['target']['name'] == 'flock_live'][-1])") || return 1
  cp "$exe" "$D.tmp/flock-live-tests" && rm -rf "$D" && mv "$D.tmp" "$D"
}

lean_verify() {   # lean_verify D O TAG: the Lean verifier of record on D's --zk session (a shape is no Program's, so
                  # it is refused for want of one, ProvedScope.NO_PROGRAM, once every check has passed)
  local D=$1 O=$2 tag=$3 s n=$4
  s=$(ls -d "$D"/sessions-zk/*/ 2>/dev/null | tail -1)
  [ -n "$s" ] || { log "  lean $tag: no --zk session"; return; }
  $TIME "$O/lean-$tag.time" bash -c "$(vlimit $STAGE_GB) exec $LEAN verify --zk --circuit $D/circuit.txt --statement verity/flock-circuit \
    --public $D/pub-$n.bin --session $s" > "$O/lean-$tag.out" 2> "$O/lean-$tag.err"
  log "  lean $tag: $(grep -m1 '^VERDICT ' "$O/lean-$tag.out" | cut -c1-300) $(tail -1 "$O/lean-$tag.err" | cut -c1-160)"
}

run_shapes() {
  local shape K N n tag D O f TD z
  TD=$(cached tests "$(tree_key verity/ml/flock/live verity/ml/flock/*.patch)" build_tests)
  [ -x "$TD/flock-live-tests" ] || { log "tests build FAILED"; tail -20 "$OUT/build-tests.log" | tee -a "$OUT/job.log"; }
  for shape in $SHAPES; do
    IFS=: read -r K N n <<< "$shape"
    tag=k$K-n$N-i$n D=$W/$tag O=$OUT/$tag
    mkdir -p "$D" "$O"
    for f in c b; do
      log "$tag: staging $f"
      $TIME "$O/stage-$f.time" bash -c "$(vlimit $STAGE_GB) exec $PY $MU/stage.py $f $K $N $n $D" > "$O/stage-$f.out" 2> "$O/stage-$f.err" \
        || log "$tag: staging $f FAILED: $(tail -1 "$O/stage-$f.err")"
    done
    for f in c b-writer b-reader; do
      [ -e "$D/$f/stage.json" ] || continue
      cp "$D/$f/stage.json" "$O/$f-stage.json"
      for z in "" --zk; do
        log "$tag: $f session ${z:-without --zk}"
        $TIME "$O/$f-prove$z.time" bash -c "$(vlimit $PROVE_GB) exec $PY $MU/prove.py $BIN $D/$f $n $z" > "$O/$f-prove$z.out" 2> "$O/$f-prove$z.err"
        cp "$D/$f/prove${z:+-zk}.json" "$O/$f-prove${z:+-zk}.json" 2>/dev/null
        log "  $(tail -1 "$O/$f-prove$z.out" | cut -c1-400)"
      done
      lean_verify "$D/$f" "$O" "$f" "$n"
    done
    if [ -x "$TD/flock-live-tests" ] && [ -e "$D/c/circuit.txt" ]; then
      log "$tag: a (tables), $(( n * N )) stages"
      $TIME "$O/a.time" bash -c "$(vlimit $PROVE_GB) FC_MU_CIRCUIT=$D/c/circuit.txt FC_MU_STAGES=$(( n * N )) exec $TD/flock-live-tests \
        --ignored --nocapture --exact tables::tests::multi_unit_as_tables" > "$O/a.out" 2> "$O/a.err"
      grep '^MU_TABLES ' "$O/a.out" | cut -c11- > "$O/a.json"
      log "  a: $(cat "$O/a.json") $(cat "$O/a.time")"
    fi
    if [[ " $SELFTEST " == *" $shape "* ]] && [ -e "$D/c/circuit.txt" ]; then
      if [[ " $SELFTEST_ALL " == *" $shape "* ]]; then
        log "$tag: c selftest (every case)"
        $TIME "$O/c-selftest.time" bash -c "$(vlimit $PROVE_GB) exec $PY $MU/prove.py $BIN $D/c $n --selftest" > "$O/c-selftest.out" 2> "$O/c-selftest.err"
      else
        log "$tag: c selftest ($MU_CASES)"
        $TIME "$O/c-selftest.time" bash -c "$(vlimit $PROVE_GB) exec $PY $MU/prove.py $BIN $D/c $n --selftest $MU_CASES" > "$O/c-selftest.out" 2> "$O/c-selftest.err"
      fi
      cp "$D/c/selftest.json" "$O/c-selftest.json" 2>/dev/null
      log "  $(tail -1 "$O/c-selftest.out" | cut -c1-400)"
    fi
    "$PY" "$MU/summarize.py" "$OUT" > /dev/null
    rm -rf "$D"
  done
  "$PY" "$MU/summarize.py" "$OUT" | tee -a "$OUT/job.log"
  log "multi-unit: done"
}

setup
run_shapes
