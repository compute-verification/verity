#!/usr/bin/env bash
# Pc4RowA / Pc4RowB under C-Flock --zk with the PoUW anchors as public input regions, on CPUs. Run from the source tree's root
# (`research run --cwd source`); results go to $RESEARCH_RUN_DIR/out (`--declared-output 'out/*'`), everything else to a
# scratch directory under $TMPDIR that the job removes.
#   1. builds flock-circuit (check_build.sh's Flock b684b12 checkout and patches; sha512, glue, seed-injection: the selftest
#      build) and the Lean flock-verify (`lake build` of a copy of backends/flock/verifier/lean), and a venv with numpy, blake3
#   2. per config ROLE:K in PC4_CONFIGS: stage.py (N instances), a --zk loopback session (1 warm + 3 timed), the selftest
#      controls, a non-ZK session (K in PC4_NZ_KS) for the Lean verify, and the Lean verifier's statement with our own
#      --public-inputs, with one anchor bit flipped and with none (and verify of the non-ZK session with our own)
#   3. summarize.py: out/<role>-k<K>/summary.json and out/summary.json
# A config is skipped, not run, when its staging peak extrapolated from the same role's last one (linear in K) passes the
# staging cap. Every step runs under `ulimit -v`, well inside the job's MemoryMax, so a step that doesn't fit fails alone.
# env: PC4_CONFIGS="row_a:64 row_b:64 row_a:256 row_b:256 row_a:1024 row_b:1024"  PC4_N=16  PC4_MEM_GB=192 (the job's
#      --mem-gb)  PC4_NZ_KS="64"  FLOCK_CIRCUIT / FLOCK_VERIFY / PC4_PYTHON (prebuilt: skip that build)
set -uo pipefail
SRC=$(pwd)
J=$SRC/scratch/pc4_zk
OUT=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$OUT"
W=$(mktemp -d "${TMPDIR:-/tmp}/pc4-zk.XXXXXX")
trap 'rm -rf "$W"' EXIT
CONFIGS=${PC4_CONFIGS:-row_a:64 row_b:64 row_a:256 row_b:256 row_a:1024 row_b:1024}
N=${PC4_N:-16}
MEM_GB=${PC4_MEM_GB:-192}
NZ_KS=" ${PC4_NZ_KS:-64} "
STAGE_GB=$(( MEM_GB * 85 / 100 ))
PROVE_GB=$(( MEM_GB * 55 / 100 ))
SERVE_GB=$(( MEM_GB * 25 / 100 ))
TH=$(nproc)
export RAYON_NUM_THREADS=$TH
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/job.log"; }

# ---- 1. builds -----------------------------------------------------------------------------------------------------------
build_circuit() {
  export PATH="$HOME/.cargo/bin:$PATH"
  local F=$W/flock
  git clone -q https://github.com/succinctlabs/flock "$F" && cd "$F" && git checkout -q b684b12 || return 1
  for p in link gpu-link sha512 glue zk vtime; do git apply "$SRC/backends/flock/flock-$p-b684b12.patch" || return 1; done
  cp -r "$SRC/backends/flock/live" crates/flock-live
  grep -q '"crates/flock-live"' Cargo.toml || sed -i 's|    "crates/flock-transcript",|    "crates/flock-transcript",\n    "crates/flock-live",|' Cargo.toml
  CARGO_TARGET_DIR=$W/target cargo build -q --release -p flock-live --features sha512,glue,seed-injection --bin flock-circuit -j "$TH"
}
build_lean() {
  command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
  mkdir -p "$W/lean" && (cd "$SRC/backends/flock/verifier/lean" && tar --exclude=.lake -cf - .) | tar -xf - -C "$W/lean" || return 1
  cd "$W/lean" && lake build
}
build_python() {
  uv venv -q --python 3.12 "$W/venv" && uv pip install -q --python "$W/venv/bin/python" numpy==2.5.3 blake3==1.0.9
}
setup() {
  log "pc4-zk: source ${RESEARCH_SOURCE_SHA:-$(git -C "$SRC" rev-parse HEAD 2>/dev/null)}, $TH CPUs, mem $MEM_GB GB (stage cap" \
      "$STAGE_GB, prove $PROVE_GB, serve $SERVE_GB), N=$N, configs: $CONFIGS"
  { lscpu | grep -E 'Model name|^CPU\(s\)'; echo "threads $TH"; grep -E 'MemTotal|MemAvailable' /proc/meminfo; cat /proc/self/cgroup; } \
    > "$OUT/host.txt" 2>&1
  local pids=() fail=0 p
  if [ -z "${FLOCK_CIRCUIT:-}" ]; then (build_circuit > "$OUT/build-circuit.log" 2>&1) & pids+=($!); FLOCK_CIRCUIT=$W/target/release/flock-circuit; fi
  if [ -z "${FLOCK_VERIFY:-}" ]; then (build_lean > "$OUT/build-lean.log" 2>&1) & pids+=($!); FLOCK_VERIFY=$W/lean/.lake/build/bin/flock-verify; fi
  if [ -z "${PC4_PYTHON:-}" ]; then (build_python > "$OUT/build-python.log" 2>&1) & pids+=($!); PC4_PYTHON=$W/venv/bin/python; fi
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  BIN=$FLOCK_CIRCUIT LEAN=$FLOCK_VERIFY PY=$PC4_PYTHON
  [ -x "$BIN" ] && [ -x "$LEAN" ] && [ -x "$PY" ] || fail=1
  if [ $fail = 1 ]; then
    log "build FAILED"; for f in "$OUT"/build-*.log; do echo "== $f"; tail -20 "$f"; done | tee -a "$OUT/job.log"; exit 1
  fi
  sha256sum "$BIN" "$LEAN" | tee "$OUT/binaries.sha256" | tee -a "$OUT/job.log"
  "$PY" -c "import numpy, blake3, verity_pouw, verity_flock; print('python', numpy.__version__, blake3.__version__)" | tee -a "$OUT/job.log"
  TIME="$PY $J/timed.py"
  log "built"
}

# ---- 2. one config ---------------------------------------------------------------------------------------------------------
free_port() { "$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

session() {   # session DIR OUT ZKFLAG RUNS WARM SUFFIX: one loopback session (the verifier serving its own pub file; prove)
  local D=$1 O=$2 Z=$3 runs=$4 warm=$5 suf=$6 port pin srv rc
  pin=$(cat "$D/pin.txt"); port=$(free_port)
  $TIME "$O/serve$suf.time" bash -c "ulimit -v $(( SERVE_GB * 1024 * 1024 )); exec $BIN serve $Z --listen 127.0.0.1:$port \
    --out $D/sessions$suf --instances $D/pub-$N.bin --circuit $D/circuit.txt --pin $pin --operator loopback" > "$O/serve$suf.log" 2>&1 &
  srv=$!
  until grep -q '^SERVING ' "$O/serve$suf.log" 2>/dev/null; do
    kill -0 $srv 2>/dev/null || { log "  serve$suf exited before SERVING: $(tail -2 "$O/serve$suf.log")"; return 1; }
    sleep 1
  done
  $TIME "$O/prove$suf.time" bash -c "ulimit -v $(( PROVE_GB * 1024 * 1024 )); exec $BIN prove $Z --verifier 127.0.0.1:$port \
    --instances $D/inst-$N.bin --circuit $D/circuit.txt --runs $runs --warm $warm" > "$O/prove$suf.out" 2> "$O/prove$suf.err"
  rc=$?
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
  return $rc
}

controls() {  # the selftest controls under --zk, each its own process
  local D=$1 O=$2 c rc
  for c in honest public_input_claim_false unit_internal_bit_flipped output_row_committed_false output_claim_false; do
    $TIME "$O/selftest-$c.time" bash -c "ulimit -v $(( PROVE_GB * 1024 * 1024 )); exec $BIN selftest --zk --instances $D/inst-$N.bin \
      --circuit $D/circuit.txt --only $c" > "$O/selftest-$c.out" 2> "$O/selftest-$c.err"
    rc=$?
    log "  selftest $c exit $rc: $(grep -E '^NEG' "$O/selftest-$c.out" | cut -c1-160)"
  done
}

lean_jobs() {   # the Lean verifier's checks of one config, as many at once as fit beside each other
  local D=$1 O=$2 k=$3 each par cap running=0 c tag args s f
  each=$(( $(stat -c %s "$D/circuit.txt") * 8 / 1024 / 1024 / 1024 + 4 ))       # GB: about 6x the circuit text at K=64
  par=$(( STAGE_GB / each )); [ $par -lt 1 ] && par=1; [ $par -gt 4 ] && par=4
  cap=$(( STAGE_GB / par ))
  local base="--circuit $D/circuit.txt --public $D/pub-$N.bin --partition $D/partition.json"
  local cmds=("statement-own|statement $base --public-inputs $D/public-inputs-$N.bin"
              "statement-flipped|statement $base --public-inputs $D/public-inputs-flipped.bin"
              "statement-none|statement $base")
  if [[ $NZ_KS == *" $k "* ]] && [ -d "$D/sessions-nz" ]; then
    s=$(ls -d "$D"/sessions-nz/*/ | head -1)
    cmds+=("verify-own|verify --statement verity/flock-circuit+seed-injection $base --public-inputs $D/public-inputs-$N.bin --session $s")
  fi
  log "  lean: ${#cmds[@]} checks, $par at a time, cap $cap GB each"
  for c in "${cmds[@]}"; do
    tag=${c%%|*} args=${c#*|}
    $TIME "$O/lean-$tag.time" bash -c "ulimit -v $(( cap * 1024 * 1024 )); exec $LEAN $args" > "$O/lean-$tag.out" 2> "$O/lean-$tag.err" &
    running=$(( running + 1 ))
    if [ $running -ge $par ]; then wait -n; running=$(( running - 1 )); fi
  done
  wait
  for f in "$O"/lean-*.out; do
    log "  $(basename "$f" .out) exit $("$PY" -c "import json; print(json.load(open('${f%.out}.time'))['exit'])"):" \
        "$(head -c 200 "$f" | tr '\n' ' ') $(tail -1 "${f%.out}.err" | cut -c1-200)"
  done
}

prove_all() {  # everything after staging, on a staged directory
  local D=$1 O=$2 k=$3 tag=$4
  log "$tag: --zk session (1 warm + 3 timed)"
  session "$D" "$O" --zk 3 1 "" || log "$tag: --zk session FAILED: $(tail -2 "$O/prove.err")"
  controls "$D" "$O"
  if [[ $NZ_KS == *" $k "* ]]; then
    log "$tag: non-ZK session (for the Lean verify)"
    session "$D" "$O" "" 1 0 -nz || log "$tag: non-ZK session FAILED: $(tail -2 "$O/prove-nz.err")"
  fi
  lean_jobs "$D" "$O" "$k"
  "$PY" "$J/summarize.py" "$OUT" > /dev/null
}

run_configs() {
  local cfg role k tag D O pk pgb est
  declare -A PEAK
  for cfg in $CONFIGS; do
    role=${cfg%%:*} k=${cfg#*:} tag=${cfg%%:*}-k${cfg#*:}
    D=$W/$tag O=$OUT/$tag
    mkdir -p "$D" "$O"
    if [ -n "${PEAK[$role]:-}" ]; then
      read -r pk pgb <<< "${PEAK[$role]}"
      est=$("$PY" -c "print(round($pgb * $k / $pk, 1))")
      if "$PY" -c "import sys; sys.exit(0 if $est > $STAGE_GB else 1)"; then
        echo "staging peak extrapolated from K=$pk ($pgb GB): $est GB > the $STAGE_GB GB cap" > "$O/SKIPPED"
        log "$tag: SKIPPED, $(cat "$O/SKIPPED")"; continue
      fi
    fi
    log "$tag: staging"
    if ! $TIME "$O/stage.time" bash -c "ulimit -v $(( STAGE_GB * 1024 * 1024 )); exec $PY $J/stage.py $role $k $N $D" \
        > "$O/stage.out" 2> "$O/stage.err"; then
      log "$tag: staging FAILED: $(tail -1 "$O/stage.err")"; rm -rf "$D"; "$PY" "$J/summarize.py" "$OUT" > /dev/null; continue
    fi
    cp "$D/stage.json" "$D/partition.json" "$D/pin.txt" "$D/public-inputs-$N.bin" "$D/public-inputs-flipped.bin" "$O/"
    PEAK[$role]="$k $("$PY" -c "import json; print(json.load(open('$O/stage.time'))['maxrss_kb'] / 2**20)")"
    log "$tag: staged: $("$PY" -c "import json; s = json.load(open('$O/stage.json')); print(s['definition'], s['ands'], 'ANDs, k_log', s['k_log'], s['circuit_bytes'], 'B, peak', s['maxrss_gb'], 'GB,', s['seconds']['total'], 's')")"
    prove_all "$D" "$O" "$k" "$tag"
    rm -rf "$D"
  done
  "$PY" "$J/summarize.py" "$OUT" | tee -a "$OUT/job.log"
  log "pc4-zk: done"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  setup
  run_configs
fi
