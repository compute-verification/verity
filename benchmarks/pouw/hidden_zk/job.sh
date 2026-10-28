#!/usr/bin/env bash
# Pearl-C4's hidden-check units under C-Flock --zk on CPUs, each proved by Rust `prove --zk` and verified by the Lean
# `verify --zk`, a row unit's anchors held to the verifier's own `TileAnchors.public_inputs` file. Run from the source tree's
# root (`research run --cwd source --declared-output 'out/*'`); results go to $RESEARCH_RUN_DIR/out, scratch to a
# `mktemp -d` directory the job removes, builds to $HZ_CACHE (kept, keyed by their inputs) or the scratch directory.
#   1. builds flock-circuit twice from Flock b684b12 and verity/ml/flock's patches (the proving build: sha512, glue; the
#      selftest build: + seed-injection), the Lean flock-verify, and a venv with numpy and blake3
#   2. per config ROLE:K in HZ_CONFIGS (ROLE row_a, row_b or tile, stage.py's): stage N instances, one loopback --zk session
#      drawing all N (subset:N; HZ_WARM warm + HZ_RUNS timed proofs) with the proving build, the selftest build's recorded
#      controls, drawn the same way, and the Lean
#      verifier on all of them:
#        honest-own     the honest session, the verifier's own anchors file                         ACCEPT
#        honest-wrong   the honest session, one bit of one anchor wrong in the verifier's file      REJECT
#        honest-chosen  the honest session, the same anchor bit chosen by the prover (public file)  REJECT
#        st-<case>      the selftest build's records under verity/flock-circuit+seed-injection: honest ACCEPT;
#                       output_row_committed_false (a tampered output row), output_claim_false, unit_internal_bit_flipped
#                       and public_input_claim_false (a false anchor claim, the verifier's file agreeing with it) REJECT
#      until T4 the verifier refuses --public-inputs, so summarize.py expects each row-unit check REFUSED; once every
#      check has passed it refuses a plain tile's output row of two output ports, so a plain tile's honest checks are
#      REFUSED, and so are a hiding tile's, whose rows (row-seg/v1, row/v2) are covered, for the tile's want of a Program
#      (no --program, NO_PROGRAM)
#      A tile has no public input, so it runs honest-own without --public-inputs and none of the anchor controls.
#   3. summarize.py: out/<role>-k<K>/summary.json and out/summary.json
# A staged config is kept under $HZ_CACHE/stage, keyed by its role, K, N, HZ_STAGE, the venv and every source stage.py
# reads; a later run with the same key proves the kept files (stage-cache.json says which, stage.json their own staging).
# The row units lower BitAnd_v1, E4m3ToF32_v1 and F32ToE2m1Sat_v1 through core3_scratch.py, a development binding: their
# results are labelled 'core3 scratch'.
# env: HZ_CONFIGS="row_a:256 row_b:256 tile:256"  HZ_N=16  HZ_MEM_GB (the job's --mem-gb)  HZ_RUNS=2  HZ_WARM=1
#      HZ_CASES (selftest cases)  HZ_CACHE (build and stage cache dir)  FLOCK_PROVE / FLOCK_SELFTEST / FLOCK_VERIFY / HZ_PYTHON
#      (prebuilt)  HZ_STAGE (stage.py's: fast, the default, or library)  HZ_STAGE_ONLY=1 (stage and keep, nothing else)
#      HZ_LEAN (the Lean checks to run, as globs over their tags or <config>/<tag>: "st-* tile-k4096/honest-own"; default
#      every one)
set -uo pipefail
SRC=$(pwd)
J=$SRC/benchmarks/pouw/hidden_zk
OUT=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$OUT"
mkdir -p "${TMPDIR:-/tmp}" && W=$(mktemp -d "${TMPDIR:-/tmp}/hidden-zk.XXXXXX")
trap 'rm -rf "$W"' EXIT
CONFIGS=${HZ_CONFIGS:?}
N=${HZ_N:-16}
MEM_GB=${HZ_MEM_GB:?}
RUNS=${HZ_RUNS:-2}
WARM=${HZ_WARM:-1}
CASES=${HZ_CASES:-honest output_row_committed_false output_claim_false unit_internal_bit_flipped public_input_claim_false}
LEAN_ONLY=${HZ_LEAN:-*}
export HZ_STAGE=${HZ_STAGE:-fast}
STAGE_GB=$(( MEM_GB * 90 / 100 ))
PROVE_GB=$(( MEM_GB * 60 / 100 ))
SERVE_GB=$(( MEM_GB * 30 / 100 ))
TH=$(nproc)
export RAYON_NUM_THREADS=$TH
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/job.log"; }
vlimit() { echo "ulimit -v $(( $1 * 1024 * 1024 ));"; }

# ---- 1. builds -----------------------------------------------------------------------------------------------------------
tree_key() { (cd "$SRC" && find "$@" -type f -not -path '*/.lake/*' -not -path '*/__pycache__/*' -print0 | sort -z \
  | xargs -0 sha256sum | sha256sum | cut -c1-16); }

build_circuit() {   # build_circuit DEST: DEST/flock-circuit-prove and DEST/flock-circuit-selftest
  export PATH="$HOME/.cargo/bin:$PATH"
  local D=$1 F=$W/flock
  rm -rf "$F" "$D.tmp"
  git clone -q https://github.com/succinctlabs/flock "$F" && cd "$F" && git checkout -q b684b12 || return 1
  for p in link gpu-link; do git apply "$SRC/verity/ml/flock/flock-$p-b684b12.patch" || return 1; done
  cp -r "$SRC/verity/ml/flock/live" crates/flock-live
  grep -q '"crates/flock-live"' Cargo.toml || sed -i 's|    "crates/flock-transcript",|    "crates/flock-transcript",\n    "crates/flock-live",|' Cargo.toml
  rm -f crates/flock-merkle/src/hm96.rs
  for p in sha512 glue zk vtime; do git apply "$SRC/verity/ml/flock/flock-$p-b684b12.patch" || return 1; done
  mkdir -p "$D.tmp"
  CARGO_TARGET_DIR=$W/target cargo build -q --release -p flock-live --features sha512,glue --bin flock-circuit -j "$TH" \
    && cp "$W/target/release/flock-circuit" "$D.tmp/flock-circuit-prove" || return 1
  CARGO_TARGET_DIR=$W/target cargo build -q --release -p flock-live --features sha512,glue,seed-injection --bin flock-circuit \
    -j "$TH" && cp "$W/target/release/flock-circuit" "$D.tmp/flock-circuit-selftest" || return 1
  rm -rf "$D" && mv "$D.tmp" "$D"
}
build_lean() {      # build_lean DEST: DEST/.lake/build/bin/flock-verify
  command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
  local D=$1
  rm -rf "$D.tmp" && mkdir -p "$D.tmp" && (cd "$SRC/verity/core/service/flock" && tar --exclude=.lake -cf - .) | tar -xf - -C "$D.tmp" \
    && (cd "$D.tmp" && lake build) && rm -rf "$D" && mv "$D.tmp" "$D"
}
build_python() {    # build_python DEST: a venv
  local D=$1
  rm -rf "$D" && uv venv -q --python 3.12 "$D" && uv pip install -q --python "$D/bin/python" numpy==2.5.3 blake3==1.0.9
}
cached() {          # cached NAME KEY BUILDER: the build's directory, built once per key under $HZ_CACHE's lock
  local name=$1 key=$2 builder=$3 base=${HZ_CACHE:-$W/cache}
  mkdir -p "$base" && local D=$base/$name-$key
  (
    flock 9
    [ -e "$D/.done" ] || { $builder "$D" > "$OUT/build-$name.log" 2>&1 && touch "$D/.done"; }
  ) 9> "$base/.lock-$name-$key"
  echo "$D"
}

setup() {
  log "hidden-zk: source ${RESEARCH_SOURCE_SHA:-$(git -C "$SRC" rev-parse HEAD 2>/dev/null)}, $TH CPUs, mem $MEM_GB GB" \
      "(stage cap $STAGE_GB, prove $PROVE_GB, serve $SERVE_GB), N=$N, runs $RUNS + $WARM warm, configs: $CONFIGS"
  { lscpu | grep -E 'Model name|^CPU\(s\)'; echo "threads $TH"; grep -E 'MemTotal|MemAvailable' /proc/meminfo; cat /proc/self/cgroup
    taskset -pc $$; df -h "${HZ_CACHE:-$W}" "$W"; } > "$OUT/host.txt" 2>&1
  local fail=0 p
  [ -n "${FLOCK_PROVE:-}" ] || cached flock "$(tree_key verity/ml/flock/live verity/ml/flock/*.patch)" build_circuit > "$W/flock.dir" &
  [ -n "${FLOCK_VERIFY:-}" ] || cached lean "$(tree_key verity/core/service/flock)" build_lean > "$W/lean.dir" &
  [ -n "${HZ_PYTHON:-}" ] || cached venv numpy2.5.3-blake3-1.0.9 build_python > "$W/venv.dir" &
  wait
  [ -n "${FLOCK_PROVE:-}" ] || { FLOCK_PROVE=$(cat "$W/flock.dir")/flock-circuit-prove; FLOCK_SELFTEST=$(cat "$W/flock.dir")/flock-circuit-selftest; }
  [ -n "${FLOCK_VERIFY:-}" ] || FLOCK_VERIFY=$(cat "$W/lean.dir")/.lake/build/bin/flock-verify
  [ -n "${HZ_PYTHON:-}" ] || HZ_PYTHON=$(cat "$W/venv.dir")/bin/python
  BIN=$FLOCK_PROVE SBIN=${FLOCK_SELFTEST:?} LEAN=$FLOCK_VERIFY PY=$HZ_PYTHON
  for p in "$BIN" "$SBIN" "$LEAN" "$PY"; do [ -x "$p" ] || { log "missing: $p"; fail=1; }; done
  if [ $fail = 1 ]; then
    log "build FAILED"; for f in "$OUT"/build-*.log; do echo "== $f"; tail -20 "$f"; done | tee -a "$OUT/job.log"; exit 1
  fi
  sha256sum "$BIN" "$SBIN" "$LEAN" | tee "$OUT/binaries.sha256" | tee -a "$OUT/job.log"
  "$PY" -c "import numpy, blake3, verity.core.protocols.pouw, verity_flock; print('python', numpy.__version__, blake3.__version__)" | tee -a "$OUT/job.log"
  TIME="$PY $J/timed.py"
  log "built"
}

# ---- 2. one config ---------------------------------------------------------------------------------------------------------
free_port() { "$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

session() {   # session D O: one loopback --zk session with the proving build, the verifier serving its own public file and
  local D=$1 O=$2 port pin srv rc try up= tu=   # drawing all N instances (subset:N), as every soundness claim is about a drawn session
  pin=$(cat "$D/pin.txt")
  # a templated statement's unit rows, placed by flock-tmpl-unit (h1_lanes/job.sh), which serve and prove read
  [ -L "$D/unit.placed" ] || [ -e "$D/unit.placed" ] && tu="--tmpl-unit $D/unit.placed"
  for try in 1 2 3 4 5; do   # the ephemeral port free_port picked can be taken by another worker before serve binds it: retake one
    port=$(free_port)
    $TIME "$O/serve.time" bash -c "$(vlimit $SERVE_GB) exec $BIN serve --zk --listen 127.0.0.1:$port --out $D/sessions \
      --instances $D/pub.bin --circuit $D/circuit.txt $tu --pin $pin --operator loopback --draw subset:$N" > "$O/serve.log" 2>&1 &
    srv=$!
    until grep -q '^SERVING ' "$O/serve.log" 2>/dev/null; do
      kill -0 $srv 2>/dev/null || break
      sleep 1
    done
    grep -q '^SERVING ' "$O/serve.log" 2>/dev/null && { up=1; break; }
    log "  serve exited before SERVING (try $try): $(tail -1 "$O/serve.log")"; wait $srv 2>/dev/null
  done
  [ -n "$up" ] || { log "  serve never came up after $try tries"; return 1; }
  $TIME "$O/prove.time" bash -c "$(vlimit $PROVE_GB) exec $BIN prove --zk --verifier 127.0.0.1:$port --instances $D/inst.bin \
    --circuit $D/circuit.txt $tu --runs $RUNS --warm $WARM" > "$O/prove.out" 2> "$O/prove.err"
  rc=$?
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
  du -sb "$D/sessions"/*/ > "$O/sessions.du" 2>/dev/null
  return $rc
}

controls() {  # controls D O ROLE: the selftest build's cases, recorded for the Lean verifier, drawn as session's are
  local D=$1 O=$2 role=$3 c rc
  mkdir -p "$D/rec"
  for c in $CASES; do
    [[ $role == tile* || $role == hidden* ]] && [ "$c" = public_input_claim_false ] && continue
    $TIME "$O/selftest-$c.time" bash -c "$(vlimit $PROVE_GB) exec $SBIN selftest --zk --instances $D/inst.bin \
      --circuit $D/circuit.txt --only $c --record-dir $D/rec --draw subset:$N" > "$O/selftest-$c.out" 2> "$O/selftest-$c.err"
    rc=$?
    log "  selftest $c exit $rc: $(grep -E '^NEG' "$O/selftest-$c.out" | cut -c1-200)"
  done
  "$PY" "$SRC/verity/ml/flock/verifier/selftest_records.py" "$D/rec" "$D/st" > /dev/null
}

lean_jobs() {   # lean_jobs D O ROLE: every Lean check of one config, as many at once as fit beside each other
  local D=$1 O=$2 role=$3 each par cap c tag args s case pub own
  each=$(( $(stat -L -c %s "$D/circuit.txt") * 6 / 1024 / 1024 / 1024 + 4 ))       # GB: its peak is about 5x the circuit text at K=256
  par=$(( STAGE_GB / each )); [ $par -lt 1 ] && par=1; [ $par -gt 8 ] && par=8
  local base="verify --zk --circuit $D/circuit.txt"
  [ -e "$D/partition.json" ] && base+=" --partition $D/partition.json"   # the verifier's own copy of the partition the circuit binds
  local cmds=()
  s=$(ls -d "$D"/sessions/*/ 2>/dev/null | tail -1)
  if [ -n "$s" ]; then
    if [[ $role == tile* || $role == hidden* ]]; then
      cmds+=("honest-own|$base --statement verity/flock-circuit --public $D/pub.bin --session $s")
    else
      cmds+=("honest-own|$base --statement verity/flock-circuit --public $D/pub.bin --public-inputs $D/public-inputs.bin --session $s"
             "honest-wrong|$base --statement verity/flock-circuit --public $D/pub.bin --public-inputs $D/public-inputs-wrong.bin --session $s"
             "honest-chosen|$base --statement verity/flock-circuit --public $D/pub-chosen.bin --public-inputs $D/public-inputs.bin --session $s")
    fi
  fi
  for s in "$D"/st/*/; do
    [ -d "$s" ] || continue
    case=$(basename "$s"); pub=$D/pub.bin own=$D/public-inputs.bin
    [ -e "$s/pub.bin" ] && pub=$s/pub.bin
    if [[ $role == tile* || $role == hidden* ]]; then own=""
    elif [ "$case" = public_input_claim_false ]; then   # the verifier's file agreeing with the false claim
      own=$s/own.bin
      "$PY" -c "import sys; o = open(sys.argv[1], 'rb').read(); h, b = o.split(b'\n', 1); p = open(sys.argv[2], 'rb').read(); \
open(sys.argv[3], 'wb').write(h + b'\n' + p[len(p) - len(b):])" "$D/public-inputs.bin" "$pub" "$own"
    fi
    cmds+=("st-$case|$base --statement verity/flock-circuit+seed-injection --public $pub ${own:+--public-inputs $own} --session $s")
  done
  local keep=() globs g cfg
  cfg=$(basename "$D")
  read -ra globs <<< "$LEAN_ONLY"
  for c in "${cmds[@]}"; do
    tag=${c%%|*}
    for g in "${globs[@]}"; do [[ $tag == $g || $cfg/$tag == $g ]] && { keep+=("$c"); break; }; done
  done
  cmds=("${keep[@]}")
  # the checks that run at once split the whole budget: `ulimit -v` bounds address space, which runs well past the resident
  # peak (a 2.6 GB circuit's check, 10.8 GB resident, ran out under an 18 GB cap)
  [ ${#cmds[@]} -gt 0 ] && [ $par -gt ${#cmds[@]} ] && par=${#cmds[@]}
  cap=$(( STAGE_GB / par ))
  log "  lean: ${#cmds[@]} checks ($LEAN_ONLY), $par at a time, cap $cap GB each"
  for c in "${cmds[@]}"; do
    tag=${c%%|*} args=${c#*|}
    # counted from the running jobs: `wait -n` also returns for a job that ended earlier unreported (setup's builds)
    while [ "$(jobs -rp | wc -l)" -ge "$par" ]; do wait -n; done
    $TIME "$O/lean-$tag.time" bash -c "$(vlimit $cap) exec $LEAN $args" > "$O/lean-$tag.out" 2> "$O/lean-$tag.err" &
  done
  wait
  for c in "${cmds[@]}"; do
    tag=${c%%|*}
    log "  lean $tag: $("$PY" -c "import json, sys; t = json.load(open(sys.argv[1])); \
v = [json.loads(l[8:]) for l in open(sys.argv[2]) if l.startswith('VERDICT ')]; \
print('exit', t['exit'], t['wall_s'], 's', round(t['maxrss_kb'] / 2**20, 1), 'GB:', \
('ACCEPT' if v[0]['accepted'] else 'REJECT ' + v[0]['why'][:160]) if v else 'no VERDICT')" "$O/lean-$tag.time" "$O/lean-$tag.out") \
$(tail -1 "$O/lean-$tag.err" | cut -c1-160)"
  done
}

stage_config() {   # stage_config ROLE K TAG D O: D holds the config's staged files, from the stage cache or staged now
  local role=$1 k=$2 tag=$3 D=$4 O=$5 key S rc=0
  key=$(tree_key benchmarks/pouw/hidden_zk/stage.py benchmarks/pouw/hidden_zk/fast_stage.py benchmarks/pouw/hidden_zk/core3_scratch.py \
    benchmarks/pouw/hidden_zk/hash_scratch.py verity/ml/flock/python verity verity/ml/circuits verity/ml/kernels \
    verity/core)
  S=${HZ_CACHE:-$W/cache}/stage/$tag-n$N-$HZ_STAGE-$(basename "$(dirname "$(dirname "$PY")")")-$key
  mkdir -p "$(dirname "$S")"
  (
    flock 9
    if [ -e "$S/.done" ]; then
      echo '{"hit": true, "dir": "'"$S"'"}' > "$O/stage-cache.json"
    else
      rm -rf "$S" && mkdir -p "$S"
      $TIME "$O/stage.time" bash -c "$(vlimit $STAGE_GB) exec $PY $J/stage.py $role $k $N $S" > "$O/stage.out" 2> "$O/stage.err" || exit 1
      cp "$O/stage.time" "$S/stage.time" && touch "$S/.done"
      echo '{"hit": false, "dir": "'"$S"'"}' > "$O/stage-cache.json"
    fi
  ) 9> "$S.lock" || rc=1
  [ $rc = 0 ] || { rm -rf "$S"; return 1; }
  cp "$S/stage.time" "$O/stage.time"
  for f in "$S"/*; do [ "$(basename "$f")" = stage.time ] || ln -s "$f" "$D/"; done
}

run_configs() {
  local cfg role k tag D O
  for cfg in $CONFIGS; do
    role=${cfg%%:*} k=${cfg#*:} tag=${cfg%%:*}-k${cfg#*:}
    D=$W/$tag O=$OUT/$tag
    mkdir -p "$D" "$O"
    log "$tag: staging ($HZ_STAGE)"
    if ! stage_config "$role" "$k" "$tag" "$D" "$O"; then
      log "$tag: staging FAILED: $(tail -1 "$O/stage.err")"; rm -rf "$D"; "$PY" "$J/summarize.py" "$OUT" > /dev/null; continue
    fi
    cp "$D/stage.json" "$D/pin.txt" "$O/"
    [ -e "$D/public-inputs.bin" ] && cp "$D/public-inputs.bin" "$D/public-inputs-wrong.bin" "$O/"
    log "$tag: staged: $("$PY" -c "import json; s = json.load(open('$O/stage.json')); c = json.load(open('$O/stage-cache.json')); print(s['definition'], s['ands'], 'ANDs, k_log', s['k_log'], s['circuit_bytes'], 'B, sha512', s['circuit_sha512'][:16], 'peak', s['maxrss_gb'], 'GB,', s['seconds']['total'], 's,', s.get('staging'), 'staging, cache', 'hit' if c['hit'] else 'miss')")"
    if [ "${HZ_STAGE_ONLY:-0}" = 1 ]; then "$PY" "$J/summarize.py" "$OUT" > /dev/null; rm -rf "$D"; continue; fi
    log "$tag: --zk session ($WARM warm + $RUNS timed)"
    session "$D" "$O" || log "$tag: --zk session FAILED: $(tail -2 "$O/prove.err")"
    controls "$D" "$O" "$role"
    lean_jobs "$D" "$O" "$role"
    "$PY" "$J/summarize.py" "$OUT" > /dev/null
    rm -rf "$D"
  done
  "$PY" "$J/summarize.py" "$OUT" | tee -a "$OUT/job.log"
  log "hidden-zk: done"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  setup
  run_configs
fi
