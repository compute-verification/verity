#!/usr/bin/env bash
# The drawn-unit replay bundle proved (prove_bundle.py; PRF-171's M1, and with PB_K M3's rehearsal): the torch loop's window
# and its tampered twin written under the fixture's test key, and the bundle made from the window (`prove_bundle.py make`).
# Without PB_K the draw is the fixture's FIXTURE_DRAW under its registered subset:4; with PB_K=K the windows register
# subset:K, the challenger logs its receipt of the window's registration, and Lean `flock-verify draw --registration`
# draws the registered law's units uniformly over the window's whole partition on its own coins.
# Then each drawn member is proved by C-Flock `prove --zk` (CPU) behind the Lean firewall, one outer proof at a time, and
# decided by Lean `verify --zk`. The negatives are the first session's one-bit proof edits, refused by Lean, and the tampered
# loop's member, stopped by the firewall and refused by Lean without it.
#   research run --on vy-nebius-cpu-4 --pool proofs --as proofs --project verity --source . --cwd clone \
#     -- bash benchmarks/one_stage/prove_bundle.sh [PB_K=8] [PB_MEMBERS=N] [PB_FLIPS=1] [PB_BINARIES=DIR] [PB_BUILD_ONLY=1]
#        [PB_IN=DIR | PB_IN_ART=art:… | PB_STEP=1] [PB_LEAN_JOBS=2]
# PB_IN (or PB_IN_ART, fetched) is a delivered tree instead of the torch loop's: a registered step (step-registration.json,
# receipt.json, draw.json, blocks/<index>/{window/, bundle.{json,bin}, tampered/?}: prove_bundle.py --step), else window/
# (Window.write's files), bundle.json and bundle.bin beside it with a live draw, and optionally tampered/; the layouts are
# PRF-172's. PB_STEP=1 with PB_K=K: the torch loop's step as such a tree, its K blocks drawn live over the step registration. PB_LEAN_JOBS: Lean verifies at
# once beside the one session proving.
# PB_BUILD_ONLY=1 builds the three binaries, keeps them in $OUT/bin with the stamp of the trees they're built from
# (verity/ml/flock and verity/core/service/flock), and stops. PB_BINARIES=DIR (on the node, e.g. that run's out/bin) uses them
# when DIR's stamp is this source's, and otherwise builds here: M3 builds before its window, not in it.
# Kept in $OUT: the summary, the logs, the session records and a live draw's receipt and record; never a window's private
# file or the bundle's payload.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH PYTHONDONTWRITEBYTECODE=1
{ lscpu | grep -E 'Model name|^CPU\(s\)'; nproc; free -g; git rev-parse HEAD; } > "$O/host.txt" 2>&1
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }
PY=$REPO/.venv/bin/python

STAMP=$(git rev-parse HEAD:verity/ml/flock HEAD:verity/core/service/flock 2>/dev/null | tr '\n' ' ')
[ -n "${STAMP// }" ] || { echo "prove-bundle: no git trees to stamp binaries with"; exit 4; }
log "prove-bundle: source $(git rev-parse HEAD), draw ${PB_K:+live, k }${PB_K:-FIXTURE_DRAW}, builds in $W"
if [ -n "${PB_BINARIES:-}" ] && [ "$(cat "$PB_BINARIES/stamp" 2>/dev/null)" = "$STAMP" ] \
    && [ -x "$PB_BINARIES/flock-circuit" ] && [ -x "$PB_BINARIES/flock-verify" ] && [ -x "$PB_BINARIES/flock-firewall" ]; then
  BIN=$PB_BINARIES/flock-circuit LEAN=$PB_BINARIES/flock-verify FWL=$PB_BINARIES/flock-firewall
  log "binaries: prebuilt in $PB_BINARIES, stamp $STAMP"
else
  # a finished run's out/bin is evicted once custody has preserved it (run-files.evicted.json), leaving only its stamp
  [ -n "${PB_BINARIES:-}" ] && log "binaries: $PB_BINARIES has not all three of this source's ($STAMP), so they're built here"
  (cd verity/core/service/flock && lake build flock-verify flock-firewall) > "$O/build-lean.log" 2>&1 &
  LP=$!
  mkdir -p "$O/m0-build"
  RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$W MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh > "$O/m0-build/build.log" 2>&1 \
    || { tail -40 "$O/m0-build/build.log"; exit 6; }
  BIN=$W/flock-gpu-link/flock/target/release/flock-circuit
  wait $LP || { tail -40 "$O/build-lean.log"; exit 7; }
  LEAN=$REPO/verity/core/service/flock/.lake/build/bin/flock-verify
  FWL=${LEAN%/*}/flock-firewall
fi
sha256sum "$BIN" "$LEAN" "$FWL" | tee "$O/binaries.sha256" | tee -a "$O/job.log"
if [ "${PB_BUILD_ONLY:-0}" = 1 ]; then
  mkdir -p "$O/bin" && cp "$BIN" "$LEAN" "$FWL" "$O/bin/" && echo "$STAMP" > "$O/bin/stamp" || exit 9
  log "build only: $O/bin, stamp $STAMP"
  exit 0
fi
export PYTHONPATH=$REPO:$REPO/verity/ml/flock/python:$REPO/integrations/torch_loop:$REPO/benchmarks/e2e_demo:$REPO/benchmarks/numerical/python
export RAYON_NUM_THREADS=$(nproc)
# flock-circuit's rayon pool keeps Rust's default 2 MiB thread stack, which a --zk prover can overflow on a many-core node
# (prove_path/run.sh)
export RUST_MIN_STACK=$((64 << 20))

STEP=
if [ -n "${PB_IN_ART:-}" ] || [ -n "${PB_IN:-}" ]; then
  # A delivered tree (PRF-172's M3), live draw required: a registered step (step-registration.json, receipt.json, draw.json,
  # blocks/<index>/{window/, bundle.{json,bin}, tampered/?}), else one bundle (window/, bundle.{json,bin}, tampered/?)
  IN=${PB_IN:-$W/in}
  if [ -n "${PB_IN_ART:-}" ]; then
    research data fetch "$PB_IN_ART" --to "$IN" > "$O/fetch.log" 2>&1 || { tail -20 "$O/fetch.log"; exit 10; }
  fi
  REG=$(find "$IN" -name step-registration.json | head -1)
  if [ -n "$REG" ]; then
    STEP=${REG%/*}
    log "delivered: ${PB_IN_ART:-$PB_IN}, registered step $STEP, $(ls "$STEP/blocks" 2>/dev/null | wc -l) blocks"
  else
    BUNDLE=$(find "$IN" -name bundle.json -not -path '*/tampered/*' | head -1)
    [ -n "$BUNDLE" ] && [ -d "${BUNDLE%/*}/window" ] || { echo "prove-bundle: no step-registration.json, nor bundle.json beside a window/, in $IN"; exit 10; }
    WINDOW=${BUNDLE%/*}/window TAMPERED=${BUNDLE%/*}/tampered LIVE=1
    [ -d "$TAMPERED" ] || TAMPERED=
    log "delivered: ${PB_IN_ART:-$PB_IN}, bundle $BUNDLE, tampered ${TAMPERED:-none}"
  fi
elif [ "${PB_STEP:-0}" = 1 ]; then
  log "step: the torch loop's step registered, drawn live (k ${PB_K:?PB_STEP needs PB_K}) over every (slot, layout unit), a bundle a drawn block"
  "$PY" benchmarks/one_stage/prove_bundle.py make --step --k "$PB_K" --lean "$LEAN" --scratch "$W" --out "$W/step" \
    > "$O/make.log" 2>&1 || { tail -30 "$O/make.log"; exit 8; }
  grep '^MAKE' "$O/make.log" | tee -a "$O/job.log"
  STEP=$W/step
else
  log "windows: the torch loop's (honest) and the tampered loop's, under the fixture's test key; then the bundle"
  "$PY" benchmarks/one_stage/prove_bundle.py make --scratch "$W" --out "$W/bundle" ${PB_K:+--k "$PB_K" --lean "$LEAN"} \
    > "$O/make.log" 2>&1 || { tail -30 "$O/make.log"; exit 8; }
  grep '^MAKE' "$O/make.log" | tee -a "$O/job.log"
  cp "$W/bundle/receipt.json" "$W/bundle/draw.json" "$W/bundle/draw.log" "$O/" 2>/dev/null
  BUNDLE=$W/bundle/bundle.json WINDOW=$W/window TAMPERED=$W/tampered LIVE=${PB_K:+1}
fi
if [ -n "$STEP" ]; then
  cp "$STEP/step-registration.json" "$STEP/receipt.json" "$STEP/draw.json" "$O/" 2>/dev/null
  log "prove: each drawn block held to the registered step, then proved, then the negatives"
  "$PY" benchmarks/one_stage/prove_bundle.py --step "$STEP" \
    --bin "$BIN" --lean "$LEAN" --firewall "$FWL" --out "$W/pb" ${PB_FLIPS:+--flips $PB_FLIPS} \
    --lean-jobs "${PB_LEAN_JOBS:-1}" > "$O/prove_bundle.log" 2> "$O/prove_bundle.err"
else
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1]))["draw"]; print(json.dumps({k: d.get(k) for k in ("source","law","receipt","units")}, sort_keys=True))' \
    "$BUNDLE" > "$O/bundle-draw.json" 2>&1
  log "prove: the bundle's drawn members, then the negatives"
  "$PY" benchmarks/one_stage/prove_bundle.py --bundle "$BUNDLE" --window "$WINDOW" ${TAMPERED:+--tampered "$TAMPERED"} \
    --bin "$BIN" --lean "$LEAN" --firewall "$FWL" --out "$W/pb" ${PB_MEMBERS:+--members $PB_MEMBERS} ${LIVE:+--live} ${PB_FLIPS:+--flips $PB_FLIPS} \
    --lean-jobs "${PB_LEAN_JOBS:-1}" > "$O/prove_bundle.log" 2> "$O/prove_bundle.err"
fi
rc=$?
grep '^PROVEBUNDLE\|^DRAWN \|^CHECK \|^VERDICT ' "$O/prove_bundle.log" | tee -a "$O/job.log"
[ $rc = 0 ] || tail -30 "$O/prove_bundle.err"
cp "$W/pb/summary.json" "$O/" 2>/dev/null
(cd "$W/pb" && { find honest blocks tampered -type f 2>/dev/null \( -name 'session.json' -o -name '*.log' -o -name 'draw.json' \) -size -8M -print0
  find flipped -maxdepth 2 -name zk_mutants.json -print0; } | xargs -0 -r -I{} cp --parents {} "$O/")
log "prove-bundle: done, rc $rc"
exit $rc
